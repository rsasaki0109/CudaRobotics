# CudaRobotics Plan / Handoff (for Codex / Claude)

Last updated: 2026-05-24 JST

This document is the long-form handoff for the next coding agent (Codex).
It captures: (1) where the repo is right now, (2) what was just done over
the 2026-05-21 → 2026-05-24 sprint, (3) house rules and conventions, (4)
known sharp edges and lessons learned from the last few attempts, and (5)
a prioritised menu of candidate next tasks with enough specificity that a
fresh agent can pick one and start.

Mainline was in sync with `origin/master` at commit
`1c06801 Add GPU SfM mini demo` before the current GPU PCG feature
branch.

There were no open GitHub PRs at the start of the GPU PCG branch.
Parked local branches remain for `feat/gaussian-splat-renderer`
(checked out in `/tmp/CudaRobotics-gaussian`) and `feat/repro-report`; treat them as
separate parked work, not active blockers for new feature work.

---

## TL;DR for the impatient

- Repo is in a clean, "between sprints" state. **No mandatory cleanup
  before starting new work.**
- The 2026-05-21..24 sprint added 39 PRs (#40 → #78): ESDF JFA (2D + 3D),
  3D voxel map, massive collision check, realistic 3D LiDAR, ROS2 nodes,
  Bundle Adjustment, 2D pose-graph SLAM backend, LiDAR SLAM frontend,
  Multi-robot planner, Gaussian Splatting, NeRF volume, diffusion planner,
  online SLAM, shared headers refactor, **and the full 2D/3D NDT + 2D/3D
  GICP scan matching family**, followed by **multi-resolution NDT 3D** and
  **GPU Hungarian-class assignment**, then **GPU CMA-ES**. PR #75 adds
  **GPU MCTS kinodynamic planning**. PR #76 adds **GPU assignment
  tracking**. PR #77 adds **10K-agent GPU crowd swarm**. PR #78 adds
  **GPU SfM mini**.
- PR #78 / branch `feat/gpu-sfm-mini` added a compact multi-view geometry
  demo: 2048 synthetic ORB-like features across 4 views, GPU descriptor
  matching, stereo triangulation, and point-only BA.
- PR #79 / branch `feat/gpu-pcg` adds a generic CSR Jacobi-PCG sparse SPD
  solver demo: 262K unknowns, 1.31M nonzeros, 33 iterations, 13.4x vs CPU.
- The "scan matching 4 siblings" are NDT 2D (#67), NDT 3D (#68), GICP 2D
  (#69), GICP 3D (#70). All present and merged.
- Follow-up #71 adds coarse-to-fine NDT 3D and fixes the old #68 outlier
  frame; #72 opens the combinatorial optimisation chapter with batched
  64x64 dense assignment; #74 adds GPU CMA-ES black-box optimisation; #75
  adds root-parallel GPU MCTS; #76 adds gated multi-object tracking.
- Two attempts during the sprint were aborted: **3D pose-graph SLAM**
  (Gauss-Newton step direction uphill at GT, root cause not identified,
  file deleted) and **LiDAR SLAM with constant-velocity ICP init**
  (Lissajous reversal made drift 9x worse, change reverted).
- Recommended next directions are listed at the bottom; safe defaults
  after this branch are a finite-difference verified retry of 3D
  pose-graph SLAM, **GPU EM clustering**, or a mechanical shared-header
  cleanup.

---

## Repo State (2026-05-24)

- **Main branch**: `master`, currently at `1c06801` before the GPU PCG
  feature branch.
- **gh-pages branch**: hosts gif assets referenced from `readme.md`.
  Files live at the branch ROOT (not `gif/`), so the URL
  `https://rsasaki0109.github.io/CudaRobotics/<name>.gif` resolves.
  Every new gif must be pushed there to render in the readme.
- 125 CUDA source files (`src/*.cu`), 15 C++ files (`src/*.cpp`) on the
  GPU PCG branch.
- Build:
  ```bash
  rtk bash -lc 'cd build && cmake .. && make -j$(nproc)'
  # single target:
  rtk bash -lc 'cd build && make <target_name> -j$(nproc)'
  ```
- Run binaries from repo root (they look for / write to `gif/`):
  ```bash
  rtk ./bin/<target>
  ```
- Shared CUDA headers landed in #66, under `include/`:
  - `cuda_check.cuh` — `CUDA_CHECK(...)` error wrapper macro.
  - `cuda_blas.cuh` — small device matrix helpers.
  - `cuda_video.h` — `cudabot::avi_to_gif(avi_path, gif_path, fps, width)`
    using ffmpeg palettegen + paletteuse. This is the canonical GIF
    pipeline.

---

## What Was Done (2026-05-21 → 2026-05-24)

Compact PR list. Format: `#PR  Title  | headline number`.

### 2026-05-21 — ESDF / voxel / collision burst
| PR | Title | Headline |
|---|---|---|
| #40 | DPF kidnap injection rate / ESS trigger ablation | Fixed 6% best, ESS-12% matches with 32% firing rate |
| #41 | GPU ESDF via Jump Flooding (2D) | **53,404x** per cell vs CPU |
| #42 | GPU 3D voxel map (log-odds occupancy) | 256³ scale |
| #43 | Massive parallel collision checker | **1,277x** per candidate (2D DDA) |
| #44 | Capability matrix + speedups in docs | readme `Capability matrix` table |
| #45 | ESDF-MPPI (JFA + bilinear lookup) | Distance-field aware MPPI cost |
| #46 | ROS2 esdf_node | GPU JFA ESDF as ROS2 component |

### 2026-05-22 — 3D + SLAM + multi-robot + visual research
| PR | Title | Headline |
|---|---|---|
| #47 | 3D ESDF via Jump Flooding (3D JFA) | **86,613x** per voxel |
| #48 | ROS2 voxel_node | GPU 3D log-odds voxel map as ROS2 node |
| #49 | Massive-parallel RRT* rewire | CPU 2K → CUDA 200K nodes, **62x rewire** |
| #50 | GPU perf regression CI stub | scripts/ + workflow |
| #51 | PF with ESDF-lookup observation | PF likelihood from precomputed ESDF |
| #52 | Realistic 3D LiDAR simulator (5 physical effects) | noise + divergence + multi-path + reflectivity + rolling shutter |
| #53 | GPU bundle adjustment (GN + Schur + Jacobi-PCG) | initial: 200 poses × 800 LM, ~0.5 ms/iter |
| #54 | GPU 2D LiDAR SLAM frontend (scan-to-scan ICP + log-odds map) | 0.68 ms/frame |
| #55 | PF on realistic LiDAR obs (Gaussian / Cauchy / learned MLP) | 3-panel comparison |
| #56 | Visibility-aware Diff-MPPI demo | baseline vs −W·V(x,y) visibility |
| #57 | Add GIFs for recent demos in readme | gh-pages catch-up |
| #58 | GPU 2D pose-graph SLAM backend (GN + Jacobi-PCG) | RMSE 4.88 → 0.56 m |
| #59 | GPU 3D Gaussian Splatting renderer | isotropic, alpha-composite, 0.94 ms/frame for ~1k Gaussians |
| #60 | GPU massive multi-robot planner | 200 robots, parallel BF distance fields + flow |
| #61 | Differentiable end-to-end SLAM (Adam-tuned robust noise scale) | DPF + diff-update + Adam |

### 2026-05-23 — perception + SLAM + scan matching family
| PR | Title | Headline |
|---|---|---|
| #62 | Scale up GPU BA | **1000 poses × 8000 landmarks**, 60k obs, 0.5 ms/iter |
| #63 | GPU online SLAM (sliding-window backend + iSAM-style global on loop) | 1.7 ms/step, RMSE 3.0 → 0.4 m |
| #64 | GPU NeRF-style volumetric renderer | 720×480, 128 samples/ray, 0.83 ms/frame |
| #65 | GPU diffusion-based motion planner | 512 trajectories × 64 waypoints, 120 Langevin steps, 0.03 ms/step |
| #66 | Extract shared CUDA helpers to include/ | `cuda_check / cuda_blas / cuda_video` |
| #67 | GPU NDT 2D scan matching | Newton on NDT grid, 0.54 ms/scenario, ~0.02 m typical |
| #68 | GPU NDT 3D point cloud registration | 16³ voxel NDT + 6-DOF GN on SE(3), 6.7 ms/scenario, ~0.03 m typical |
| #69 | GPU GICP 2D scan matching | per-point cov + NN match, 1.9 ms/scenario, ~0.08 m typical |
| #70 | GPU GICP 3D point cloud registration | Cardano eigendecomp + 6-DOF GN on SE(3), 4.7 ms/scenario, ~1 mm typical |

### 2026-05-24 — scan-matching polish + combinatorial optimisation
| PR | Title | Headline |
|---|---|---|
| #71 | GPU multi-resolution NDT 3D | 8x8x4 coarse -> 16x16x6 fine, 9.47 ms/frame, avg fine 0.0155 m / 0.0072 rad |
| #72 | GPU Hungarian assignment | 512 batched 64x64 dense assignments, 0.082 ms/batch, **158.2x** vs CPU Hungarian |
| #73 | Update handoff plan | Refreshed `plan.md` after #72 |
| #74 | GPU CMA-ES optimiser demo | 3 x 32768 candidates x 10D, 0.025 ms/generation eval, **1254x** objective eval |
| #75 | GPU MCTS planner | 64 scenes x 4096 rollouts x 48 horizon, 1.82 ms/plan, **712x** vs CPU |
| #76 | GPU assignment tracking | 128 scenes x 48 tracks x 72 detections, 0.093 ms/update, **14.0x** vs CPU |
| #77 | GPU crowd swarm | 10,000 agents, 120x80 uniform grid, 0.275 ms/step, **105x** vs CPU |
| #78 | GPU SfM mini | 2048 features x 4 views, match + point BA, **217x** vs CPU |

(39 merged PRs in 4 days; cadence was sustained because each demo was a single
~500-700 LOC `.cu` file plus a few lines in `CMakeLists.txt` and
`readme.md`.)

### Bigger architectural things landed in this sprint
- **Shared CUDA headers (`include/`)** — #66. New `.cu` files should
  `#include "cuda_check.cuh"` and `#include "cuda_video.h"` instead of
  re-defining the wrappers locally. Older files have not been
  back-migrated yet — that is a small mechanical cleanup waiting in the
  open threads list.
- **`readme.md` Sensors / perception section** now has the scan matching
  family plus the multi-resolution NDT 3D tile. The Planning / Control
  section has Hungarian assignment, CMA-ES, MCTS, assignment tracking, and
  crowd swarm tiles, and the capability matrix includes the swarm /
  assignment / tracking / optimisation / MCTS rows.

### Attempts that did NOT land (for context, so we do not repeat)
- **3D pose-graph SLAM** — Wrote `src/gpu_pose_graph_slam_3d.cu` (~700
  LOC, SE(3) GN + Jacobi-PCG, right-perturbation Jacobians, Levenberg-
  Marquardt). At GT initialisation the GN step direction was uphill
  (cost 2525 → 293199) regardless of sign of the b vector. Tried sign
  flips, did not isolate the H-matrix construction bug. **File was
  deleted.** Verified Jacobians but not via finite differences — that
  is the recommended next move if anyone retries this.
- **LiDAR SLAM with constant-velocity ICP init** — Edit to
  `src/gpu_lidar_slam.cu` to cache prev `(dx, dy, dyaw)` and use it as
  ICP init. The demo trajectory is a Lissajous figure whose velocity
  reverses at every sinusoid peak, so the CV init was systematically
  pushed into the wrong basin. Drift went 6.5 m → 56.6 m (9x worse).
  Halving the CV gain still made it worse (8.7 m). **Reverted.**
  A real CV-init benefit would need a less adversarial trajectory.

---

## House Rules

These are the conventions every PR in this sprint followed. Skipping
them in one PR is the kind of avoidable rework that wastes a session.

- **Git**: NO `Co-Authored-By` lines in commit messages. Commits are
  user-authored only. This is set in `~/.claude/CLAUDE.md` global rules.
- **Shell commands**: this checkout has AGENTS.md -> `/home/sasaki/.codex/RTK.md`.
  Prefix shell commands with `rtk`; use `rtk proxy <cmd>` when RTK filtering
  hides detail or breaks pathspec-style commands.
- **PR body**: NO "Generated with Claude Code" / "🤖" / any AI
  attribution footer. Per global rule.
- **PR body style**: Short Summary, a results table, a Test plan
  checklist, and the deployed GIF inline as
  `![demo](https://rsasaki0109.github.io/CudaRobotics/<name>.gif)`.
- **Language**: Code comments and PR titles are English. Commit
  messages are English. Chat with the user is 日本語.
- **License**: repo is MIT. If you port from an LGPL/GPL upstream, write
  from the paper, not from upstream source. Annotate attribution in the
  source header (see `src/expansion_reset_mcl.cu` for the form).
- **CUDA compile options**: every new target needs
  `target_compile_options(<name> PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)`
  in `CMakeLists.txt`. The four scan matching targets are good copy-paste
  starting points.
- **One demo = one `.cu` file**. Do not split a demo into multiple files
  unless there is a reason. Helpers shared across demos go in
  `include/`. Demo-local helpers stay inside `namespace cudabot {}` in
  the `.cu`.
- **GIF size budget**: keep each GIF ≤ ~3 MB. The `cuda_video::avi_to_gif`
  helper produces moderately compressed output. For tighter compression
  use ffmpeg manually with palettegen + paletteuse:
  ```bash
  rtk ffmpeg -y -i gif/<name>.avi \
    -vf "fps=8,scale=460:-1:flags=lanczos,split[s0][s1];\
         [s0]palettegen=max_colors=128[p];\
         [s1][p]paletteuse=dither=bayer:bayer_scale=5" \
    gif/<name>.gif
  ```
  Lower fps / smaller scale = smaller GIF.
- **Comparison gifs at >100K samples**: do not use `cv::circle` per
  sample in the visualisation loop — it is the bottleneck. Use direct
  pixel splatting (see `src/comparison_lidar_sim.cu::draw_dense_hits`).
  Write a 2×2 block per sample so points survive ffmpeg lanczos
  downscale + gif palette quantisation.

### gh-pages deploy workflow (this is the only fiddly part)

GIFs must live at the **root** of the `gh-pages` branch, not `gif/`.
The procedure that survived multiple sprints without breaking:

```bash
# 1. Move the GIF out of the working tree (so checkout doesn't clobber)
rtk mv gif/<name>.gif /tmp/<name>.gif
rtk cp /tmp/<name>.gif /tmp/<name>_keep.gif       # extra safety copy

# 2. Stash any other untracked files
rtk git stash push -u -m "untracked gifs"

# 3. Switch to gh-pages, copy the GIF in, commit, push
rtk git checkout gh-pages
rtk cp /tmp/<name>_keep.gif <name>.gif
rtk git add <name>.gif
rtk git commit -m "Add <name>.gif"
rtk git push origin gh-pages

# 4. Switch back and restore
rtk git checkout <feature-branch>
rtk git stash pop
rtk mv /tmp/<name>.gif gif/<name>.gif
```

GitHub Pages serves the new file within ~30 seconds typically.

**Failure mode seen**: skipping step 1 and doing `cp gif/foo.gif foo.gif`
while still on the feature branch (because the `checkout gh-pages` failed
silently due to a dirty tree) creates an accidental commit on the feature
branch. Recovery is `rtk git reset --soft HEAD~1`, then redo properly.

---

## Lessons learned this sprint (sharp edges)

These are the non-obvious things that ate time. Read them before starting
any new scan-matching / SLAM / optimisation work.

1. **NDT basin width is dominated by cell size × variance regularisation.**
   First NDT 2D attempt used 40×40 cells with reg `1e-3` and got error
   `1.34 m / 0.35 rad` (basin too narrow). 20×20 cells + reg `0.15`
   gave `0.107 m / 0.054 rad`. NDT 3D first tried 32×32×12 + reg `0.03`
   → mediocre; ended at 16×16×6 + reg `0.4` → `0.026 m / 0.010 rad`.
   **For demos, coarser cells + higher reg = wider basin but coarser
   final accuracy. Tune both, not just one.**

2. **GICP per-point covariance needs eigendecomposition.** In 2D it is
   closed-form via the 2×2 quadratic; in 3D use Cardano's trig form for
   3×3 symmetric (see `smallest_eigvec_3x3_sym` in
   `src/gpu_gicp_3d.cu`). The smallest eigenvalue's eigenvector is the
   surface normal; the regularised cov is `I - (1 - eps) * n n^T`.
   GICP 3D ran at near machine precision (~1 mm) because source and
   target are the SAME cloud rotated + noise. A more honest demo would
   raycast scans from different viewpoints (see "view-dependent
   occlusion" caveat in `gpu_gicp_3d.cu` PR body).

3. **6-DOF Cholesky on a 6×6 SPD H with adaptive LM damping is the right
   solver for NDT 3D / GICP 3D / pose-graph SE(3) backends.** The
   `cholesky_solve_6 + H_OFF` pair in `src/gpu_ndt_3d.cu` and
   `src/gpu_gicp_3d.cu` is identical and should probably be lifted into a
   shared header next time someone touches this code.

4. **For multi-iteration GN, ALWAYS implement adaptive LM** (`lambda *= 0.5`
   on improvement, `*= 4` on rejection, clamped). The fixed-`lambda`
   version of NDT 3D had higher variance across runs.

5. **Brute-force NN is fine for N ≤ 2500 in 3D, N ≤ 1000 in 2D per
   iteration** on a modest GPU. Above that, use a uniform grid index
   (see `accum_grid_kernel` in `gpu_ndt_3d.cu` for the bucketing pattern)
   or KD-tree. GICP 3D used N=2500 specifically to keep N² matching
   tractable.

6. **Right-perturbation SE(3) update**: `t_new = t + delta_t` (WORLD
   translation), `R_new = R * Exp(delta_w)` (BODY rotation). Used
   consistently in `src/gpu_ndt_3d.cu`, `src/gpu_gicp_3d.cu`, the BA
   stack, and the 2D pose-graph SLAM backend. Mixing this convention
   with a left-perturbation update is the most common source of "GN step
   goes uphill" bugs.

7. **Auto-mode classifier blocks merges and force-pushes without
   explicit authorisation.** When the user says "マージ！" / "merge!" /
   "iiyo!" it is authorisation. Without that, do not act.

---

## Open Threads (parked but not abandoned)

### A. Mechanical cleanup
- **Refresh the `readme.md` Headline benchmarks table** for the newer
  demos: NeRF, online SLAM, diffusion planner, multi-resolution NDT 3D,
  GICP 2D/3D, and Hungarian assignment. CMA-ES, MCTS, assignment
  tracking, crowd swarm, SfM mini, and PCG are now present, but the
  lower headline table still lags several perception / SLAM demos.
- **Back-migrate older `.cu` files to use `include/cuda_check.cuh`** and
  friends from #66, instead of their private wrappers. Mechanical, low
  risk, good first task.
- **Lift `cholesky_solve_6` + `H_OFF` + `so3_exp` + `mat3_mul` into a
  shared header** (`include/se3_helpers.cuh` perhaps). Currently
  duplicated across `gpu_ndt_3d.cu`, `gpu_ndt_3d_multires.cu`, and
  `gpu_gicp_3d.cu`, and would be needed in any retried 3D pose-graph
  SLAM.

### B. New algorithm candidates (ranked rough order, see "Recommended Next" below)

1. **Retry 3D pose-graph SLAM** — this time with finite-difference
   verification of the Jacobians + H matrix before launching GN. The
   abandoned file was ~700 LOC; expect ~1000 LOC including the FD scaffold.
   The unidentified H-matrix bug is the only thing blocking this.
2. **GPU EM clustering (GMM)** — classical clustering demo. ~400 LOC.
3. **GPU diffusion policy / behaviour cloning** — extension of #65
    (motion planner) into a learned planner. ~800 LOC.

### C. Older items still parked (carried over from previous handoff)
- DPF research line — from-scratch tracking-loss MLP training, harder
  scenes, EKF/AMCL accuracy comparison. See previous plan.md sections.
- Topology bench Day 4+ — failure taxonomy CSV expansion, Day 5
  consolidated report.

---

## Recommended Next Session

After GPU CMA-ES, GPU MCTS, assignment tracking, crowd swarm, PR #78 GPU
SfM mini, and PR #79 GPU PCG, the natural next move depends on user
goal:

- **Hard but high value**: retry 3D pose-graph SLAM with FD-verified
  Jacobians (B1). This unblocks any future global SLAM work and
  completes the 2D / 3D pose-graph pair (only the 2D backend exists,
  PR #58).
- **Tying off loose ends**: refresh the lower headline benchmark table
  and back-migrate to shared headers (Open Threads A). One PR,
  mechanical, removes drift.
- **Small classical demo**: GPU EM clustering (B2), if the user wants
  another compact algorithm PR rather than backend work.

If unsure after this branch, start with a mechanical shared-header cleanup
if they want a low-risk maintenance PR, or GPU EM clustering if they want
another compact visual algorithm.

Suggested starting commands:

```bash
rtk git switch -c feat/gpu-pose-graph-3d-v2    # B1
rtk git switch -c feat/gpu-em-gmm              # B2
rtk git switch -c chore/shared-cuda-cleanup    # A: cleanup
```

---

## File Map (key entry points)

### Top-level
- `readme.md` — top of repo. Top showcase grid, Capability matrix,
  SLAM / Multi-view geometry, Planning / Control, Differentiable /
  learning, Sensors / perception. New demos land in the most relevant
  section.
- `CLAUDE.md` — project rules. Applies to Codex too.
- `plan.md` — this file.
- `CMakeLists.txt` — every new `.cu` needs an `add_executable` +
  `target_link_libraries(${OpenCV_LIBS})` + `target_compile_options(...
  --expt-relaxed-constexpr)` triplet. Pattern is uniform — find an
  existing target and copy.

### Shared headers (`include/`)
- `cuda_check.cuh` — `CUDA_CHECK(...)` macro (use this, not custom
  wrappers).
- `cuda_blas.cuh` — small matrix helpers.
- `cuda_video.h` — `cudabot::avi_to_gif(avi, gif, fps, width)`.
- `autodiff_engine.cuh` — dual-number forward-mode autodiff used by
  Diff-MPPI and DPF.
- `gpu_mlp.cuh` — flat-array MLP, used by Neural SDF, Neuroevolution,
  DPF MLP observation.

### Scan matching family (this sprint's centrepiece)
- `src/gpu_ndt_2d.cu` (PR #67) — 20×20 NDT grid, 720 rays, 3-DOF GN.
- `src/gpu_ndt_3d.cu` (PR #68) — 16×16×6 voxel NDT, 16k pts, 6-DOF GN
  on SE(3) with Cholesky + LM + step capping. **Reference for any
  future SE(3) GN backend.**
- `src/gpu_ndt_3d_multires.cu` (PR #71) — coarse 8x8x4 -> fine
  16x16x6 NDT 3D. Use this version when basin width matters.
- `src/gpu_gicp_2d.cu` (PR #69) — k=10 per-point cov + brute-force NN
  + 3-DOF GN. 2x2 eigendecomp closed-form.
- `src/gpu_gicp_3d.cu` (PR #70) — k=15 per-point cov via Cardano 3×3
  eigendecomp + brute-force NN + 6-DOF GN. Reuses NDT 3D's Cholesky
  scaffold.

### SLAM / multi-view geometry (this sprint)
- `src/gpu_sfm_mini.cu` (PR #78) — 2048 ORB-like feature tracks
  across 4 views, GPU brute-force descriptor matching, stereo
  triangulation, and fixed-camera point-only BA.
- `src/gpu_bundle_adjustment.cu` (PR #62) — 1000 poses × 8000 LM,
  Schur + Jacobi-PCG.
- `src/gpu_lidar_slam.cu` (PR #54) — scan-to-scan ICP + log-odds map.
- `src/gpu_pose_graph_slam.cu` (PR #58) — 2D GN + Jacobi-PCG. **3D
  version does not exist** (see Lessons #6, Threads B2).
- `src/gpu_online_slam.cu` (PR #63) — sliding-window W=60 + iSAM-style
  global pass on loop closure.

### Solver / infrastructure
- `src/gpu_pcg_solver.cu` (PR #79) — generic CSR Jacobi-PCG for
  sparse SPD systems; 262K unknowns, 1.31M nonzeros, convergence GIF, and
  direct CPU PCG comparison.

### Visualisation / rendering
- `src/gpu_gaussian_splatting.cu` (PR #59) — isotropic alpha-composite.
- `src/gpu_nerf_volume.cu` (PR #64) — volumetric ray-march renderer.

### Planning / control
- `src/gpu_diffusion_planner.cu` (PR #65) — 512 trajectories × 64
  waypoints, 120 Langevin steps.
- `src/gpu_hungarian_assignment.cu` (PR #72) — 512 batched 64x64 dense
  assignments solved with a shared-memory parallel auction kernel and
  checked against CPU Hungarian.
- `src/gpu_assignment_tracking.cu` (PR #76) — 128 batched
  multi-object tracking scenes with gated assignment, missed detections,
  and clutter.
- `src/gpu_cma_es.cu` (PR #74) — 3 objective families x 32768 candidates
  x 10D, host covariance update with GPU objective evaluation.
- `src/gpu_mcts_planner.cu` (PR #75) — root-parallel MCTS for 64
  kinodynamic planning scenes, 4096 rollouts x 48 horizon per scene.
- `src/gpu_multi_robot_planner.cu` (PR #60) — 200 robots, parallel BF
  distance fields.
- `src/gpu_crowd_swarm.cu` (PR #77) — 10,000 boids with
  uniform-grid neighbour search, group goals, and obstacle avoidance.
- `src/visibility_mppi.cu` (PR #56) — visibility-aware MPPI variant.
- `src/esdf_mppi.cu` (PR #45) — ESDF-aware MPPI variant.

### Sensors / perception
- `src/comparison_lidar3d_realistic.cu` (PR #52) — 5 physical effects.
- `src/comparison_voxel_map.cu` / 3D voxel (PR #42).
- `src/comparison_esdf.cu` / 2D JFA (PR #41).
- `src/comparison_esdf_3d.cu` / 3D JFA (PR #47).
- `src/comparison_collision_check.cu` (PR #43) — 1M segments / scan.
- `src/comparison_rrtstar_rewire.cu` (PR #49) — 200K nodes.

### Earlier (carried over)
- `src/diff_pf.cu` / `src/diff_pf_mlp.cu` — DPF base + MLP obs model.
- `src/diff_e2e_slam.cu` (PR #61) — differentiable end-to-end SLAM.
- `src/benchmark_diff_mppi.cu` — 12-planner sweep + topology suite.

### ROS2 (`ros2_ws/`)
- `src/esdf_node.cpp` (PR #46) — GPU JFA ESDF node.
- `src/voxel_node.cpp` (PR #48) — GPU 3D log-odds voxel map node.
- Both build via `colcon build --packages-select cuda_robotics` from
  `ros2_ws/`. Not run in normal sessions; only touch if user asks for
  ROS2 work specifically.

---

## Quick command reference

```bash
# Build everything
rtk bash -lc 'cd build && cmake .. && make -j$(nproc)'

# Single target
rtk bash -lc 'cd build && make gpu_ndt_3d -j$(nproc)'

# Run from repo root (binaries write to gif/)
rtk ./bin/gpu_ndt_3d

# Regenerate a tighter GIF
rtk ffmpeg -y -i gif/<name>.avi \
  -vf "fps=8,scale=460:-1:flags=lanczos,split[s0][s1];\
       [s0]palettegen=max_colors=128[p];\
       [s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  gif/<name>.gif

# Push GIF to gh-pages (see "House Rules" for the full safe procedure)
rtk mv gif/<name>.gif /tmp/<name>.gif
rtk git stash push -u -m "untracked"
rtk git checkout gh-pages
rtk bash -lc 'cp /tmp/<name>.gif <name>.gif && git add <name>.gif'
rtk bash -lc 'git commit -m "Add <name>.gif" && git push origin gh-pages'
rtk git checkout <feature-branch>
rtk git stash pop
rtk mv /tmp/<name>.gif gif/<name>.gif

# Open PR (gh CLI is at ~/.local/bin/gh, already authed)
rtk git push -u origin <feature-branch>
rtk bash -lc 'export PATH="$HOME/.local/bin:$PATH"; gh pr create --title "Add ..." --body-file /tmp/pr-body.md'

# Merge (only after user authorisation)
rtk bash -lc 'export PATH="$HOME/.local/bin:$PATH"; gh pr merge <N> --squash --delete-branch'
```

---

## Environment notes

- **CUDA**: driver / library, runtime healthy. No
  driver/library mismatch as of this writing.
- **GPU**: CUDA-capable (single GPU). Compute capability ≥ 7.5.
- **OpenCV 4.5+** required for `cv::VideoWriter` and OpenCV drawing
  primitives.
- **Eigen 3** required.
- **CMake ≥ 3.18, CUDA Toolkit ≥ 12.0, C++14 / CUDA C++14.**
- **ffmpeg** must be on PATH for `cuda_video::avi_to_gif`.
- **GitHub CLI** at `~/.local/bin/gh`, authenticated.
- **Jira CLI** at `~/.local/bin/jira`. Not used in normal CudaRobotics
  work; only relevant if user explicitly asks.

---

End of handoff. Good hunting.
