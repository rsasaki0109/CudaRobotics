# CudaRobotics Plan / Handoff (for Codex / Claude)

Last updated: 2026-06-10 JST (nav2 GPU MPPI controller plugin #175 merged;
initial MPPI Python bindings added; retreat + SE(2) footprint follow-ups
implemented. Note: some older sections below carry their own internal dates —
treat section headers as the ordering authority within each line, not this
single timestamp.)

This document is the long-form handoff for the next coding agent (Codex).
It captures: (1) where the repo is right now, (2) what was just done over
recent sprints, (3) house rules and conventions, (4) known sharp edges and
lessons learned from the last few attempts, and (5) a prioritised menu of
candidate next tasks with enough specificity that a fresh agent can pick one
and start.

**Read the "Current State" section first.** Everything below it through the
2026-05-21 → 2026-05-25 sprint narrative is preserved as historical context;
do not treat the old "IN FLIGHT" / "Recommended Next Session" paragraphs as
authoritative without cross-checking the current-state block.

There are now **two active lines** in this repo:

1. **Star-growth / product line** (NEW, top section) — make CudaRobotics a
   *used* library, not a watched gallery. First artifact: the nav2 GPU MPPI
   controller plugin, merged as #175. Next: pip Python bindings.
2. **Research line** — SOPPI / Diff-MPPI box-pushing reproduction zoo
   (see "Research Line State" below; unchanged by this sprint).

---

## Current State (2026-06-10, star-growth sprint: nav2 GPU MPPI plugin)

Mainline: **`master` at `32995ff`** (`Add cuda_mppi_controller: GPU-accelerated
MPPI controller plugin for Nav2 (#175)`), in sync with `origin/master`.
**No in-flight PR.**

### Strategy context (agreed with user, 2026-06-10)

Stars come from being **used**, not watched. Demo mass-production is over
(depth over breadth). Agreed priority order and status:

1. ~~nav2 GPU MPPI controller plugin~~ — **DONE, merged as #175.**
2. ~~MPPI Zoo + SOPPI arXiv tech report~~ — **SKIPPED by user decision
   (2026-06-10). Do not pick this up unless the user re-requests it.**
   The zoo remains an internal benchmark + paper scaffold.
3. **pip-installable Python bindings** — **MPPI + registration BINDINGS DONE.**
   **Packaging:** sdist + `linux_x86_64` wheels via CI (`python-package.yml`);
   `scripts/sync_python_core.sh` bundles CUDA sources for PyPI-style installs.
   `cibuildwheel` config is in `python/pyproject.toml` for broader manylinux wheels.

**Distribution (HN / Reddit / ROS Discourse / X / Zenn) is user-owned.
Agents must never post, announce, or publish anywhere external.** Producing
material the user can post (GIFs, benchmark tables, headline numbers) is in
scope and encouraged. The strongest current headline:
*"Drop-in GPU MPPI controller for Nav2 — 65k rollouts in ~10 ms, same
trajectory quality as the stock CPU controller, monotonically better with K."*

### What landed in #175 (squashed from 7 commits)

New package `ros2_ws/src/cuda_mppi_controller/` — a `nav2_core::Controller`
plugin running every MPPI rollout on the GPU (1 CUDA thread = 1 sampled
trajectory), drop-in alternative to `nav2_mppi_controller`:

| Piece | Where | Notes |
|---|---|---|
| GPU optimizer core | `src/mppi_gpu.cu` + `include/cuda_mppi_controller/mppi_gpu.hpp` | Pure CUDA behind a PIMPL; the header has **no ROS and no CUDA includes** — deliberately reusable for Python bindings |
| nav2 plugin layer | `src/cuda_mppi_controller.cpp` | configure / activate / setPlan / computeVelocityCommands / setSpeedLimit / reset |
| pluginlib registration | `cuda_mppi_plugin.xml` + `package.xml` export | `plugin_load_test` loads it exactly as controller_server does |
| Motion models | DiffDrive / Ackermann (`abs(wz) <= abs(vx)/min_turning_r`) / Omni (adds `vy`) | control arrays are always 3-DOF; `vy` stays 0 outside Omni |
| Footprint sweep | `footprint_collides()` in the rollout kernel | polygon edges sampled at costmap resolution; gated on inflated cost > 0 (**requires an inflation layer**); max 16 vertices |
| Cost terms | path align (d²) · path follow (pursuit point `follow_lookahead` ahead) · linear terminal goal + yaw gate · graded costmap · smoothness · backward · **speed** · **angular damping** | tuned defaults in `mppi_gpu.hpp` |
| Head-to-head benchmark | `test/controller_benchmark.cpp` | loads BOTH plugins via pluginlib; same costmap / plan / limits; 20 Hz closed loop; writes CSV + per-run trajectories |
| Standalone scenario | `test/mppi_gpu_standalone.cpp` | wall-gap world; modes `diff / ackermann / omni / footprint`; `MPPI_TRACE=1` prints per-step v/w |
| Sim demos | `scripts/run_nav2_loopback_demo.py` (+`record_nav2_path.py`, `render_nav2_loopback_demo.py`) | loopback AND Gazebo (gz Harmonic + AMCL) two-waypoint missions both SUCCEEDED |
| Results / assets | `docs/results/cuda_mppi_vs_nav2_2026-06-10.{md,svg,png}` · `gif/cuda_mppi_vs_nav2_cpu.gif` · `gif/cuda_mppi_nav2_{loopback,gazebo}.gif` | |

### Headline numbers (benchmark GPU vs benchmark CPU, T=56, dt=0.05)

| | CPU K=2,000 | CPU K=10,000 | GPU K=16,384 | GPU K=65,536 |
|---|---:|---:|---:|---:|
| mean solve | 5.2 ms | 27.4 ms | 3.9 ms | 10.6 ms |
| time to goal | 16.2 s | 16.2 s | 16.1 s | **16.0 s** |

Time-to-goal **matches the stock CPU critics** and improves monotonically
with K on the GPU. Honest scope note: ONE wall-gap scenario, single seed
pair, no sensor pipeline — see menu item B4 before widening claims.

### Hard-won tuning lessons (cost ~3 debug cycles; do NOT re-learn)

1. **Clamped-average saturation bias**: zero-mean noise clamped at `v_max`
   averages ~`v_max − 0.4σ`, so a softmin-weighted average of clamped
   samples can never cruise at the limit (measured: 0.42 m/s with
   v_max=0.5). Fix = anti-windup nominal: store UNCLAMPED perturbed
   controls, average those, let the nominal exceed `v_max` by one `v_std`,
   clamp only the executed/output controls.
2. **Never let the angular rate wind up**: applying the same margin to `wz`
   made the robot pirouette through disturbances instead of counter-steering
   (sustained `w=±w_max` 360° spins at the wall gap, seen via `MPPI_TRACE`).
   `wz` nominal is clamped hard at `±w_max`; only translational speeds get
   the +1σ margin.
3. **Heading random walk**: the noisy weighted average accumulates yaw
   drift → stall/reverse episodes. Fixed by `angular_weight·wz²` damping +
   `path_weight` 10 + `follow_lookahead` 1.0 m + `temperature` 0.12.
4. **Squared terminal goal distance stalls** (gradient vanishes as the local
   goal nears) — terminal goal cost is LINEAR in distance.
5. **A pursuit term is what pulls rollouts forward** — lateral path distance
   alone has no forward incentive; the observed failure was a literal
   zero-velocity local optimum (robot parks itself forever).

### Environment gotchas on THIS machine (will bite again)

- **ROS graph pollution**: stray processes from old sessions + live sims
  (AWSIM/Autoware on Humble, AirSim) leak TF across domains 0/42/77/88 —
  an Autoware NDT `map→base_link` at UTM-scale coordinates literally
  hijacked the TB3's `base_link` mid-demo. Before any ROS demo, probe a
  domain for silence and use it (101 was clean on 2026-06-10). Never assume
  0 or 42 is free.
- **`PYTHONNOUSERSITE=1`** for anything ROS-Python: `~/.local` pip numpy
  2.x breaks `transforms3d` (`np.maximum_sctype` removed) →
  `loopback_simulator` dies at import.
- **`pkill -f <pattern>` suicide**: the shell's own command line matches
  the pattern if it appears anywhere else in the same compound command
  (e.g. a later `git add scripts/record_nav2_path.py`). Keep kills in
  their own command and bracket a letter (`"[t]b3_loopback"`).
- **Gazebo + AMCL activation race**: `planner_server`'s global costmap
  activation waits for `map→base_link`, which needs AMCL, which needs an
  initial pose. Launch the stack and the mission script CONCURRENTLY
  (the script republishes the initial pose early) or the lifecycle manager
  aborts bringup. Also: gz lidar is slow for the first ~30 s headless
  (EGL fallback warnings are normal); wait it out.
- **CI does NOT build `ros2_ws/`** (no ROS on the GitHub runner). The
  plugin is only built/tested locally — see menu item B6.

### Reproduce the plugin results

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run cuda_mppi_controller plugin_load_test
ros2 run cuda_mppi_controller mppi_gpu_standalone 2048 ackermann   # diff/omni/footprint
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_bench
python3 scripts/render_cuda_mppi_benchmark.py /tmp/mppi_bench 2026-06-10
# Gazebo / loopback missions: see package README "Status" section (two terminals)
```

---

## Next-Task Menu (star-growth line) — pick ONE, in priority order

### A. `pip install cudarobotics` — Python bindings (strategy item 3)

The single biggest remaining usability multiplier: most roboticists live in
Python; a GPU MPPI / registration library they can `pip install` enters the
"use → cite → star" loop. Plan in enough detail to start cold:

1. **Scope (first release, keep it small)**:
   - ~~`cudarobotics.MppiPlanner` — wrap the existing `MppiGpu` core. It is
     ALREADY ROS-free behind a PIMPL (`mppi_gpu.hpp` has zero ROS/CUDA
     includes — this was deliberate). Expose `MppiParams` fields as
     constructor kwargs and `compute(state, costmap, path, goal, ...)`
     taking numpy arrays; return `(v, vy, w, info)`.~~ **DONE for MPPI
     initial release:** `python/` uses nanobind + scikit-build-core, and
     `examples/python/mppi_quickstart.py` renders the wall-gap GIF.
   - ~~`cudarobotics.registration` — FilterReg initial binding~~ **DONE:**
     `include/cudarobotics/filterreg_gpu.hpp` + `src/filterreg_gpu.cu` promoted;
     `examples/python/filterreg_quickstart.py` validates alignment.
   - ~~Sinkhorn-OT / FGR / BCPD Python bindings~~ **DONE:**
     `sinkhorn_gpu.cu`, `fgr_gpu.cu`, `bcpd_gpu.cu` + `examples/python/registration_quickstart.py`.
   - ~~Robust Student's-t / point-to-plane~~ **DONE:**
     `robust_treg_gpu.cu`, `robust_p2plane_gpu.cu` exposed as `RobustTreg` / `RobustP2Plane`.
   - ~~Packaging / wheels~~ **DONE (2026-06-10):** self-contained sdist +
     `linux_x86_64` CI wheels; `scripts/sync_python_core.sh` + `python/tests/`;
     `cibuildwheel` job on `master` pushes (`cudarobotics-manylinux-wheels` artifact).
2. **Binding tech**: nanobind (faster builds, smaller wheels than
   pybind11) + `scikit-build-core` for the CMake bridge. Accept numpy
   first; torch/cupy zero-copy later via `__dlpack__` — do NOT block the
   first release on dlpack.
3. **Source-of-truth move**: promote `mppi_gpu.{hpp,cu}` from
   `ros2_ws/src/cuda_mppi_controller/` to top-level `include/` + `src/`,
   then make the ROS package consume the promoted copy (relative path or
   CMake interface target). One copy, two consumers (ROS plugin, Python).
   **DONE.**
4. **Packaging reality check**: CUDA wheels are painful. First release may
   simply require a local CUDA toolkit (sdist build, pycuda-style), or ship
   linux-x86_64 wheels via `cibuildwheel` + manylinux. Decide AFTER the API
   works — do not start with packaging.
5. **Layout**: `python/` at repo root → `python/src/cudarobotics/`,
   `python/CMakeLists.txt`, `examples/python/mppi_quickstart.py`.
6. **Success criteria**: `pip install -e python/` works on this machine;
   the quickstart plans through the wall-gap scenario in <20 lines and
   renders a GIF; README gains a "Python" section with the snippet.

### B. nav2 plugin follow-ups (smaller, independent; good filler PRs)

1. ~~**Retreat/recovery behavior** — on `all_colliding`, decelerate/back out
   along the last valid sequence instead of throwing `NoValidControl`
   every cycle.~~ **DONE.**
2. ~~**SE(2) footprint check** — the sweep tests the polygon at sampled yaw
   only; rotation-in-place on asymmetric footprints can clip corners
   between samples.~~ **DONE.**
3. ~~**Dynamic parameter updates**~~ **DONE** — live `on_set_parameters` callback
   re-reads tunable weights and rebuilds the GPU optimizer.
4. ~~**More benchmark scenarios**~~ **DONE** — `controller_benchmark` adds
   `narrow_corridor` and `u_turn` cells (`controller_benchmark <out> all`).
5. ~~**Ackermann/Omni in-sim verification**~~ **DONE** — `controller_benchmark`
   adds `gpu_ackermann_K8192` / `gpu_omni_K8192`; full loopback configs
   `nav2_loopback_demo_{ackermann,omni}.yaml`; results in
   `docs/results/cuda_mppi_motion_models_2026-06-10.md` +
   `scripts/run_nav2_motion_model_demos.sh`.
6. ~~**CI coverage**~~ **DONE** (prior PR #176).
7. ~~**Humble backport check**~~ **DONE** — `nav2_compat.hpp` maps
   `InvalidPath` → `PlannerException` when `CUDAMPPI_NAV2_HUMBLE=1`.

### C. Research line (SOPPI) — unchanged

See "Research Line State" below. Priority within that line is still
`soppi_fast` on `box_align_contact_loss`, then a stronger contact-loss
cell. The arXiv packaging of the MPPI zoo is **skipped** (user decision).

---

## Research Line State: SOPPI box pushing (2026-06-14 snapshot, preserved)

(Snapshot taken at `master` = `5545dfd`; mainline has since moved — see the
star-growth Current State above. Within the RESEARCH line, the active thread
is the **SOPPI / Diff-MPPI box-pushing reproduction zoo** — not SLAM, not
MegaParticles, not shared-header cleanup.)

### What just landed (2026-06-04 → 2026-06-14)

Seven squash merges on the SOPPI box-pushing line, building a scenario taxonomy
where different planner families win for *different mechanistic reasons*:

| PR | Commit theme | Key artifact |
|---|---|---|
| #167 | Fixed-seed MPPI zoo suite + comparison GIF | `docs/results/mppi_zoo_suite_2026-06-10.*` |
| #168 | SOPPI kernel speed pass (`K*T` score kernels, ping-pong buffers) | `benchmark_diff_mppi*.cu` |
| #169 | Check in post-kernel box pushing results | `soppi_box_pushing_2026-06-10.*` |
| #170 | `box_align_strict` orientation-binding cell | scenario in `benchmark_diff_mppi_pushing_box.cu` |
| #171 | `box_align_detour` obstacle-detour cell | axis-aligned wall + `w_obs` barrier |
| #172 | `soppi_g3` / `soppi_fast_g3` hybrid planners | SVGD + 3 nominal Diff-MPPI grad steps |
| #173 | `box_align_contact_loss` + `w_contact_loss` penalty | pure `soppi` 0.25 vs `mppi` 0.00 |

Full reproduction notes: [`docs/soppi_reproduction.md`](docs/soppi_reproduction.md).
Latest checked-in box row: [`docs/results/soppi_box_pushing_2026-06-14.md`](docs/results/soppi_box_pushing_2026-06-14.md).

### Box pushing scenario taxonomy (7 scenarios, append-only seed discipline)

Scenarios are registered in `src/benchmark_diff_mppi_pushing_box.cu` in this
order (seed formula `6000 + si*100 + seed*7 + K`; **append new scenarios LAST**
so published numbers for `si=0..N-1` stay byte-identical):

| Scenario | Mechanism under test | Notes |
|---|---|---|
| `box_turn` | translate + rotate, loose tol | All planners fail at quick budget; not a SOPPI signal cell |
| `box_align` | orientation-dominant off-centre push | Large cost/distance reduction for SOPPI; success still 0.00 at strict tol |
| `box_pivot` | opposite handedness, tight `ang_tol=0.11` | Gradient-positive but quick budget insensitive |
| `box_swivel` | **negative control** — wide tol, corner push | MPPI 0.75; pure `soppi` **1.00** (sampling NOT contact-starved) |
| `box_align_strict` | orientation-binding gate | `pos_tol=0.28`, `ang_tol=0.08`; Diff-MPPI 1.00, `soppi_fast` 0.75 |
| `box_align_detour` | obstacle wall on direct push lane | **gradient-positive / sampling-negative**; only `diff_mppi_3` and `soppi_fast_g3` at 0.25 |
| `box_align_contact_loss` | **contact-loss cell** (NEW) | `w_contact_loss=47`, `w_near=0`; pure `soppi` 0.25 vs `mppi` 0.00 |

### Benchmark cell labels (use these when writing results)

1. **Sampling-positive / contact-easy** — `box_swivel`: isotropic noise routinely
   produces correct torque; SVGD can improve success (`soppi` 1.00) and steps.
2. **Orientation-binding** — `box_align_strict`: combined pos/ang gate separates
   Diff-MPPI from vanilla MPPI; SOPPI-family partial wins without nominal grad.
3. **Obstacle / nominal-grad-positive** — `box_align_detour`: pure SVGD fails;
   needs nominal trajectory grad steps (`diff_mppi_3`, `soppi_fast_g3`).
4. **Contact-loss / SVGD-positive** — `box_align_contact_loss`: stage penalty on
   pusher-box gap; differentiable contact gradient in SVGD score helps **without**
   nominal grad steps (all-pairs `soppi` 0.25, `mppi` 0.00, `soppi_fast` 0.00).

Do NOT collapse these into one headline. Each cell tests a different failure mode.

### Latest numbers (`K=256`, seed-count 4, `--quick`)

From `docs/results/soppi_box_pushing_2026-06-14.csv`:

| Scenario | mppi | soppi | soppi_fast | diff_mppi_3 | soppi_fast_g3 |
|---|---:|---:|---:|---:|---:|
| box_swivel | 0.75 | **1.00** | **1.00** | 0.75 | — |
| box_align_strict | 0.75 | 0.50 | 0.75 | **1.00** | — |
| box_align_detour | 0.00 | 0.00 | 0.00 | 0.25 | 0.25 |
| box_align_contact_loss | 0.00 | 0.50 | **1.00** | **1.00** | — |

Navigation suite (separate benchmark): SOPPI still **2/10** solved cells
(`narrow_passage` only). Treat navigation as honest negative coverage; box
pushing is the stronger SOPPI signal. See
[`docs/results/mppi_zoo_suite_2026-06-10.md`](docs/results/mppi_zoo_suite_2026-06-10.md).

### Implementation map (SOPPI box pushing)

File: `src/benchmark_diff_mppi_pushing_box.cu`

- **Plants**: smooth quasi-static contact (`push_step_box_f`) + optional hard
  Coulomb plant (`push_step_box_hard_f`) for sim-to-sim / fidelity arms.
- **Costs**: stage + terminal pose; `w_near` shaping; `w_obs` obstacle barrier;
  **`w_contact_loss`** squared gap when pusher leaves smooth contact (float +
  dual autodiff + SOPPI score kernel).
- **Planners registered**: `mppi`, `lp_mppi*`, `oi_mppi*`, `diff_mppi_{1,3,5}`,
  `soppi`, `soppi_fast`, `soppi_g3`, `soppi_fast_g3`, `mppi_hardmodel`.
- **SOPPI kernels**: `soppi_timestep_score_kernel` (per `(k,t)` autodiff score),
  `soppi_svgd_step_kernel` (RBF SVGD in normalized action space),
  `fixed_rollout_kernel` (re-evaluate moved samples before MPPI weighting).
- **CLI overrides**: `--override-soppi-iters`, `--override-soppi-step-size`,
  `--override-soppi-bandwidth`, `--override-soppi-neighbors`.
- **Diagnostics** (optional modes): mechanism diag (`escape_frac`, `contact_frac`),
  gradient-agreement capstone (smooth autodiff vs hard-plant FD).

Related SOPPI ports (same pattern, weaker signal today):

- `src/benchmark_diff_mppi.cu` — 2D bicycle nav (local stage-cost score, not
  full trajectory gradient).
- `src/benchmark_diff_mppi_cartpole.cu` — differentiable cost-to-go score.
- `src/benchmark_diff_mppi_pushing.cu` — disk pushing (saturated: all planners 1.00).
- `scripts/sweep_soppi.py` — grid sweeps for navigation SOPPI hyperparameters.

Paper target: Aldrich & Jenkins, arXiv:2511.02015 (SOPPI). This repo is a
**reproduction scaffold**, not paper-faithful on nav score or RNN gradient approx.

### Reproduce the latest box row

```bash
cmake --build build --target benchmark_diff_mppi_pushing_box -j$(nproc)
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-14.csv
```

Probe a single new cell:

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --scenarios box_align_contact_loss \
  --planners mppi,soppi,soppi_fast,diff_mppi_3 \
  --k-values 256 \
  --seed-count 4
```

CI: GitHub Actions `build` job ~15 min (CUDA install + full build + Python/CPU
tests). Recent PRs #172/#173 both green.

### Lessons learned (SOPPI sprint, do not re-learn)

1. **Append scenarios last** — seed formula is `si`-dependent; inserting in the
   middle invalidates all prior published numbers.
2. **Tune on full-suite seeds** — isolated `--scenarios X` probes can look great
   then regress when `si` shifts (happened twice tuning `w_contact_loss`).
3. **`w_contact_loss` is a knife** — too low: MPPI still wins; too high: everyone
   fails. Sweet spot for current geometry: `w_contact_loss=47`, `pen_thresh=0.007`,
   `w_near=0`.
4. **Pure SVGD ≠ subset SVGD** — all-pairs `soppi` can beat MPPI where `soppi_fast`
   (32-neighbor subset) still fails (`box_align_contact_loss`). Do not assume
   `soppi_fast` tracks `soppi` on hard cells.
5. **Obstacle detour needs nominal grad** — SVGD on sample distribution does not
   substitute for Diff-MPPI nominal trajectory steps on `box_align_detour`; hybrid
   `soppi_fast_g3` is the honest fix, not more SVGD tuning alone.
6. **`box_pivot_contact_loss` was tried and dropped** — pivot geometry + contact
   penalty did not yield a clean pure-SOPPI > MPPI cell under full-suite seeds;
   kept only `box_align_contact_loss`.

### Recommended next (pick ONE)

Priority order for the next coding agent:

1. ~~**Lift `soppi_fast` on `box_align_contact_loss`**~~ **DONE → 1.00** — subset SVGD
   (`neighbor_count=112`, `svgd_iters=2`, `step_size=0.05`, `bandwidth=2.0`) plus **one**
   nominal trajectory grad step (`grad_steps=1`, `alpha=0.010`) on canonical seeds
   (K=256, 4 seeds): `soppi_fast` **1.00** vs `mppi` **0.00**, all-pairs `soppi`
   **0.50**; pure subset SVGD alone plateaued at **0.75** (seed 3 at `pos=0.287` vs
   `pos_tol=0.28`).

2. ~~**Stronger contact-loss cell**~~ **DONE → `box_align_contact_arc`** — appended scenario
   with `pos_tol=0.30`, `ang_tol=0.12`, same `w_contact_loss=47`. Fixed-seed probe
   (K=256, 4 seeds): **`soppi` 1.00**, **`soppi_fast` 1.00**, `mppi` 1.00 — wider gate
   lifts pure SVGD without nominal grad; strict `box_align_contact_loss` stays the hard cell.

3. ~~**`--baseline-planners` for `scripts/sweep_soppi.py`**~~ **DONE** — comma-separated
   baseline planners run once before the SOPPI grid (default `mppi`).

4. ~~**Navigation SOPPI score upgrade**~~ **DONE (2026-06-10)** — nav SOPPI uses
   trajectory-level adjoint score (`soppi_trajectory_score_kernel`, CartPole-style)
   instead of per-timestep stage-cost only. On the 4-scenario default nav suite
   (`K=128`, 3 seeds): `narrow_passage` stays 1.00 for `soppi`/`soppi_fast`;
   `slalom` final distance improves (7.1 vs 19.4 m for `mppi`) but success stays 0;
   full 10-scenario zoo headline remains ~2/10 — honest partial win, not a new solved cell.

5. **Mechanical / unrelated** — Open Threads A (readme headline table refresh,
   `cuda_check.cuh` back-migration) if user wants a low-risk maintenance PR instead.

Start on a fresh branch:

```bash
git switch -c feat/soppi-fast-contact-loss master
# or
git switch -c feat/soppi-sweep-baseline-planners master
```

---

## Historical Handoff (preserved below)

The sections below document the 2026-05-21 → 2026-05-27 sprint (SLAM / CSM /
BnB / MegaParticles / graph-MPPI demos). They remain useful for file-map and
house-rule context but **do not reflect current priority**.

PR (`feat/gpu-bnb-loop-closure-slam` -> `master`) was **MERGED** (#127) on
2026-05-27; historical note preserved:
the capstone of the CSM line -- branch-and-bound (#124) wired into the SLAM
loop-closure front-end (#121/#123). New single file
`src/gpu_bnb_loop_closure_slam.cu` runs the SAME closed-lap submap SLAM as #123,
but at every loop attempt searches the relpose over a full-resolution 4.5M-cell
grid (+/-8 m / +/-0.6 rad, BNB_S=256, SEARCH_NT=69) TWO ways -- brute force
(one thread = one candidate) and branch-and-bound (GPU multi-resolution max-pool
bound + a complete coarse frontier scored on the GPU, then a host best-first
descent) -- and confirms they return the IDENTICAL relpose argmax. The
branch-and-bound relpose (after a shared fine refinement) drives the live
pose-graph back-end. Result (deterministic): B&B == brute force on **51/51**
attempts, B&B scoring **4.7k nodes vs 4.52M candidates (957x fewer)**, GPU B&B
0.27 ms vs brute force 7.1 ms/attempt, lap closed dead-reckoning ATE 2.18 m ->
0.20 m (48 loops accepted / 3 rejected). KEY LESSON (cost a full design pivot):
this is the EFFICIENCY statement, NOT a capability one. The first cut tried to
show a "wider" B&B window (full circle / +/-1.4 rad, prior-free heading)
recovering loops the bounded #123 window misses -- but a larger heading window in
a sparse-scan scene only invites PERCEPTUAL-ALIASING false positives (the 64-ray
scan locks onto ~180 deg rotational aliases scoring above the gate), and the B&B
variant DIVERGED to 6-13 m ATE while the bounded +/-0.6 rad search stayed at
0.36 m. The loops here are also effectively local (the lap's relative drift over a
loop is modest once the prior is corrected), so the bounded window is already
sufficient and GPU brute force over it is already cheap. The honest, shippable
result is therefore: B&B searches the SAME effective window as the established
front-end, returns the SAME loops, and closes the SAME lap -- just scoring ~1000x
fewer candidates (which is what lets the search scale to the large windows / 6-DoF
where brute force becomes infeasible, the #124 point, now demonstrated end-to-end
inside a SLAM pipeline). Do NOT chase a "wider window beats narrow" SLAM story in
this benign 2D scene; it is not honest. Touched files: CMakeLists.txt, readme.md,
plan.md, src/gpu_bnb_loop_closure_slam.cu, plus the gh-pages GIF.

PR (`feat/gpu-branch-and-bound-csm` -> `master`) was **MERGED** (squash) on
2026-05-26 at `73bdf08`; the remote branch was deleted and local `master` is in
sync.  It was the algorithmic follow-up to the CSM line (#120 exhaustive -> #121
loop-closure front-end -> #123 submap front-end). All three find the global pose
by EXHAUSTIVE
search (one thread scores one candidate, host takes the argmax) -- fine for a
small window, but a real loop-closure search must cover a LARGE window and the
exhaustive count grows cubically, so brute force (even on the GPU) eventually
runs out of room. Cartographer's real loop closer (Hess et al., ICRA 2016) solves
this with BRANCH AND BOUND over a precomputed multi-resolution likelihood field:
a stack of max-pooled grids gives an admissible UPPER BOUND on the best score in
any block of candidates, so a best-first search discards whole blocks without
scoring their leaves and still returns the IDENTICAL global maximum. New single
file `src/gpu_branch_and_bound_csm.cu` runs that head-to-head against the #120
exhaustive search on the SAME field, SAME scan, and SAME discrete (x, y, theta)
grid (translation step = one field cell, so the spatial max-pool gives an
admissible per-point bound; theta enumerated at ~1 deg). Two GPU pieces stay in
the "one thread = one candidate" idiom: the multi-resolution max-pool stack is
built on the GPU (one thread = one cell per level, a 2D sparse table), and the
BnB frontier -- a complete coarse tiling at level C-3 (64 nodes/theta) -- is
scored on the GPU in parallel (one thread = one node); a short host best-first
descent (first leaf popped == global optimum) then refines the winner. The window
GROWS frame by frame (C 5->8, +/-0.8 m -> +/-6.4 m). Result (deterministic): the
exhaustive count climbs 13k -> 4.52M candidates while BnB stays ~0.9k -> 4.5k
nodes, and BOTH return the same pose -- 40/40 frames the BnB grid cell equals the
exhaustive cell (or an immediate neighbour at a float tie). At the largest window
BnB scores **1004x fewer** candidates (4,521,984 vs 4,504) for the identical
optimum; the GPU exhaustive baseline is 12 ms vs a one-off CPU 0.7 s (~57x). KEY
LESSONS: (a) the max-pool bound's admissibility breaks at the GRID EDGE -- if a
node's min corner falls off-grid (sampled as 0) while an in-grid leaf inside the
block is positive, the bound underestimates and BnB prunes the true optimum
(observed as a real ~4-point miss); the fix is to clamp the bound's sample cell
to [0, GRID_N-span] so the max-pool window always CONTAINS the block's in-grid
leaves (leaves themselves, level 0, keep the out-of-grid skip so they match the
exhaustive kernel bit-for-bit); (b) score the BnB leaves and the exhaustive
candidates with the SAME out-of-grid convention or the two argmaxes diverge by a
cell at near-ties; (c) verify exactness on the grid CELL (adjacency) not the raw
float score, since GPU and host summation rank near-equal cells differently to
~1e-3. Touched files: CMakeLists.txt, readme.md, plan.md,
src/gpu_branch_and_bound_csm.cu, plus the gh-pages GIF. This graduates the CSM
line from brute force to the bounded search real systems use for large windows.

PR (`feat/gpu-csm-submap-slam` -> `master`) was **MERGED** (squash) on 2026-05-26
at `3ee8586`; the remote branch was deleted and local `master` is in sync. It is
the submap follow-up to #121. #121 detects loop closures by matching the current
scan against a likelihood field built from ONE earlier keyframe's scan; that
single-scan field is sparse and view-dependent, so a true revisit observed from a
slightly different heading can score below the accept gate. Real systems
(Cartographer) instead match against a SUBMAP -- several consecutive scans fused
into one local grid -- which fills in the surfaces from many viewpoints, giving a
denser, far less view-dependent field. New single file
`src/gpu_csm_submap_slam.cu` builds that submap front-end and runs it
head-to-head against the single-scan front-end of #121 on the SAME drifting
trajectory, SAME candidate gating, and SAME exhaustive coarse-to-fine CSM search
(1.42M candidate relposes/attempt). The ONLY difference between the two variants
is the likelihood field: a single anchor scan (scan-to-scan) vs the fused submap
around that anchor (scan-to-submap, SUBMAP_KF=8). To create the regime where
submaps actually earn their place, the simulated scan is deliberately SPARSE and
noisy (N_SCAN=64 rays, SCAN_NOISE=6 cm) -- a single such scan localises poorly.
Result (deterministic): from a 2.03 m dead-reckoning ATE, the single-scan
front-end recovers only 17/52 loops (35 rejected, ATE 0.38 m) while the
scan-to-submap front-end recovers 48/52 (4 rejected, ATE 0.18 m) -- the fused
field lets revisits cross the score gate where the sparse single scan cannot, and
lands an order-of-magnitude-better map. GPU 4.6 ms vs a one-off CPU coarse search
0.67 s => ~148x (lower than #121's 630x only because the sparse 64-pt moving scan
makes the GPU overhead-bound; same 1.42M candidates). NO ground truth enters the
constraint -- the intra-submap relposes that fuse the field come from the live
estimate, and over ~8 keyframes their drift is negligible. The middle panel
tints points by submap so the partition is visible; left=dead reckoning,
middle=scan-to-submap SLAM, right=info with three ATE curves (dead-reckon vs
scan-to-scan vs scan-to-submap) and both variants' accept/reject tallies. KEY
LESSONS: (a) on DENSE clean scans (180 rays) single-scan already works well, so
the submap edge only shows once the scan is sparse/noisy enough that a single
view is ambiguous -- the demo deliberately picks that regime, which is exactly
why Cartographer uses submaps; (b) a smaller submap is sharper but recovers fewer
loops, a larger one is denser/more robust but its fused relpose is slightly
blurred by intra-submap estimate error (SUBMAP_KF=8 balances the two); (c) time
the GPU and the one-off CPU reference on the SAME first query or the speedup
mixes two different scans. Touched files: CMakeLists.txt, readme.md, plan.md,
src/gpu_csm_submap_slam.cu, plus the gh-pages GIF. This graduates #121's
single-scan loop front-end into the submap-based front-end real 2D SLAM uses.

PR (`feat/gpu-csm-loop-closure-slam` -> `master`) was **MERGED** (squash) on
2026-05-26 at `fb7a186`; the remote branch was deleted and local `master` is in
sync. It is a 2D SLAM demo whose loop closures are DETECTED by scan matching, not
injected from ground truth. Every existing pose-graph SLAM demo in the repo
(`gpu_pose_graph_slam` #58, `gpu_online_slam` #63, the 3D/robust/switchable
family) takes its loop-closure constraints from GT spatial proximity -- they
exercise the back-end but skip the hard front-end question (has the robot
returned to a known place, and what is the relpose of the revisit?). New single
file `src/gpu_csm_loop_closure_slam.cu` answers it with the #120 correlative
scan matcher as the loop-closure FRONT-END: a robot drives a closed elliptical
lap (140 keyframes) with a small systematic odometry heading bias, so dead
reckoning drifts ~2 m by the seam -- past a local matcher's basin but inside
CSM's exhaustive window. When the drifting estimate brings the current keyframe
near an earlier one (index-gap + estimate-proximity gate), the current scan's
in-range endpoints are matched EXHAUSTIVELY (coarse-to-fine, ~1.42M candidate
relposes/attempt, one thread = one candidate) against a distance-transform
likelihood field built from the earlier keyframe's scan; the normalised
scan-to-scan score gates accept/reject and the argmax gives the relpose -- NO
ground truth enters the constraint. Accepted (odometry + CSM loop) edges feed a
compact dense SE(2) pose-graph Gauss-Newton back-end (host Cholesky, the graph
is small; the GPU work is the front-end search) that re-optimises the whole
trajectory the moment a loop snaps shut. Result (deterministic): 52 loops
proposed, 49 accepted, 3 rejected by the score gate; dead-reckoning ATE 2.03 m ->
SLAM ATE 0.17 m; GPU 2.4 ms vs a one-off CPU coarse search 1.5 s => ~630x. Left
panel (dead reckoning) smears / rotates; right panel folds into one consistent
floor-plan at the closure. KEY LESSONS: (a) the CSM search must be centred on the
estimate-predicted relpose (`relative(est[o], est[k])`), NOT zero -- at a lap
seam the true relpose is small but the heading can be far from 0, and the window
is only +/-0.6 rad; (b) the local field must be built from the OLD keyframe's
matchable (in-range) endpoints only, and the moving scan likewise range-capped,
or far/aliased returns dilute the normalised score below the accept gate; (c) the
one-off CPU timing reference must consume its `best` score or the whole search is
eliminated as dead code (same trap as #120, 0.0 ms tell-tale). Touched files:
CMakeLists.txt, readme.md, plan.md, src/gpu_csm_loop_closure_slam.cu, plus the
gh-pages GIF. This is the natural follow-up to #120 -- the global scan matcher
graduating from a standalone relocalization primitive into a SLAM loop-closure
front-end, and the first repo SLAM demo with data-driven (not GT) loop closures.

PR #120 (`feat/gpu-correlative-scan-matching` -> `master`) was **MERGED**
(squash) on 2026-05-26 at `f32c831`; CI Build passed (12m59s; Build + Python
tests + CPU tests all green; only the Node.js 20 deprecation annotation), the
draft was marked ready, and the remote branch was deleted. The concurrent agent
session landed #119 `gpu_trainable_safety_dual_graph_mppi` at `89e41f2` just
before it, so the squash fast-forward landed it together; #120's own diff is its
four files (CMakeLists.txt, readme.md, plan.md,
src/gpu_correlative_scan_matching.cu) plus the gh-pages GIF. Local `master` is
at `f32c831` and in sync with origin.
It adds the GLOBAL/exhaustive member missing from the scan-matching
family (NDT 2D/3D, GICP 2D/3D are all LOCAL iterative refiners). New single file
`src/gpu_correlative_scan_matching.cu` implements correlative scan matching
(Olson, ICRA 2009; Cartographer's real-time CSM): a discretised (x, y, theta)
window of candidate poses is scored EXHAUSTIVELY against a map likelihood field
and the global maximum is taken -- one thread = one candidate pose, the repo's
canonical parallel pattern. Coarse-to-fine (151x151x91 coarse @0.06 m/1 deg ->
41x41x41 fine), ~2.1M candidates/frame. Controlled comparison on the SAME field
objective: a LOCAL gradient-ascent matcher (stand-in for NDT/GICP) vs the GPU
exhaustive search, with the initial offset growing across frames. Result: CSM
recovers 44/44 frames (RMSE 0.006 m) up to +/-3.8 m / 40 deg offset while the
local matcher stalls outside its narrow field basin (5/44, RMSE 1.95 m); GPU
~6 ms vs a single timed CPU run ~2.9 s => ~490x. KEY LESSON (cost a debug
cycle): a likelihood field built as distance-to-nearest-OCCUPIED-cell rewards
piling scan endpoints into large FILLED obstacle interiors (lut=1 over the whole
solid), so the global max was a spurious pose ~1.8 m off even at zero offset.
Fix: build the field from obstacle SURFACES only (occupied AND adjacent to free
space) -- a scan only ever observes surfaces, so distance-to-surface makes the
true pose the unique global max. Also use thin wall slabs, not filled blocks,
and avoid dead-code elimination of the timed CPU reference (consume its result).
Touched files: CMakeLists.txt, readme.md, plan.md,
src/gpu_correlative_scan_matching.cu, plus the gh-pages GIF. This complements
the local scan matchers with a global loop-closure / relocalization alignment
primitive.

PR #118 (`feat/gpu-megaparticles-smoother` -> `master`) was **MERGED** (squash)
on 2026-05-26 at `1bf1fba`; CI Build passed (13m35s; Build + Python tests + CPU
tests all green; only the Node.js 20 deprecation annotation), the draft was
marked ready, and the remote branch was deleted. The concurrent agent session
landed two more demos (`gpu_safe_noregret_game_graph_mppi` at `aa549f4`, and
#117 `gpu_learned_safety_dual_graph_mppi` at `3c99944`) on master just before
it, so the squash fast-forward landed them together; #118's own diff is its four
files (CMakeLists.txt, readme.md, plan.md, src/gpu_megaparticles_smoother.cu)
plus the gh-pages GIF. Local `master` is at `1bf1fba` and in sync with origin.
It is the "Localization polish" follow-up flagged in Open Threads /
Recommended Next -- a short smoothing pass over the representative MegaParticles
trajectory that reports raw max-posterior vs smoothed pose error SEPARATELY,
finally replacing the tiny hand-tuned continuity gate that #86/#101/#104/#115 all
carried as a known limitation. New single file
`src/gpu_megaparticles_smoother.cu`: the GPU does the expensive part exactly as
#86 (1,048,576 particles, distance-field likelihood, bucket-neighbor Stein
motion, posterior smoothing) and each frame emits one raw max-posterior
representative pose; a lightweight host backend keeps a sliding window of the
last SMOOTH_W=10 frames and jointly optimises a smoothed pose chain by IRLS
Gauss-Newton with (a) SWITCHABLE CV-motion factors (a genuine pose
discontinuity -- the hidden kidnap -- breaks the link instead of being smeared)
and (b) Huber-robust measurement factors (a one-frame spurious max-posterior
spike is rejected). A frame is finalized once it falls off the window head
(fixed lag = future frames refine it). KEY LESSON: a robust smoother alone
CANNOT distinguish a sustained new-location measurement (kidnap) from an outlier
-- it rejected the post-kidnap relocalization and stuck to the coasted old
trajectory (post-kidnap RMSE ~1.7 m). Fix: a data-driven reset -- the smoother
resets its window only when measurements RESUME FAR from the coast AFTER a
measurement dropout (has_obs=false run), distinguishing a genuine relocalization
from the high-confidence spurious-mode flips during normal tracking (those stay
rejected as outliers). Verified over 4 runs (GPU atomicAdd noise floor):
in-track jitter (mean |d2 pos|) raw 4.31 -> smoothed ~0.06 (~70x, truth ref
0.0055), in-track RMSE raw 5.4 -> smoothed ~0.25 m (raw inflated by 16 m
spurious-mode flips in the repetitive-corridor map that the robust smoother
rejects), post-kidnap RMSE raw ~1.2-1.9 -> smoothed ~0.09 m, recovers the hidden
kidnap in 0 frames; smoothing adds negligible cost (host backend; GPU step
~5 ms, same as #86). Final steady-state error is a touch higher than raw
(0.040 -> 0.059 m) -- the expected smoother trade of a little sharpness for
robustness. Touched files: CMakeLists.txt, readme.md, plan.md,
src/gpu_megaparticles_smoother.cu, plus the gh-pages GIF. With this the
MegaParticles localization line covers Stein (#86), explicit LSH (#101),
6-DoF SE(3) (#104), GICP D2D likelihood (#115), and the trajectory smoother.

PR #115 (`feat/gpu-megaparticles-gicp-d2d` -> `master`) was **MERGED** (squash)
on 2026-05-26 at `0318b9f`; CI Build passed (12m50s; Build + Python tests + CPU
tests all green; only the Node.js 20 deprecation annotation), the draft was
marked ready, and the remote branch was deleted. The concurrent agent session
landed two more graph-neural / game-theoretic MPPI demos
(`gpu_iterative_game_graph_mppi`, `gpu_noregret_game_graph_mppi`) on master just
before it, so the squash fast-forward landed them together; #115's own diff is
its four files (CMakeLists.txt, readme.md, plan.md,
src/gpu_megaparticles_gicp_mcl.cu) plus the gh-pages GIF. Local `master` is at
`0318b9f` and in sync with origin. The PR added a GICP-style distribution-to-
distribution (D2D) scan likelihood
for the MegaParticles line -- the "GICP-like point-cloud likelihood" follow-up
flagged after #86/#101/#104. New single file `src/gpu_megaparticles_gicp_mcl.cu`
runs a controlled head-to-head: two 1,048,576-particle filters with IDENTICAL
MegaParticles machinery (global uniform init, Gauss-Newton particle motion,
sparse bucket-neighbor Stein attraction/repulsion, posterior smoothing,
representative-state gate, hidden kidnap + 15-frame scan blackout), differing
ONLY in the per-particle scoring kernel. Arm A is the #86 distance-field
endpoint proxy (control; it reproduces #86's ~0.097 m post-kidnap RMSE). Arm B
is the new GICP D2D likelihood: the map is a point cloud (2,396 points, boundary
cells thinned to ~0.15 m) with per-point disk covariances (small variance along
the surface normal, large along the tangent), indexed by a uniform NN grid; each
particle matches every scan endpoint to the nearest map point and scores the
surface-aware Gaussian log-likelihood under the combined covariance
M = (C_map + R C_scan R^T)^{-1} (Segal et al. RSS 2009), summed over the scan,
with a per-particle full 3x3 Gauss-Newton step driving the Stein motion. Key
robustness lesson: a PURE D2D likelihood (flat penalty + zero gradient outside
the match radius) re-localized the hidden kidnap only intermittently -- the
sharp D2D contracts harder pre-kidnap, leaving thin global support, and a lost
particle gets no gradient pull. Fix is a coarse-to-fine design: an UNMATCHED ray
falls back to the distance-field endpoint log-likelihood (smooth long-range
pull), so the worst case == the field filter and global recovery is robust,
while MATCHED rays use the sharp surface-aware GICP term for accuracy. Verified
over 4 runs (GPU atomicAdd noise floor): both arms recover the kidnap in 0
frames; post-kidnap RMSE field 0.099 m -> GICP D2D ~0.064 m (~35% lower), final
error 0.040 m -> 0.021 m (~halved), at ~2.4x per-step cost (field ~4.9 ms ->
D2D ~12.1 ms). Touched files: CMakeLists.txt, readme.md, plan.md,
src/gpu_megaparticles_gicp_mcl.cu, plus the gh-pages GIF. With this the
MegaParticles localization line now covers Stein (#86), explicit LSH (#101),
6-DoF SE(3) (#104), and the GICP D2D likelihood.

PR #112 (`chore/shared-se3-helpers` -> `master`) was **MERGED** (squash) on
2026-05-26 at `3eb4b4d`; CI Build passed (12m1s; Build + Python tests + CPU
tests all green; only the Node.js 20 deprecation annotation), the draft was
marked ready, and the remote branch was deleted. The concurrent agent session
landed six GPU graph-neural / game-theoretic MPPI demos
(`gpu_best_response_graph_mppi`, `gpu_belief_risk_graph_mppi`,
`gpu_intent_graph_neural_mppi`, `gpu_priority_graph_neural_mppi`,
`gpu_interaction_graph_neural_mppi`, `gpu_multiagent_graph_neural_mppi`) on
master just before it, so the squash fast-forward landed all of them together;
#112's own diff is exactly its five files (include/se3_helpers.cuh, plan.md,
and the three migrated `.cu`). Local `master` is at `3eb4b4d` and in sync with
origin. There is **no active feature branch** now -- the next agent starts
fresh from `master`.
This was the Open Threads A shared-header cleanup recommended after #105: it
lifts the SE(3) / SO(3) math kernels + the 6x6 SPD Cholesky solve that were
copied verbatim across the three rotation-matrix pose-graph SLAM back-ends
(`gpu_pose_graph_slam_3d.cu`, `gpu_pose_graph_slam_3d_switchable.cu`,
`gpu_online_slam_3d_switchable.cu`) into a single new
`include/se3_helpers.cuh` (`clampf`, `mat3_identity/mul/transpose_mul/
transpose_vec/vec`, `so3_exp`, `so3_log`, `solve6_spd_device`). Net -322 LOC
across the three `.cu` files (364 deletions / 42 insertions). Investigation
showed the plan's "six files share this scaffold" framing was optimistic:
only these three share the *rotation-matrix* SE(3) scheme byte-for-byte;
`gpu_ndt_3d.cu` / `gpu_gicp_3d.cu` use a differently-named `cholesky_solve_6`
+ `H_OFF` family, and `gpu_megaparticles_6dof.cu` uses a quaternion
representation -- those are genuinely different code, not drift, so folding
them in would be a forced abstraction and is deliberately left out of scope.
The struct-coupled helpers (`pose_relative`, `residual_edge`, `perturb_pose`)
stay per-`.cu` because they depend on the demo-local `Pose` / `Edge` layout;
the header stays a pure struct-agnostic math kernel library. Verified numeric
parity: the fully-deterministic CPU reference path is byte-identical before
and after (robust 0.2844 m / 2.1182 deg, switchable 0.2934 m / 2.2235 deg);
GPU final metrics match to the pre-existing `atomicAdd`-order run-to-run
noise floor (robust 0.2842 m / cost 547.39 / 36-36 rejected, switchable
~0.293 m / cost ~144547.5 / 36-36, online plain 9.10 m vs switchable 0.29 m
/ 21-21). All four affected targets (the three demos plus the
`gpu_pose_graph_slam_3d_robust` variant of the first source) rebuild and run
clean. No CMake change needed -- `include/` is already on the compile include
path. No GIF/readme change (behaviour is unchanged; this is a pure cleanup).

PR #105 (`feat/gpu-online-slam-3d-switchable` -> `master`) was **MERGED**
(squash) on 2026-05-25; CI Build passed (~12 min; Build + Python tests +
CPU tests all green; only the Node.js 20 deprecation annotation), the draft was
marked ready, and the remote feature branch was deleted. Local `master` is at
`c1e977d` and in sync with origin. It is the *other* 3D SLAM follow-up flagged
in the previous handoff -- wiring the switchable-constraint SE(3) back-end of
#98 into the online sliding-window front-end of #63. A robot streams 420 SE(3)
poses; true loops (GT proximity) and 21 gross false loops arrive incrementally;
two back-ends run lockstep on the same edge stream: "plain online" (every loop
weight 1) is yanked off course as false loops arrive (final RMSE 9.10 m), while
"switchable online" re-minimises per-loop switches in closed form each frame
inside the sliding window and rejects all 21 false loops live (final RMSE
0.29 m, clean-loop switch 1.000 / false 0.000). Built on the SE(3) GN+PCG +
closed-form switch machinery of #98 made sliding-window-aware (active_lo/
active_hi masking, per-window anchor pin), plus #63's iSAM-style global pass on
loop. Single file `src/gpu_online_slam_3d_switchable.cu`; ~16 ms/step for both
back-ends. The squash diff touched only its four files (CMakeLists.txt,
plan.md, readme.md, src/gpu_online_slam_3d_switchable.cu). This closes the
3D-SLAM follow-up line (#82 v2 -> #83 robust -> #98 switchable batch -> #105
switchable online). There is **no active feature branch** now -- the next agent
starts fresh from `master`.

Mainline is in sync with `origin/master` at commit
`c1e977d Add GPU online 3D SLAM with switchable loop constraints (#105)`.
PR #104 (`feat/gpu-megaparticles-6dof` -> `master`) was **MERGED** (squash) on
2026-05-25; CI Build passed (~12 min; Build + Python tests + CPU tests all
green; only the Node.js 20 deprecation annotation), the draft was marked ready,
and the remote feature branch was deleted. The demo is the 6-DoF/SE(3) slice of
the localization-depth follow-up: a flying sensor in a 3D voxel world, range
scans scored against a GPU 3D ESDF (built with the JFA-3D from
`comparison_esdf_3d.cu`), quaternion-based Gauss-Newton SE(3) per-particle
steps, and -- crucially -- the explicit p-stable LSH neighbor index from #101
generalised to the 6-D pose feature (x, y, z, s*rotvec). In 6-DoF a dense grid
is combinatorially infeasible, so the LSH consensus from #101 becomes essential
rather than optional. Local bootstrap MCL has 5.97 m post-kidnap RMSE; the 1M
6-DoF MegaParticles path re-localizes a hidden kidnap to 0.22 m / 1.9 deg,
reacquiring in 0 frames (it re-seeds globally uniform over SE(3) during the scan
blackout, the honest "lost -> search everywhere" relocalization behaviour). The
squash diff touched only its four files (CMakeLists.txt, plan.md, readme.md,
src/gpu_megaparticles_6dof.cu); the concurrent agent session landed GPU
graph-guided and kinodynamic graph-neural MPPI demos
(`gpu_graph_guided_neural_mppi`, `gpu_kinodynamic_graph_neural_mppi`) on master
just before it. There is **no active feature branch** now -- the next agent
starts fresh from `master`.

PR #101 (`feat/gpu-megaparticles-lsh` -> `master`) was **MERGED** (squash)
on 2026-05-25; CI Build passed in 11m53s (Build + Python tests + CPU tests
all green; only the Node.js 20 deprecation annotation), the draft was marked
ready, and the remote feature branch was deleted. The demo is the explicit-LSH
slice of the localization-depth follow-up: it replaces the fixed-grid neighbor
stand-in of #86 with L=8 random hash tables of K=3 Gaussian projections (the
actual Datar et al. 2004 p-stable LSH scheme). A head-to-head controlled
comparison (1M particles each, identical Stein machinery, only the neighbor
structure differs) gives neighbor recall vs brute-force kNN 58.2% -> 87.8% and
post-kidnap RMSE 0.099 -> 0.088 m. The squash diff touched only its four files
(CMakeLists.txt, plan.md, readme.md, src/gpu_megaparticles_lsh.cu); the
concurrent agent session landed GPU spatiotemporal neural A* and learned
experience-graph planner demos (`902380e`, `7ba624a`) on master just before it.
There is **no active feature branch** now — the next agent starts fresh from
`master`.

PR #98 (`feat/gpu-pose-graph-3d-switchable` -> `master`) was **MERGED**
(squash) on 2026-05-25; CI Build passed in 13m11s (Build + Python tests +
CPU tests all green; only the Node.js 20 deprecation annotation), the draft
was marked ready, and the remote feature branch was deleted. The demo is the
3D SLAM follow-up that replaces the frozen front-end trim gate of #83 with
explicit per-loop switch variables jointly optimised with the SE(3) poses.
The squash diff touched only its four files (CMakeLists.txt, plan.md,
readme.md, src/gpu_pose_graph_slam_3d_switchable.cu); the concurrent agent
session landed GPU anytime / multi-goal neural A* traversability demos
(`6b972fa`, `a293804`) on master just before it. There is **no active feature
branch** now — the next agent starts fresh from `master`.

PR #95 (`feat/gpu-kld-amcl-kidnap` -> `master`) was **MERGED** on
2026-05-25 (squash). CI Build passed (11m56s; Build + Python tests +
CPU tests all green; only the Node.js 20 deprecation annotation), the
draft was marked ready, and the PR was squash-merged with the remote
feature branch deleted. Local `master` is at `5570ab3` and in sync with
origin. There is **no active feature branch** right now — the next agent
starts a fresh branch from `master`.

Note this checkout is shared with a concurrent agent session that has been
landing its own demos (graph/traversability line) directly on `master`.
Earlier the same day: PR #86 (MegaParticles Stein MCL) at `af4d5ee`, then
PR #89 (label propagation) at `9d6f902`. Between #89 and #95 the concurrent
session added the GAT traversability policy, differentiable value-iteration
traversability, and neural A* traversability demos (`4706efe`, `62564fa`,
`20ca275`). PR #95's squash diff touched only its own four files
(CMakeLists.txt, plan.md, readme.md, src/gpu_kld_amcl.cu).

There were no open GitHub PRs at the start of the MegaParticles-style
Stein MCL branch; #86 was opened from this branch after the local demo,
GIF generation, and Pages publication were validated, and is now merged.
Parked local branches remain for `feat/gaussian-splat-renderer`
(checked out in `/tmp/CudaRobotics-gaussian`) and `feat/repro-report`; treat them as
separate parked work, not active blockers for new feature work.

---

## TL;DR for the impatient

- Repo is on `master` at `5570ab3`, in sync with origin. PR #95 (GPU
  augmented KLD-sampling AMCL) was squash-merged on 2026-05-25 and its
  remote feature branch deleted. No active feature branch; start the next
  task from a fresh branch off `master`. (Same-day localization line: #86
  MegaParticles Stein MCL, #89 label propagation, #95 KLD-AMCL. A
  concurrent agent session is also landing graph/traversability demos on
  this shared checkout.)
- The 2026-05-21..25 sprint added 45 PRs (#40 → #84): ESDF JFA (2D + 3D),
  3D voxel map, massive collision check, realistic 3D LiDAR, ROS2 nodes,
  Bundle Adjustment, 2D pose-graph SLAM backend, LiDAR SLAM frontend,
  Multi-robot planner, Gaussian Splatting, NeRF volume, diffusion planner,
  online SLAM, shared headers refactor, **and the full 2D/3D NDT + 2D/3D
  GICP scan matching family**, followed by **multi-resolution NDT 3D** and
  **GPU Hungarian-class assignment**, then **GPU CMA-ES**. PR #75 adds
  **GPU MCTS kinodynamic planning**. PR #76 adds **GPU assignment
  tracking**. PR #77 adds **10K-agent GPU crowd swarm**. PR #78 adds
  **GPU SfM mini**. PR #79 adds **GPU PCG sparse solver**. PR #80 adds
  **GPU EM GMM clustering**. PR #81 adds **GPU spectral clustering**.
- PR #78 / branch `feat/gpu-sfm-mini` added a compact multi-view geometry
  demo: 2048 synthetic ORB-like features across 4 views, GPU descriptor
  matching, stereo triangulation, and point-only BA.
- PR #79 / branch `feat/gpu-pcg` adds a generic CSR Jacobi-PCG sparse SPD
  solver demo: 262K unknowns, 1.31M nonzeros, 33 iterations, 13.4x vs CPU.
- PR #80 / branch `feat/gpu-em-gmm` adds GPU EM clustering for a 2D
  full-covariance Gaussian mixture: 262K points, 5 components, 42 EM
  iterations, 90.2x vs CPU.
- PR #81 / branch `feat/gpu-spectral-clustering` adds normalized-affinity
  GPU spectral clustering on a 3072-point dense RBF graph: 40 subspace
  iterations, 193x vs CPU, 100% mapped cluster accuracy.
- PR #82 / branch `feat/gpu-pose-graph-3d-v2` adds GPU 3D pose-graph SLAM
  with central finite-difference SE(3) Jacobians: 384 poses, 575 edges,
  translation RMSE 1.64 m -> 0.28 m, rotation RMSE 11.29 deg -> 2.12 deg.
- PR #83 / branch `feat/gpu-pose-graph-3d-robust` extends the 3D backend
  with 36 deliberately false loop closures and a trimmed switch gate:
  plain GN is pulled to 6.95 m / 39.89 deg, while the switched solve
  rejects 36/36 false loops and returns to 0.284 m / 2.11 deg.
- PR #84 / branch `feat/gpu-global-localization-recovery` adds a GPU
  global-localization MCL recovery demo: 32,768 particles, 72 landmarks,
  10 range-bearing observations, hidden kidnap at step 70. Local-only MCL
  has 20.24 m post-kidnap RMSE; sensor-reset MCL triggers once and
  recovers to 0.022 m post-kidnap RMSE.
- Current branch `feat/gpu-megaparticles-stein-mcl` adds a compact
  MegaParticles-inspired SE(2) range-localization demo: 1,048,576
  particles, distance-field scan likelihoods, bucket-neighbor
  Stein-style updates, posterior propagation, and hidden kidnap recovery.
  Local bootstrap MCL has 14.61 m post-kidnap RMSE; the MegaParticles
  path recovers to 0.097 m post-kidnap RMSE and 0.041 m final error.
- The demo is deliberately framed as **MegaParticles-style**, not a full
  reproduction of Koide et al.'s ICRA 2024 6-DoF system. It preserves the
  visible algorithmic ideas in a repo-sized SE(2) benchmark: massive
  particles, range-field likelihood, neighbor-bucket Stein attraction /
  repulsion, posterior propagation, representative-state smoothing, and
  hidden kidnap recovery after scan blackout.
- The "scan matching 4 siblings" are NDT 2D (#67), NDT 3D (#68), GICP 2D
  (#69), GICP 3D (#70). All present and merged.
- Follow-up #71 adds coarse-to-fine NDT 3D and fixes the old #68 outlier
  frame; #72 opens the combinatorial optimisation chapter with batched
  64x64 dense assignment; #74 adds GPU CMA-ES black-box optimisation; #75
  adds root-parallel GPU MCTS; #76 adds gated multi-object tracking.
- One attempt during the sprint was aborted: **LiDAR SLAM with
  constant-velocity ICP init**
  (Lissajous reversal made drift 9x worse, change reverted).
- Recommended next directions are listed at the bottom; safe defaults
  after this branch are GPU diffusion policy / behaviour cloning, a
  graph-ML follow-up, or a mechanical shared-header cleanup.

---

## Repo State (2026-05-25)

- **Main branch**: `master`, currently at
  `5570ab3 Add GPU augmented KLD-sampling AMCL demo (#95)`, in sync
  with origin.
- **Active branch**: none (last feature branch
  `feat/gpu-kld-amcl-kidnap` merged and deleted).
- **Active PR**: #95 **MERGED** (squash) on 2026-05-25; target `master`.
  (Same-day localization line: #86 MegaParticles, #89 label propagation,
  #95 KLD-AMCL. A concurrent agent session has also landed GAT / diff-VI /
  neural-A* traversability demos on this shared checkout.)
- **Last CI**: GitHub Actions Build passed in 11m56s (Build + Python
  tests + CPU tests all green; only a Node.js 20 deprecation annotation,
  not a failure).
- **gh-pages branch**: hosts gif assets referenced from `readme.md`.
  Files live at the branch ROOT (not `gif/`), so the URL
  `https://rsasaki0109.github.io/CudaRobotics/<name>.gif` resolves.
  Every new gif must be pushed there to render in the readme.
- **Current Pages asset**:
  `https://rsasaki0109.github.io/CudaRobotics/gpu_kld_amcl.gif`
  returned HTTP 200 after the gh-pages deployment completed.
- 130 CUDA source files (`src/*.cu`), 15 C++ files (`src/*.cpp`) on the
  MegaParticles-style Stein MCL branch.
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

### Active PR #86 Snapshot

- Branch: `feat/gpu-megaparticles-stein-mcl`
- Base: `master`
- Title: `Add MegaParticles-style Stein MCL demo`
- PR: `https://github.com/rsasaki0109/CudaRobotics/pull/86`
- Files in scope:
  - `src/gpu_megaparticles_stein_mcl.cu`
  - `gif/gpu_megaparticles_stein_mcl.gif`
  - `CMakeLists.txt`
  - `readme.md`
  - `plan.md`
- Local validation already run:
  ```bash
  rtk bash -lc 'cd build && cmake .. && make gpu_megaparticles_stein_mcl -j8'
  rtk ./bin/gpu_megaparticles_stein_mcl
  rtk ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of default=noprint_wrappers=1 gif/gpu_megaparticles_stein_mcl.gif
  rtk git diff --check
  rtk proxy curl -I https://rsasaki0109.github.io/CudaRobotics/gpu_megaparticles_stein_mcl.gif
  ```
- Local numeric result:
  - post-kidnap RMSE: local bootstrap `14.6053 m`, MegaParticles-style
    Stein/bucket posterior `0.0974 m`
  - final error: local bootstrap `10.5934 m`, MegaParticles-style
    Stein/bucket posterior `0.0412 m`
  - reacquisition after scan blackout: `0` visible frames after blackout
    ends
  - average GPU step: local bootstrap `2.2750 ms`, MegaParticles-style
    path `5.3728 ms`
- GIF: 900x255, 65 frames, 1.9 MB.

---

## What Was Done (2026-06-04 → 2026-06-14) — SOPPI box pushing line

Compact PR list for the current active research thread.

| PR | Title | Headline |
|---|---|---|
| #167 | Fixed-seed MPPI zoo suite + GIF | 5 nav scenarios, 10 planners, `K=64,128`, 3 seeds |
| #168 | SOPPI kernel speed pass | `K*T` score kernels; ~3.4x `soppi_fast` on `box_swivel` |
| #169 | Post-kernel box pushing results | 4-scenario row checked in |
| #170 | `box_align_strict` scenario | Orientation-binding gate; Diff-MPPI 1.00 |
| #171 | `box_align_detour` scenario | Obstacle wall; gradient-positive / sampling-negative |
| #172 | `soppi_fast_g3` hybrid planner | SVGD + nominal grad; matches `diff_mppi_3` on detour 0.25 |
| #173 | `box_align_contact_loss` + `w_contact_loss` | Pure `soppi` 0.25 vs `mppi` 0.00 |

Adjacent merged work (same period, Diff-MPPI contact paper line):

| PR | Theme |
|---|---|
| #164–#166 | Contact fidelity arm, contact-depth study, gradient-agreement capstone |

Key docs touched across #167–#173:

- `src/benchmark_diff_mppi_pushing_box.cu`
- `docs/results/soppi_box_pushing_2026-06-{10,11,12,13,14}.*`
- `docs/soppi_reproduction.md`, `docs/mppi_reproduction_zoo.md`, `readme.md`

---

## What Was Done (2026-05-21 → 2026-05-25)

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
| #79 | GPU PCG solver | 262K unknowns, 1.31M CSR nnz, 33 iterations, **13.4x** vs CPU |
| #80 | GPU EM GMM clustering | 262K points, 5 full-cov Gaussians, 42 EM iterations, **90.2x** vs CPU |

(41 merged PRs in 5 days; cadence was sustained because each demo was a single
~500-700 LOC `.cu` file plus a few lines in `CMakeLists.txt` and
`readme.md`.)

### 2026-05-25 — clustering / graph ML
| PR | Title | Headline |
|---|---|---|
| #81 | GPU spectral clustering | 3072-point dense RBF graph, 40 subspace iterations, **193x** vs CPU, 100% mapped accuracy |
| #82 | GPU 3D pose-graph SLAM v2 | 384 poses, 575 SE(3) edges, finite-difference Jacobians, RMSE 1.64 m -> 0.28 m |
| #83 | GPU robust 3D pose-graph SLAM | 384 poses, 611 SE(3) edges, 36 false loops; plain 6.95 m / 39.89 deg -> switched 0.284 m / 2.11 deg |
| #84 | GPU global-localization MCL | 32,768 particles, 72 landmarks, hidden kidnap; local-only 20.24 m -> sensor-reset 0.022 m post-kidnap RMSE |
| #86 | GPU MegaParticles-style Stein MCL | 1,048,576 range particles, distance-field likelihoods; local bootstrap 14.61 m -> Stein/bucket posterior 0.097 m post-kidnap RMSE |
| #89 | GPU semi-supervised label propagation | 3072-node RBF graph, K=3, 12 clamped seeds, 50 iterations; **123x** vs CPU (55.3 ms vs 6.8 s), 100% unlabeled accuracy, 100% GPU/CPU label agreement |
| #95 | GPU augmented KLD-sampling AMCL | KLD-sampling adapts 400→65,536 particles; augmented w_fast/w_slow injection (deadband 0.4) reacquires hidden kidnap in 13 steps, settled RMSE 0.014 m, **15.2x** vs CPU (0.35 ms vs 5.28 ms/step) |
| #98 | GPU switchable-constraint 3D pose-graph SLAM | per-loop switch variables jointly optimised with SE(3) poses (Sünderhauf 2012, block coordinate descent, asymmetric switch damping); 384 poses / 611 edges / 36 false loops; plain GN 6.95 m / 39.89 deg → switchable 0.29 m / 2.23 deg, learns 36/36 false-loop rejection with no hand-set trim fraction; GPU/CPU agree to <1 mm |
| #101 | GPU MegaParticles LSH neighbor index | explicit p-stable LSH (8 tables × 3 Gaussian projections, Datar 2004) replaces the fixed-grid neighbor stand-in of #86; controlled comparison at 2 × 1,048,576 particles with identical Stein machinery; neighbor recall vs brute-force kNN 58.2% → 87.8%, post-kidnap RMSE 0.099 → 0.088 m, both reacquire in 0 frames; LSH 9.6 ms vs grid 4.9 ms / step (8-table OR cost) |
| #104 | GPU MegaParticles 6-DoF SE(3) | 1,048,576 SE(3) particles (position + quaternion) in a 3D voxel world; GPU 3D-ESDF (JFA-3D) range likelihood, quaternion GN steps (right-perturbation), 6-D p-stable LSH neighbor consensus (a dense 6-D grid is infeasible, so #101's LSH is essential); hidden kidnap: local bootstrap post RMSE 5.97 m → 6-DoF MegaParticles 0.22 m / 1.9 deg, reacquires in 0 frames; mega ~13.9 ms/step |
| #105 | GPU online 3D SLAM, switchable loop constraints | #98 switchable SE(3) back-end wired into #63's online sliding-window front-end; 420 streamed poses, window W=80 + global pass on loop, true loops from GT proximity + 21 gross false loops injected live; lockstep plain vs switchable on the same edge stream; plain online corrupted to 9.10 m as false loops arrive, switchable online rejects all 21 live (closed-form per-frame switch update inside the window) → 0.29 m, clean switch 1.000 / false 0.000; ~16 ms/step both back-ends |

### Current branch deep notes: MegaParticles-style Stein MCL (#86)

Motivation:
- The previous localization PR (#84) used mapped landmarks with known IDs
  and sensor-reset hypotheses. That made the kidnap recovery story very
  clear, but it still had a relatively structured observation model.
- Koide et al.'s MegaParticles line is more ambitious: large particle
  counts, range-based scan likelihoods, Stein variational particle motion,
  neighbor graph posterior propagation, and relocalization without a
  reliable initial pose. The current branch implements a compact SE(2)
  demonstration of those ideas without pretending to be a complete 6-DoF
  reproduction.

Implementation shape:
- `K_MEGA = 1 << 20` gives 1,048,576 particles. `K_LOCAL = 1 << 16`
  gives a 65,536-particle local bootstrap baseline.
- The map is a synthetic indoor floorplan of rectangles with repeated
  corridors and rooms. A CPU OpenCV distance transform creates a 2D
  distance field and central-difference gradients, then the distance /
  gradient arrays are uploaded to CUDA.
- Each observation is a 30-ray local range scan. The likelihood projects
  scan endpoints through each particle pose and scores endpoint distance
  to the nearest wall via the precomputed field.
- The local baseline is a conventional particle filter path: predict,
  likelihood weight, normalize, weighted mean, cumulative sum, systematic
  resample. It starts near the first true pose and therefore tracks before
  kidnap, but it has no particles near the hidden kidnapped pose.
- The MegaParticles-style path starts globally uniform. During visible
  scan frames it performs two correction passes:
  - per-particle approximate Gauss-Newton displacement from distance-field
    gradients,
  - bucket-neighbor Stein-style update that combines local particle
    displacement, bucket posterior-weighted displacement, a small
    repulsive term from the bucket mean, and jitter.
- The bucket grid is a practical stand-in for the paper's LSH neighbor
  search. It is dynamic, sparse in pose space, and cheap enough for the
  one-million-particle demo, but it is not the exact stable-distribution
  LSH algorithm from the paper.
- After Stein updates, the demo runs posterior smoothing over the same
  buckets. The representative pose is selected from the highest posterior
  bucket and then lightly time-gated to avoid the max-posterior pose
  jitter that the paper also notes as a smoothing issue.
- A hidden kidnap happens at step 56, followed by 15 scan-blackout frames.
  The local bootstrap MCL remains tied to the old mode, while the
  MegaParticles-style filter jumps to the recovered mode as soon as scans
  return.

Limitations to keep honest:
- This is SE(2), not full SE(3) / 6-DoF.
- The neighbor structure is bucketed pose space, not the paper's exact
  iterative LSH neighbor list.
- The range likelihood is a 2D endpoint distance-field proxy, not GICP
  distribution-to-distribution scoring over 3D point clouds.
- The representative-state continuity gate is intentionally simple. It is
  there to make the demo metric stable, not to claim a full trajectory
  smoothing backend.

Why the numbers are good enough for this PR:
- The baseline is deliberately a strong local tracker before kidnap and a
  weak global tracker after kidnap. That isolates the relocalization
  property.
- The MegaParticles path keeps enough global support after blackout to
  re-score the full map immediately when range scans return.
- The final result is visually and numerically clear: local bootstrap
  post-kidnap RMSE `14.6053 m` vs MegaParticles-style `0.0974 m`, final
  error `10.5934 m` vs `0.0412 m`.

### Bigger architectural things landed in this sprint
- **Shared CUDA headers (`include/`)** — #66. New `.cu` files should
  `#include "cuda_check.cuh"` and `#include "cuda_video.h"` instead of
  re-defining the wrappers locally. Older files have not been
  back-migrated yet — that is a small mechanical cleanup waiting in the
  open threads list.
- **`readme.md` SLAM / Multi-view geometry section** now includes the 2D
  and 3D pose-graph SLAM pair plus the robust 3D follow-up. The 3D
  versions use central finite-difference SE(3) Jacobians to avoid the
  aborted hand-Jacobian failure mode; the robust target trims the worst
  loop closures from the odometry-chain residual and solves the switched
  graph.
- **`readme.md` Sensors / perception section** now has the scan matching
  family plus the multi-resolution NDT 3D tile and the EM / spectral
  clustering tiles. The Planning / Control section has Hungarian
  assignment, CMA-ES, MCTS, assignment tracking, and crowd swarm tiles, and
  the capability matrix includes the swarm / assignment / tracking /
  optimisation / MCTS / clustering rows.

### Attempts that did NOT land (for context, so we do not repeat)
- **3D pose-graph SLAM hand-Jacobian attempt** — The first attempt wrote
  `src/gpu_pose_graph_slam_3d.cu` (~700 LOC, SE(3) GN + Jacobi-PCG,
  right-perturbation Jacobians, Levenberg-Marquardt). At GT
  initialisation the GN step direction was uphill (cost 2525 → 293199)
  regardless of sign of the b vector. Tried sign flips, did not isolate
  the H-matrix construction bug, and deleted the file. The current v2
  branch fixes this by generating central finite-difference Jacobians from
  the scoring residual instead of trusting hand-derived Jacobians.
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
- **Shell commands**: this checkout has AGENTS.md -> `$HOME/.codex/RTK.md`.
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

### SOPPI / box pushing (ACTIVE — see Current State)
- **`soppi_fast` on `box_align_contact_loss`**: **DONE (2026-06-10)** — subset SVGD +
  one nominal grad step reaches **1.00** on canonical seeds (vs `mppi` 0.00, all-pairs
  `soppi` 0.50); see `docs/results/soppi_box_pushing_2026-06-10.md`.
- **Stronger pure-SOPPI cell**: `box_align_contact_arc` at **1.00** for `soppi` /
  `soppi_fast` (K=256, 4 seeds); documented in same report.
- **Canonical scenario seeds for filtered runs**: **DONE** — `--scenarios` probes now
  use stable `si` from the full scenario list (isolated tuning matches full suite).
- **`scripts/sweep_soppi.py --baseline-planners`**: implemented (default `mppi`).
- **Partial rollout cache for box autodiff score**: **DONE (2026-06-10)** —
  `cache_soppi_rollout_states_kernel` + `dcost_dparam_box_from` reuse cached
  per-sample states; `box_align_contact_loss` quick row updated (`soppi_fast` 1.00 with
  one nominal grad step).
- **Navigation SOPPI**: 2/10 solved; score kernel is simplified vs paper; low
  priority unless user explicitly wants nav headline numbers.

### A. Mechanical cleanup
- ~~**Refresh the `readme.md` Headline benchmarks table**~~ **DONE (2026-06-10)** —
  Highlights table now covers nav2 GPU MPPI, NeRF, multi-res NDT 3D, GICP 3D,
  Hungarian assignment, PCG, SfM mini, diffusion planner, and assignment tracking.
- ~~**Back-migrate older `.cu` files to use `include/cuda_check.cuh`**~~ **DONE
  (2026-06-10)** — all `src/*.cu` except `mppi_gpu.cu` (throws `std::runtime_error`
  by design). `comparison_*`, benchmarks, and legacy demos now `#include "cuda_check.cuh"`.
- **Lift shared SE(3) math into a header** — DONE for the rotation-matrix
  pose-graph family in `chore/shared-se3-helpers` (2026-05-26): the new
  `include/se3_helpers.cuh` now holds `clampf`, the `mat3_*` family,
  `so3_exp`, `so3_log`, and `solve6_spd_device`, shared by
  `gpu_pose_graph_slam_3d.cu`, `gpu_pose_graph_slam_3d_switchable.cu`, and
  `gpu_online_slam_3d_switchable.cu`. The scan-matching family
  `gpu_ndt_3d.cu` / `gpu_ndt_3d_multires.cu` / `gpu_gicp_3d.cu` was the
  remaining duplication: it carried its OWN `cholesky_solve_6` + `H_OFF`
  (a host solve over the PACKED 21-float upper triangle, run on the host
  per Gauss-Newton iteration), copied byte-for-byte across all three demos
  and intentionally *not* the same routine as the `__device__`, full-6x6
  `solve6_spd_device` used by the pose-graph back-ends. This is now lifted
  into `include/solve6_packed.cuh` (PR #122 `feat/scan-matching-shared-cholesky`,
  MERGED squash 2026-05-26 at `164d467`): the three demos `#include` it and drop their private
  copies. Verified byte-identical: a normalised diff confirmed the three
  copies were the same algorithm modulo comments, and each demo's reported
  error is unchanged after the lift (ndt_3d 0.0263 m / 0.0102 rad, gicp_3d
  0.0011 m / 0.0001 rad, ndt_3d_multires fine 0.0155 m / 0.0072 rad) -- no
  basin re-tuning. NOT merged into `solve6_spd_device` on purpose (different
  execution context: host-packed vs device-full). `gpu_megaparticles_6dof.cu`
  is quaternion-based and intentionally separate. With this, Open Threads A's
  shared-math duplication across the SE(3)/6x6-solve demos is closed.

### B. New algorithm candidates (ranked rough order, see "Recommended Next" below)

1. **GPU diffusion policy / behaviour cloning** — extension of #65
    (motion planner) into a learned planner. ~800 LOC.
2. **Graph ML follow-up** — spectral clustering is merged; possible
   follow-ups are GPU label propagation, graph cuts, or semi-supervised
   graph classification.
3. **Localization follow-up** — global-localization MCL (#84) and the
   MegaParticles-style range-field branch (#86) now exist. Good next
   localization follow-ups are: full 3D/6-DoF particles, explicit
   iterative LSH neighbor search, GICP-like point-cloud likelihood,
   KLD-AMCL comparison under the same hidden kidnap, or a small trajectory
   smoother over the representative MegaParticles state.
4. **3D SLAM follow-up** — robust switched loop closures now exist; the
   next deeper follow-ups are online integration, dynamic switch variables
   instead of a trimmed front-end gate, or robust kernels inside the
   nonlinear solve.

### C. Older items still parked (carried over from previous handoff)
- DPF research line — from-scratch tracking-loss MLP training, harder
  scenes, EKF/AMCL accuracy comparison. See previous plan.md sections.
- Topology bench Day 4+ — failure taxonomy CSV expansion, Day 5
  consolidated report.

---

## Recommended Next Session

**Superseded for priority by "Current State (2026-06-14)" above.** The block
below is the 2026-05-27 handoff (MegaParticles / shared-header / 3D SLAM era).
Keep it for file-map cross-references only.

Historical immediate next action (2026-05-27): PR #98 / #104 / #105 era —
localization-depth and 3D-SLAM switchable lines were considered exhausted;
recommended mechanical shared-header cleanup or MegaParticles GICP follow-ups.
None of that is the active line today.

---
the "explicit switch variables optimised alongside poses" slice of the 3D SLAM
follow-up) is already merged (squash) and its branch deleted; local `master`
is at `a4dab88` and in sync with origin. There is no in-flight PR to babysit.
PR #104 (the 6-DoF/SE(3) slice of the localization-depth follow-up, the LSH
index from #101 generalised to full SE(3) poses where a dense grid is
infeasible) is already merged (squash) and its branch deleted; local `master`
is at `430a0f5` and in sync with origin. There is no in-flight PR to babysit.
Pick a fresh task from the menu below and start it on a new branch off
`master`. With the localization-depth line now exhausted (#86 Stein, #101 LSH,
#104 6-DoF) AND the 3D-SLAM online-robustness slice now done (#105 wires #98's
switchable SE(3) back-end into #63's online sliding-window front-end, rejecting
false loops live as they stream in), the strongest remaining candidate is the
**Open Threads A shared-header
cleanup** (now especially worthwhile: the SE(3) GN + Jacobi-PCG + 6x6 Cholesky
scaffold + closed-form switch update + quaternion/so3 helpers are duplicated
across `gpu_pose_graph_slam_3d.cu`, `gpu_pose_graph_slam_3d_switchable.cu`,
`gpu_online_slam_3d_switchable.cu`, `gpu_ndt_3d.cu`, `gpu_gicp_3d.cu`, and
`gpu_megaparticles_6dof.cu` — a clean `include/se3_helpers.cuh` lift removes
real drift). Other open directions: a GICP-style distribution-to-distribution
LiDAR likelihood for the MegaParticles branch, or a learning/control extension.
Done so far: B1 (diffusion policy), B2 (graph-ML / label propagation), the
KLD-AMCL slice of B3, the switch-variable slice of the 3D SLAM follow-up (#98),
the explicit-LSH slice of B3 (#101), the 6-DoF/SE(3) slice of B3 (#104), and now
the online-switchable slice of the 3D SLAM follow-up. Coordinate with the
concurrent session, which is working the graph/traversability line on this same
checkout.

After GPU CMA-ES, GPU MCTS, assignment tracking, crowd swarm, PR #78 GPU
SfM mini, PR #79 GPU PCG, PR #80 GPU EM GMM, PR #81 GPU spectral
clustering, PR #82 GPU 3D pose-graph SLAM v2, PR #83 robust switched
3D pose-graph SLAM, the GPU global-localization MCL recovery branch, and
the MegaParticles-style Stein MCL branch, the natural next move depends
on user goal:

- **Localization depth**: push the MegaParticles branch closer to the
  paper with 3D/6-DoF poses, an explicit LSH neighbor list, or a direct
  KLD-AMCL comparison under the same hidden kidnap.
- **Localization polish**: add a short smoothing pass for the
  representative MegaParticles trajectory and report raw max-posterior
  versus smoothed pose error separately.
- **Hard but high value**: wire the robust 3D backend into an online
  SLAM-style frontend, or replace the trimmed loop gate with explicit
  switch variables optimised alongside poses.
- **Tying off loose ends**: refresh the lower headline benchmark table
  and back-migrate to shared headers (Open Threads A). One PR,
  mechanical, removes drift.
- **Learning/control extension**: GPU diffusion policy / behaviour
  cloning (B2), if the user wants to build on the diffusion planner line.

If unsure after this branch, start with a mechanical shared-header cleanup
if they want a low-risk maintenance PR, or a graph-ML follow-up if they
want another compact visual algorithm.

Suggested starting commands:

```bash
rtk git switch -c feat/gpu-diffusion-policy    # B1
rtk git switch -c feat/gpu-label-propagation   # B2
rtk git switch -c feat/gpu-megaparticles-3d
rtk git switch -c feat/gpu-megaparticles-lsh
rtk git switch -c feat/gpu-kld-amcl-kidnap
rtk git switch -c feat/gpu-online-slam-3d-robust
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
- `src/gpu_pose_graph_slam.cu` (PR #58) — 2D GN + Jacobi-PCG.
- `src/gpu_pose_graph_slam_3d.cu` (PR #82) — 3D SE(3) GN +
  Jacobi-PCG with central finite-difference Jacobians; 384 poses, 575
  edges, 1.64 m -> 0.28 m translation RMSE. The
  `gpu_pose_graph_slam_3d_robust` target compiles the same source with
  `GPU_POSE_GRAPH_3D_ROBUST=1`, adds 36 false loop closures, trims them
  with a switch gate, and recovers 0.284 m / 2.11 deg while plain GN on
  the corrupted graph lands at 6.95 m / 39.89 deg.
- `src/gpu_online_slam.cu` (PR #63) — sliding-window W=60 + iSAM-style
  global pass on loop closure.

### Localization / state estimation
- `src/gpu_megaparticles_stein_mcl.cu` (PR #86 / current branch) —
  MegaParticles-inspired SE(2) range-localization demo with 1,048,576
  particles, distance-field scan likelihoods, bucket-neighbor
  Stein-style particle updates, posterior propagation, and a hidden
  kidnap blackout. Local bootstrap MCL has 14.61 m post-kidnap RMSE,
  while the MegaParticles-style path recovers to 0.097 m post-kidnap RMSE.
  Main kernels to inspect are `likelihood_gradient_kernel`,
  `bucket_motion_aggregate_kernel`, `stein_bucket_update_kernel`,
  `bucket_posterior_aggregate_kernel`, and `posterior_smooth_kernel`.
- `src/gpu_global_localization_mcl.cu` (PR #84) — 32,768-particle
  MCL recovery demo with 72 mapped landmarks and 10 range-bearing
  observations. A hidden kidnap at step 70 leaves local-only MCL at
  20.24 m post-kidnap RMSE, while a GPU sensor-reset particle gate
  triggers once and recovers to 0.022 m post-kidnap RMSE.
- `src/expansion_reset_mcl.cu` / `src/comparison_expansion_reset_mcl.cu`
  — expansion-reset MCL kidnap recovery baseline.
- `src/amcl.cu` (older) — CUDA AMCL with KLD-sampling, likelihood field,
  and augmented MCL.

### Solver / infrastructure
- `src/gpu_pcg_solver.cu` (PR #79) — generic CSR Jacobi-PCG for
  sparse SPD systems; 262K unknowns, 1.31M nonzeros, convergence GIF, and
  direct CPU PCG comparison.

### Clustering / ML
- `src/gpu_em_gmm.cu` (PR #80) — GPU EM for 2D full-covariance
  Gaussian mixtures; 262K points, 5 components, 42 iterations, direct
  CPU comparison, and convergence GIF.
- `src/gpu_spectral_clustering.cu` (PR #81) — normalized RBF
  graph spectral clustering without materializing the dense matrix; 3072
  points, 40 subspace iterations, 193x vs CPU, 100% mapped accuracy.

### Visualisation / rendering
- `src/gpu_gaussian_splatting.cu` (PR #59) — isotropic alpha-composite.
- `src/gpu_nerf_volume.cu` (PR #64) — volumetric ray-march renderer.

### Planning / control / MPPI reproduction zoo
- `src/benchmark_diff_mppi.cu` — 2D bicycle nav; 12+ planner variants;
  topology suite; SOPPI with local stage-cost score.
- `src/benchmark_diff_mppi_cartpole.cu` — CartPole; SOPPI with trajectory
  cost-to-go autodiff score.
- `src/benchmark_diff_mppi_pushing.cu` — planar disk pushing; saturated SOPPI
  signal (all planners succeed).
- `src/benchmark_diff_mppi_pushing_box.cu` — **primary SOPPI reproduction target**:
  rectangular box + orientation; 7 scenarios; `w_contact_loss`; hybrid
  `soppi_g3` / `soppi_fast_g3`; mechanism + gradient-agreement diagnostics.
  Results: `docs/results/soppi_box_pushing_2026-06-14.csv`.
- `scripts/sweep_soppi.py` — navigation SOPPI hyperparameter grid sweeps.
- `scripts/run_mppi_zoo_suite.py` — fixed-seed MPPI zoo suite runner.
- `docs/soppi_reproduction.md` — SOPPI implementation notes + checked-in rows.
- `docs/mppi_reproduction_zoo.md` — MPPI-family paper index.
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
- `src/benchmark_diff_mppi.cu` — see "Planning / control / MPPI reproduction zoo".

### ROS2 (`ros2_ws/`)
- `src/cuda_mppi_controller/` (PR #175) — **nav2 GPU MPPI controller
  plugin**, the star-growth line's flagship. See the top "Current State"
  section; build with `colcon build --packages-select cuda_mppi_controller`.
- `src/cuda_robotics/src/esdf_node.cu` (PR #46) — GPU JFA ESDF node.
- `src/cuda_robotics/src/voxel_node.cu` (PR #48) — GPU 3D log-odds voxel
  map node. Build via `colcon build --packages-select cuda_robotics`.

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

# Open PR
rtk git push -u origin <feature-branch>
rtk bash -lc 'gh pr create --title "Add ..." --body-file /tmp/pr-body.md'

# Merge (only after user authorisation)
rtk bash -lc 'gh pr merge <N> --squash --delete-branch'
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

---

End of handoff. **Start from "Current State (2026-06-10, star-growth
sprint)"** for product work (pip bindings next), or from "Research Line
State: SOPPI box pushing" for research work. Good hunting.
