# CudaRobotics Plan / Handoff (for Codex / Claude)

Last updated: 2026-05-21 JST

Short handoff for the next coding agent. The local checkout is currently on
`feat/diff-pf-injection-trigger`. The CUDA driver/library mismatch that had
blocked earlier validation has been resolved (kernel and userspace are now
both `driver-version-redacted`, `nvidia-smi` healthy), the DPF MLP injection-rate / ESS
trigger sweep has been run end-to-end, and the resulting numbers, gif, and
docs are committed locally; the remaining work is gh-pages publish + PR.

PR #39 (Gaussian splatting renderer) is still open as a draft with CI green.

---

## Repo State (2026-05-21)

- **Main branch**: `master`
- **gh-pages branch**: hosts gif assets referenced from `readme.md`. Update it
  whenever you add or regenerate a gif so the readme image URLs resolve.
- 100+ build targets (`87 .cu` / `14 .cpp` source files)
- Build:
  ```bash
  cd build && cmake .. && make -j$(nproc)
  ```
- The 2026-05-20/21 session merged PRs #16〜#23 and #25〜#38. PR #39 is open
  as a draft and CI is green. The repo's "Why CUDA?" top showcase is now a
  curated visual grid; the Research Extensions table includes DPF, 3D LiDAR,
  and Gaussian-splatting-style map rendering work.

---

## What Was Just Done (2026-05-20/21 session)

| PR | Title | Headline number |
|---|---|---|
| #16 | Rename `emcl2` → `expansion_reset_mcl`, readme 3-tile curation | License-clean (LGPL upstream vs MIT repo) |
| #17 | Per-variant `t_horizon`, split `hybrid_astar_mppi` for topology vs open-dynamic | open-dynamic 27/30 cells, `_long` solves bottleneck |
| #18 | Massive Lidar Simulator (`comparison_lidar_sim`) | per-ray **1737x** faster on GPU |
| #19 | Massive Reeds-Shepp Fan (`comparison_reeds_shepp_fan`) | per-path **5236x** faster on GPU |
| #20 | Expansion Reset MCL → 10K particles, top-showcase tile | Visual showcase |
| #21 | Differentiable Particle Filter base (`diff_pf`) | α-trained DPF RMSE 18.8m → 8.1m (**57% reduction**) |
| #22 | Fix Lidar Sim showcase: 2x2 hit blocks survive gif downscale | Visual bug fix |
| #23 | DPF + learnable MLP observation model (`diff_pf_mlp`) | MLP RMSE = handcrafted × 0.96 (drop-in replacement) |
| #25 | DPF MLP tracking-loss fine-tuning | Clean scene tracking-tuned MLP 6.16m vs Gaussian 6.97m (**0.88x**) |
| #26 | DPF MLP range-outlier hard scene | Tracking-tuned MLP 6.91m vs Gaussian 10.27m (**0.67x**) |
| #27 | DPF MLP occlusion+kidnap stress scene | Tracking-tuned MLP 7.56m vs Gaussian 10.38m (**0.73x**) |
| #28 | DPF observation scenario table | Paper-facing clean/outlier/occlusion matrix |
| #29 | DPF MLP biased-range scene | Supervised MLP 6.13m vs Gaussian 8.33m (**0.74x**) |
| #30 | DPF MLP biased-range tuning | Tracking-tuned MLP 6.64m vs Gaussian 8.33m (**0.80x**) |
| #31 | DPF direct observation surrogate | Direct-surrogate MLP 5.81m vs Gaussian 8.33m (**0.70x**) |
| #32 | DPF calibrated biased observation | Calibrated-surrogate MLP 6.03m vs Gaussian 8.41m (**0.72x**) |
| #33 | DPF calibrated outlier surrogate | Calibrated-surrogate MLP 7.04m vs Gaussian 9.80m (**0.72x**) |
| #34 | DPF calibrated occlusion surrogate | Calibrated-surrogate MLP 6.93m vs Gaussian 8.25m (**0.84x**) |
| #35 | DPF occlusion/kidnap ablations | Split occlusion-only vs kidnap-only recovery limits |
| #36 | DPF kidnap particle injection | Gaussian+injection 1.01m vs no-injection Gaussian 7.30m (**0.14x**) |
| #37 | Show expansion reset particles in MCL comparison | Merged visual fix: orange reset-cloud overlay is now visible after kidnap |
| #38 | Add 3D LiDAR simulator comparison | Merged: CPU 16x512 vs CUDA 64x2048 multi-ring raycast; ~651x faster per ray in animated sweep |
| #39 | Add Gaussian splatting map renderer | **Open draft**, CI green: CPU sparse vs CUDA dense Gaussian surfels; ~1381x faster per Gaussian |

All four findings docs are under `docs/`: `topology_bench_day1_findings.md`
through `topology_bench_day4_findings.md`.

Merged PR #25 picked up the DPF follow-up: `src/diff_pf_mlp.cu` now clones
the supervised MLP and fine-tunes the weights through tracking loss using
central finite differences over a soft-resampling DPF rollout plus Adam.
Latest local run:
handcrafted Gaussian **6.97 m**, supervised MLP **7.19 m**, tracking-tuned MLP
**6.16 m** (**0.88x** handcrafted). The gif
`gif/comparison_diff_pf_mlp.gif` was regenerated as a 3-panel comparison.

Merged PR #26 adds an intentionally
misspecified observation setting to `src/diff_pf_mlp.cu`: **18%** of visible
range measurements get uniform **±9 m** outliers. Latest local run:
handcrafted Gaussian **10.27 m**, supervised MLP **7.45 m**, tracking-tuned
MLP **6.91 m** (**0.67x** handcrafted). New gif:
`gif/comparison_diff_pf_mlp_hard.gif`.

Merged PR #27 adds a second DPF
stress scene: **30%** landmark dropout + **25%** short returns from
**t=1.0..16.0s**, plus a hidden pose jump of **(-4m, +3m)** at **t=3.0s**.
Latest local run: handcrafted Gaussian **10.38 m**, supervised MLP **7.66 m**,
tracking-tuned MLP **7.56 m** (**0.73x** handcrafted). New gif:
`gif/comparison_diff_pf_mlp_occlusion_kidnap.gif`.

Merged PR #28 added `paper/diff_pf_observation_scenarios.md`, consolidating
clean Gaussian, range-outlier, and occlusion+kidnap DPF observation-model
results into a single paper-facing matrix.

Merged PR #29 added the next ablation: distance-dependent range bias,
`z = d + N(0, 1.0) + 0.35 * max(0, d - 10.0)`. The first local run found
handcrafted Gaussian **8.33 m**, supervised MLP **6.13 m** (**0.74x**
handcrafted), tracking-tuned MLP **7.20 m** (**0.86x** handcrafted). New gif:
`gif/comparison_diff_pf_mlp_biased_range.gif`.

Merged PR #30 reduced tracking fine-tuning variance by averaging the
finite-difference gradient over two rollout seeds and restoring the best
held-out validation checkpoint. That run found handcrafted Gaussian **8.33 m**,
supervised MLP **6.76 m** (**0.81x** handcrafted), tracking-tuned MLP
**6.64 m** (**0.80x** handcrafted).

Merged PR #31 added a direct biased observation surrogate trained with
ordinary GPU backprop. That run found handcrafted Gaussian **8.33 m**,
supervised MLP **6.21 m** (**0.75x**), tracking-tuned MLP **6.94 m**
(**0.83x**), direct-surrogate MLP **5.81 m** (**0.70x**). Direct surrogate
training took **~2.9s**, versus **~86s** for the finite-difference tracking
update in the same run.

Merged PR #32 removed the hand-coded bias formula from the direct training
target. It fits a 24-bin bias curve and residual sigma from **8,192**
known-distance calibration traces, then trains the MLP surrogate with ordinary
GPU backprop. Latest local run: calibration bias RMSE **0.064 m**, estimated
sigma **1.002 m**, handcrafted Gaussian **8.41 m**, supervised MLP **7.11 m**
(**0.85x**), tracking-tuned MLP **7.03 m** (**0.84x**),
calibrated-surrogate MLP **6.03 m** (**0.72x**). Calibrated surrogate training
took **~2.8s**.

Merged PR #33 applied the same
trace-learned surrogate idea to the range-outlier scene. It fits a 96-bin
residual likelihood from **8,192** known-distance calibration traces, then
trains the MLP surrogate with ordinary GPU backprop. Latest local run:
residual RMSE **2.301 m**, tail log-likelihood at +/-6m **-3.464**,
handcrafted Gaussian **9.80 m**, supervised MLP **7.18 m** (**0.73x**),
tracking-tuned MLP **7.22 m** (**0.74x**), calibrated-surrogate MLP
**7.04 m** (**0.72x**). Calibrated surrogate training took **~2.8s**.

Merged PR #34 extended that
calibration path to the occlusion+kidnap scene. It samples known-distance
measurements during the occlusion window, skips dropped landmarks as invalid,
and fits the valid short-return residual tail with the same 96-bin likelihood.
Latest local run: **8,192** valid samples from **11,713** attempts, residual
RMSE **2.301 m**, log-likelihood at **-6m = -2.553**, handcrafted Gaussian
**8.25 m**, supervised MLP **7.40 m** (**0.90x**), tracking-tuned MLP
**8.08 m** (**0.98x**), calibrated-surrogate MLP **6.93 m** (**0.84x**).
Calibrated surrogate training took **~2.9s**.

Merged PR #35 split that stress scene
into occlusion-only and kidnap-only runs. `src/diff_pf_mlp.cu` now has
separate `OBS_OCCLUSION_ONLY` and `OBS_KIDNAP_ONLY` modes and runs both from
the same supervised pre-training checkpoint. Latest local run from the
follow-up branch: occlusion-only Gaussian **7.98 m**, supervised MLP
**7.27 m** (**0.91x**), tracking-tuned MLP **6.69 m** (**0.84x**),
calibrated-surrogate MLP **6.99 m** (**0.88x**); kidnap-only Gaussian
**7.66 m**, supervised MLP **8.24 m** (**1.08x**), tracking-tuned MLP
**12.27 m** (**1.60x**), calibrated-surrogate MLP **6.90 m** (**0.90x**).
Gifs:
`gif/comparison_diff_pf_mlp_occlusion_only.gif` and
`gif/comparison_diff_pf_mlp_kidnap_only.gif`.

Merged PR #36 added explicit
particle-support recovery to the kidnap-only ablation. After the hidden jump,
12% of particles are range-reset by sampling on valid landmark-measurement
circles before the next likelihood update. Latest local run:
no-injection Gaussian **7.30 m**, Gaussian + range-reset injection **1.01 m**
(**0.14x**), calibrated-surrogate MLP **5.87 m** (**0.80x**),
calibrated-surrogate + injection **1.08 m** (**0.15x**). New gif:
`gif/comparison_diff_pf_mlp_kidnap_injection.gif`.

Merged PR #37 fixed the Expansion Reset MCL showcase. The comparison used to
draw only post-likelihood/post-resample particles, so the actual expansion
reset cloud was not visible even though recovery worked.
`src/comparison_expansion_reset_mcl.cu` now snapshots reset-time particles and
overlays them in orange for a short window;
`gif/comparison_expansion_reset_mcl.gif` was regenerated and published to
`gh-pages`.

Merged PR #38 added `src/comparison_lidar3d_sim.cu`, a PR-sized 3D extension
of the 2D massive lidar simulator. It uses analytic 3D primitives, a spinning
multi-ring LiDAR model, and one CUDA thread per ray. Outputs include a
3-panel GIF (`gif/comparison_lidar3d_sim.gif`) with CPU sparse point cloud,
CUDA dense point cloud, and CUDA range image. Latest local run:
`64x2048` CPU **63.00 ms** vs CUDA **0.088 ms** (**715.7x** same-ray speedup),
`128x4096` CPU **248.19 ms** vs CUDA **0.116 ms** (**2144.9x**), correctness
max range error **0.000381 m**, label match **100%**, animated sweep about
**651x** faster per ray. GIF was published to `gh-pages`.

Open PR #39 adds `src/comparison_gaussian_splatting.cu`, a forward-only
Gaussian surfel renderer inspired by the EasyGaussianSplatting direction but
implemented as a small robotics map-visualization demo, not a training stack.
It renders CPU sparse vs CUDA dense Gaussian maps plus an opacity/density
panel. Latest local run: correctness check on **2,048** Gaussians, CPU
**35.07 ms**, CUDA **29.624 ms**, accumulator MAE **0.000000**; animated
average CPU **66.84 ms/frame** for **4,096** Gaussians vs CUDA
**0.77 ms/frame** for **65,536** Gaussians, about **1381x** faster per
Gaussian. `gif/comparison_gaussian_splatting.gif` was generated and published
to `gh-pages`. PR #39 status as of this handoff: draft, mergeable, GitHub
Actions build **SUCCESS**.

Commands already run for #39:

```bash
rtk cmake -S . -B build
rtk cmake --build build --target comparison_gaussian_splatting -j$(nproc)
rtk ./bin/comparison_gaussian_splatting
rtk git diff --check
```

To finish #39:

```bash
rtk gh pr ready 39
rtk gh pr merge 39 --squash --delete-branch \
  --subject "Add Gaussian splatting map renderer" --body ""
```

Current WIP branch `feat/diff-pf-injection-trigger` adds an injection-rate /
ESS-trigger ablation on the kidnap-only scene. `src/diff_pf_mlp.cu` now:

- computes normalized-weight effective sample size, `ESS = 1 / sum(w_i^2)`
  (clamped to `[1, N]` to stay sane on collapse);
- returns ESS and the actually applied injection rate per step via `PFStep`;
- sweeps fixed-rate policies at **1% / 3% / 6% / 12% / 24%**;
- sweeps ESS-triggered policies at **3%** and **12%** with threshold
  `ESS < 0.35 N`;
- generates a 4-panel GIF
  `gif/comparison_diff_pf_mlp_injection_trigger.gif` showing no injection,
  fixed 3%, fixed 12%, and ESS-triggered 12%.

Latest local run (Gaussian likelihood, kidnap-only, 240 frames):

| Policy | RMSE | Inj. steps | Mean ESS | Mean applied rate |
|---|---:|---:|---:|---:|
| No injection | 12.255 m | 0/240 | 0.47 N | 0.000 |
| Fixed 1% | 1.271 m | 210/240 | 0.37 N | 0.009 |
| Fixed 3% | 1.079 m | 210/240 | 0.40 N | 0.026 |
| **Fixed 6%** | **0.972 m** | 210/240 | 0.41 N | 0.052 |
| Fixed 12% | 0.994 m | 210/240 | 0.37 N | 0.105 |
| Fixed 24% | 0.979 m | 210/240 | 0.30 N | 0.210 |
| ESS 3%@0.35N | 1.130 m | 98/240 | 0.36 N | 0.012 |
| **ESS 12%@0.35N** | **1.081 m** | **77/240** | 0.39 N | 0.038 |

Recovery saturates near 6% fixed rate; ESS-triggered 12% matches the
order-of-magnitude recovery while firing on only 32% of post-jump steps and
cutting mean applied perturbation from 0.105 to 0.038. Local CUDA runtime is
healthy again (driver/library both `driver-version-redacted`).

Remaining work for this branch: copy the new GIF to `gh-pages`, commit the
code/doc/gif changes, push, open the PR, wait for CI, merge.

---

## Open Threads (parked but not abandoned)

### 1. Topology bench Day 4+ items (deferred during the visual-showcase sprint)

- **Failure taxonomy CSV expansion**: per-episode CSV currently has `success`
  / `final_dist` / `collisions`. Day 2 hooks call for adding
  `failure_type` (`stuck` / `collision` / `timeout` / `goal_miss`),
  `time_to_goal`, `min_clearance`. The new scenes (`dynamic_bottleneck`,
  `dynamic_crossing_with_topology`) distinguish "stuck at gate" vs
  "collided in gate" but the summary table cannot see that.
- **Day 5 consolidated report**: `paper/hybrid_astar_matrix_report.md`
  draft tying together Day 1〜4 findings + the 30-cell open-dynamic
  baseline + the new topology suite. Skeleton is implicit in the four
  Day N findings docs.

### 2. DPF follow-up (after scenario-table branch)

- **From-scratch tracking-loss MLP training**. The implemented branch uses
  supervised pre-training as the initialization, then does true end-to-end
  finite-difference fine-tuning. Scratch-only training decreases rollout loss
  but did not yet beat the supervised initialization on the 240-frame eval.
  Next work: longer horizons, multi-seed gradient averaging, or smoother
  resampling relaxations.
- **Particle-injection trigger ablation**: fixed 12% range-reset injection
  closes the kidnap-only recovery gap. A WIP branch now compares fixed rates
  against an effective-sample-size trigger, but runtime validation is blocked
  by the local NVIDIA driver/library mismatch described above.
- **Harder scenes beyond current stress tests**: longer kidnap recovery with
  injection trigger/rate tuning, or smoother resampling relaxations where
  observation-model learning and recovery mechanics can be separated cleanly.
- **EKF / AMCL baseline**: compare DPF tracking RMSE against the existing
  `amcl` and `pf` baselines for a clean accuracy / compute trade-off plot.

### 3. Brainstorm leftovers / next showcase options

- **Gaussian Splatting follow-ups**: after PR #39 merges, possible next work
  is loading point-cloud outputs into Gaussian surfels, adding approximate
  depth ordering / tile binning, or rendering real PLY/KITTI inputs. Keep this
  forward-rendering-only unless deliberately starting a larger training PR.
- **MathematicalRobotics-inspired optimization demos**: both
  `scomup/MathematicalRobotics` and `scomup/EasyGaussianSplatting` are MIT.
  Good repo-fitting ports would be GPU Gauss-Newton / pose graph / bundle
  adjustment demos. Do not directly copy large external implementations; keep
  PRs small and write CudaRobotics-native CUDA/C++ demos.
- **10K Boids / Crowd Swarm**: scale `multi_robot_planner` from 500 → 10000
  agents + flocking. Quick win for one more order-of-magnitude visual.
- **GPU MCTS Planner**: tree expansion parallelised across threads,
  grid-world or Sokoban; CudaRobotics has no MCTS yet.

---

## House Rules

- **Git**: no `Co-Authored-By` lines in commit messages — commits are
  user-authored only.
- **PR body**: no "Generated with Claude Code" or similar AI-attribution
  footers. Per `~/.claude/CLAUDE.md` global rule.
- **gh-pages workflow**: when you add a gif, push it to the `gh-pages`
  branch root (not `gif/` subfolder) so the published URL
  `https://rsasaki0109.github.io/CudaRobotics/<name>.gif` resolves. Pattern:
  ```bash
  cp gif/<name>.gif /tmp/x.gif && git checkout gh-pages
  cp /tmp/x.gif <name>.gif && git add <name>.gif
  git commit -m "..." && git push origin gh-pages
  git checkout <feature-branch>
  ```
  Pages build deploys in ~1–3 minutes.
- **License**: repo is MIT. If you port an algorithm from an LGPL/GPL
  upstream, write the implementation from the paper, not from upstream
  source. Document attribution in the source header (see
  `src/expansion_reset_mcl.cu` for the form).
- **Comparison gifs at >100K samples**: do not use `cv::circle` per
  sample in the visualisation loop — it is the bottleneck. Use direct
  pixel splatting (see `src/comparison_lidar_sim.cu::draw_dense_hits`).
  Write a 2×2 block per sample so points survive ffmpeg lanczos
  downscale + gif palette quantisation.

---

## Recommended Next Session

The 2026-05-20/21 session leaned hard into visual showcases and the DPF
research line. The most natural next moves:

**Immediate cleanup**: PR #39 is already open, draft, mergeable, and CI green.
Merge it first unless the user wants to inspect it.

**If goal = paper / research value**: first unblock CUDA runtime validation,
then finish `feat/diff-pf-injection-trigger`. Fixed 12% range-reset injection
works; the current WIP asks how low the injection rate can be, and whether an
ESS-triggered reset keeps the same recovery with less steady-state noise.

**If goal = OSS visibility**: after #39, the strongest next visual continuation
is Gaussian/point-cloud integration: render real point-cloud samples or 3D
LiDAR hits as Gaussian surfels, then optionally add depth sorting or tiled
accumulation.

**If goal = benchmark completeness**: pick up the **failure taxonomy CSV
expansion** (Day 4+ item 1). Mechanical change to
`src/benchmark_diff_mppi.cu`, useful for the eventual Day 5 paper draft.

Recommended starting branches:

```bash
git switch feat/diff-pf-injection-trigger      # current research WIP
git switch -c feat/gaussian-pointcloud-map     # possible next showcase track
git switch -c feat/failure-taxonomy-csv       # bench track
```

---

## File Map (key entry points)

- `readme.md` — top of repo. Top showcase is the 2×3 grid at line 11–18.
- `CLAUDE.md` — project-local rules for Claude Code (this file applies
  to Codex too).
- `paper/cudarobotics_systems_paper.md` — systems paper draft.
- `docs/topology_bench_day{1,2,3,4}_findings.md` — Day-by-day topology
  benchmark results.
- `include/autodiff_engine.cuh` — dual-number forward-mode autodiff used
  by Diff-MPPI and Differentiable PF.
- `include/gpu_mlp.cuh` — flat-array MLP with `forward_batch` and
  `train_step_backprop`, used by Neural SDF, Neuroevolution, and the
  new DPF MLP observation model.
- `src/diff_pf.cu` — DPF base (α-only learnable).
- `src/diff_pf_mlp.cu` — DPF with learnable MLP observation model.
- `src/benchmark_diff_mppi.cu` — 12-planner sweep + topology suite.
- `src/comparison_lidar_sim.cu` — 1M-ray comparison demo.
- `src/comparison_lidar3d_sim.cu` — merged 3D multi-ring LiDAR comparison demo.
- `src/comparison_gaussian_splatting.cu` — PR #39 forward-only Gaussian surfel renderer.
- `src/comparison_reeds_shepp_fan.cu` — 1M-path comparison demo.
