# CudaRobotics Plan / Handoff (for Codex)

Last updated: 2026-05-20 JST

Short handoff for the next coding agent. The repo is on `master`, synced with
`origin/master`, working tree clean before this file update.

---

## Repo State (2026-05-20)

- **Main branch**: `master`
- **gh-pages branch**: hosts gif assets referenced from `readme.md`. Update it
  whenever you add or regenerate a gif so the readme image URLs resolve.
- 100+ build targets (`87 .cu` / `14 .cpp` source files)
- Build:
  ```bash
  cd build && cmake .. && make -j$(nproc)
  ```
- The 2026-05-20 session merged PRs #16〜#23 and #25〜#29 (13 PRs). The repo's "Why CUDA?"
  top showcase is now a curated 2x3 grid; the Research Extensions table has
  two new entries (Differentiable Particle Filter + DPF with MLP observation).

---

## What Was Just Done (2026-05-20 session)

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

Current feature branch `feat/diff-pf-biased-range-tuning` reduces tracking
fine-tuning variance by averaging the finite-difference gradient over two
rollout seeds and restoring the best held-out validation checkpoint. Latest
local run: handcrafted Gaussian **8.33 m**, supervised MLP **6.76 m**
(**0.81x** handcrafted), tracking-tuned MLP **6.64 m** (**0.80x**
handcrafted).

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
- **Direct differentiable MLP observation tuning**: the current biased-range
  tuning branch uses two-seed finite differences and a held-out checkpoint,
  but this still costs roughly one minute for a 65-parameter MLP. Next work:
  smoother resampling relaxations or a direct differentiable surrogate.
- **Harder scenes beyond current stress tests**: longer kidnap recovery with
  particle injection, or smoother resampling relaxations where
  observation-model learning and recovery mechanics can be separated cleanly.
- **EKF / AMCL baseline**: compare DPF tracking RMSE against the existing
  `amcl` and `pf` baselines for a clean accuracy / compute trade-off plot.

### 3. Brainstorm leftovers (offered during the session, not picked)

- **3D Lidar Simulator**: 3D extension of `comparison_lidar_sim`. Same
  ray-cast kernel pattern, rotating sensor, dense indoor point cloud,
  flows into the point-cloud pipeline.
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

The 2026-05-20 session leaned hard into visual showcases and the DPF
research line. The most natural next moves:

**If goal = paper / research value**: replace the current finite-difference
MLP observation update with a smoother resampling relaxation or direct
differentiable surrogate. The biased-range scene now works, but the optimizer
is too expensive for a tiny MLP.

**If goal = OSS visibility**: pick up **3D Lidar Simulator**. The 2D
version is a strong tile on the readme; a 3D version paired with the
existing point-cloud pipeline closes a natural product loop.

**If goal = benchmark completeness**: pick up the **failure taxonomy CSV
expansion** (Day 4+ item 1). Mechanical change to
`src/benchmark_diff_mppi.cu`, useful for the eventual Day 5 paper draft.

Recommended starting branches:

```bash
git switch -c feat/diff-pf-direct-obs-grad    # research track
git switch -c feat/lidar-sim-3d               # showcase track
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
- `src/comparison_reeds_shepp_fan.cu` — 1M-path comparison demo.
