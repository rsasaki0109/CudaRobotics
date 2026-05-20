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
- The 2026-05-20 session merged PRs #16〜#23 (8 PRs). The repo's "Why CUDA?"
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

All four findings docs are under `docs/`: `topology_bench_day1_findings.md`
through `topology_bench_day4_findings.md`.

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

### 2. DPF follow-up (after PR #23 merge)

- **End-to-end MLP weight learning via finite-difference**. Right now the
  MLP is pre-trained supervised against the analytic likelihood. The natural
  next step is to fine-tune the MLP weights through tracking loss using
  finite-difference gradient over the soft-resample DPF chain — true
  end-to-end DPF training. Tractable because the MLP is tiny (~50 weights);
  see `src/diff_pf_mlp.cu` for the existing scaffolding.
- **Harder scenes**: kidnap recovery comparison (analytic vs MLP under noise
  model mismatch), or sensor occlusion / non-Gaussian noise where the
  analytic form is no longer optimal and the MLP can outperform.
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

**If goal = paper / research value**: pick up **DPF end-to-end MLP
weight learning** (item 2 above). The scaffolding is in `src/diff_pf_mlp.cu`;
need to add finite-difference gradient computation over tracking loss with
the MLP weights as the perturbed parameter, plus an Adam loop on those
weights. Story: "DPF learns observation model from scratch via tracking
loss" — paper-worthy.

**If goal = OSS visibility**: pick up **3D Lidar Simulator**. The 2D
version is a strong tile on the readme; a 3D version paired with the
existing point-cloud pipeline closes a natural product loop.

**If goal = benchmark completeness**: pick up the **failure taxonomy CSV
expansion** (Day 4+ item 1). Mechanical change to
`src/benchmark_diff_mppi.cu`, useful for the eventual Day 5 paper draft.

Recommended starting branches:

```bash
git switch -c feat/diff-pf-end-to-end-mlp     # research track
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
