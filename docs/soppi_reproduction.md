# SOPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **Stein-based Optimization of Sampling Distributions in Model Predictive Path Integral Control** by Jace Aldrich and Odest Chadwicke Jenkins.

- arXiv: https://arxiv.org/abs/2511.02015
- Latest arXiv metadata: submitted 2025-11-03, revised 2026-03-30.
- Core idea: Stein-Optimized Path-Integral Inference (SOPPI) applies SVGD updates to MPPI action samples so the proposal distribution spreads toward lower-cost regions before the standard MPPI weighted update.

Step-MPPI (`arXiv:2604.01539`) was checked first and is already represented by the existing `step_mppi` variant in `src/benchmark_diff_mppi.cu`, so SOPPI was selected as the next missing MPPI-family reproduction.

## Implementation

Implemented a lightweight SOPPI variant in `src/benchmark_diff_mppi.cu`:

- Added `PlannerVariant::use_soppi_sampling` and hyperparameters:
  - `soppi_svgd_iters`
  - `soppi_step_size`
  - `soppi_bandwidth`
  - `soppi_neighbor_count` (`0` = all-pairs, `>0` = deterministic particle subset)
- Added `soppi` planner registration.
- Added `soppi_fast` planner registration with `soppi_neighbor_count=32`.
- Added `soppi_svgd_step_kernel`, which updates each per-timestep action particle with an RBF-kernel SVGD step in normalized `(accel, steer)` action space.
- Added `rollout_fixed_controls_kernel` to re-evaluate moved particles before MPPI weighting.
- Added CLI overrides:
  - `--override-soppi-iters`
  - `--override-soppi-step-size`
  - `--override-soppi-bandwidth`
  - `--override-soppi-neighbors`

Added `scripts/sweep_soppi.py` for reproducible parameter sweeps against MPPI. The runner executes MPPI once, then executes SOPPI across the requested grid, and writes per-run CSVs, a combined CSV, a manifest, and a Markdown summary. The script supports `--neighbors` to compare all-pairs SOPPI against the faster subset approximation.

Implemented a CartPole SOPPI variant in `src/benchmark_diff_mppi_cartpole.cu`:

- Added `soppi` and `soppi_fast` planner registrations.
- Added the same CLI overrides as the 2D navigation benchmark.
- Added per-sample rollout-state storage and fixed-control re-rollout after SVGD.
- Added `sample_action_gradient_kernel`, which computes a per-sample cost-to-go action gradient by backpropagating through the CartPole dynamics. This is closer to the SOPPI paper's trajectory-cost score than the lightweight 2D navigation implementation.
- Fixed CartPole benchmark seeds so planners share the same initial state for a scenario/K/seed cell. The old seed formula included the planner index, which made planner comparisons depend on different initial states.

Implemented a planar pushing SOPPI variant in `src/benchmark_diff_mppi_pushing.cu`:

- Added `soppi` and `soppi_fast` planner registrations.
- Added the same SOPPI CLI overrides.
- Added `push_sample_grad_kernel`, which evaluates the existing differentiable contact rollout gradient for every sampled control sequence.
- Added `push_fixed_rollout_kernel` to re-evaluate moved samples before MPPI weighting.
- Fixed pushing benchmark seeds so planners share the same scenario/K/seed initial condition.

Implemented a box pushing SOPPI variant in `src/benchmark_diff_mppi_pushing_box.cu`:

- Added `soppi` and `soppi_fast` planner registrations.
- Added `soppi_g3` and `soppi_fast_g3` hybrid registrations that combine SVGD
  sampling with three nominal Diff-MPPI grad steps (`grad_steps=3`, `alpha=0.010`).
- Added the same SOPPI CLI overrides.
- Added `sample_grad_kernel`, which evaluates the existing differentiable box-contact rollout gradient for every sampled control sequence.
- Added `fixed_rollout_kernel` to re-evaluate moved samples before MPPI weighting.
- Fixed box-pushing benchmark seeds so planners share the same scenario/K/seed initial condition within a run.
- Added `w_contact_loss` stage penalty (squared pusher-box gap) to `BoxParams`, wired
  through float rollout cost, dual autodiff, and the SOPPI score kernel.
- Added `box_align_contact_loss` scenario (`box_align_strict` geometry,
  `w_near=0`, `w_contact_loss=47`).

## Scope Caveats

This is a reproduction scaffold, not yet a full paper-faithful reproduction:

- The 2D navigation SVGD score uses the local stage-cost gradient at each rollout state/action, not a full differentiable dynamics-through-time gradient.
- The CartPole SVGD score uses differentiable cost-to-go gradients through the sampled rollout, but it is still not the paper's RNN gradient approximation for black-box dynamics.
- The planar pushing SVGD score uses differentiable contact cost gradients through each sampled control sequence, but the current built-in pushing scenarios are easy enough that all planners already succeed.
- The box-pushing SVGD score also uses differentiable contact cost gradients through each sampled control sequence. It exposes stronger sample-distribution effects than disk pushing, but the built-in success thresholds are still strict enough that many cells improve final error/cost without reaching success.
- The implementation applies SVGD independently at each horizon timestep to keep the kernel dimension meaningful, matching the paper's motivation, but it does not reproduce the paper's RNN gradient approximation for non-differentiable dynamics.
- Runtime is `O(K^2 * T)` when `soppi_neighbor_count=0`. `soppi_fast` reduces this to `O(K * N * T)` with a deterministic particle subset of size `N`, which is much faster but approximate.
- The 2026-06-09 kernel pass precomputes per-timestep SVGD scores in `O(K * T)` before the neighbor aggregation, and removes per-iteration control-buffer copies.
- Current validation is on the existing 2D bicycle navigation, CartPole, planar pushing, and box-pushing benchmarks, not the paper's 7-DOF arm pushing or planar biped tasks.
- Scenario filtering can change the scenario index used in the benchmark seed formula. Planner comparisons within a single command share seeds, but separate commands with different `--scenarios` ordering should not be treated as exact seed-matched reruns.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Initial grid sweep:

```bash
python3 scripts/sweep_soppi.py \
  --bin bin/benchmark_diff_mppi \
  --output-dir build-docker-smoke/soppi_sweep \
  --scenarios dynamic_crossing,cluttered \
  --k-values 128,256 \
  --seed-count 1 \
  --step-sizes 0.015,0.025,0.045,0.075 \
  --bandwidths 1.0,2.0,4.0 \
  --iters 1
```

Confirmation sweep:

```bash
python3 scripts/sweep_soppi.py \
  --bin bin/benchmark_diff_mppi \
  --output-dir build-docker-smoke/soppi_sweep_confirm \
  --scenarios dynamic_crossing,cluttered \
  --k-values 128,256 \
  --seed-count 3 \
  --step-sizes 0.025,0.075 \
  --bandwidths 1.0,2.0 \
  --iters 1
```

Fast neighbor sweep:

```bash
python3 scripts/sweep_soppi.py \
  --bin bin/benchmark_diff_mppi \
  --output-dir build-docker-smoke/soppi_fast_sweep \
  --scenarios dynamic_crossing,cluttered \
  --k-values 128,256 \
  --seed-count 3 \
  --step-sizes 0.075 \
  --bandwidths 2.0 \
  --iters 1 \
  --neighbors 0,16,32,64
```

Direct planner smoke:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing \
  --planners soppi_fast,soppi,mppi \
  --k-values 256 \
  --seed-count 1 \
  --csv build-docker-smoke/soppi_fast_check.csv
```

CartPole comparison:

```bash
./bin/benchmark_diff_mppi_cartpole \
  --quick \
  --scenarios cartpole_recover,cartpole_large_angle \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256,512 \
  --seed-count 3 \
  --csv build-docker-smoke/soppi_cartpole_compare.csv
```

CartPole SOPPI sweep:

```bash
python3 scripts/sweep_soppi.py \
  --bin bin/benchmark_diff_mppi_cartpole \
  --output-dir build-docker-smoke/soppi_cartpole_sweep \
  --scenarios cartpole_recover,cartpole_large_angle \
  --k-values 256,512 \
  --seed-count 3 \
  --step-sizes 0.005,0.015,0.03,0.06,0.12 \
  --bandwidths 0.5,1.0,2.0 \
  --iters 1 \
  --neighbors 0,32
```

Planar pushing comparison:

```bash
./bin/benchmark_diff_mppi_pushing \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv build-docker-smoke/soppi_pushing_compare.csv
```

Planar pushing SOPPI sweep:

```bash
python3 scripts/sweep_soppi.py \
  --bin bin/benchmark_diff_mppi_pushing \
  --output-dir build-docker-smoke/soppi_pushing_sweep \
  --scenarios push_straight,push_diagonal \
  --k-values 128,256 \
  --seed-count 4 \
  --step-sizes 0.03,0.06,0.12 \
  --bandwidths 1.0,2.0 \
  --iters 1 \
  --neighbors 0,32
```

Box pushing comparison:

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-10.csv
```

Box pushing SOPPI sweep:

```bash
python3 scripts/sweep_soppi.py \
  --bin bin/benchmark_diff_mppi_pushing_box \
  --output-dir build-docker-smoke/soppi_pushing_box_sweep \
  --scenarios box_turn,box_align,box_pivot,box_swivel \
  --k-values 128,256 \
  --seed-count 4 \
  --step-sizes 0.03,0.06,0.12 \
  --bandwidths 1.0,2.0 \
  --iters 1 \
  --neighbors 0,32
```

## Confirmation Results

Artifacts:

- `build-docker-smoke/soppi_sweep_confirm/soppi_sweep_summary.md`
- `build-docker-smoke/soppi_sweep_confirm/soppi_sweep_combined.csv`
- `build-docker-smoke/soppi_sweep_confirm/manifest.csv`

Best SOPPI by scenario/K from the seed-count 3 confirmation sweep:

| Scenario | K | Best SOPPI | Success | Final Distance | Cost | Avg ms | MPPI Final Distance | MPPI Cost | MPPI Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | s=0.025, b=2, i=1 | 0.00 | 38.50 | 49939.8 | 1.96 | 38.50 | 49955.5 | 0.16 |
| cluttered | 256 | s=0.075, b=2, i=1 | 0.00 | 38.50 | 49885.2 | 3.66 | 38.52 | 49927.6 | 0.20 |
| dynamic_crossing | 128 | s=0.025, b=1, i=1 | 0.00 | 3.38 | 46204.8 | 1.39 | 3.39 | 46253.3 | 0.13 |
| dynamic_crossing | 256 | s=0.075, b=2, i=1 | 0.00 | 2.77 | 45400.5 | 2.51 | 2.93 | 45545.0 | 0.16 |

Observed pattern:

- SOPPI reduces cumulative cost consistently in the tested cells.
- `dynamic_crossing, K=256` shows the clearest final-distance improvement: `2.93 -> 2.77`.
- `cluttered` mostly shows cost improvement without meaningful final-distance movement.
- No tested configuration reached the goal in these two smoke scenarios.
- The all-pairs SVGD kernel is much slower than MPPI, so the next engineering step is reducing interaction cost before broad benchmarking.

## Fast Neighbor Results

Artifacts:

- `build-docker-smoke/soppi_fast_sweep/soppi_sweep_summary.md`
- `build-docker-smoke/soppi_fast_sweep/soppi_sweep_combined.csv`
- `build-docker-smoke/soppi_fast_sweep/manifest.csv`
- `build-docker-smoke/soppi_fast_check.csv`

Best SOPPI by scenario/K from the seed-count 3 neighbor sweep:

| Scenario | K | Best SOPPI | Success | Final Distance | Cost | Avg ms | MPPI Final Distance | MPPI Cost | MPPI Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | s=0.075, b=2, i=1, n=32 | 0.00 | 38.50 | 49922.2 | 0.61 | 38.50 | 49955.5 | 0.23 |
| cluttered | 256 | s=0.075, b=2, i=1, n=16 | 0.00 | 38.50 | 49887.4 | 0.44 | 38.52 | 49927.6 | 0.27 |
| dynamic_crossing | 128 | s=0.075, b=2, i=1, n=64 | 0.00 | 3.41 | 46142.7 | 0.70 | 3.39 | 46253.3 | 0.20 |
| dynamic_crossing | 256 | s=0.075, b=2, i=1, n=0 | 0.00 | 2.77 | 45400.5 | 2.39 | 2.93 | 45545.0 | 0.32 |

Direct planner smoke on `dynamic_crossing, K=256, seed-count=1`:

| Planner | Final Distance | Cost | Avg ms |
|---|---:|---:|---:|
| mppi | 3.20 | 46129.6 | 0.19 |
| soppi | 3.08 | 45775.3 | 2.68 |
| soppi_fast | 2.97 | 45669.5 | 0.48 |

Observed pattern:

- `neighbors=16/32` keeps the cost reduction close to all-pairs SOPPI in the tested `K=256` cells while cutting average control time by roughly 5x to 8x versus `neighbors=0`.
- `dynamic_crossing, K=256` still gets the strongest final-distance improvement from all-pairs SOPPI, but `neighbors=32/64` is nearly identical in final distance and cost at much lower runtime.
- `cluttered` remains insensitive in final distance; the improvement is mostly lower cumulative cost.
- Existing smoke reproducibility suite passed after the `soppi_fast` change: `build-docker-smoke/repro_suite_after_soppi_fast/report.md`.

## CartPole Results

Artifacts:

- `build-docker-smoke/soppi_cartpole_compare.csv`
- `build-docker-smoke/soppi_cartpole_compare_summary.md`
- `build-docker-smoke/soppi_cartpole_sweep/soppi_sweep_summary.md`
- `build-docker-smoke/repro_suite_cartpole_soppi/report.md`

Default planner comparison, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| cartpole_recover | 256 | mppi | 0.00 | 0.68 | 1124.0 | 0.19 |
| cartpole_recover | 256 | diff_mppi_3 | 0.00 | 0.52 | 1019.1 | 0.58 |
| cartpole_recover | 256 | soppi | 0.00 | 0.61 | 1094.6 | 0.46 |
| cartpole_recover | 256 | soppi_fast | 0.00 | 0.62 | 1134.3 | 0.38 |
| cartpole_recover | 512 | mppi | 0.33 | 0.81 | 661.6 | 0.16 |
| cartpole_recover | 512 | diff_mppi_3 | 0.33 | 0.80 | 586.6 | 0.63 |
| cartpole_recover | 512 | soppi | 0.33 | 0.83 | 643.1 | 0.60 |
| cartpole_recover | 512 | soppi_fast | 0.33 | 0.80 | 621.9 | 0.42 |
| cartpole_large_angle | 256 | mppi | 0.00 | 1.28 | 2417.7 | 0.12 |
| cartpole_large_angle | 256 | diff_mppi_3 | 0.00 | 1.27 | 2362.6 | 0.58 |
| cartpole_large_angle | 256 | soppi | 0.00 | 1.28 | 2388.3 | 0.49 |
| cartpole_large_angle | 256 | soppi_fast | 0.00 | 1.27 | 2384.5 | 0.42 |
| cartpole_large_angle | 512 | mppi | 0.00 | 1.33 | 2417.8 | 0.18 |
| cartpole_large_angle | 512 | diff_mppi_3 | 0.00 | 1.29 | 2378.0 | 0.61 |
| cartpole_large_angle | 512 | soppi | 0.00 | 1.35 | 2397.9 | 0.66 |
| cartpole_large_angle | 512 | soppi_fast | 0.00 | 1.31 | 2392.1 | 0.43 |

Best SOPPI from the CartPole sweep:

| Scenario | K | Best SOPPI | Success | Final Distance | Cost | Avg ms | MPPI Final Distance | MPPI Cost | MPPI Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cartpole_large_angle | 256 | s=0.12, b=1, i=1, n=32 | 0.00 | 1.25 | 2378.1 | 0.39 | 1.28 | 2417.7 | 0.18 |
| cartpole_large_angle | 512 | s=0.03, b=0.5, i=1, n=32 | 0.00 | 1.28 | 2394.4 | 0.44 | 1.33 | 2417.8 | 0.21 |
| cartpole_recover | 256 | s=0.12, b=1, i=1, n=0 | 0.33 | 0.63 | 1093.9 | 0.46 | 0.68 | 1124.0 | 0.17 |
| cartpole_recover | 512 | s=0.06, b=2, i=1, n=32 | 0.33 | 0.80 | 621.9 | 0.43 | 0.81 | 661.6 | 0.22 |

Observed pattern:

- CartPole SOPPI is now a meaningful MPPI improvement in several cells, especially lower cumulative cost.
- `soppi_fast` is the better default CartPole tradeoff: it is close to tuned SOPPI quality and much cheaper than all-pairs.
- Diff-MPPI remains stronger than default SOPPI in the current CartPole setup, so SOPPI is not yet a replacement for the adjoint-refinement baseline.
- The harder `cartpole_large_angle` scenario remains unsolved by all methods under this quick budget.
- CartPole repro suite passed after the change: `build-docker-smoke/repro_suite_cartpole_soppi/report.md`.

## Planar Pushing Results

Artifacts:

- `build-docker-smoke/soppi_pushing_compare.csv`
- `build-docker-smoke/soppi_pushing_compare_summary.md`
- `build-docker-smoke/soppi_pushing_sweep/soppi_sweep_summary.md`

Default planner comparison, seed-count 4, `K=256`:

| Scenario | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---|---:|---:|---:|---:|---:|
| push_diagonal | mppi | 1.00 | 31.8 | 0.19 | 3.3 | 0.12 |
| push_diagonal | diff_mppi_1 | 1.00 | 29.8 | 0.18 | 3.2 | 0.23 |
| push_diagonal | diff_mppi_3 | 1.00 | 27.2 | 0.18 | 3.0 | 0.48 |
| push_diagonal | soppi | 1.00 | 31.0 | 0.19 | 3.3 | 0.40 |
| push_diagonal | soppi_fast | 1.00 | 31.0 | 0.18 | 3.2 | 0.30 |
| push_straight | mppi | 1.00 | 30.8 | 0.19 | 2.2 | 0.14 |
| push_straight | diff_mppi_1 | 1.00 | 25.8 | 0.19 | 1.9 | 0.33 |
| push_straight | diff_mppi_3 | 1.00 | 25.0 | 0.18 | 1.8 | 0.69 |
| push_straight | soppi | 1.00 | 30.0 | 0.19 | 2.2 | 0.58 |
| push_straight | soppi_fast | 1.00 | 30.0 | 0.18 | 2.2 | 0.34 |

Best SOPPI from the planar pushing sweep:

| Scenario | K | Best SOPPI | Success | Final Distance | Cost | Avg ms | MPPI Final Distance | MPPI Cost | MPPI Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| push_diagonal | 128 | s=0.12, b=2, i=1, n=32 | 1.00 | 0.18 | 3.5 | 0.24 | 0.19 | 3.6 | 0.09 |
| push_diagonal | 256 | s=0.06, b=1, i=1, n=32 | 1.00 | 0.18 | 3.3 | 0.27 | 0.19 | 3.4 | 0.11 |
| push_straight | 128 | s=0.12, b=1, i=1, n=32 | 1.00 | 0.18 | 2.1 | 0.24 | 0.18 | 2.1 | 0.08 |
| push_straight | 256 | s=0.12, b=1, i=1, n=0 | 1.00 | 0.18 | 2.2 | 0.38 | 0.18 | 2.2 | 0.12 |

Observed pattern:

- The SOPPI implementation works on differentiable contact rollouts, but the current pushing scenarios are saturated: MPPI already has `1.00` success.
- SOPPI can shave a small amount off final distance/cost in the sweep, but the effect is not large enough to be a strong reproduction result.
- Diff-MPPI remains the stronger pushing baseline because it reduces steps more clearly.
- The next useful task is not more tuning on these two scenarios; it is a harder pushing setup with contact loss, obstacle detours, or box orientation.

## Box Pushing Results

Latest checked-in fixed-seed run (seven scenarios, includes `box_align_contact_loss`):

- Report: [`results/soppi_box_pushing_2026-06-14.md`](results/soppi_box_pushing_2026-06-14.md)
- CSV: [`results/soppi_box_pushing_2026-06-14.csv`](results/soppi_box_pushing_2026-06-14.csv)

Predecessor hybrid-detour row:

- Report: [`results/soppi_box_pushing_2026-06-13.md`](results/soppi_box_pushing_2026-06-13.md)
- CSV: [`results/soppi_box_pushing_2026-06-13.csv`](results/soppi_box_pushing_2026-06-13.csv)

Predecessor obstacle-only row (no hybrid planners):

- Report: [`results/soppi_box_pushing_2026-06-12.md`](results/soppi_box_pushing_2026-06-12.md)
- CSV: [`results/soppi_box_pushing_2026-06-12.csv`](results/soppi_box_pushing_2026-06-12.csv)

Predecessor five-scenario run:

- Report: [`results/soppi_box_pushing_2026-06-11.md`](results/soppi_box_pushing_2026-06-11.md)
- CSV: [`results/soppi_box_pushing_2026-06-11.csv`](results/soppi_box_pushing_2026-06-11.csv)

Predecessor four-scenario run:

- Report: [`results/soppi_box_pushing_2026-06-10.md`](results/soppi_box_pushing_2026-06-10.md)
- CSV: [`results/soppi_box_pushing_2026-06-10.csv`](results/soppi_box_pushing_2026-06-10.csv)

Legacy smoke artifacts (pre-check-in):

- `build-docker-smoke/soppi_pushing_box_compare.csv`
- `build-docker-smoke/soppi_pushing_box_compare_summary.md`
- `build-docker-smoke/soppi_pushing_box_sweep/soppi_sweep_summary.md`

Default planner comparison, seed-count 4, `K=256`:

| Scenario | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---|---:|---:|---:|---:|---:|
| box_align | mppi | 0.00 | 240.0 | 0.43 | 7.8 | 0.12 |
| box_align | diff_mppi_3 | 0.50 | 149.8 | 0.40 | 7.1 | 2.36 |
| box_align | soppi | 0.00 | 240.0 | 0.28 | 4.2 | 0.50 |
| box_align | soppi_fast | 0.00 | 240.0 | 0.28 | 4.2 | 0.28 |
| box_pivot | mppi | 0.00 | 240.0 | 0.11 | 1.2 | 0.13 |
| box_pivot | diff_mppi_3 | 0.00 | 240.0 | 0.12 | 1.0 | 2.28 |
| box_pivot | soppi | 0.00 | 240.0 | 0.11 | 1.1 | 0.47 |
| box_pivot | soppi_fast | 0.00 | 240.0 | 0.11 | 1.1 | 0.31 |
| box_swivel | mppi | 0.75 | 103.0 | 0.28 | 1.8 | 0.16 |
| box_swivel | diff_mppi_3 | 0.75 | 100.0 | 0.27 | 1.8 | 2.35 |
| box_swivel | soppi | 1.00 | 98.0 | 0.22 | 1.7 | 0.55 |
| box_swivel | soppi_fast | 0.75 | 100.8 | 0.24 | 1.8 | 0.29 |
| box_turn | mppi | 0.00 | 260.0 | 0.41 | 4.6 | 0.15 |
| box_turn | diff_mppi_3 | 0.00 | 260.0 | 0.39 | 4.1 | 2.36 |
| box_turn | soppi | 0.00 | 260.0 | 0.40 | 4.5 | 0.43 |
| box_turn | soppi_fast | 0.00 | 260.0 | 0.40 | 4.5 | 0.28 |
| box_align_strict | mppi | 0.75 | 121.0 | 0.28 | 4.5 | 0.09 |
| box_align_strict | diff_mppi_3 | 1.00 | 71.0 | 0.27 | 4.0 | 1.75 |
| box_align_strict | soppi | 0.50 | 159.0 | 0.28 | 4.4 | 0.31 |
| box_align_strict | soppi_fast | 0.75 | 119.0 | 0.28 | 4.1 | 0.19 |

`box_align_strict` reuses the `box_align` geometry with `pos_tol=0.28 m` and
`ang_tol=0.08 rad`. The combined gate turns the parent near-misses into partial
success for sampling planners and full success for Diff-MPPI.

| Scenario | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---|---:|---:|---:|---:|---:|
| box_align_detour | mppi | 0.00 | 280.0 | 0.29 | 2.9 | 0.42 |
| box_align_detour | diff_mppi_3 | 0.25 | 219.5 | 0.26 | 2.1 | 3.06 |
| box_align_detour | soppi | 0.00 | 280.0 | 0.29 | 2.9 | 0.85 |
| box_align_detour | soppi_fast | 0.00 | 280.0 | 0.30 | 2.9 | 0.74 |

`box_align_detour` adds a narrow axis-aligned wall on the direct push lane and
requires collision-free success. Only `diff_mppi_3` clears a seed in the checked-in
run; treat this as a gradient-positive / sampling-negative obstacle cell.

| Scenario | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---|---:|---:|---:|---:|---:|
| box_align_contact_loss | mppi | 0.00 | 240.0 | 0.29 | 4.9 | 0.29 |
| box_align_contact_loss | diff_mppi_3 | 1.00 | 44.0 | 0.28 | 2.8 | 2.77 |
| box_align_contact_loss | soppi | 0.25 | 216.2 | 0.29 | 4.7 | 0.87 |
| box_align_contact_loss | soppi_fast | 0.00 | 240.0 | 0.29 | 4.9 | 0.55 |

`box_align_contact_loss` penalizes pusher-box gap during rollout. Pure all-pairs
`soppi` reaches `0.25` success while vanilla `mppi` stays at `0.00` — a
contact-loss cell where SVGD helps without nominal Diff-MPPI grad steps.

Best SOPPI from the box-pushing sweep:

| Scenario | K | Best SOPPI | Success | Final Distance | Cost | Avg ms | MPPI Final Distance | MPPI Cost | MPPI Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| box_align | 128 | s=0.03, b=2, i=1, n=0 | 0.00 | 0.28 | 3.4 | 0.88 | 0.28 | 3.6 | 0.10 |
| box_align | 256 | s=0.12, b=1, i=1, n=32 | 0.00 | 0.27 | 3.7 | 0.99 | 0.28 | 4.2 | 0.13 |
| box_pivot | 128 | s=0.03, b=1, i=1, n=0 | 0.00 | 0.11 | 1.2 | 0.86 | 0.11 | 1.2 | 0.10 |
| box_pivot | 256 | s=0.12, b=2, i=1, n=0 | 0.00 | 0.11 | 1.1 | 1.04 | 0.11 | 1.1 | 0.11 |
| box_swivel | 128 | s=0.12, b=1, i=1, n=0 | 1.00 | 0.21 | 1.6 | 0.87 | 0.27 | 2.2 | 0.09 |
| box_swivel | 256 | s=0.06, b=2, i=1, n=32 | 0.75 | 0.27 | 2.2 | 1.00 | 0.27 | 2.3 | 0.11 |
| box_turn | 128 | s=0.12, b=1, i=1, n=0 | 0.00 | 0.40 | 4.5 | 0.85 | 0.40 | 4.6 | 0.09 |
| box_turn | 256 | s=0.06, b=1, i=1, n=0 | 0.00 | 0.39 | 4.3 | 1.06 | 0.40 | 4.4 | 0.10 |

Observed pattern:

- Box pushing is a better SOPPI reproduction target than disk pushing. All-pairs
  `soppi` improves `box_swivel` from `0.75` MPPI success to `1.00` in the
  checked-in run; `soppi_fast` matches MPPI at `0.75` on this cell.
- `box_align` shows a large final-distance/cost reduction (`0.43 -> 0.28`,
  `7.8 -> 4.2` for `soppi_fast`) even though it does not cross the strict
  success threshold in this quick run.
- `box_align_strict` is the new orientation-binding cell: Diff-MPPI reaches
  `1.00` success; `soppi_fast` ties MPPI at `0.75` with lower cost (`4.1` vs `4.5`).
- `box_turn` and `box_pivot` are mostly insensitive under the current quick budget.
- Post-kernel `soppi_fast` is about **3.4x faster** than the pre-optimization note
  on `box_swivel` (`1.00 ms -> 0.29 ms`) and about **1.8x slower than MPPI**,
  down from roughly 9x slower before the kernel pass.

## Navigation Suite, 2026-06-10

Checked-in fixed-seed suite row:

- Report: [`results/mppi_zoo_suite_2026-06-10.md`](results/mppi_zoo_suite_2026-06-10.md)
- CSV: [`results/mppi_zoo_suite_2026-06-10.csv`](results/mppi_zoo_suite_2026-06-10.csv)

Reproduce just the SOPPI cells:

```bash
python3 scripts/run_mppi_zoo_suite.py \
  --bin bin/benchmark_diff_mppi \
  --planners mppi,soppi,soppi_fast \
  --stem mppi_zoo_soppi_nav_check
```

Aggregate over five scenarios and `K=64,128`:

| Planner | Solved | Success | Final d | Avg ms |
|---|---:|---:|---:|---:|
| mppi | 2/10 | 0.20 | 4.95 | 0.126 |
| soppi | 2/10 | 0.20 | 4.46 | 0.302 |
| soppi_fast | 2/10 | 0.20 | 4.51 | 0.251 |

Observed pattern:

- Both SOPPI variants clear only `narrow_passage` at `K=64` and `K=128`.
- On dynamic stress scenes (`dynamic_crossing`, `dynamic_pincer`,
  `uncertain_crossing`, `model_mismatch_crossing`) SOPPI matches vanilla MPPI
  failure modes in this lightweight score implementation.
- `soppi_fast` is about 1.2x to 1.7x faster than all-pairs `soppi`, but still
  1.5x to 3x slower than `step_mppi_smooth` on the same cells.
- Treat this row as honest negative coverage on navigation; box pushing remains
  the stronger reproduction target.

## Kernel Optimization, 2026-06-09

The first SOPPI pass spent too much time inside the SVGD loop:

- Navigation (`benchmark_diff_mppi.cu`) recomputed `stage_cost_grad` inside every neighbor lookup.
- Box/pushing benchmarks (`benchmark_diff_mppi_pushing_box.cu`, `benchmark_diff_mppi_pushing.cu`) computed full `2T` autodiff gradients from only `K` threads, then copied controls with `cudaMemcpy` every SVGD iteration.

Changes:

- Navigation now uses `soppi_stage_score_kernel` (`K*T` threads, one stage score each) plus a score-only `soppi_svgd_step_kernel`.
- Box/pushing now use `soppi_timestep_score_kernel` / `push_soppi_timestep_score_kernel` with `K*T` parallelism.
- All three benchmarks ping-pong control buffers instead of device-to-device copying every SVGD iteration.

The checked-in four-scenario run above supersedes the earlier single-scenario
`box_swivel` smoke note. Speedup versus the pre-kernel box-pushing artifacts:

- `soppi_fast`: about **3.4x faster** on `box_swivel` (`1.00 ms -> 0.29 ms`).
- `soppi` all-pairs: about **2.0x faster** on `box_swivel` (`1.08 ms -> 0.55 ms`).
- Navigation at `K=256` is now about **1.5x to 1.7x slower than MPPI**, not an
  order of magnitude slower.

Quality check: all-pairs `soppi` still reaches `1.00` success on `box_swivel`
with a step-count advantage over MPPI.

## Next Steps

1. Add a `--baseline-planners` option to `scripts/sweep_soppi.py` if repeated comparisons against Diff-MPPI are needed.
2. ~~Lift `soppi_fast` on `box_align_contact_loss`~~ **DONE (2026-06-10)** — tuned subset
   SVGD reaches `0.75` on strict cell; `box_align_contact_arc` documents pure-SOPPI at
   `1.00` (see `docs/results/soppi_box_pushing_2026-06-10.md`).
3. Consider caching partial rollout states for the box autodiff score kernel if another speed pass is needed.
