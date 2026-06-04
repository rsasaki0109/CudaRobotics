# dsMPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **dsMPPI: Deterministic Sampling for Model Predictive Path Integral Control**.

- arXiv: https://arxiv.org/abs/2601.03893
- Web search checked on 2026-06-04 for a dedicated public `dsMPPI` GitHub implementation; none was found in the top results.
- Core idea: replace pure random MPPI sampling with deterministic sample sets and run multiple proposal-distribution refinement iterations, improving sample efficiency while preserving the MPPI weighting structure.

## Implementation

Implemented a lightweight dsMPPI variant in `src/benchmark_diff_mppi.cu`:

- Added `PlannerVariant::use_deterministic_sampling`.
- Added deterministic-sampling hyperparameters:
  - `ds_iterations`
  - `ds_alpha`
  - `ds_noise_scale`
  - `ds_momentum`
  - `ds_stride`
- Added `rollout_deterministic_kernel`, which:
  - reserves sample `k=0` as the nominal rollout,
  - generates antithetic deterministic Gaussian samples using Halton sequences and Box-Muller,
  - offsets the deterministic sequence by seed, MPC step, and MPPI refinement pass,
  - applies one-pole temporal coloring along the horizon,
  - evaluates the sampled controls with the existing bicycle rollout cost.
- Added `blend_controls_with_previous_kernel` for optional proposal momentum.
- Added planner registrations:
  - `ds_mppi`: two deterministic refinement passes, `lambda=4`, balanced cost/smoothness.
  - `ds_mppi_smooth`: same sample shape with `lambda=6`, lower roughness.
  - `ds_mppi_cov`: experimental per-timestep weighted-variance update.
  - `ds_mppi_cov_smooth`: more aggressively damped covariance update.
  - `ds_mppi_elite`: experimental CEM-style elite mean/sigma update.
  - `ds_mppi_elite_smooth`: larger elite set with stronger sigma damping.
- Added CLI overrides:
  - `--override-lambda`
  - `--override-ds-iters`
  - `--override-ds-alpha`
  - `--override-ds-noise-scale`
  - `--override-ds-momentum`
  - `--override-ds-stride`

Implemented the same lightweight dsMPPI sampling path in `src/benchmark_diff_mppi_cartpole.cu`:

- Added `ds_mppi` and `ds_mppi_smooth`.
- Added the same `--override-lambda` and `--override-ds-*` CLI controls.
- Added a 1D-action `rollout_deterministic_kernel` for CartPole.
- Added optional proposal momentum and `sample_budget` accounting for deterministic refinement passes.
- Added experimental per-timestep covariance adaptation in `src/benchmark_diff_mppi.cu`:
  - `d_ds_sigma_` stores acceleration/steering sigma for each horizon step.
  - `update_deterministic_sigma_kernel` updates sigma from weighted rollout variance.
  - `shift_deterministic_sigma_kernel` shifts sigma with the receding horizon.
- Added experimental elite/CEM-style distribution update in `src/benchmark_diff_mppi.cu`:
  - `update_deterministic_elite_kernel` selects the lowest-cost rollouts per update.
  - It replaces the nominal sequence with the elite mean, instead of the MPPI weighted mean.
  - It updates diagonal per-timestep sigma from elite variance with a floor/ceiling clamp.
  - This keeps the deterministic sampler simple while testing whether dsCEM-like proposal updates help the local benchmarks.

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- The paper's deterministic sampler is based on modified Cramer-von Mises optimization / dsCEM-style sampling. This implementation uses Halton low-discrepancy samples plus antithetic pairing.
- The paper updates distribution parameters beyond just the mean. This implementation keeps the repo's existing fixed MPPI noise scales and only changes the sampled control sequences before the weighted mean update.
- The validation here is on the existing 2D bicycle navigation and CartPole benchmarks, not the paper's full benchmark suite.
- The current variant intentionally avoids extra buffers for covariance and best-sample history. That keeps the implementation small and easy to compare against MPPI, LP-MPPI, pi-MPPI, and SOPPI.
- The experimental `ds_mppi_cov*` variants add only a diagonal per-timestep weighted variance update. They are not a faithful dsCEM/CvM covariance optimizer and were not adopted as the preferred result.
- The experimental `ds_mppi_elite*` variants use a simple per-step elite mean/variance update. They do not keep elite history, optimize a deterministic sample set, or enforce temporal projection/smoothing the way a full dsCEM-style implementation should.
- `ds_iterations > 1` increases rollout work per controller update. `sample_budget` now accounts for this in `benchmark_diff_mppi`.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

CartPole build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi_cartpole -j$(nproc)
```

Default comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,ds_mppi,ds_mppi_smooth,lp_mppi,lp_mppi_smooth,pi_mppi,pi_mppi_smooth \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/ds_mppi_compare.csv
```

Parameter sweep around the selected default:

```bash
for cfg in \
  l2n20a35i2:2:0.35:2.0:2.0 \
  l4n20a35i2:2:0.35:2.0:4.0 \
  l6n20a35i2:2:0.35:2.0:6.0 \
  l4n18a35i3:3:0.35:1.8:4.0 \
  l4n20a35i3:3:0.35:2.0:4.0 \
  l4n22a35i3:3:0.35:2.2:4.0 \
  l4n20a25i3:3:0.25:2.0:4.0 \
  l4n20a45i3:3:0.45:2.0:4.0
do
  IFS=: read name iters alpha noise lambda <<< "$cfg"
  ./bin/benchmark_diff_mppi \
    --quick \
    --scenarios dynamic_crossing \
    --planners ds_mppi \
    --k-values 256 \
    --seed-count 1 \
    --override-ds-iters "$iters" \
    --override-ds-alpha "$alpha" \
    --override-ds-noise-scale "$noise" \
    --override-ds-momentum 0.0 \
    --override-lambda "$lambda" \
    --csv "build-docker-smoke/ds_mppi_sweep2_${name}.csv"
done
```

CartPole comparison:

```bash
./bin/benchmark_diff_mppi_cartpole \
  --quick \
  --scenarios cartpole_recover,cartpole_large_angle \
  --planners mppi,ds_mppi,ds_mppi_smooth,lp_mppi,lp_mppi_smooth,diff_mppi_3,soppi_fast \
  --k-values 256,512 \
  --seed-count 3 \
  --csv build-docker-smoke/ds_mppi_cartpole_compare.csv
```

CartPole parameter sweep:

```bash
for cfg in \
  a20n06l08i1:1:0.20:0.6:0.8 \
  a20n08l10i1:1:0.20:0.8:1.0 \
  a35n06l08i1:1:0.35:0.6:0.8 \
  a35n08l10i1:1:0.35:0.8:1.0 \
  a50n06l08i1:1:0.50:0.6:0.8 \
  a20n06l08i2:2:0.20:0.6:0.8 \
  a20n08l10i2:2:0.20:0.8:1.0 \
  a35n06l08i2:2:0.35:0.6:0.8 \
  a35n08l10i2:2:0.35:0.8:1.0 \
  a50n08l12i2:2:0.50:0.8:1.2
do
  IFS=: read name iters alpha noise lambda <<< "$cfg"
  ./bin/benchmark_diff_mppi_cartpole \
    --quick \
    --scenarios cartpole_recover \
    --planners ds_mppi \
    --k-values 512 \
    --seed-count 3 \
    --override-ds-iters "$iters" \
    --override-ds-alpha "$alpha" \
    --override-ds-noise-scale "$noise" \
    --override-lambda "$lambda" \
    --csv "build-docker-smoke/ds_mppi_cartpole_sweep_${name}.csv"
done
```

Adaptive-sigma comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,ds_mppi,ds_mppi_smooth,ds_mppi_cov,ds_mppi_cov_smooth,lp_mppi_smooth,pi_mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/ds_mppi_cov_compare.csv
```

Elite/CEM-style comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,ds_mppi,ds_mppi_smooth,ds_mppi_cov,ds_mppi_elite,ds_mppi_elite_smooth,lp_mppi_smooth,pi_mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/ds_mppi_elite_compare.csv
```

## Results

Artifacts:

- `build-docker-smoke/ds_mppi_compare.csv`
- `build-docker-smoke/ds_mppi_compare_summary.md`
- `build-docker-smoke/ds_mppi_sweep2_*.csv`
- `build-docker-smoke/ds_mppi_cartpole_compare.csv`
- `build-docker-smoke/ds_mppi_cartpole_sweep_*.csv`
- `build-docker-smoke/ds_mppi_cov_compare.csv`
- `build-docker-smoke/ds_mppi_cov_sweep_*.csv`
- `build-docker-smoke/ds_mppi_elite_compare.csv`
- `build-docker-smoke/ds_mppi_elite_compare_summary.md`

Default comparison, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | mppi | 0.00 | 38.57 | 50093.7 | 0.600 | 0.563 | 0.16 |
| cluttered | 128 | lp_mppi_smooth | 0.00 | 39.60 | 48982.5 | 0.291 | 0.139 | 0.16 |
| cluttered | 128 | pi_mppi | 0.00 | 32.21 | 46980.9 | 0.147 | 0.037 | 0.24 |
| cluttered | 128 | ds_mppi | 0.00 | 39.85 | 49216.5 | 0.111 | 0.024 | 0.46 |
| cluttered | 128 | ds_mppi_smooth | 0.00 | 39.60 | 49411.5 | 0.105 | 0.018 | 0.35 |
| cluttered | 256 | mppi | 0.00 | 38.55 | 49918.5 | 0.401 | 0.248 | 0.16 |
| cluttered | 256 | lp_mppi_smooth | 0.00 | 39.67 | 49052.3 | 0.204 | 0.068 | 0.17 |
| cluttered | 256 | pi_mppi | 0.00 | 31.92 | 46829.1 | 0.117 | 0.024 | 0.32 |
| cluttered | 256 | ds_mppi | 0.00 | 39.75 | 49243.3 | 0.073 | 0.016 | 0.37 |
| cluttered | 256 | ds_mppi_smooth | 0.00 | 39.53 | 49517.1 | 0.069 | 0.011 | 0.37 |
| dynamic_crossing | 128 | mppi | 0.00 | 3.40 | 46083.3 | 0.598 | 0.550 | 0.14 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 1.96 | 41593.8 | 0.258 | 0.110 | 0.13 |
| dynamic_crossing | 128 | pi_mppi_smooth | 1.00 | 1.91 | 41833.7 | 0.128 | 0.027 | 0.30 |
| dynamic_crossing | 128 | ds_mppi | 1.00 | 1.97 | 41319.5 | 0.122 | 0.029 | 0.30 |
| dynamic_crossing | 128 | ds_mppi_smooth | 1.00 | 1.99 | 42514.6 | 0.111 | 0.020 | 0.30 |
| dynamic_crossing | 256 | mppi | 0.00 | 3.06 | 45851.7 | 0.420 | 0.284 | 0.16 |
| dynamic_crossing | 256 | lp_mppi_smooth | 1.00 | 1.86 | 41623.1 | 0.185 | 0.057 | 0.16 |
| dynamic_crossing | 256 | pi_mppi_smooth | 1.00 | 1.91 | 41783.4 | 0.099 | 0.019 | 0.47 |
| dynamic_crossing | 256 | ds_mppi | 1.00 | 1.98 | 41607.7 | 0.084 | 0.018 | 0.34 |
| dynamic_crossing | 256 | ds_mppi_smooth | 1.00 | 1.94 | 43013.1 | 0.075 | 0.011 | 0.34 |

Sweep, `dynamic_crossing K=256`, seed-count 1:

| Setting | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---:|---:|---:|---:|---:|
| lambda 2, noise 2.0, alpha 0.35, iters 2 | 1.00 | 1.98 | 40559.9 | 0.105 | 0.033 | 0.55 |
| lambda 4, noise 2.0, alpha 0.35, iters 2 | 1.00 | 1.97 | 41594.1 | 0.081 | 0.018 | 0.49 |
| lambda 6, noise 2.0, alpha 0.35, iters 2 | 1.00 | 1.93 | 42985.1 | 0.071 | 0.011 | 0.49 |
| lambda 4, noise 1.8, alpha 0.35, iters 3 | 1.00 | 1.97 | 41213.6 | 0.090 | 0.023 | 0.68 |
| lambda 4, noise 2.0, alpha 0.25, iters 3 | 1.00 | 1.96 | 40783.1 | 0.089 | 0.026 | 0.56 |
| lambda 4, noise 2.0, alpha 0.35, iters 3 | 1.00 | 1.95 | 41199.5 | 0.092 | 0.023 | 0.66 |

CartPole comparison, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Err | Cost | Mean du | Roughness | Avg ms | Track Loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cartpole_recover | 256 | mppi | 0.00 | 173.0 | 0.683 | 1124.0 | 0.307 | 0.183 | 0.21 | 0.33 |
| cartpole_recover | 256 | lp_mppi_smooth | 0.00 | 179.0 | 0.577 | 1135.9 | 0.270 | 0.142 | 0.19 | 0.33 |
| cartpole_recover | 256 | ds_mppi | 0.33 | 139.0 | 0.909 | 1146.7 | 0.242 | 0.185 | 0.31 | 0.33 |
| cartpole_recover | 512 | mppi | 0.33 | 122.0 | 0.811 | 661.6 | 0.248 | 0.138 | 0.18 | 0.33 |
| cartpole_recover | 512 | lp_mppi | 0.33 | 133.0 | 0.684 | 600.9 | 0.212 | 0.089 | 0.19 | 0.33 |
| cartpole_recover | 512 | ds_mppi | 0.33 | 73.7 | 0.907 | 727.2 | 0.255 | 0.150 | 0.35 | 0.67 |
| cartpole_large_angle | 256 | mppi | 0.00 | 94.7 | 1.283 | 2417.7 | 0.282 | 0.217 | 0.15 | 1.00 |
| cartpole_large_angle | 256 | ds_mppi_smooth | 0.00 | 92.0 | 1.265 | 2403.2 | 0.377 | 0.271 | 0.28 | 1.00 |
| cartpole_large_angle | 512 | mppi | 0.00 | 69.3 | 1.327 | 2417.8 | 0.235 | 0.181 | 0.19 | 1.00 |
| cartpole_large_angle | 512 | lp_mppi | 0.00 | 66.3 | 1.322 | 2408.4 | 0.163 | 0.118 | 0.19 | 1.00 |
| cartpole_large_angle | 512 | ds_mppi | 0.00 | 180.7 | 0.975 | 2230.2 | 0.171 | 0.106 | 0.36 | 0.67 |
| cartpole_large_angle | 512 | ds_mppi_smooth | 0.00 | 137.7 | 1.105 | 2149.8 | 0.253 | 0.163 | 0.36 | 0.67 |

Adaptive-sigma comparison, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | ds_mppi | 0.00 | 39.85 | 49216.5 | 0.111 | 0.024 | 0.45 |
| cluttered | 128 | ds_mppi_cov | 0.00 | 50.65 | 59643.4 | 0.130 | 0.037 | 0.35 |
| cluttered | 128 | ds_mppi_cov_smooth | 0.00 | 53.66 | 61256.6 | 0.106 | 0.017 | 0.34 |
| cluttered | 256 | ds_mppi | 0.00 | 39.75 | 49243.3 | 0.073 | 0.016 | 0.39 |
| cluttered | 256 | ds_mppi_cov | 0.00 | 52.40 | 60488.0 | 0.049 | 0.015 | 0.40 |
| cluttered | 256 | ds_mppi_cov_smooth | 0.00 | 54.55 | 61682.7 | 0.035 | 0.006 | 0.41 |
| dynamic_crossing | 128 | ds_mppi | 1.00 | 1.97 | 41319.4 | 0.122 | 0.029 | 0.31 |
| dynamic_crossing | 128 | ds_mppi_cov | 0.00 | 48.14 | 71861.1 | 0.128 | 0.033 | 0.33 |
| dynamic_crossing | 128 | ds_mppi_cov_smooth | 0.00 | 52.30 | 74486.1 | 0.103 | 0.017 | 0.31 |
| dynamic_crossing | 256 | ds_mppi | 1.00 | 1.98 | 41607.7 | 0.084 | 0.018 | 0.36 |
| dynamic_crossing | 256 | ds_mppi_cov | 0.00 | 50.87 | 73392.9 | 0.048 | 0.016 | 0.39 |
| dynamic_crossing | 256 | ds_mppi_cov_smooth | 0.00 | 53.78 | 75347.2 | 0.032 | 0.006 | 0.39 |

Elite/CEM-style comparison, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms | Collisions |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | ds_mppi | 0.00 | 39.85 | 49216.5 | 0.111 | 0.024 | 0.45 | 0.00 |
| cluttered | 128 | ds_mppi_cov | 0.00 | 50.65 | 59643.4 | 0.130 | 0.037 | 0.38 | 0.00 |
| cluttered | 128 | ds_mppi_elite | 0.00 | 19.67 | 43374.2 | 0.529 | 0.470 | 0.60 | 0.00 |
| cluttered | 128 | ds_mppi_elite_smooth | 0.00 | 18.01 | 52883.5 | 0.480 | 0.410 | 0.81 | 6.33 |
| cluttered | 128 | pi_mppi | 0.00 | 26.06 | 44785.1 | 0.156 | 0.057 | 0.28 | 0.00 |
| cluttered | 256 | ds_mppi | 0.00 | 39.75 | 49243.3 | 0.073 | 0.016 | 0.39 | 0.00 |
| cluttered | 256 | ds_mppi_elite | 0.00 | 28.01 | 48487.3 | 1.285 | 3.389 | 0.62 | 0.00 |
| cluttered | 256 | ds_mppi_elite_smooth | 0.00 | 22.24 | 45017.6 | 0.824 | 1.544 | 0.93 | 0.00 |
| dynamic_crossing | 128 | ds_mppi | 1.00 | 1.97 | 41319.4 | 0.122 | 0.029 | 0.32 | 0.00 |
| dynamic_crossing | 128 | ds_mppi_elite | 0.00 | 3.08 | 45212.6 | 0.446 | 0.360 | 0.57 | 0.00 |
| dynamic_crossing | 128 | ds_mppi_elite_smooth | 0.00 | 5.37 | 47806.2 | 0.448 | 0.337 | 0.78 | 0.00 |
| dynamic_crossing | 128 | pi_mppi | 1.00 | 1.93 | 43350.0 | 0.145 | 0.031 | 0.27 | 0.00 |
| dynamic_crossing | 256 | ds_mppi | 1.00 | 1.98 | 41607.7 | 0.084 | 0.018 | 0.37 | 0.00 |
| dynamic_crossing | 256 | ds_mppi_elite | 1.00 | 1.81 | 42914.5 | 0.590 | 1.208 | 0.66 | 0.00 |
| dynamic_crossing | 256 | lp_mppi_smooth | 1.00 | 1.93 | 41606.3 | 0.190 | 0.061 | 0.17 | 0.00 |
| dynamic_crossing | 256 | pi_mppi | 1.00 | 1.91 | 43324.5 | 0.107 | 0.018 | 0.31 | 0.00 |

Observed pattern:

- dsMPPI is a clear positive signal on `dynamic_crossing`: vanilla MPPI fails at `K=128/256`, while dsMPPI reaches `1.00` success at both budgets.
- Compared with LP-MPPI, dsMPPI has similar cost on `dynamic_crossing` and much lower applied-control roughness, but it spends more runtime because it performs two rollout/update passes per controller update.
- `ds_mppi_smooth` further reduces roughness but sacrifices cost.
- On `cluttered`, dsMPPI does not solve the scenario and is weaker than `pi_mppi` on final distance/cost. This points to deterministic sample diversity being too narrow for cluttered geometry in the lightweight implementation.
- On `cartpole_large_angle K=512`, dsMPPI is a useful positive signal: it extends survival, reduces final error/cost, and lowers roughness relative to vanilla MPPI and LP-MPPI, although it still does not meet the success window.
- On `cartpole_recover`, dsMPPI is mixed: `K=256` reaches one success where MPPI/LP-MPPI do not, but `K=512` is worse than LP-MPPI on cost, final error, and track loss.
- The naive adaptive-sigma update is a negative result: it collapses or biases exploration enough that `dynamic_crossing` fails completely despite fixed-noise dsMPPI succeeding. This should not be used as the preferred dsMPPI reproduction.
- The elite/CEM-style update is a partial positive signal. It improves `cluttered K=128` final distance from `39.85` to `19.67` and reaches the best `dynamic_crossing K=256` final distance at `1.81`, but it is much rougher, costs more runtime, and fails `dynamic_crossing K=128` where fixed dsMPPI succeeds.
- `ds_mppi_elite_smooth` is not preferred: it reduces `cluttered` final distance further in some rows, but introduces collisions at `cluttered K=128` and still fails `dynamic_crossing`.
- The most useful next step for a more faithful reproduction is implementing the paper's actual deterministic sample optimization / dsCEM-style sampler and covariance update, including elite/best-sample history, a non-collapsing covariance schedule, and temporal smoothing/projection before re-running CartPole and pushing.
