# LP-MPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **LP-MPPI: Low-Pass Filtering for Efficient Model Predictive Path Integral Control** by Piotr Kicki.

- arXiv: https://arxiv.org/abs/2503.11717
- Latest arXiv metadata checked: submitted 2025-03-13, revised 2026-02-03.
- arXiv comments: accepted at ICRA 2026.
- Core idea: filter sampled MPPI control perturbations before rollout so exploration is lower-frequency, smoother, and more sample-efficient.

Local search found no existing LP-MPPI-equivalent implementation in this repo. The existing `stomp_*` variants smooth the cost-weighted control update after sampling; LP-MPPI instead filters the sampled trajectory perturbations before rollout.

## Implementation

Implemented a lightweight LP-MPPI variant in `src/benchmark_diff_mppi.cu`:

- Added `PlannerVariant::use_low_pass_sampling`.
- Added `PlannerVariant::lp_alpha`, where `1.0` recovers vanilla MPPI-style white perturbations and smaller values apply stronger horizon-wise low-pass filtering.
- Added `rollout_low_pass_kernel`, which applies a one-pole IIR filter to Gaussian acceleration/steering perturbations along the horizon before rollout.
- Added variance normalization for the IIR-filtered perturbations so marginal exploration scale is not simply reduced by filtering.
- Added planner registrations:
  - `lp_mppi` with `lp_alpha=0.35`
  - `lp_mppi_smooth` with `lp_alpha=0.20`
- Added CLI override:
  - `--override-lp-alpha`
- Added CSV metrics for applied-control smoothness:
  - `mean_control_delta`
  - `control_roughness`

Implemented the same lightweight LP-MPPI sampling path in `src/benchmark_diff_mppi_cartpole.cu`:

- Added `lp_mppi` and `lp_mppi_smooth`.
- Added `--override-lp-alpha`.
- Added `rollout_low_pass_kernel` for one-dimensional CartPole action perturbations.
- Added the same applied-control smoothness metrics to the CartPole CSV output.

Implemented the same lightweight LP-MPPI sampling path in `src/benchmark_diff_mppi_pushing.cu`:

- Added `lp_mppi` and `lp_mppi_smooth`.
- Added `--override-lp-alpha`.
- Added `push_low_pass_rollout_kernel` for two-dimensional end-effector velocity perturbations.
- Added the same applied-control smoothness metrics to the planar pushing CSV output.

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- The paper uses explicit low-pass filtering with direct frequency-spectrum control. This repo implementation uses a first-order IIR approximation controlled by `lp_alpha`.
- The current validation is on the existing 2D bicycle navigation, CartPole, and planar pushing benchmarks, not Gymnasium, quadruped locomotion, or F1TENTH.
- CartPole and planar pushing validation now exist, but they are still local benchmarks rather than the paper tasks.
- Smoothness is measured from the applied control stream, not by spectral analysis.
- `lp_mppi` has essentially MPPI-level runtime because filtering is done inside the rollout thread.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Default LP-MPPI comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,lp_mppi,lp_mppi_smooth,step_mppi,soppi_fast \
  --k-values 256 \
  --seed-count 3 \
  --csv build-docker-smoke/lp_mppi_compare.csv
```

Alpha sweep:

```bash
mkdir -p build-docker-smoke/lp_mppi_sweep/csv
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/lp_mppi_sweep/csv/mppi_baseline.csv
for a in 0.15 0.20 0.35 0.55 0.80; do
  ./bin/benchmark_diff_mppi \
    --quick \
    --scenarios dynamic_crossing,cluttered \
    --planners lp_mppi \
    --override-lp-alpha "$a" \
    --k-values 128,256 \
    --seed-count 3 \
    --csv "build-docker-smoke/lp_mppi_sweep/csv/lp_alpha_${a}.csv"
done
```

CartPole comparison:

```bash
./bin/benchmark_diff_mppi_cartpole \
  --quick \
  --scenarios cartpole_recover,cartpole_large_angle \
  --planners mppi,lp_mppi,lp_mppi_smooth,diff_mppi_3,soppi_fast \
  --k-values 256,512 \
  --seed-count 3 \
  --csv build-docker-smoke/lp_mppi_cartpole_compare.csv
```

CartPole alpha sweep:

```bash
mkdir -p build-docker-smoke/lp_mppi_cartpole_sweep/csv
./bin/benchmark_diff_mppi_cartpole \
  --quick \
  --scenarios cartpole_recover,cartpole_large_angle \
  --planners mppi \
  --k-values 256,512 \
  --seed-count 3 \
  --csv build-docker-smoke/lp_mppi_cartpole_sweep/csv/mppi_baseline.csv
for a in 0.15 0.20 0.35 0.55 0.80; do
  ./bin/benchmark_diff_mppi_cartpole \
    --quick \
    --scenarios cartpole_recover,cartpole_large_angle \
    --planners lp_mppi \
    --override-lp-alpha "$a" \
    --k-values 256,512 \
    --seed-count 3 \
    --csv "build-docker-smoke/lp_mppi_cartpole_sweep/csv/lp_alpha_${a}.csv"
done
```

Planar pushing comparison:

```bash
./bin/benchmark_diff_mppi_pushing \
  --quick \
  --planners mppi,lp_mppi,lp_mppi_smooth,diff_mppi_3,soppi_fast \
  --k-values 128,256 \
  --seed-count 4 \
  --csv build-docker-smoke/lp_mppi_pushing_compare.csv
```

Planar pushing alpha sweep:

```bash
mkdir -p build-docker-smoke/lp_mppi_pushing_sweep/csv
./bin/benchmark_diff_mppi_pushing \
  --quick \
  --planners mppi \
  --k-values 128,256 \
  --seed-count 4 \
  --csv build-docker-smoke/lp_mppi_pushing_sweep/csv/mppi_baseline.csv
for a in 0.15 0.20 0.35 0.55 0.80; do
  ./bin/benchmark_diff_mppi_pushing \
    --quick \
    --planners lp_mppi \
    --override-lp-alpha "$a" \
    --k-values 128,256 \
    --seed-count 4 \
    --csv "build-docker-smoke/lp_mppi_pushing_sweep/csv/lp_alpha_${a}.csv"
done
```

## Results

Artifacts:

- `build-docker-smoke/lp_mppi_compare.csv`
- `build-docker-smoke/lp_mppi_compare_summary.md`
- `build-docker-smoke/lp_mppi_sweep/csv/`
- `build-docker-smoke/lp_mppi_cartpole_compare.csv`
- `build-docker-smoke/lp_mppi_cartpole_compare_summary.md`
- `build-docker-smoke/lp_mppi_cartpole_sweep/csv/`
- `build-docker-smoke/lp_mppi_pushing_compare.csv`
- `build-docker-smoke/lp_mppi_pushing_compare_summary.md`
- `build-docker-smoke/lp_mppi_pushing_sweep/csv/`

Default comparison, seed-count 3, `K=256`:

| Scenario | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---|---:|---:|---:|---:|---:|---:|
| cluttered | mppi | 0.00 | 38.57 | 49917.8 | 0.411 | 0.266 | 0.18 |
| cluttered | step_mppi | 0.00 | 38.56 | 49908.1 | 0.405 | 0.252 | 0.23 |
| cluttered | soppi_fast | 0.00 | 38.45 | 49821.3 | 0.383 | 0.227 | 0.77 |
| cluttered | lp_mppi | 0.00 | 39.48 | 49079.0 | 0.257 | 0.108 | 0.25 |
| cluttered | lp_mppi_smooth | 0.00 | 39.59 | 48986.2 | 0.200 | 0.070 | 0.22 |
| dynamic_crossing | mppi | 0.00 | 2.99 | 45772.6 | 0.419 | 0.274 | 0.15 |
| dynamic_crossing | step_mppi | 0.00 | 3.02 | 45803.2 | 0.424 | 0.290 | 0.17 |
| dynamic_crossing | soppi_fast | 0.00 | 3.14 | 45822.6 | 0.405 | 0.255 | 0.48 |
| dynamic_crossing | lp_mppi | 1.00 | 1.90 | 41860.6 | 0.248 | 0.100 | 0.18 |
| dynamic_crossing | lp_mppi_smooth | 1.00 | 1.94 | 41633.0 | 0.194 | 0.062 | 0.14 |

Alpha sweep, seed-count 3:

| Scenario | K | Planner | LP alpha | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | mppi | baseline | 0.00 | 38.50 | 49955.5 | 0.583 | 0.530 | 0.16 |
| cluttered | 128 | lp_mppi | 0.15 | 0.00 | 39.77 | 48998.6 | 0.256 | 0.111 | 0.14 |
| cluttered | 128 | lp_mppi | 0.35 | 0.00 | 39.43 | 49015.4 | 0.352 | 0.205 | 0.14 |
| cluttered | 128 | lp_mppi | 0.80 | 0.00 | 38.89 | 49537.0 | 0.519 | 0.425 | 0.13 |
| dynamic_crossing | 128 | mppi | baseline | 0.00 | 3.39 | 46253.3 | 0.574 | 0.519 | 0.15 |
| dynamic_crossing | 128 | lp_mppi | 0.15 | 1.00 | 1.88 | 41496.2 | 0.223 | 0.084 | 0.12 |
| dynamic_crossing | 128 | lp_mppi | 0.35 | 1.00 | 1.90 | 41918.0 | 0.318 | 0.168 | 0.13 |
| dynamic_crossing | 128 | lp_mppi | 0.80 | 0.67 | 1.93 | 44265.2 | 0.512 | 0.423 | 0.13 |
| dynamic_crossing | 256 | mppi | baseline | 0.00 | 2.93 | 45545.0 | 0.442 | 0.301 | 0.16 |
| dynamic_crossing | 256 | lp_mppi | 0.15 | 1.00 | 1.91 | 41522.0 | 0.175 | 0.054 | 0.14 |
| dynamic_crossing | 256 | lp_mppi | 0.35 | 1.00 | 1.90 | 41860.6 | 0.248 | 0.100 | 0.15 |
| dynamic_crossing | 256 | lp_mppi | 0.80 | 1.00 | 1.94 | 43762.5 | 0.387 | 0.235 | 0.16 |

Observed pattern:

- LP-MPPI is a strong reproduction signal on `dynamic_crossing`: vanilla MPPI fails in all tested `K=128/256` cells, while low-pass sampling reaches `1.00` success for most alpha settings.
- Strong filtering (`alpha=0.15` to `0.35`) gives the best cost and smoothness on `dynamic_crossing`.
- `cluttered` is not solved by either MPPI or LP-MPPI. LP-MPPI reduces cost and control roughness but drifts to a worse final distance, so this scenario is a caveat rather than a win.
- Runtime stays close to MPPI, unlike SOPPI, because the filter adds only a few scalar operations inside each rollout thread.
- `lp_mppi_smooth` is the smoother default, while `lp_mppi` is the better balanced default for maintaining exploration.

## CartPole Results

Default CartPole comparison, seed-count 3:

| Scenario | K | Planner | Success | Final Err | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cartpole_recover | 256 | mppi | 0.00 | 0.683 | 1124.0 | 0.307 | 0.183 | 0.18 |
| cartpole_recover | 256 | lp_mppi | 0.00 | 0.585 | 1144.4 | 0.287 | 0.158 | 0.18 |
| cartpole_recover | 256 | lp_mppi_smooth | 0.00 | 0.577 | 1135.9 | 0.270 | 0.142 | 0.15 |
| cartpole_recover | 256 | diff_mppi_3 | 0.00 | 0.523 | 1019.1 | 0.657 | 0.800 | 0.63 |
| cartpole_recover | 512 | mppi | 0.33 | 0.811 | 661.6 | 0.248 | 0.138 | 0.16 |
| cartpole_recover | 512 | lp_mppi | 0.33 | 0.684 | 600.9 | 0.212 | 0.089 | 0.17 |
| cartpole_recover | 512 | lp_mppi_smooth | 0.33 | 0.714 | 624.6 | 0.216 | 0.107 | 0.17 |
| cartpole_large_angle | 256 | mppi | 0.00 | 1.283 | 2417.7 | 0.282 | 0.217 | 0.13 |
| cartpole_large_angle | 256 | lp_mppi | 0.00 | 1.277 | 2451.9 | 0.278 | 0.215 | 0.13 |
| cartpole_large_angle | 256 | lp_mppi_smooth | 0.00 | 1.268 | 2439.0 | 0.252 | 0.179 | 0.13 |
| cartpole_large_angle | 512 | mppi | 0.00 | 1.327 | 2417.8 | 0.235 | 0.181 | 0.17 |
| cartpole_large_angle | 512 | lp_mppi | 0.00 | 1.322 | 2408.4 | 0.163 | 0.118 | 0.18 |
| cartpole_large_angle | 512 | lp_mppi_smooth | 0.00 | 1.302 | 2417.1 | 0.165 | 0.127 | 0.18 |

CartPole alpha sweep, seed-count 3:

| Scenario | K | Planner | LP alpha | Success | Final Err | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cartpole_recover | 256 | mppi | baseline | 0.00 | 0.683 | 1124.0 | 0.307 | 0.183 | 0.20 |
| cartpole_recover | 256 | lp_mppi | 0.15 | 0.00 | 0.578 | 1131.1 | 0.259 | 0.130 | 0.16 |
| cartpole_recover | 256 | lp_mppi | 0.35 | 0.00 | 0.585 | 1144.4 | 0.287 | 0.158 | 0.14 |
| cartpole_recover | 512 | mppi | baseline | 0.33 | 0.811 | 661.6 | 0.248 | 0.138 | 0.25 |
| cartpole_recover | 512 | lp_mppi | 0.35 | 0.33 | 0.684 | 600.9 | 0.212 | 0.089 | 0.18 |
| cartpole_recover | 512 | lp_mppi | 0.55 | 0.33 | 0.700 | 588.3 | 0.232 | 0.127 | 0.17 |
| cartpole_large_angle | 256 | mppi | baseline | 0.00 | 1.283 | 2417.7 | 0.282 | 0.217 | 0.19 |
| cartpole_large_angle | 256 | lp_mppi | 0.20 | 0.00 | 1.268 | 2439.0 | 0.252 | 0.179 | 0.15 |
| cartpole_large_angle | 512 | mppi | baseline | 0.00 | 1.327 | 2417.8 | 0.235 | 0.181 | 0.25 |
| cartpole_large_angle | 512 | lp_mppi | 0.15 | 0.00 | 1.286 | 2417.9 | 0.163 | 0.105 | 0.19 |

CartPole observed pattern:

- LP-MPPI improves `cartpole_recover` final error and smoothness versus MPPI, especially at `K=512` (`0.811 -> 0.684`, cost `661.6 -> 600.9`, roughness `0.138 -> 0.089`).
- `cartpole_large_angle` remains unsolved; LP-MPPI reduces control roughness and sometimes final error, but does not create a strong success-rate improvement.
- Diff-MPPI is still the stronger CartPole quality baseline at `K=256`, but it is much rougher and slower in these runs.
- CartPole confirms LP-MPPI's smooth-control behavior, but the strongest reproduction signal remains the 2D `dynamic_crossing` task.

## Planar Pushing Results

Default planar pushing comparison, seed-count 4:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| push_straight | 256 | mppi | 1.00 | 30.8 | 0.187 | 2.21 | 0.492 | 0.395 | 0.08 |
| push_straight | 256 | lp_mppi | 1.00 | 26.0 | 0.186 | 2.02 | 0.514 | 0.375 | 0.10 |
| push_straight | 256 | lp_mppi_smooth | 1.00 | 25.2 | 0.180 | 1.97 | 0.412 | 0.268 | 0.10 |
| push_straight | 256 | diff_mppi_3 | 1.00 | 25.0 | 0.176 | 1.82 | 0.672 | 0.769 | 0.66 |
| push_straight | 256 | soppi_fast | 1.00 | 30.0 | 0.182 | 2.15 | 0.493 | 0.410 | 0.27 |
| push_diagonal | 256 | mppi | 1.00 | 31.8 | 0.188 | 3.33 | 0.546 | 0.438 | 0.12 |
| push_diagonal | 256 | lp_mppi | 1.00 | 26.0 | 0.178 | 2.88 | 0.509 | 0.398 | 0.11 |
| push_diagonal | 256 | lp_mppi_smooth | 1.00 | 25.5 | 0.181 | 2.87 | 0.393 | 0.245 | 0.12 |
| push_diagonal | 256 | diff_mppi_3 | 1.00 | 27.2 | 0.184 | 2.95 | 0.749 | 0.841 | 0.46 |
| push_diagonal | 256 | soppi_fast | 1.00 | 31.0 | 0.183 | 3.24 | 0.486 | 0.399 | 0.29 |

Planar pushing alpha sweep, aggregated over both scenarios and `K=128/256`, seed-count 4:

| Planner | LP alpha | Success | Steps | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mppi | baseline | 1.00 | 31.9 | 0.187 | 2.87 | 0.580 | 0.505 | 0.11 |
| lp_mppi | 0.15 | 1.00 | 25.8 | 0.181 | 2.58 | 0.437 | 0.289 | 0.11 |
| lp_mppi | 0.20 | 1.00 | 26.1 | 0.182 | 2.58 | 0.478 | 0.340 | 0.11 |
| lp_mppi | 0.35 | 1.00 | 26.5 | 0.184 | 2.55 | 0.553 | 0.462 | 0.11 |
| lp_mppi | 0.55 | 1.00 | 27.3 | 0.184 | 2.61 | 0.560 | 0.460 | 0.11 |
| lp_mppi | 0.80 | 1.00 | 29.5 | 0.185 | 2.75 | 0.595 | 0.524 | 0.11 |

Planar pushing observed pattern:

- The quick pushing scenarios are saturated: every planner reaches `1.00` success, so success rate is not a discriminating metric here.
- LP-MPPI improves MPPI's step count and cost while keeping MPPI-level runtime. At `K=256`, `lp_mppi_smooth` reduces roughness from `0.395 -> 0.268` on `push_straight` and `0.438 -> 0.245` on `push_diagonal`.
- Diff-MPPI reaches slightly lower final distance and cost, but it is much rougher and slower in these runs.
- The alpha sweep shows strongest smoothing around `alpha=0.15` to `0.20`; `alpha=0.80` approaches vanilla MPPI behavior.

## Next Steps

1. Replace the one-pole IIR approximation with a paper-closer discrete low-pass filter and expose cutoff frequency directly.
2. Add a frequency-domain smoothness metric for sampled and applied controls.
3. Port LP-MPPI to box pushing and add harder pushing scenarios where success rate is not saturated.
