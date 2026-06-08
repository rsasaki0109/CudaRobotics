# Step-MPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **Toward Single-Step MPPI via Differentiable Predictive Control**.

- arXiv: https://arxiv.org/abs/2604.01539
- Submitted on 2026-04-02.
- Core idea: learn an MPPI proposal distribution in a self-supervised way from long-horizon objectives, so single-step or short-lookahead MPPI can keep long-horizon foresight while reducing runtime and sample burden.

## Implementation

Implemented a lightweight Step-MPPI reproduction in `src/benchmark_diff_mppi.cu`:

- Existing `step_mppi` path:
  - `d_sampling_bias_` stores per-timestep action-distribution mean shift.
  - `apply_sampling_bias_kernel` shifts the nominal before rollout sampling.
  - `update_sampling_bias_kernel` updates the bias from MPPI's cost-weighted control update.
- Added adaptive proposal variance:
  - `PlannerVariant::use_learned_sigma`.
  - `d_step_sigma_` stores per-timestep acceleration/steering sigma.
  - `rollout_learned_sampling_kernel` samples with the learned sigma.
  - `update_deterministic_sigma_kernel` is reused to update sigma from weighted rollout variance.
- Added planners:
  - `step_mppi`: original slow EMA bias baseline.
  - `step_mppi_fast`: faster EMA bias update.
  - `step_mppi_smooth`: faster EMA bias plus low-pass sampled action noise.
  - `step_mppi_adaptive`: faster EMA bias plus learned per-timestep sigma.
  - `step_mppi_single`: one-step `T=1` learned mean/sigma variant.

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- The paper uses a neural network to parameterize the proposal distribution. This implementation uses per-horizon buffers updated online by EMA.
- The paper trains with long-horizon objectives, constraint penalties, and entropy regularization. This implementation uses MPPI's cost-weighted update and weighted rollout variance as the self-supervised signal.
- `step_mppi_single` tests the one-step idea, but the repo's long 2D navigation tasks and bicycle dynamics still need temporal lookahead.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Main comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,lp_mppi_smooth,step_mppi,step_mppi_fast,step_mppi_adaptive,step_mppi_single,ds_mppi,pi_mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/step_mppi_compare.csv
```

Smooth variant check:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing \
  --planners mppi,lp_mppi_smooth,step_mppi_fast,step_mppi_smooth,step_mppi_adaptive,ds_mppi,pi_mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/step_mppi_smooth_dynamic.csv
```

## Results

Artifacts:

- `build-docker-smoke/step_mppi_compare.csv`
- `build-docker-smoke/step_mppi_compare_summary.md`
- `build-docker-smoke/step_mppi_smooth_dynamic.csv`
- `build-docker-smoke/step_mppi_smooth_dynamic_summary.md`

Dynamic crossing, smooth-variant check, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_crossing | 128 | mppi | 0.00 | 260.0 | 2.75 | 45557.9 | 0.598 | 0.562 | 0.11 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 252.0 | 1.89 | 41520.3 | 0.249 | 0.103 | 0.12 |
| dynamic_crossing | 128 | pi_mppi | 1.00 | 256.3 | 1.88 | 43297.8 | 0.135 | 0.028 | 0.20 |
| dynamic_crossing | 128 | ds_mppi | 1.00 | 254.3 | 1.95 | 41322.1 | 0.117 | 0.027 | 0.39 |
| dynamic_crossing | 128 | step_mppi_fast | 1.00 | 258.0 | 1.86 | 43865.3 | 0.679 | 0.752 | 0.13 |
| dynamic_crossing | 128 | step_mppi_smooth | 1.00 | 251.0 | 1.88 | 41178.3 | 0.281 | 0.135 | 0.14 |
| dynamic_crossing | 128 | step_mppi_adaptive | 0.33 | 260.0 | 2.05 | 44763.5 | 0.790 | 0.966 | 0.14 |
| dynamic_crossing | 256 | mppi | 0.00 | 260.0 | 3.15 | 45870.7 | 0.439 | 0.290 | 0.14 |
| dynamic_crossing | 256 | lp_mppi_smooth | 1.00 | 252.0 | 1.94 | 41575.2 | 0.187 | 0.060 | 0.14 |
| dynamic_crossing | 256 | pi_mppi | 1.00 | 256.3 | 1.91 | 43403.5 | 0.097 | 0.015 | 0.29 |
| dynamic_crossing | 256 | step_mppi_smooth | 1.00 | 251.3 | 1.91 | 41242.2 | 0.209 | 0.074 | 0.17 |
| dynamic_crossing | 256 | step_mppi_fast | 1.00 | 256.3 | 1.89 | 43332.3 | 0.517 | 0.421 | 0.17 |

Main comparison, cluttered, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | mppi | 0.00 | 38.54 | 49828.4 | 0.562 | 0.519 | 0.13 |
| cluttered | 128 | step_mppi_fast | 0.00 | 38.93 | 49412.2 | 0.707 | 0.814 | 0.15 |
| cluttered | 128 | step_mppi_adaptive | 0.00 | 38.47 | 49588.0 | 0.751 | 0.913 | 0.16 |
| cluttered | 128 | step_mppi_single | 0.00 | 56.06 | 62485.6 | 0.154 | 0.036 | 0.12 |
| cluttered | 256 | pi_mppi | 0.00 | 26.04 | 44902.4 | 0.110 | 0.021 | 0.33 |
| cluttered | 256 | step_mppi_fast | 0.00 | 38.96 | 49445.9 | 0.507 | 0.417 | 0.18 |

Observed pattern:

- `step_mppi_fast` and `step_mppi_smooth` are positive on `dynamic_crossing`: vanilla MPPI fails, while both reach `1.00` success at `K=128/256`.
- `step_mppi_smooth` is the preferred lightweight Step-MPPI variant: it keeps success, has lower roughness than `step_mppi_fast`, and reaches the lowest cumulative cost in the dynamic comparison.
- `step_mppi_adaptive` is mixed. The learned sigma helps at `K=256`, but is too aggressive at `K=128` and produces rough control.
- `step_mppi_single` is a negative result in this benchmark. One-step learned sampling cannot solve these long nonholonomic navigation tasks and often drives away from the goal.
- On `cluttered`, Step-MPPI does not solve the geometry and is weaker than the existing pi-MPPI / CDF-MPPI directions.
- The most useful next step for a more faithful Step-MPPI reproduction is replacing the EMA buffers with a small state-conditioned proposal network or table indexed by local state features, then training it across episodes rather than resetting per episode.
