# CDF-MPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **One-Step Model Predictive Path Integral for Manipulator Motion Planning Using Configuration Space Distance Fields**.

- arXiv: https://arxiv.org/abs/2509.00836
- Web search checked on 2026-06-04 for a clear standalone public implementation; no dedicated repository was found in the top results.
- Core idea: use Configuration-space Distance Fields (CDFs) so collision distance and its gradient are expressed directly in configuration space, allowing MPPI to use very short horizons, down to one step in the paper's manipulator setting.

## Implementation

Implemented a lightweight CDF-MPPI reproduction in `src/benchmark_diff_mppi.cu`:

- Added CDF planner parameters to `PlannerVariant`.
- Added `rollout_cdf_kernel`:
  - samples controls as in MPPI,
  - optionally applies low-pass action noise,
  - evaluates a smooth C-space margin cost against static and dynamic circular obstacles.
- Added `seed_cdf_nominal`:
  - computes a host-side CDF vector field from goal attraction plus obstacle repulsion,
  - seeds the nominal acceleration/steering sequence before MPPI sampling,
  - slows slightly near obstacles, but does not stop completely.
- Added planner registrations:
  - `cdf_mppi`: short-horizon CDF-guided MPPI, `T=16`.
  - `cdf_lp_mppi`: short-horizon CDF-guided MPPI with low-pass action noise, `T=16`.
  - `cdf_mppi_one_step`: one-step CDF-guided MPPI, `T=1`, kept to test the paper's one-step claim in this 2D bicycle setting.

## Scope Caveats

This is not a paper-faithful implementation:

- The paper targets manipulator motion planning in joint-space CDFs. This reproduction uses the repo's 2D navigation state as a configuration space.
- The CDF is analytic from the existing circular obstacles, not a learned or precomputed high-dimensional field.
- The one-step version is structurally faithful to the paper's short-horizon motivation, but the bicycle model and long navigation tasks still need temporal lookahead.
- Dynamic obstacles are not a natural fit for the paper's static CDF framing; the dynamic CDF terms here are a lightweight extension and produced a negative result.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Static/dynamic comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,lp_mppi_smooth,cdf_mppi,cdf_lp_mppi,cdf_mppi_one_step,ds_mppi,pi_mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/cdf_mppi_compare_tuned.csv
```

Dynamic retune check:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing \
  --planners mppi,lp_mppi_smooth,cdf_mppi,cdf_lp_mppi,cdf_mppi_one_step,ds_mppi,pi_mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/cdf_mppi_dynamic_retuned.csv
```

## Results

Artifacts:

- `build-docker-smoke/cdf_mppi_compare.csv`
- `build-docker-smoke/cdf_mppi_compare_summary.md`
- `build-docker-smoke/cdf_mppi_compare_tuned.csv`
- `build-docker-smoke/cdf_mppi_compare_tuned_summary.md`
- `build-docker-smoke/cdf_mppi_dynamic_retuned.csv`
- `build-docker-smoke/cdf_mppi_dynamic_retuned_summary.md`

Static cluttered comparison, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms | Collisions |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | mppi | 0.00 | 38.54 | 50034.2 | 0.514 | 0.420 | 0.14 | 0.00 |
| cluttered | 128 | lp_mppi_smooth | 0.00 | 39.66 | 49039.0 | 0.293 | 0.144 | 0.14 | 0.00 |
| cluttered | 128 | pi_mppi | 0.00 | 32.20 | 46801.8 | 0.146 | 0.036 | 0.25 | 0.00 |
| cluttered | 128 | ds_mppi | 0.00 | 39.84 | 49201.3 | 0.109 | 0.023 | 0.45 | 0.00 |
| cluttered | 128 | cdf_mppi | 0.00 | 32.28 | 101442.0 | 0.388 | 0.327 | 0.18 | 0.00 |
| cluttered | 128 | cdf_lp_mppi | 0.00 | 17.76 | 58713.7 | 0.157 | 0.043 | 0.21 | 0.00 |
| cluttered | 128 | cdf_mppi_one_step | 0.00 | 27.42 | 86311.1 | 0.175 | 0.078 | 0.15 | 20.33 |
| cluttered | 256 | mppi | 0.00 | 38.51 | 49787.1 | 0.393 | 0.248 | 0.19 | 0.00 |
| cluttered | 256 | pi_mppi | 0.00 | 26.10 | 44990.7 | 0.112 | 0.023 | 0.34 | 0.00 |
| cluttered | 256 | cdf_lp_mppi | 0.00 | 17.82 | 59303.7 | 0.116 | 0.024 | 0.19 | 0.00 |

Dynamic crossing retune, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms | Collisions |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_crossing | 128 | mppi | 0.00 | 3.39 | 46253.3 | 0.574 | 0.519 | 0.14 | 0.00 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 1.91 | 41633.7 | 0.260 | 0.107 | 0.14 | 0.00 |
| dynamic_crossing | 128 | ds_mppi | 1.00 | 1.98 | 41271.8 | 0.118 | 0.026 | 0.38 | 0.00 |
| dynamic_crossing | 128 | pi_mppi | 1.00 | 1.89 | 43024.5 | 0.142 | 0.031 | 0.24 | 0.00 |
| dynamic_crossing | 128 | cdf_lp_mppi | 0.00 | 43.40 | 257892.9 | 0.200 | 0.066 | 0.20 | 0.00 |
| dynamic_crossing | 128 | cdf_mppi_one_step | 0.00 | 24.84 | 128604.5 | 0.159 | 0.039 | 0.16 | 56.33 |
| dynamic_crossing | 256 | mppi | 0.00 | 2.93 | 45545.0 | 0.442 | 0.301 | 0.16 | 0.00 |
| dynamic_crossing | 256 | lp_mppi_smooth | 1.00 | 1.91 | 41604.9 | 0.193 | 0.062 | 0.17 | 0.00 |
| dynamic_crossing | 256 | pi_mppi | 1.00 | 1.88 | 43547.4 | 0.100 | 0.016 | 0.34 | 0.00 |
| dynamic_crossing | 256 | cdf_lp_mppi | 0.00 | 43.35 | 257923.4 | 0.152 | 0.040 | 0.15 | 0.00 |

Observed pattern:

- `cdf_lp_mppi` is a strong static-clutter positive signal: it cuts `cluttered K=128` final distance from MPPI's `38.54` to `17.76`, and beats `pi_mppi` on final distance at both `K=128` and `K=256`.
- The static improvement does not translate into success because the existing `cluttered` scenario remains hard under the episode step budget and goal tolerance.
- `cdf_mppi_one_step` is not acceptable in this bicycle benchmark. It makes progress but collides frequently; one-step CDF guidance is too myopic for the current nonholonomic dynamics and clutter geometry.
- Dynamic obstacles are a negative result. Even after reducing dynamic CDF pull/cost, `cdf_lp_mppi` stalls on `dynamic_crossing`, while LP-MPPI, dsMPPI, and pi-MPPI solve it. The lightweight CDF field acts like a local static avoidance potential and lacks timing-aware crossing behavior.
- The most useful next step for a faithful CDF-MPPI reproduction would be a separate grid or joint-space CDF planner with a one-step update law, tested on static configuration-space mazes rather than dynamic-crossing navigation.
