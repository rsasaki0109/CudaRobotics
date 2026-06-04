# pi-MPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **pi-MPPI: A Projection-based Model Predictive Path Integral Scheme for Smooth Optimal Control of Fixed-Wing Aerial Vehicles** by Edvin Martin Andrejev, Amith Manoharan, Karl-Eerik Unt, and Arun Kumar Singh.

- arXiv: https://arxiv.org/abs/2504.10962
- Official OSS: https://github.com/edvinmandrejev/Pi-MPPI
- arXiv metadata checked on 2026-06-04: submitted 2025-04-15, revised 2025-04-16, journal reference IEEE RA-L Vol. 10 No. 6, June 2025.
- Core idea: project sampled MPPI control sequences onto constraints for control magnitude and higher-order control derivatives before the MPPI weighted update, so the sampled and averaged sequences stay smooth.

The paper has public OSS, so this is not an OSS-missing paper globally. The gap addressed here is local: this repo had MPPI, LP-MPPI, SOPPI, and Step-MPPI-style variants, but no projection-filtered MPPI path.

## Implementation

Implemented a lightweight pi-MPPI variant in `src/benchmark_diff_mppi.cu`:

- Added `PlannerVariant::use_projection_sampling`.
- Added projection hyperparameters:
  - `projection_passes`
  - `projection_max_accel_delta`
  - `projection_max_steer_delta`
  - `projection_max_accel_ddelta`
  - `projection_max_steer_ddelta`
- Added `project_control_component`, which projects a sampled control sequence by repeated clamping against:
  - control box constraints
  - first finite-difference bounds
  - second finite-difference bounds
- Added `rollout_projection_kernel`, which samples MPPI controls, projects each sequence before rollout, then evaluates the projected controls.
- Added `project_nominal_controls_kernel`, which projects the weighted-average nominal control sequence after the MPPI update.
- Added planner registrations:
  - `pi_mppi`: balanced projection constraints.
  - `pi_mppi_smooth`: tighter projection constraints selected by a small sweep.
- Added CLI overrides:
  - `--override-pi-passes`
  - `--override-pi-accel-delta`
  - `--override-pi-steer-delta`
  - `--override-pi-accel-ddelta`
  - `--override-pi-steer-ddelta`

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- The paper uses a projection filter formulated as a constrained optimization problem, with a neural accelerated custom optimizer. This implementation uses repeated forward/backward finite-difference clamping as a cheap approximation.
- The paper validates on fixed-wing aerial vehicle tasks. This validation is on the existing 2D bicycle navigation benchmark.
- The projection bounds are manually tuned for this benchmark's acceleration/steering controls.
- `pi_mppi_smooth` spends more runtime than LP-MPPI because the projection loop runs inside each rollout thread.
- This implementation is intended to test whether projection-filtered sampling is useful in this codebase before investing in a paper-faithful optimizer.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Default comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,pi_mppi,pi_mppi_smooth,lp_mppi,lp_mppi_smooth,soppi_fast \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/pi_mppi_compare.csv
```

Projection sweep:

```bash
for cfg in \
  loose:1:1.6:0.14:1.2:0.10 \
  default:2:1.2:0.10:1.0:0.08 \
  smooth:3:0.8:0.065:0.6:0.045 \
  tight:4:0.6:0.045:0.4:0.03
do
  IFS=: read name passes ad sd add sdd <<< "$cfg"
  ./bin/benchmark_diff_mppi \
    --quick \
    --scenarios dynamic_crossing,cluttered \
    --planners pi_mppi \
    --k-values 256 \
    --seed-count 3 \
    --override-pi-passes "$passes" \
    --override-pi-accel-delta "$ad" \
    --override-pi-steer-delta "$sd" \
    --override-pi-accel-ddelta "$add" \
    --override-pi-steer-ddelta "$sdd" \
    --csv "build-docker-smoke/pi_mppi_sweep_${name}.csv"
done
```

## Results

Artifacts:

- `build-docker-smoke/pi_mppi_compare.csv`
- `build-docker-smoke/pi_mppi_compare_summary.md`
- `build-docker-smoke/pi_mppi_sweep_loose.csv`
- `build-docker-smoke/pi_mppi_sweep_default.csv`
- `build-docker-smoke/pi_mppi_sweep_smooth.csv`
- `build-docker-smoke/pi_mppi_sweep_tight.csv`

Default comparison, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cluttered | 128 | mppi | 0.00 | 38.54 | 49828.4 | 0.562 | 0.519 | 0.18 |
| cluttered | 128 | lp_mppi_smooth | 0.00 | 39.81 | 49070.4 | 0.270 | 0.121 | 0.19 |
| cluttered | 128 | pi_mppi | 0.00 | 38.53 | 49114.0 | 0.141 | 0.034 | 0.28 |
| cluttered | 128 | pi_mppi_smooth | 0.00 | 39.24 | 49080.2 | 0.131 | 0.031 | 0.32 |
| cluttered | 256 | mppi | 0.00 | 38.57 | 49917.8 | 0.411 | 0.266 | 0.19 |
| cluttered | 256 | lp_mppi_smooth | 0.00 | 39.59 | 48986.2 | 0.200 | 0.070 | 0.18 |
| cluttered | 256 | pi_mppi | 0.00 | 26.04 | 44902.4 | 0.110 | 0.021 | 0.32 |
| cluttered | 256 | pi_mppi_smooth | 0.00 | 39.07 | 48977.1 | 0.102 | 0.022 | 0.49 |
| dynamic_crossing | 128 | mppi | 0.00 | 3.20 | 45976.6 | 0.623 | 0.599 | 0.13 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 1.87 | 41532.9 | 0.262 | 0.113 | 0.13 |
| dynamic_crossing | 128 | pi_mppi | 1.00 | 1.89 | 43458.2 | 0.134 | 0.029 | 0.22 |
| dynamic_crossing | 128 | pi_mppi_smooth | 1.00 | 1.88 | 41748.8 | 0.133 | 0.030 | 0.30 |
| dynamic_crossing | 256 | mppi | 0.00 | 2.99 | 45772.6 | 0.419 | 0.274 | 0.16 |
| dynamic_crossing | 256 | lp_mppi_smooth | 1.00 | 1.94 | 41633.0 | 0.194 | 0.062 | 0.16 |
| dynamic_crossing | 256 | pi_mppi | 1.00 | 1.92 | 43404.1 | 0.098 | 0.015 | 0.31 |
| dynamic_crossing | 256 | pi_mppi_smooth | 1.00 | 1.87 | 41965.0 | 0.101 | 0.018 | 0.44 |

Projection sweep, `K=256`, seed-count 3:

| Setting | Scenario | Success | Final Dist | Cost | Mean du | Roughness | Avg ms |
|---|---|---:|---:|---:|---:|---:|---:|
| loose | cluttered | 0.00 | 39.23 | 49596.2 | 0.120 | 0.024 | 0.37 |
| default | cluttered | 0.00 | 25.91 | 44746.7 | 0.113 | 0.021 | 0.36 |
| smooth | cluttered | 0.00 | 38.65 | 48950.0 | 0.107 | 0.022 | 0.43 |
| tight | cluttered | 0.00 | 38.99 | 48932.2 | 0.105 | 0.025 | 0.49 |
| loose | dynamic_crossing | 0.33 | 2.05 | 44033.9 | 0.123 | 0.024 | 0.29 |
| default | dynamic_crossing | 1.00 | 1.91 | 43182.3 | 0.100 | 0.016 | 0.34 |
| smooth | dynamic_crossing | 1.00 | 1.88 | 42271.4 | 0.097 | 0.016 | 0.41 |
| tight | dynamic_crossing | 1.00 | 1.85 | 41736.2 | 0.098 | 0.018 | 0.48 |

Observed pattern:

- On `dynamic_crossing`, vanilla MPPI and `soppi_fast` fail at `K=128/256`, while both pi-MPPI variants reach `1.00` success.
- `pi_mppi_smooth` nearly matches LP-MPPI cost on `dynamic_crossing` while keeping applied-control roughness much lower (`0.018` vs `0.062` at `K=256` against `lp_mppi_smooth`).
- `pi_mppi` is the better balanced default for `cluttered K=256`, improving final distance and cost substantially without solving the scenario.
- Tighter projection helps the dynamic crossing task, but can over-smooth exploration in cluttered geometry.
- The result is promising enough to keep pi-MPPI as a local benchmark variant, but the next step should be either a paper-faithful projection optimizer or extension to CartPole/pushing before making broad claims.
