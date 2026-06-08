# Shield-MPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **Shield Model Predictive Path Integral: A Computationally Efficient Robust MPC Approach Using Control Barrier Functions**.

- arXiv: https://arxiv.org/abs/2302.11719
- Submitted on 2023-02-23.
- Core idea: MPPI can violate constraints when the horizon or sample count is too small. Shield-MPPI augments MPPI trajectory costs with a control-barrier-function style safety term and adds a local repair step so a smaller trajectory population can still preserve safety.

Related OSS / implementation references checked during web research:

- https://github.com/shaoanlu/mppi_cbf
- This is a notebook playground with MPPI + CBF cost and CBF-QP style safety filters. It is not the Shield-MPPI paper's official implementation, but it is useful for reproducing the CBF-cost and safety-filter pattern in a compact form.

Related papers:

- Model Predictive Path Integral Methods with Reach-Avoid Tasks and Control Barrier Functions: https://arxiv.org/abs/2407.13693
- Path Integral Methods with Stochastic Control Barrier Functions: https://arxiv.org/abs/2206.11985

## Implementation

Implemented a lightweight Shield-MPPI reproduction in `src/benchmark_diff_mppi.cu`:

- Added `PlannerVariant::use_shield_cost` and `use_shield_repair`.
- Added shield tuning fields:
  - `shield_safe_margin`
  - `shield_cbf_alpha`
  - `shield_cbf_weight`
  - `shield_repair_steps`
  - `shield_repair_grid`
  - `shield_repair_accel_delta`
  - `shield_repair_steer_delta`
  - `shield_repair_safety_weight`
- Added `min_obstacle_margin_device`.
- Added `rollout_shield_kernel`.
- The shield rollout uses a discrete CBF-like condition:
  - `h(x) = obstacle_margin(x) - safe_margin`
  - violation if `h(x_next) < (1 - alpha) * h(x)`
  - cost adds squared violation plus extra penalty if `h(x_next) < 0`
- Added host-side local repair:
  - `apply_shield_repair`
  - `shield_candidate_score`
  - It runs after `sync_nominal_from_device()` and before executing `h_nominal_[0:2]`.
  - It evaluates a small candidate grid around the first MPPI action, plus explicit brake candidates, over a short rollout.
- Added planners:
  - `shield_mppi`: CBF rollout cost plus local first-action repair, short horizon `T=12`.
  - `shield_mppi_smooth`: same but with low-pass sampled noise.
  - `shield_mppi_repair`: standard MPPI rollout plus local first-action repair only.

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- No CBF-QP solve.
- No formal safety guarantee.
- No continuous-time CBF derivation for the bicycle dynamics.
- The repair is a small discrete candidate search, not gradient-based repair.
- The CBF only uses circular static/dynamic obstacle margins already available in the benchmark.
- The local repair uses the planning scenario's nominal dynamics and obstacle prediction, so it does not solve hidden model mismatch.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Main comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_bottleneck,dynamic_pincer,dynamic_crossing,uncertain_crossing \
  --planners mppi,lp_mppi_smooth,shield_mppi,shield_mppi_smooth,shield_mppi_repair,step_mppi_smooth,pi_mppi,ds_mppi \
  --k-values 32,64,128 \
  --seed-count 3 \
  --csv build-docker-smoke/shield_mppi_compare.csv
```

Summary:

```bash
python3 scripts/summarize_diff_mppi.py \
  --csv build-docker-smoke/shield_mppi_compare.csv \
  --markdown-out build-docker-smoke/shield_mppi_compare_summary.md \
  --time-caps 0.25,0.5,1.0 \
  --time-targets 0.25,0.5
```

## Results

Artifacts:

- `build-docker-smoke/shield_mppi_compare.csv`
- `build-docker-smoke/shield_mppi_compare_summary.md`
- `build-docker-smoke/shield_mppi_compare_summary.tex`

Dynamic bottleneck, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Collisions | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | mppi | 0.33 | 309.7 | 11.03 | 42306.8 | 0.00 | 0.11 |
| dynamic_bottleneck | 32 | shield_mppi | 0.33 | 305.0 | 14.66 | 43217.1 | 0.00 | 0.10 |
| dynamic_bottleneck | 32 | shield_mppi_repair | 0.33 | 305.3 | 15.43 | 43372.5 | 0.00 | 0.10 |
| dynamic_bottleneck | 32 | step_mppi_smooth | 0.33 | 318.0 | 17.26 | 44442.8 | 0.00 | 0.13 |
| dynamic_bottleneck | 64 | mppi | 0.00 | 320.0 | 24.38 | 46948.0 | 0.00 | 0.12 |
| dynamic_bottleneck | 64 | shield_mppi | 0.33 | 317.3 | 15.14 | 44032.5 | 0.00 | 0.12 |
| dynamic_bottleneck | 64 | step_mppi_smooth | 0.00 | 320.0 | 24.91 | 46330.7 | 0.00 | 0.13 |
| dynamic_bottleneck | 128 | mppi | 0.33 | 312.0 | 16.83 | 43649.1 | 0.00 | 0.12 |
| dynamic_bottleneck | 128 | lp_mppi_smooth | 0.00 | 320.0 | 24.81 | 46498.9 | 0.00 | 0.12 |
| dynamic_bottleneck | 128 | pi_mppi | 0.00 | 320.0 | 24.51 | 46292.9 | 0.00 | 0.21 |
| dynamic_bottleneck | 128 | shield_mppi | 1.00 | 292.0 | 1.86 | 37620.2 | 0.00 | 0.11 |
| dynamic_bottleneck | 128 | shield_mppi_repair | 1.00 | 272.0 | 1.83 | 34602.6 | 0.00 | 0.12 |
| dynamic_bottleneck | 128 | shield_mppi_smooth | 0.00 | 320.0 | 22.30 | 86671.9 | 20.33 | 0.12 |
| dynamic_bottleneck | 128 | step_mppi_smooth | 0.00 | 320.0 | 24.72 | 46514.3 | 0.00 | 0.14 |

Dynamic crossing, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing | 32 | mppi | 0.00 | 260.0 | 4.46 | 47190.6 | 0.11 |
| dynamic_crossing | 32 | lp_mppi_smooth | 1.00 | 253.3 | 1.88 | 41867.4 | 0.11 |
| dynamic_crossing | 32 | shield_mppi | 0.00 | 260.0 | 4.66 | 47795.1 | 0.11 |
| dynamic_crossing | 32 | shield_mppi_smooth | 1.00 | 257.0 | 1.89 | 43972.9 | 0.11 |
| dynamic_crossing | 32 | step_mppi_smooth | 1.00 | 251.3 | 1.94 | 41310.3 | 0.12 |
| dynamic_crossing | 128 | mppi | 0.00 | 260.0 | 3.20 | 45976.6 | 0.12 |
| dynamic_crossing | 128 | shield_mppi | 0.00 | 260.0 | 4.49 | 47569.9 | 0.14 |
| dynamic_crossing | 128 | shield_mppi_smooth | 1.00 | 256.0 | 1.99 | 43871.4 | 0.14 |
| dynamic_crossing | 128 | step_mppi_smooth | 1.00 | 250.3 | 1.96 | 41120.1 | 0.15 |

Dynamic pincer, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_pincer | 32 | mppi | 0.00 | 260.0 | 19.36 | 54674.3 | 0.13 |
| dynamic_pincer | 32 | lp_mppi_smooth | 1.00 | 257.7 | 1.89 | 42275.9 | 0.13 |
| dynamic_pincer | 32 | shield_mppi_smooth | 0.00 | 260.0 | 2.66 | 44990.1 | 0.13 |
| dynamic_pincer | 32 | step_mppi_smooth | 1.00 | 257.0 | 1.95 | 41723.9 | 0.14 |
| dynamic_pincer | 64 | shield_mppi_smooth | 0.67 | 260.0 | 2.20 | 44416.2 | 0.10 |
| dynamic_pincer | 128 | shield_mppi_smooth | 0.67 | 260.0 | 1.93 | 44173.0 | 0.12 |
| dynamic_pincer | 128 | step_mppi_smooth | 1.00 | 256.3 | 1.90 | 41707.6 | 0.14 |

Observed pattern:

- `shield_mppi` and `shield_mppi_repair` are positive on `dynamic_bottleneck`, the timing-gate scene that defeats low-pass, pi-MPPI, dsMPPI, and Step-MPPI in this run. At `K=128`, both reach `1.00` success while vanilla MPPI is `0.33`.
- The local repair-only variant is the best bottleneck result in this sweep: `success=1.00`, `steps=272.0`, `cost=34602.6`.
- `shield_mppi_smooth` is positive on open dynamic scenes (`dynamic_crossing`, `uncertain_crossing`) and partially positive on `dynamic_pincer`.
- `shield_mppi` without low-pass is negative on open dynamic scenes: the CBF/repair combination is too conservative and misses the goal threshold.
- `shield_mppi_smooth` is negative on `dynamic_bottleneck`: low-pass action noise plus barrier repair can still collide or stall in the narrow timing gate.
- The practical rule from this scaffold is scenario-dependent:
  - use `shield_mppi_repair` or `shield_mppi` for bottleneck/timing-gate safety;
  - use `shield_mppi_smooth`, `lp_mppi_smooth`, or `step_mppi_smooth` for open dynamic avoidance;
  - do not treat this as a universal replacement for the smoother MPPI variants.

Next faithful step:

- Replace the discrete repair grid with a CBF-QP or differentiable local repair.
- Make the CBF margin account for predicted swept volume, not only point-state obstacle distance.
- Tune the safety filter against actual model mismatch and uncertain obstacle predictions.
