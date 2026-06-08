# SC-MPPI Lightweight Reproduction

## Target

- Paper: "Safe Importance Sampling in Model Predictive Path Integral Control"
- Method name: Safety Controlled Model Predictive Path Integral Control (SC-MPPI)
- Source: https://arxiv.org/abs/2303.03441
- Public reference implementation: none found in the web/GitHub search performed
  for this pass.

The paper's core idea is to run MPPI forward sampling under an embedded safety
controller. Instead of only adding penalties after a rollout becomes unsafe, the
sampling dynamics are biased so that sampled trajectories remain inside the
domain where the safety controller can keep them feasible.

## Implemented Scope

This repository now has a lightweight nav reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_safety_controlled_sampling` plus safety-controller tuning parameters in
  `PlannerVariant`.
- `rollout_safety_controlled_kernel`, a rollout kernel that samples actions,
  predicts the next state, checks obstacle clearance, and locally adjusts the
  action before stepping the dynamics.
- `sc_mppi`, `sc_mppi_smooth`, and `sc_mppi_timing` planner variants.

The embedded controller is a margin-based local policy:

1. Sample a candidate action as in MPPI, optionally with low-pass action noise.
2. Predict the one-step next state under the raw action.
3. Find the nearest static or dynamic obstacle at the predicted time.
4. If the predicted margin is below `sc_safe_margin`, steer away from the
   obstacle and reduce acceleration, with bounded action deltas.
5. Store the corrected action in `d_perturbed`, roll out that corrected action,
   and use standard MPPI weighting.

This reproduces the forward-sampling placement of SC-MPPI, not the full paper
derivation. It does not implement the original barrier-state controller or the
full information-theoretic importance-sampling correction. The controller is a
small analytic obstacle-margin safety filter fitted to this benchmark.

## Build And Benchmark

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Benchmark:

```bash
./bin/benchmark_diff_mppi --quick \
    --scenarios dynamic_bottleneck,dynamic_pincer,dynamic_crossing,uncertain_crossing \
    --planners mppi,lp_mppi_smooth,sc_mppi,sc_mppi_smooth,sc_mppi_timing,shield_mppi,shield_mppi_repair,shield_mppi_smooth,step_mppi_smooth,bc_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/sc_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/sc_mppi_compare.csv \
      --markdown-out build-docker-smoke/sc_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5
```

Artifacts:

- `build-docker-smoke/sc_mppi_compare.csv`
- `build-docker-smoke/sc_mppi_compare_summary.md`
- `build-docker-smoke/sc_mppi_compare_summary.tex`

All table values below are means over 3 seeds. Standard deviations are omitted
here for compactness.

## Positive Result: Dynamic Crossing

`sc_mppi_smooth` solves the open crossing scene with the same success rate as
LP/Step/BC, but it is much faster than BC because it does not run a
single-thread trajectory probability scan.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing | 32 | mppi | 0.00 | 260.0 | 4.46 | 47190.6 | 0.11 |
| dynamic_crossing | 32 | lp_mppi_smooth | 1.00 | 253.3 | 1.88 | 41867.4 | 0.11 |
| dynamic_crossing | 32 | step_mppi_smooth | 1.00 | 251.7 | 1.93 | 41300.6 | 0.12 |
| dynamic_crossing | 32 | bc_mppi_smooth | 1.00 | 251.7 | 1.94 | 41308.0 | 0.62 |
| dynamic_crossing | 32 | sc_mppi_smooth | 1.00 | 252.0 | 1.92 | 41530.6 | 0.13 |
| dynamic_crossing | 128 | sc_mppi_smooth | 1.00 | 251.3 | 1.97 | 41359.5 | 0.13 |

## Positive Result: Dynamic Pincer

`sc_mppi_smooth` is positive on `dynamic_pincer` at `K=32` and `K=128`. The
`K=64` cell is partial in this seed set, so this is not as stable as LP/BC.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_pincer | 32 | mppi | 0.00 | 260.0 | 19.36 | 54674.3 | 0.12 |
| dynamic_pincer | 32 | lp_mppi_smooth | 1.00 | 257.7 | 1.89 | 42275.9 | 0.11 |
| dynamic_pincer | 32 | step_mppi_smooth | 1.00 | 257.0 | 1.92 | 41810.8 | 0.13 |
| dynamic_pincer | 32 | bc_mppi_smooth | 1.00 | 257.7 | 1.95 | 41867.8 | 0.74 |
| dynamic_pincer | 32 | sc_mppi_smooth | 1.00 | 257.3 | 1.91 | 41875.3 | 0.13 |
| dynamic_pincer | 64 | sc_mppi_smooth | 0.67 | 258.7 | 7.94 | 43354.8 | 0.15 |
| dynamic_pincer | 128 | sc_mppi_smooth | 1.00 | 257.7 | 1.88 | 41938.9 | 0.16 |

## Positive Result: Uncertain Crossing

`sc_mppi_smooth` keeps the same 1.00 success pattern under the uncertain
dynamic obstacle scene.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| uncertain_crossing | 32 | mppi | 0.00 | 260.0 | 4.08 | 46819.9 | 0.12 |
| uncertain_crossing | 32 | lp_mppi_smooth | 1.00 | 252.3 | 1.92 | 41794.3 | 0.12 |
| uncertain_crossing | 32 | step_mppi_smooth | 1.00 | 252.0 | 1.92 | 41337.1 | 0.13 |
| uncertain_crossing | 32 | bc_mppi_smooth | 1.00 | 252.3 | 1.92 | 41435.9 | 0.61 |
| uncertain_crossing | 32 | sc_mppi_smooth | 1.00 | 251.7 | 1.94 | 41436.7 | 0.14 |
| uncertain_crossing | 128 | sc_mppi_smooth | 1.00 | 252.0 | 1.89 | 41439.5 | 0.13 |

## Partial/Negative Result: Dynamic Bottleneck

The narrow timing-gate scene remains hard. `sc_mppi_timing` and `sc_mppi`
produce partial success at `K=128`, but do not improve over the best Shield
direction in this benchmark.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | mppi | 0.33 | 309.7 | 11.03 | 42306.8 | 0.11 |
| dynamic_bottleneck | 32 | sc_mppi_smooth | 0.00 | 320.0 | 24.39 | 46384.1 | 0.15 |
| dynamic_bottleneck | 32 | sc_mppi_timing | 0.00 | 320.0 | 22.59 | 52340.2 | 0.11 |
| dynamic_bottleneck | 128 | mppi | 0.33 | 312.0 | 16.83 | 43649.1 | 0.14 |
| dynamic_bottleneck | 128 | sc_mppi | 0.33 | 309.7 | 16.80 | 43198.2 | 0.16 |
| dynamic_bottleneck | 128 | sc_mppi_timing | 0.33 | 304.7 | 15.25 | 42578.8 | 0.12 |
| dynamic_bottleneck | 128 | shield_mppi | 0.33 | 318.0 | 14.84 | 43656.9 | 0.13 |
| dynamic_bottleneck | 128 | shield_mppi_repair | 0.33 | 302.7 | 14.86 | 41298.3 | 0.12 |
| dynamic_bottleneck | 128 | bc_mppi_smooth | 0.00 | 320.0 | 24.77 | 46542.7 | 2.19 |

## Takeaways

- `sc_mppi_smooth` is a good safety-controlled sampling reproduction for open
  dynamic avoidance: success matches LP/Step/BC while runtime stays around
  `0.13-0.16 ms`.
- The local safety controller is useful only when paired with low-pass sampling.
  `sc_mppi` without low-pass remains mixed.
- SC-MPPI is a much better runtime fit than the current BC-MPPI reproduction.
  The BC safety scan is `0.61-2.43 ms`; SC stays close to standard MPPI cost.
- The bottleneck/timing-gate scene is still not solved. A one-step margin
  controller is too local; the scene needs either timed waiting or a stronger
  CBF/repair hybrid.

## Next Faithful Steps

1. Replace the heuristic away-steering controller with a true barrier-state
   feedback controller.
2. Add an importance-sampling correction term for the controller-induced drift.
3. Combine SC forward sampling with the existing Shield first-action repair.
4. Add safe-sample-rate metrics, since SC-MPPI is mainly about making the
   sampled rollout distribution safer before weighting.
