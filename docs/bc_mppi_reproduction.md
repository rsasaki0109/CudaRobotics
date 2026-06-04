# BC-MPPI Lightweight Reproduction

## Target

- Paper: "BC-MPPI: A Probabilistic Constraint Layer for Safe Model-Predictive Path-Integral Control"
- Source: https://arxiv.org/abs/2510.00272
- Venue/index references found during search:
  - https://discovery.ucl.ac.uk/id/eprint/10217409
  - https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202502218169465888
- Public reference implementation: none found in the web/GitHub search performed for this pass.

The paper adds a Bayesian/probabilistic constraint layer to MPPI. Each rollout
gets the usual MPPI exponential cost weight and an additional scalar feasibility
probability. Unsafe rollouts are not explicitly rejected; their contribution to
the weighted control update is suppressed by the probability factor.

## Implemented Scope

This repository now has a lightweight nav reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_bc_safety_layer` plus margin/probability parameters in `PlannerVariant`.
- `compute_bc_safety_weights_kernel`, which multiplies the standard MPPI weight
  by a rollout feasibility probability.
- `bc_mppi`, `bc_mppi_smooth`, and `bc_mppi_strict` planner variants.

The feasibility model is intentionally minimal:

1. For each rollout, scan predicted states over the horizon.
2. Compute obstacle clearance using the existing static/dynamic obstacle margin.
3. Convert margin to a per-step safety probability with a sigmoid.
4. Combine the per-step probabilities into one scalar feasibility estimate.
5. Multiply the MPPI cost weight by that scalar and renormalize.

This reproduces the control-side mechanism from BC-MPPI, but it is not a full
paper-faithful reproduction. The original method uses a trained probabilistic
surrogate, reported as a Bayesian classifier/BNN trained from offline
simulations. This implementation uses analytic simulator margins as a surrogate
classifier, has no learned uncertainty calibration, no versioned classifier
artifact, and currently runs the safety pass in one CUDA thread.

## Build And Benchmark

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Benchmark:

```bash
./bin/benchmark_diff_mppi --quick \
    --scenarios dynamic_bottleneck,dynamic_pincer,dynamic_crossing,uncertain_crossing \
    --planners mppi,lp_mppi_smooth,bc_mppi,bc_mppi_smooth,bc_mppi_strict,shield_mppi,shield_mppi_repair,shield_mppi_smooth,step_mppi_smooth,pi_mppi \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/bc_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/bc_mppi_compare.csv \
      --markdown-out build-docker-smoke/bc_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5
```

Artifacts:

- `build-docker-smoke/bc_mppi_compare.csv`
- `build-docker-smoke/bc_mppi_compare_summary.md`
- `build-docker-smoke/bc_mppi_compare_summary.tex`

All table values below are means over 3 seeds. Standard deviations are omitted
here for compactness; the generated summary has the full values.

## Positive Result: Dynamic Pincer

`bc_mppi_smooth` is positive on `dynamic_pincer`, especially at low rollout
budget where it reaches 1.00 success while vanilla MPPI fails and low-pass MPPI
is only partially successful in this run.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_pincer | 32 | mppi | 0.00 | 260.0 | 11.23 | 49364.2 | 0.12 |
| dynamic_pincer | 32 | lp_mppi_smooth | 0.67 | 259.0 | 7.91 | 43456.3 | 0.13 |
| dynamic_pincer | 32 | step_mppi_smooth | 1.00 | 257.0 | 1.92 | 41810.8 | 0.13 |
| dynamic_pincer | 32 | shield_mppi_smooth | 0.33 | 259.3 | 11.07 | 49359.8 | 0.13 |
| dynamic_pincer | 32 | bc_mppi_smooth | 1.00 | 258.0 | 1.88 | 42114.9 | 0.73 |
| dynamic_pincer | 32 | bc_mppi_strict | 0.67 | 258.3 | 2.76 | 42548.7 | 0.75 |
| dynamic_pincer | 128 | lp_mppi_smooth | 1.00 | 257.0 | 1.88 | 41994.3 | 0.18 |
| dynamic_pincer | 128 | step_mppi_smooth | 1.00 | 256.3 | 1.93 | 41768.7 | 0.14 |
| dynamic_pincer | 128 | bc_mppi_smooth | 1.00 | 257.0 | 1.93 | 41804.0 | 2.41 |
| dynamic_pincer | 128 | bc_mppi_strict | 1.00 | 257.7 | 1.90 | 42002.9 | 2.40 |

Interpretation: probability-weighting plus low-pass sampling can bias rollouts
away from moving-obstacle conflict regions without needing an additive penalty.
Runtime is the main weakness.

## Positive Result: Dynamic Crossing

`bc_mppi_smooth` and `bc_mppi_strict` also solve the open crossing scene, but
they are much slower than low-pass or Step-MPPI because of the current
single-thread safety scan.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing | 32 | mppi | 0.00 | 260.0 | 3.87 | 46583.1 | 0.12 |
| dynamic_crossing | 32 | lp_mppi_smooth | 1.00 | 253.0 | 1.89 | 41716.1 | 0.12 |
| dynamic_crossing | 32 | step_mppi_smooth | 1.00 | 251.7 | 1.93 | 41300.6 | 0.13 |
| dynamic_crossing | 32 | bc_mppi | 0.33 | 259.3 | 2.74 | 45171.1 | 0.61 |
| dynamic_crossing | 32 | bc_mppi_smooth | 1.00 | 252.3 | 1.95 | 41641.7 | 0.62 |
| dynamic_crossing | 32 | bc_mppi_strict | 1.00 | 253.3 | 1.96 | 41698.2 | 0.61 |
| dynamic_crossing | 128 | bc_mppi_smooth | 1.00 | 251.3 | 1.92 | 41329.3 | 1.75 |
| dynamic_crossing | 128 | bc_mppi_strict | 1.00 | 252.7 | 1.92 | 41643.0 | 1.84 |

## Positive Result: Uncertain Crossing

The same pattern holds under the uncertain dynamic obstacle scene.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| uncertain_crossing | 32 | mppi | 0.00 | 260.0 | 4.11 | 46840.6 | 0.12 |
| uncertain_crossing | 32 | lp_mppi_smooth | 1.00 | 252.7 | 1.91 | 41700.0 | 0.12 |
| uncertain_crossing | 32 | step_mppi_smooth | 1.00 | 252.0 | 1.92 | 41337.1 | 0.13 |
| uncertain_crossing | 32 | bc_mppi_smooth | 1.00 | 252.0 | 1.92 | 41527.7 | 0.61 |
| uncertain_crossing | 32 | bc_mppi_strict | 1.00 | 251.3 | 1.92 | 41459.7 | 0.62 |
| uncertain_crossing | 128 | bc_mppi_smooth | 1.00 | 251.7 | 1.90 | 41376.8 | 1.92 |
| uncertain_crossing | 128 | bc_mppi_strict | 1.00 | 252.7 | 1.88 | 41563.5 | 1.85 |

## Negative Result: Dynamic Bottleneck

BC-MPPI is negative on the narrow timing-gate scene. The probability layer
over-penalizes trajectories near the moving obstacle and stalls instead of
finding the tight timing solution.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | mppi | 0.67 | 303.3 | 9.48 | 40827.9 | 0.13 |
| dynamic_bottleneck | 32 | lp_mppi_smooth | 0.00 | 320.0 | 24.73 | 46434.6 | 0.13 |
| dynamic_bottleneck | 32 | bc_mppi | 0.00 | 320.0 | 23.87 | 46657.8 | 0.78 |
| dynamic_bottleneck | 32 | bc_mppi_smooth | 0.00 | 320.0 | 24.91 | 46541.3 | 0.69 |
| dynamic_bottleneck | 32 | bc_mppi_strict | 0.00 | 320.0 | 24.69 | 46573.7 | 0.70 |
| dynamic_bottleneck | 128 | mppi | 0.33 | 308.3 | 16.83 | 43352.7 | 0.14 |
| dynamic_bottleneck | 128 | shield_mppi | 0.33 | 318.0 | 14.84 | 43656.9 | 0.14 |
| dynamic_bottleneck | 128 | shield_mppi_repair | 0.33 | 302.7 | 14.86 | 41298.3 | 0.14 |
| dynamic_bottleneck | 128 | bc_mppi_smooth | 0.00 | 320.0 | 24.73 | 46485.1 | 2.17 |
| dynamic_bottleneck | 128 | bc_mppi_strict | 0.00 | 320.0 | 24.87 | 46586.2 | 2.14 |

## Takeaways

- The BC probability layer is useful in open dynamic avoidance scenes when
  paired with low-pass sampling.
- `bc_mppi` without low-pass sampling is mixed; the probability factor alone
  tends to suppress too many useful rollouts.
- The current reproduction is computationally inefficient. `bc_mppi_smooth`
  at `K=128` costs roughly `1.75-2.41 ms` in the tested dynamic scenes, while
  low-pass and Step-MPPI variants are around `0.13-0.18 ms`.
- The bottleneck result is a real negative. The BC layer is too conservative for
  narrow timing windows in this margin-sigmoid implementation.

## Next Faithful Steps

1. Replace the analytic margin sigmoid with a trained probabilistic classifier
   or BNN artifact built from offline rollouts.
2. Parallelize `compute_bc_safety_weights_kernel` across rollouts and timesteps.
3. Add probability calibration metrics and log the per-rollout feasibility
   distribution to verify that the layer is not collapsing weights.
4. Test a hybrid with Shield-MPPI: BC probability weighting for sample scoring,
   CBF/repair for the first action in narrow timing-gate scenes.
