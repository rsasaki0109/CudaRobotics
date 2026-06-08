# C2U-MPPI Lightweight Reproduction

## Target

- Paper: "Chance-Constrained Sampling-Based MPC for Collision Avoidance in
  Uncertain Dynamic Environments"
- Source: https://arxiv.org/abs/2501.08520
- Code-index check: https://www.catalyzex.com/paper/chance-constrained-sampling-based-mpc-for
- Public reference implementation: none found in the web/GitHub search
  performed for this pass.

The paper introduces C2U-MPPI, a chance-constrained extension of U-MPPI for
dynamic obstacle avoidance under uncertainty. The core idea used in this
reproduction is to convert probabilistic collision constraints into a
deterministic safety backoff, then use that risk-aware value when computing
MPPI rollout weights.

Related follow-up to investigate next: "Chance-Constrained MPPI under State and
Dynamic Object Prediction Uncertainty and the Evaluation of Collision Risk
Calibration" (arXiv:2605.28330). It appears to extend the same family with
state/localization uncertainty, dynamic-object prediction uncertainty, and risk
calibration metrics.

## Implemented Scope

This repository now has a lightweight navigation reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_c2u_chance_constraints` plus C2U tuning fields in `PlannerVariant`.
- `c2u_chance_margin_device`, a deterministic chance-constraint margin helper.
- `compute_c2u_chance_scores_kernel`, a per-rollout CUDA score kernel.
- Planner variants:
  - `c2u_mppi`: unsmoothed chance-constraint ablation.
  - `c2u_mppi_smooth`: low-pass sampling plus moderate chance backoff.
  - `c2u_mppi_strict`: higher uncertainty and risk backoff sensitivity run.

The implemented chance margin uses five sigma points for each dynamic obstacle:

```text
mean obstacle center
mean +/- obstacle_sigma_x
mean +/- obstacle_sigma_y
```

For a rollout state and predicted obstacle center distribution, the score kernel
computes:

```text
chance_margin = mean(clearance) - z * sqrt(var(clearance) + robot_sigma^2) - safe_margin
step_probability = sigmoid(chance_margin / prob_sigma)
rollout_feasibility = geometric_mean(step_probability)
score = nominal_cost + violation_weight * mean(max(0, -chance_margin)^2)
        - lambda * log(rollout_feasibility)
```

The resulting score is normalized with the existing `compute_weights_kernel`.

This is not a full paper-faithful reproduction. It does not implement full
U-MPPI covariance propagation for robot state and controls, does not use a full
layered dynamic obstacle representation, does not update an obstacle belief
state, and assumes diagonal isotropic obstacle uncertainty. The useful part
reproduced here is the CUDA-side chance-constraint control hook.

## Build And Benchmark

Build:

```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc 'cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)'
```

Crossing/pincer benchmark:

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc './bin/benchmark_diff_mppi --quick \
    --scenarios uncertain_crossing,model_mismatch_crossing,dynamic_crossing,dynamic_pincer \
    --planners lp_mppi_smooth,bc_mppi_smooth,c2u_mppi_smooth,c2u_mppi_strict,cc_mppi_smooth,tsallis_mppi_smooth,step_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/c2u_mppi_parallel_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/c2u_mppi_parallel_compare.csv \
      --markdown-out build-docker-smoke/c2u_mppi_parallel_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Slalom stress benchmark:

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc './bin/benchmark_diff_mppi --quick \
    --scenarios uncertain_slalom,dynamic_slalom,model_mismatch_slalom \
    --planners lp_mppi_smooth,bc_mppi_smooth,c2u_mppi_smooth,c2u_mppi_strict,cc_mppi_smooth,tsallis_mppi_smooth,step_mppi_smooth \
    --k-values 64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/c2u_mppi_slalom_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/c2u_mppi_slalom_compare.csv \
      --markdown-out build-docker-smoke/c2u_mppi_slalom_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Artifacts:

- `build-docker-smoke/c2u_mppi_parallel_compare.csv`
- `build-docker-smoke/c2u_mppi_parallel_compare_summary.md`
- `build-docker-smoke/c2u_mppi_parallel_compare_summary.tex`
- `build-docker-smoke/c2u_mppi_slalom_compare.csv`
- `build-docker-smoke/c2u_mppi_slalom_compare_summary.md`
- `build-docker-smoke/c2u_mppi_slalom_compare_summary.tex`

All table values below are means over 3 seeds.

## Positive Result: Cheap Open-Scene Chance Layer

`c2u_mppi_smooth` solves the open crossing scenes and is far cheaper than the
existing BC probability layer.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| uncertain_crossing | 32 | c2u_mppi_smooth | 1.00 | 252.0 | 1.95 | 0.14 |
| uncertain_crossing | 32 | bc_mppi_smooth | 1.00 | 252.3 | 1.92 | 0.61 |
| uncertain_crossing | 64 | c2u_mppi_smooth | 1.00 | 252.0 | 1.93 | 0.16 |
| uncertain_crossing | 64 | bc_mppi_smooth | 1.00 | 251.3 | 1.92 | 1.00 |
| uncertain_crossing | 128 | c2u_mppi_smooth | 1.00 | 251.7 | 1.94 | 0.16 |
| uncertain_crossing | 128 | bc_mppi_smooth | 1.00 | 251.3 | 1.96 | 1.74 |
| dynamic_crossing | 32 | c2u_mppi_smooth | 1.00 | 251.3 | 1.90 | 0.12 |
| dynamic_crossing | 64 | c2u_mppi_smooth | 1.00 | 252.0 | 1.91 | 0.12 |
| dynamic_crossing | 128 | c2u_mppi_smooth | 1.00 | 251.7 | 1.91 | 0.14 |

At the fixed 0.25 ms cap across the four crossing/pincer scenarios,
`c2u_mppi_smooth` reaches `0.92` success at about `0.15 ms` mean control time.
`bc_mppi_smooth` only appears under the 1.00 ms cap in this run because its
current probability pass is much more expensive.

## Partial Result: Dynamic Pincer Needs Rollout Budget

C2U is mixed on the pincer scene. The smooth variant needs higher rollout
budget; the strict variant helps at `K=64` but is more sensitive elsewhere.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| dynamic_pincer | 32 | c2u_mppi_smooth | 0.33 | 259.3 | 3.06 | 0.14 |
| dynamic_pincer | 64 | c2u_mppi_smooth | 0.67 | 259.0 | 7.83 | 0.14 |
| dynamic_pincer | 128 | c2u_mppi_smooth | 1.00 | 256.7 | 1.92 | 0.15 |
| dynamic_pincer | 32 | c2u_mppi_strict | 0.33 | 259.7 | 3.01 | 0.13 |
| dynamic_pincer | 64 | c2u_mppi_strict | 1.00 | 256.7 | 1.93 | 0.14 |
| dynamic_pincer | 128 | c2u_mppi_strict | 1.00 | 258.3 | 1.89 | 0.15 |

## Negative Result: Model Mismatch And Slalom

`c2u_mppi_smooth` does not solve the model-mismatch crossing reliably at low
rollout budgets. Step-MPPI and Tsallis-MPPI are stronger in that scenario.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| model_mismatch_crossing | 32 | c2u_mppi_smooth | 0.00 | 300.0 | 2.06 | 0.12 |
| model_mismatch_crossing | 64 | c2u_mppi_smooth | 0.00 | 300.0 | 2.12 | 0.12 |
| model_mismatch_crossing | 128 | c2u_mppi_smooth | 0.67 | 300.0 | 1.99 | 0.16 |
| model_mismatch_crossing | 128 | step_mppi_smooth | 1.00 | 300.0 | 1.90 | 0.15 |
| model_mismatch_crossing | 128 | tsallis_mppi_smooth | 1.00 | 298.7 | 1.95 | 0.15 |

The slalom stress benchmark is also negative. None of the tested planners
solved `dynamic_slalom`, `uncertain_slalom`, or `model_mismatch_slalom` in this
quick 3-seed run, and C2U did not produce a clear best final-distance result.

## Implementation Note

The first C2U prototype computed chance scores in one CUDA thread and took
roughly 1-6 ms depending on rollout count. The current implementation uses a
parallel per-rollout score kernel, reducing the C2U overhead to roughly
0.12-0.16 ms in the tested scenes.

## Takeaways

- Keep `c2u_mppi_smooth` as a cheap uncertainty/chance-constraint ablation for
  open dynamic obstacle scenes.
- Do not promote C2U to the default planner. Step-MPPI, Tsallis-MPPI, and
  CC-MPPI are still stronger overall in this benchmark set.
- Treat `c2u_mppi_strict` as a sensitivity variant. It can help pincer at
  moderate rollout budget, but can become overconservative in clutter.
- A more faithful next pass should add full U-MPPI state covariance propagation,
  calibrated obstacle prediction uncertainty, and risk calibration metrics.
