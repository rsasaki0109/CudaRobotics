# DUCCT-MPPI Lightweight Reproduction

## Target

- Paper: "Chance-Constrained MPPI under State and Dynamic Object Prediction
  Uncertainty and the Evaluation of Collision Risk Calibration"
- Source: https://arxiv.org/abs/2605.28330
- Public reference implementation: none found in the web/GitHub search
  performed for this pass.

The paper proposes Dual-Uncertainty Chance-Constrained Tube MPPI
(DUCCT-MPPI). It combines a one-tube Unscented Transform approximation for
robot localization uncertainty with dynamic-obstacle prediction uncertainty,
then evaluates joint collision risk and uses it as both a soft MPPI cost and a
hard chance-constraint rejection signal. The paper also emphasizes that risk
estimates must be calibrated, using metrics such as Brier score, Log Loss, and
NEES-style consistency checks.

## Implemented Scope

This repository now has a lightweight navigation reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_ducct_risk` plus DUCCT tuning fields in `PlannerVariant`.
- `ducct_margin_risk_device`, a Gaussian-overlap-style risk surrogate.
- `ducct_joint_risk_device`, which aggregates obstacle risks as
  `1 - prod(1 - p_i)`.
- `compute_ducct_risk_scores_kernel`, a per-rollout CUDA score kernel.
- Planner variants:
  - `ducct_mppi_smooth`: lightly tuned risk hook for normal use.
  - `ducct_mppi_cautious`: inflated uncertainty / stricter chance threshold.
  - `ducct_mppi_diluted`: intentionally large uncertainty to expose
    over-dispersion sensitivity.

The implemented score is:

```text
robot_sigma_t = loc_sigma0 + loc_sigma_growth * sqrt(t * dt)
pred_sigma_t = pred_sigma0 + pred_sigma_growth * t * dt
risk_t = 1 - product_i(1 - obstacle_risk_i)
score = nominal_cost
        + risk_weight * mean(risk_t)
        + reject_cost * mean(max(0, risk_t - threshold)^2)
        - lambda * survival_power * log(geometric_mean(1 - risk_t))
```

The risk helper includes a dilution factor:

```text
dilution = radius^2 / (radius^2 + sigma^2)
```

This makes extreme uncertainty able to flatten local risk instead of always
making the planner more conservative, matching the failure mode discussed in
the paper at a lightweight level.

This is not a full paper-faithful reproduction. It does not propagate a true
3x3 robot covariance with an Unscented Transform, does not perform Monte Carlo
sampling over a shared collision area, does not use a goal-directed pedestrian
belief tube, does not compute Brier score or Log Loss from predicted executed
risk, and does not run NEES calibration checks. The useful part reproduced here
is the online control hook: a shared localization-uncertainty tube plus joint
collision-risk scoring inside MPPI.

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
    --planners c2u_mppi_smooth,ducct_mppi_smooth,ducct_mppi_cautious,ducct_mppi_diluted,cc_mppi_smooth,tsallis_mppi_smooth,step_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/ducct_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/ducct_mppi_compare.csv \
      --markdown-out build-docker-smoke/ducct_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Slalom stress benchmark:

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc './bin/benchmark_diff_mppi --quick \
    --scenarios uncertain_slalom,dynamic_slalom,model_mismatch_slalom \
    --planners c2u_mppi_smooth,ducct_mppi_smooth,ducct_mppi_cautious,ducct_mppi_diluted,cc_mppi_smooth,tsallis_mppi_smooth,step_mppi_smooth \
    --k-values 64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/ducct_mppi_slalom_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/ducct_mppi_slalom_compare.csv \
      --markdown-out build-docker-smoke/ducct_mppi_slalom_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Artifacts:

- `build-docker-smoke/ducct_mppi_compare.csv`
- `build-docker-smoke/ducct_mppi_compare_summary.md`
- `build-docker-smoke/ducct_mppi_compare_summary.tex`
- `build-docker-smoke/ducct_mppi_slalom_compare.csv`
- `build-docker-smoke/ducct_mppi_slalom_compare_summary.md`
- `build-docker-smoke/ducct_mppi_slalom_compare_summary.tex`

All table values below are means over 3 seeds.

## Positive Result: Open Dynamic Scenes

`ducct_mppi_smooth` solves the easy open crossing scenes and stays in the same
sub-millisecond budget as the other parallel CUDA-side MPPI score variants.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| dynamic_crossing | 32 | ducct_mppi_smooth | 1.00 | 252.3 | 1.88 | 0.17 |
| dynamic_crossing | 64 | ducct_mppi_smooth | 1.00 | 251.7 | 1.93 | 0.17 |
| dynamic_crossing | 128 | ducct_mppi_smooth | 1.00 | 252.7 | 1.89 | 0.18 |
| uncertain_crossing | 32 | ducct_mppi_smooth | 1.00 | 252.0 | 1.93 | 0.17 |
| uncertain_crossing | 64 | ducct_mppi_smooth | 1.00 | 252.7 | 1.90 | 0.18 |
| uncertain_crossing | 128 | ducct_mppi_smooth | 1.00 | 252.0 | 1.96 | 0.18 |

At the fixed 0.25 ms cap across crossing/pincer/model-mismatch scenes,
`ducct_mppi_smooth` reaches `0.83` success at about `0.18 ms`. This is useful
as a DUCCT-style uncertainty hook, but it does not beat `step_mppi_smooth`,
`tsallis_mppi_smooth`, `cc_mppi_smooth`, or `c2u_mppi_smooth` overall.

## Partial Result: Pincer Sensitivity

After tuning, `ducct_mppi_smooth` can solve dynamic pincer, but it is more
sensitive than C2U or Step-MPPI.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| dynamic_pincer | 32 | ducct_mppi_smooth | 1.00 | 258.7 | 1.93 | 0.17 |
| dynamic_pincer | 64 | ducct_mppi_smooth | 0.67 | 259.0 | 2.00 | 0.18 |
| dynamic_pincer | 128 | ducct_mppi_smooth | 1.00 | 258.0 | 1.91 | 0.20 |
| dynamic_pincer | 128 | c2u_mppi_smooth | 1.00 | 257.3 | 1.96 | 0.18 |
| dynamic_pincer | 128 | step_mppi_smooth | 1.00 | 257.3 | 1.88 | 0.18 |

The inflated DUCCT variants expose the calibration failure mode:

| Scenario | K | Planner | Success | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|
| dynamic_pincer | 64 | ducct_mppi_cautious | 0.00 | 23.97 | 0.18 |
| dynamic_pincer | 128 | ducct_mppi_cautious | 0.00 | 24.60 | 0.19 |
| dynamic_pincer | 64 | ducct_mppi_diluted | 0.00 | 35.10 | 0.18 |
| dynamic_pincer | 128 | ducct_mppi_diluted | 0.00 | 34.63 | 0.19 |

This is a useful negative: simply inflating uncertainty produces conservative
stalling in the current benchmark rather than robust crowd navigation.

## Negative Result: Model Mismatch And Slalom

`ducct_mppi_smooth` remains weak under vehicle model mismatch.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| model_mismatch_crossing | 32 | ducct_mppi_smooth | 0.33 | 300.0 | 2.06 | 0.16 |
| model_mismatch_crossing | 64 | ducct_mppi_smooth | 0.00 | 300.0 | 2.09 | 0.18 |
| model_mismatch_crossing | 128 | ducct_mppi_smooth | 0.33 | 300.0 | 2.05 | 0.18 |
| model_mismatch_crossing | 128 | step_mppi_smooth | 1.00 | 300.0 | 1.90 | 0.16 |
| model_mismatch_crossing | 128 | tsallis_mppi_smooth | 1.00 | 298.7 | 1.95 | 0.16 |

The slalom stress benchmark is also negative. No tested planner solved
`dynamic_slalom`, `uncertain_slalom`, or `model_mismatch_slalom` in this quick
3-seed run. DUCCT did not produce the best final distance; Step-MPPI and
Tsallis-MPPI remained stronger on these layouts.

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| ducct_mppi_smooth | 64 | 0.00 | 15.96 | 0.14 |
| ducct_mppi_smooth | 128 | 0.00 | 17.74 | 0.15 |
| step_mppi_smooth | 128 | 0.00 | 10.71 | 0.13 |
| tsallis_mppi_smooth | 64 | 0.00 | 13.74 | 0.11 |

## Takeaways

- Keep `ducct_mppi_smooth` as a DUCCT-style uncertainty aggregation ablation,
  not as the default planner.
- Keep `ducct_mppi_cautious` and `ducct_mppi_diluted` as calibration-sensitivity
  negative controls. They make the conservative/freezing failure mode visible.
- The current reproduction lacks the paper's most important evaluation piece:
  predicted executed-step risk with Brier score, Log Loss, and NEES-style
  calibration checks.
- A more faithful next pass should log per-step predicted risk, compare it to
  actual collision events, and add a true one-tube UT covariance propagation
  instead of the scalar sigma schedule used here.
