# DRA-MPPI Lightweight Reproduction

## Target

- Paper: "Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient
  Monte Carlo Approximations"
- Project page: https://autonomousrobots.nl/paper_websites/dra-mppi
- arXiv: https://arxiv.org/abs/2506.21205
- PDF: https://autonomousrobots.nl/assets/files/publications/25-trevisan-iros-dra-mppi.pdf
- Public reference implementation: no dedicated DRA-MPPI implementation was
  found in the web/GitHub search performed for this pass.

DRA-MPPI estimates the joint collision probability (CP) between each sampled
MPPI rollout and multiple dynamic obstacles. The paper uses Monte Carlo
samples over a shared rectangular region at each time step, evaluates dynamic
obstacle prediction distributions at those points, then integrates over each
rollout's circular collision region. CP can be used as a soft objective and as
a hard rejection signal for sampled trajectories above a risk threshold.

## Implemented Scope

This repository now has a lightweight CUDA reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_dra_risk` plus DRA tuning fields in `PlannerVariant`.
- `dra_collision_probability_device`, a fixed low-discrepancy Monte Carlo
  stencil over the collision disk around each rollout state.
- `dra_dynamic_occupancy_device`, a Gaussian occupancy model with an optional
  two-side lateral mode to emulate non-Gaussian / multimodal predictions.
- `compute_dra_risk_scores_kernel`, a per-rollout CUDA score kernel.
- Planner variants:
  - `dra_mppi_soft`: lower CP cost, high threshold, intended to avoid freezing.
  - `dra_mppi_hard`: stricter threshold and larger rejection penalty.
  - `dra_mppi_multimodal`: two lateral dynamic-obstacle modes.

The implemented score is:

```text
pred_sigma_t = pred_sigma0 + pred_sigma_growth * t * dt
cp_t = 1 - product_o(1 - mean_j occupancy_o(sample_j))
score = nominal_cost
        + soft_weight * mean(cp_t)
        + reject_cost * mean(max(0, cp_t - threshold)^2)
        - lambda * survival_power * log(geometric_mean(1 - cp_t))
```

This is not a full paper-faithful reproduction. It does not share one
rectangular MC sampling region across all rollouts, does not use a learned
pedestrian predictor, does not model full covariance matrices or arbitrary
prediction density shapes, and uses a deterministic low-discrepancy stencil
instead of randomized MC. The reproduced control hook is the useful online
piece: dynamic-obstacle CP scoring inside MPPI, with both soft risk and
hard-threshold rejection.

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
    --planners c2u_mppi_smooth,ducct_mppi_smooth,dra_mppi_soft,dra_mppi_hard,dra_mppi_multimodal,cc_mppi_smooth,tsallis_mppi_smooth,step_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/dra_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/dra_mppi_compare.csv \
      --markdown-out build-docker-smoke/dra_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Slalom stress benchmark:

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc './bin/benchmark_diff_mppi --quick \
    --scenarios uncertain_slalom,dynamic_slalom,model_mismatch_slalom \
    --planners c2u_mppi_smooth,ducct_mppi_smooth,dra_mppi_soft,dra_mppi_hard,dra_mppi_multimodal,cc_mppi_smooth,tsallis_mppi_smooth,step_mppi_smooth \
    --k-values 64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/dra_mppi_slalom_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/dra_mppi_slalom_compare.csv \
      --markdown-out build-docker-smoke/dra_mppi_slalom_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Artifacts:

- `build-docker-smoke/dra_mppi_compare.csv`
- `build-docker-smoke/dra_mppi_compare_summary.md`
- `build-docker-smoke/dra_mppi_compare_summary.tex`
- `build-docker-smoke/dra_mppi_slalom_compare.csv`
- `build-docker-smoke/dra_mppi_slalom_compare_summary.md`
- `build-docker-smoke/dra_mppi_slalom_compare_summary.tex`

All table values below are means over 3 seeds.

## Positive Result: Dynamic Crossing And Pincer

The soft DRA variant is the most useful setting in this benchmark. It keeps the
open dynamic scenes solved and handles dynamic pincer at every tested K.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|---:|
| dynamic_crossing | 32 | dra_mppi_soft | 1.00 | 252.0 | 1.90 | 0.19 |
| dynamic_crossing | 64 | dra_mppi_soft | 1.00 | 251.3 | 1.96 | 0.19 |
| dynamic_crossing | 128 | dra_mppi_soft | 1.00 | 252.0 | 1.94 | 0.24 |
| uncertain_crossing | 32 | dra_mppi_soft | 1.00 | 252.0 | 1.88 | 0.22 |
| uncertain_crossing | 64 | dra_mppi_soft | 1.00 | 252.3 | 1.92 | 0.23 |
| uncertain_crossing | 128 | dra_mppi_soft | 1.00 | 252.0 | 1.92 | 0.25 |
| dynamic_pincer | 32 | dra_mppi_soft | 1.00 | 258.3 | 1.86 | 0.35 |
| dynamic_pincer | 64 | dra_mppi_soft | 1.00 | 258.3 | 1.93 | 0.36 |
| dynamic_pincer | 128 | dra_mppi_soft | 1.00 | 257.0 | 1.89 | 0.38 |

At a fixed 0.25 ms cap across the four crossing/pincer/model-mismatch scenes,
`dra_mppi_soft` reaches `0.89` success over the scenarios that fit the cap and
`0.92` success at the 0.50 ms cap across all four scenarios. It is useful, but
the lighter `step_mppi_smooth` and `tsallis_mppi_smooth` remain stronger in
aggregate.

| Cap ms | Planner | Scenarios | Success | Final Dist | Avg ms | Mean K |
|---:|---|---:|---:|---:|---:|---:|
| 0.25 | dra_mppi_soft | 3 | 0.89 | 1.93 | 0.22 | 64 |
| 0.25 | ducct_mppi_smooth | 4 | 0.92 | 1.90 | 0.17 | 64 |
| 0.25 | step_mppi_smooth | 4 | 1.00 | 1.90 | 0.18 | 80 |
| 0.25 | tsallis_mppi_smooth | 4 | 1.00 | 1.90 | 0.15 | 80 |
| 0.50 | dra_mppi_soft | 4 | 0.92 | 1.91 | 0.25 | 56 |
| 0.50 | step_mppi_smooth | 4 | 1.00 | 1.90 | 0.18 | 80 |
| 0.50 | tsallis_mppi_smooth | 4 | 1.00 | 1.90 | 0.15 | 80 |

## Negative Result: Threshold Sensitivity

`dra_mppi_hard` exposes the paper's risk-threshold tradeoff in a useful way.
With a low CP threshold and high reject cost, it can reject too many useful
rollouts and freeze or stall in pincer.

| Scenario | K | Planner | Success | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|
| dynamic_pincer | 32 | dra_mppi_hard | 0.33 | 7.93 | 0.42 |
| dynamic_pincer | 64 | dra_mppi_hard | 1.00 | 1.92 | 0.43 |
| dynamic_pincer | 128 | dra_mppi_hard | 1.00 | 1.94 | 0.44 |

`dra_mppi_multimodal` is robust on pincer in this quick run, but its
two-lateral-mode occupancy roughly doubles the runtime compared with the soft
single-mode variant.

| Scenario | K | Planner | Success | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|
| dynamic_pincer | 32 | dra_mppi_multimodal | 1.00 | 1.88 | 0.81 |
| dynamic_pincer | 64 | dra_mppi_multimodal | 1.00 | 1.93 | 0.81 |
| dynamic_pincer | 128 | dra_mppi_multimodal | 1.00 | 1.92 | 0.83 |

## Negative Result: Slalom And Model Mismatch

No tested planner solved `dynamic_slalom`, `uncertain_slalom`, or
`model_mismatch_slalom` in this quick 3-seed run. DRA does not fix that class
of failure, and the hard/multimodal settings can make it worse.

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| dra_mppi_soft | 64 | 0.00 | 15.92 | 0.18 |
| dra_mppi_soft | 128 | 0.00 | 15.98 | 0.19 |
| dra_mppi_hard | 64 | 0.00 | 22.07 | 0.21 |
| dra_mppi_hard | 128 | 0.00 | 15.89 | 0.21 |
| step_mppi_smooth | 128 | 0.00 | 11.44 | 0.16 |
| tsallis_mppi_smooth | 64 | 0.00 | 14.21 | 0.14 |

On `model_mismatch_crossing`, DRA soft is competitive with C2U and DUCCT at
some K values but remains behind Step/Tsallis at high K:

| Scenario | K | Planner | Success | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|
| model_mismatch_crossing | 32 | dra_mppi_soft | 0.67 | 2.02 | 0.22 |
| model_mismatch_crossing | 64 | dra_mppi_soft | 0.33 | 2.05 | 0.22 |
| model_mismatch_crossing | 128 | dra_mppi_soft | 0.67 | 2.01 | 0.25 |
| model_mismatch_crossing | 128 | step_mppi_smooth | 1.00 | 1.93 | 0.20 |
| model_mismatch_crossing | 128 | tsallis_mppi_smooth | 1.00 | 1.95 | 0.18 |

## Takeaways

- DRA-style CP scoring is worth keeping as a reproducible risk-aware MPPI
  hook for dynamic-obstacle scenes.
- The soft CP cost is the best default from this pass; hard rejection is too
  sensitive when the rollout set is small.
- Multimodal occupancy can improve pincer robustness, but it is not free:
  roughly 0.8 ms on pincer in this run versus 0.35-0.38 ms for DRA soft.
- This implementation should not be presented as a full DRA-MPPI reproduction.
  It is a benchmarkable CUDA-side approximation of the central online risk
  scoring idea.
