# Tsallis-MPPI Lightweight Reproduction

## Target

- Paper: "Variational Inference MPC using Tsallis Divergence"
- Source: https://arxiv.org/abs/2104.00241
- RSS paper mirror: https://www.roboticsproceedings.org/rss17/p073.pdf
- Public reference implementation: no dedicated Tsallis-MPPI implementation was
  found in the web/GitHub search performed for this pass. Generic MPPI
  implementations exist, but they do not expose the Tsallis VI-MPC update.

The paper generalizes VI-MPC, MPPI, CEM, and SVGD-style MPC under a Tsallis
divergence objective. The practical control hook is replacing the usual
exponential optimality likelihood with a deformed, q-exponential transform,
which changes how aggressively high-cost samples are downweighted.

## Implemented Scope

This repository now has a lightweight nav reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_tsallis_weights`, `tsallis_q`, `tsallis_temperature`, and
  `tsallis_min_weight` in `PlannerVariant`.
- `compute_tsallis_weights_kernel`, which applies a normalized q-exponential
  transform to rollout costs and renormalizes the sample weights.
- Three planner variants:
  - `tsallis_mppi_q07`: compact-support style q-exponential with `q=0.70`.
  - `tsallis_mppi_smooth`: `q=0.70` plus the existing low-pass update
    (`low_pass_alpha=0.20`).
  - `tsallis_mppi_q13`: heavy-tail style q-exponential with `q=1.30`.

The implementation intentionally keeps the existing CUDA MPPI rollout path and
only swaps the sample likelihood/weight transform. It does not reproduce the
full VI-MPC derivation, mixture policy update, covariance update, warm-up
optimization loop, or stochastic dynamics ensemble from the paper.

The q-exponential used here is:

```text
exp_q(x) = [1 + (1 - q) x]_+ ^ (1 / (1 - q))
```

with the standard exponential as the `q -> 1` fallback. Costs are normalized
against the rollout minimum before the transform for stability.

## Build And Benchmark

Build:

```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc 'cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)'
```

Benchmark:

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc './bin/benchmark_diff_mppi --quick \
    --scenarios dynamic_bottleneck,dynamic_pincer,dynamic_crossing,uncertain_crossing \
    --planners mppi,lp_mppi_smooth,tsallis_mppi_q07,tsallis_mppi_smooth,tsallis_mppi_q13,step_mppi_smooth,csc_mppi_smooth,dm_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/tsallis_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/tsallis_mppi_compare.csv \
      --markdown-out build-docker-smoke/tsallis_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Artifacts:

- `build-docker-smoke/tsallis_mppi_compare.csv`
- `build-docker-smoke/tsallis_mppi_compare_summary.md`
- `build-docker-smoke/tsallis_mppi_compare_summary.tex`

All table values below are means over 3 seeds.

## Strong Positive Result: Cheap Dynamic Bottleneck Fix

`tsallis_mppi_smooth` solves the narrow timing-gate case where vanilla MPPI,
LP-MPPI, and Step-MPPI stall. It is much cheaper and smoother than CSC/DM, but
less robust at `K=64`.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness | Collisions |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | mppi | 0.00 | 320.0 | 24.52 | 46936.6 | 0.13 | 1.86 | 0.00 |
| dynamic_bottleneck | 32 | lp_mppi_smooth | 0.00 | 320.0 | 24.55 | 46444.6 | 0.14 | 0.49 | 0.00 |
| dynamic_bottleneck | 32 | step_mppi_smooth | 0.00 | 320.0 | 24.43 | 46390.8 | 0.14 | 0.60 | 0.00 |
| dynamic_bottleneck | 32 | tsallis_mppi_q07 | 0.00 | 320.0 | 24.95 | 46769.0 | 0.13 | 2.36 | 0.00 |
| dynamic_bottleneck | 32 | tsallis_mppi_smooth | 1.00 | 287.3 | 1.91 | 36381.3 | 0.13 | 0.50 | 0.00 |
| dynamic_bottleneck | 32 | csc_mppi_smooth | 1.00 | 272.0 | 1.90 | 34622.3 | 0.68 | 2.50 | 0.00 |
| dynamic_bottleneck | 32 | dm_mppi_smooth | 1.00 | 279.3 | 1.94 | 35585.1 | 0.91 | 3.32 | 0.00 |
| dynamic_bottleneck | 64 | tsallis_mppi_smooth | 0.67 | 305.7 | 9.69 | 40647.1 | 0.14 | 0.27 | 0.00 |
| dynamic_bottleneck | 128 | tsallis_mppi_smooth | 1.00 | 307.0 | 1.89 | 38885.6 | 0.16 | 0.16 | 0.00 |

Interpretation: q-reweighted samples plus low-pass smoothing suppress the
weighted-average stall without adding the expensive representative selection
used by CSC or the influence scan used by DM.

## Positive Result: Open Dynamic Crossings

In open crossing scenes, `tsallis_mppi_smooth` is competitive with LP/Step and
keeps the same lightweight runtime envelope.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_crossing | 128 | mppi | 0.00 | 260.0 | 3.20 | 46034.9 | 0.16 | 0.60 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 252.7 | 1.89 | 41670.5 | 0.18 | 0.11 |
| dynamic_crossing | 128 | step_mppi_smooth | 1.00 | 251.0 | 1.93 | 41218.1 | 0.15 | 0.13 |
| dynamic_crossing | 128 | tsallis_mppi_q07 | 1.00 | 250.7 | 1.93 | 41033.5 | 0.18 | 0.50 |
| dynamic_crossing | 128 | tsallis_mppi_smooth | 1.00 | 249.3 | 1.96 | 40780.6 | 0.18 | 0.17 |
| dynamic_crossing | 128 | csc_mppi_smooth | 1.00 | 249.7 | 1.93 | 41010.2 | 1.40 | 3.09 |
| dynamic_crossing | 128 | dm_mppi_smooth | 1.00 | 250.0 | 1.88 | 40961.7 | 2.55 | 3.58 |
| uncertain_crossing | 128 | tsallis_mppi_smooth | 1.00 | 250.0 | 1.92 | 40919.4 | 0.15 | 0.15 |

## Partial Result: Dynamic Pincer

The pincer scene is not the main win. `tsallis_mppi_smooth` reaches full success
at `K=64` and `K=128`, but LP is already strong in this benchmark pass.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_pincer | 64 | mppi | 0.00 | 260.0 | 9.67 | 47889.6 | 0.16 | 1.16 |
| dynamic_pincer | 64 | lp_mppi_smooth | 1.00 | 257.3 | 1.95 | 42238.8 | 0.16 | 0.23 |
| dynamic_pincer | 64 | step_mppi_smooth | 0.33 | 258.7 | 2.04 | 42049.4 | 0.20 | 0.26 |
| dynamic_pincer | 64 | tsallis_mppi_q07 | 0.33 | 260.0 | 2.04 | 41739.4 | 0.18 | 0.96 |
| dynamic_pincer | 64 | tsallis_mppi_smooth | 1.00 | 259.0 | 1.95 | 41796.7 | 0.18 | 0.31 |
| dynamic_pincer | 64 | tsallis_mppi_q13 | 0.00 | 260.0 | 2.86 | 43708.8 | 0.18 | 0.88 |
| dynamic_pincer | 128 | tsallis_mppi_smooth | 1.00 | 259.0 | 1.91 | 41695.1 | 0.17 | 0.15 |

## Negative Result: q Shape Matters

The q setting is not a cosmetic parameter:

- `tsallis_mppi_q07` without low-pass over-prunes the useful sample set and
  stalls in the bottleneck at all tested rollout counts.
- `tsallis_mppi_q13` keeps too much heavy-tail mass in the bottleneck and
  collides frequently: average collisions were `11.33`, `34.33`, and `8.00`
  at `K=32`, `K=64`, and `K=128`.
- `tsallis_mppi_smooth` is the only useful variant from this pass; the low-pass
  update is part of the working recipe, not a presentation tweak.

## Fixed-Time Summary

Under the 0.25 ms wall-clock cap, `tsallis_mppi_smooth` is the only tested
planner with aggregate success `1.00` across the four stress scenarios:

| Cap ms | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Mean K |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.25 | mppi | 0.17 | 269.4 | 6.39 | 45021.0 | 0.16 | 96 |
| 0.25 | lp_mppi_smooth | 0.75 | 270.4 | 7.58 | 42941.0 | 0.15 | 56 |
| 0.25 | step_mppi_smooth | 0.75 | 270.1 | 7.54 | 42734.2 | 0.18 | 72 |
| 0.25 | tsallis_mppi_q07 | 0.58 | 270.5 | 7.66 | 42614.5 | 0.18 | 128 |
| 0.25 | tsallis_mppi_q13 | 0.50 | 273.1 | 7.28 | 57504.0 | 0.14 | 72 |
| 0.25 | tsallis_mppi_smooth | 1.00 | 267.0 | 1.90 | 40649.3 | 0.15 | 88 |

## Takeaways

- Keep `tsallis_mppi_smooth` as a low-cost timing-gate stabilizer.
- Do not use unsmoothed `q=0.70` for bottlenecks; it gets stuck.
- Do not use `q=1.30` in constrained dynamic scenes without an additional
  safety layer; its heavy-tail behavior caused collisions in the bottleneck.
- CSC/DM remain more principled robust planners for hard bottlenecks, but they
  are much more expensive in this reproduction.
- The next useful step is combining Tsallis weighting with the existing
  Shield/BC safety filters to see whether the low-cost bottleneck gain survives
  with explicit collision suppression.
