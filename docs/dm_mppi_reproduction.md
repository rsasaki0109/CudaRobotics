# DM-MPPI Lightweight Reproduction

## Target

- Paper: "DM-MPPI: Datamodel for Efficient and Safe Model Path Integral Control"
- Source: https://arxiv.org/abs/2512.00759
- arXiv status checked: v1 submitted 2025-11-30, v2 revised 2026-03-25.
- Public reference implementation: none found in the web/GitHub search performed
  for this pass.

The paper extends the Datamodels idea to MPPI. It learns a sample influence
predictor from rollout cost features, prunes low-influence samples, and monitors
the influence of constraint-violating samples to tune safety penalties.

## Implemented Scope

This repository now has a lightweight nav reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_datamodel_influence_pruning` plus datamodel surrogate parameters in
  `PlannerVariant`.
- `compute_dm_influence_weights_kernel`, which estimates sample influence from
  rollout cost and minimum obstacle margin.
- `dm_mppi`, `dm_mppi_smooth`, and `dm_mppi_safe` planner variants.

The implementation is not a trained datamodel. It uses a feature surrogate:

1. Scan each rollout and compute its minimum static/dynamic obstacle margin.
2. Build an adjusted score: rollout cost plus margin-violation penalty.
3. Convert normalized adjusted score and safety probability into an influence
   estimate.
4. Keep only the top influence fraction.
5. Renormalize the remaining influences and run the standard MPPI control
   update.

This reproduces the online control-side behavior of influence pruning and
constraint-aware weighting. It does not implement offline influence-coefficient
training, the learned predictor, or true computational sample reduction. The
current kernel still scans every rollout and is intentionally conservative.

## Build And Benchmark

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Benchmark:

```bash
./bin/benchmark_diff_mppi --quick \
    --scenarios dynamic_bottleneck,dynamic_pincer,dynamic_crossing,uncertain_crossing \
    --planners mppi,lp_mppi_smooth,dm_mppi,dm_mppi_smooth,dm_mppi_safe,bc_mppi_smooth,sc_mppi_smooth,csc_mppi_smooth,step_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/dm_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/dm_mppi_compare.csv \
      --markdown-out build-docker-smoke/dm_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5
```

Artifacts:

- `build-docker-smoke/dm_mppi_compare.csv`
- `build-docker-smoke/dm_mppi_compare_summary.md`
- `build-docker-smoke/dm_mppi_compare_summary.tex`

All table values below are means over 3 seeds. Standard deviations are omitted
here for compactness.

## Strong Positive Result: Dynamic Bottleneck

DM-style influence pruning solves the timing-gate scene at `K=32` for all three
variants. `dm_mppi_safe` and `dm_mppi_smooth` stay stable through `K=128`.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | mppi | 0.33 | 307.3 | 16.71 | 43085.4 | 0.15 | 1.86 |
| dynamic_bottleneck | 32 | lp_mppi_smooth | 0.00 | 320.0 | 24.92 | 46609.9 | 0.15 | 0.49 |
| dynamic_bottleneck | 32 | bc_mppi_smooth | 0.00 | 320.0 | 24.62 | 46544.9 | 0.80 | 0.53 |
| dynamic_bottleneck | 32 | csc_mppi_smooth | 1.00 | 275.0 | 1.89 | 34872.3 | 0.62 | 2.44 |
| dynamic_bottleneck | 32 | dm_mppi | 1.00 | 294.3 | 1.90 | 37484.4 | 0.91 | 7.48 |
| dynamic_bottleneck | 32 | dm_mppi_safe | 1.00 | 274.3 | 1.89 | 34831.1 | 0.90 | 3.90 |
| dynamic_bottleneck | 32 | dm_mppi_smooth | 1.00 | 274.7 | 1.91 | 34970.3 | 0.90 | 3.46 |
| dynamic_bottleneck | 128 | dm_mppi_safe | 1.00 | 273.0 | 1.90 | 34656.2 | 3.07 | 3.42 |
| dynamic_bottleneck | 128 | dm_mppi_smooth | 1.00 | 274.7 | 1.89 | 34969.0 | 3.05 | 3.33 |

Interpretation: pruning to high-influence, constraint-respecting rollouts avoids
the same weighted-average stall that CSC fixed. The tradeoff is higher control
roughness and much higher runtime in this surrogate implementation.

## Positive Result: Open Dynamic Crossings

DM-MPPI also solves `dynamic_crossing` and `uncertain_crossing`. It often gets
lower cumulative cost than LP/Step, but at much higher runtime.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_crossing | 32 | lp_mppi_smooth | 1.00 | 252.0 | 1.92 | 41516.5 | 0.12 | 0.35 |
| dynamic_crossing | 32 | step_mppi_smooth | 1.00 | 251.0 | 1.96 | 41136.2 | 0.13 | 0.42 |
| dynamic_crossing | 32 | dm_mppi_smooth | 1.00 | 250.7 | 1.84 | 40961.9 | 0.75 | 4.19 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 252.0 | 1.92 | 41602.6 | 0.14 | 0.11 |
| dynamic_crossing | 128 | dm_mppi_smooth | 1.00 | 249.7 | 1.89 | 40869.8 | 2.44 | 4.00 |
| uncertain_crossing | 32 | dm_mppi_smooth | 1.00 | 249.7 | 1.91 | 40924.9 | 0.74 | 3.67 |
| uncertain_crossing | 128 | dm_mppi_safe | 1.00 | 251.0 | 1.90 | 41213.0 | 2.47 | 3.64 |

## Negative/Partial Result: Dynamic Pincer

DM-MPPI is not a replacement for low-pass/Step/SC on pincer. It gets close to
the goal, but often misses the success threshold.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_pincer | 32 | lp_mppi_smooth | 1.00 | 257.7 | 1.89 | 42171.5 | 0.12 |
| dynamic_pincer | 32 | step_mppi_smooth | 1.00 | 257.7 | 1.96 | 41896.0 | 0.13 |
| dynamic_pincer | 32 | dm_mppi | 0.00 | 260.0 | 2.54 | 42509.4 | 0.99 |
| dynamic_pincer | 32 | dm_mppi_smooth | 0.67 | 259.3 | 2.05 | 41945.4 | 0.99 |
| dynamic_pincer | 128 | sc_mppi_smooth | 1.00 | 256.7 | 1.94 | 41842.7 | 0.16 |
| dynamic_pincer | 128 | dm_mppi | 0.67 | 260.0 | 1.92 | 41956.5 | 3.49 |
| dynamic_pincer | 128 | dm_mppi_smooth | 0.67 | 259.3 | 1.98 | 41820.7 | 3.44 |

## Takeaways

- The lightweight DM surrogate is positive for timing-gate safety and open
  dynamic crossings.
- It is not actually efficient yet. The paper's efficiency comes from learned
  influence prediction and sample pruning; this reproduction still evaluates and
  scans every rollout.
- Influence pruning behaves like a softer version of CSC representative
  selection: it fixes bottleneck but increases roughness.
- On pincer, the pruning is too aggressive and drops useful alternate samples.

## Next Faithful Steps

1. Train an actual influence predictor from logged rollout features instead of
   using the hand-coded score.
2. Move pruning before expensive rollout/cost evaluation for true sample-count
   reduction.
3. Add adaptive penalty state across control updates, not only per-rollout score
   adjustment.
4. Add a post-update smoothing filter for pruned high-influence controls.
