# CSC-MPPI Lightweight Reproduction

## Target

- Paper: "CSC-MPPI: A Novel Constrained MPPI Framework with DBSCAN for
  Reliable Obstacle Avoidance"
- Source: https://arxiv.org/abs/2506.16386
- Project page: https://cscmppi.github.io/
- Reference code: https://github.com/RCILab/RCI_cscmppi

The paper addresses two MPPI failure modes: samples can violate constraints,
and weighted averaging can blend distinct trajectory modes into a bad action.
CSC-MPPI first shifts sampled inputs toward feasible regions, clusters sampled
trajectories with DBSCAN, then selects representative low-cost controls from
the resulting trajectory clusters.

## Implemented Scope

This repository now has a lightweight nav reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_cluster_representative_update` plus coarse cluster-selection parameters
  in `PlannerVariant`.
- `update_controls_from_cluster_representative_kernel`, which replaces MPPI's
  weighted-average control update with a representative sample selected from
  trajectory-space clusters.
- `csc_mppi`, `csc_mppi_smooth`, and `csc_mppi_strict` planner variants.

The reproduction uses the SC-MPPI safety-controlled rollout kernel as the
constraint-shift stage, then performs a coarse trajectory clustering step:

1. Generate safety-adjusted sampled rollouts.
2. Score each rollout by cost plus a margin-violation penalty.
3. Assign each rollout to a coarse cluster using mid-horizon and final `y`.
4. Keep the best rollout in each cluster.
5. Select the lowest-scoring representative and blend its control sequence into
   the nominal controls.

This captures the main "avoid weighted averaging across modes" behavior, but it
is not a full paper-faithful implementation. It does not implement DBSCAN,
primal-dual gradient shifting, or the JAX/JIT reference pipeline.

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
    --scenarios dynamic_bottleneck,dynamic_pincer,dynamic_crossing,uncertain_crossing,dynamic_crossing_with_topology \
    --planners mppi,lp_mppi_smooth,sc_mppi_smooth,csc_mppi,csc_mppi_smooth,csc_mppi_strict,shield_mppi_smooth,step_mppi_smooth,bc_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/csc_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/csc_mppi_compare.csv \
      --markdown-out build-docker-smoke/csc_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Artifacts:

- `build-docker-smoke/csc_mppi_compare.csv`
- `build-docker-smoke/csc_mppi_compare_summary.md`
- `build-docker-smoke/csc_mppi_compare_summary.tex`

All table values below are means over 3 seeds. Standard deviations are omitted
here for compactness.

## Strong Positive Result: Dynamic Bottleneck

This is the clearest positive result so far. The cluster-representative update
solves the narrow timing-gate scene that low-pass, Step, SC, BC, and smooth
Shield variants fail in this run.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | mppi | 0.00 | 320.0 | 24.48 | 46974.2 | 0.12 | 2.00 |
| dynamic_bottleneck | 32 | lp_mppi_smooth | 0.00 | 320.0 | 24.53 | 46417.7 | 0.13 | 0.44 |
| dynamic_bottleneck | 32 | sc_mppi_smooth | 0.00 | 320.0 | 24.67 | 46477.9 | 0.14 | 0.47 |
| dynamic_bottleneck | 32 | step_mppi_smooth | 0.00 | 320.0 | 24.84 | 46563.6 | 0.13 | 0.64 |
| dynamic_bottleneck | 32 | bc_mppi_smooth | 0.00 | 320.0 | 24.62 | 46544.9 | 0.82 | 0.53 |
| dynamic_bottleneck | 32 | csc_mppi_smooth | 1.00 | 274.0 | 1.94 | 34778.7 | 0.61 | 2.14 |
| dynamic_bottleneck | 32 | csc_mppi_strict | 1.00 | 278.7 | 1.92 | 35464.6 | 0.61 | 2.03 |

At higher rollout counts the pattern stays positive:

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 64 | csc_mppi | 1.00 | 296.7 | 1.97 | 37862.0 | 1.01 |
| dynamic_bottleneck | 64 | csc_mppi_smooth | 1.00 | 272.0 | 1.90 | 34551.3 | 0.97 |
| dynamic_bottleneck | 64 | csc_mppi_strict | 1.00 | 273.3 | 1.93 | 34702.1 | 0.97 |
| dynamic_bottleneck | 128 | csc_mppi | 1.00 | 297.7 | 1.91 | 38047.1 | 1.68 |
| dynamic_bottleneck | 128 | csc_mppi_smooth | 1.00 | 269.3 | 1.91 | 34218.7 | 1.69 |
| dynamic_bottleneck | 128 | csc_mppi_strict | 1.00 | 274.0 | 1.90 | 34811.4 | 1.67 |

Interpretation: selecting one representative trajectory avoids the bad
weighted-average behavior that otherwise stalls at the gate.

## Positive Result: Open Dynamic Crossings

CSC also solves the open crossing scenes. It is slower and rougher than LP/Step,
but can produce lower cumulative cost because it commits to one low-cost sample
instead of averaging.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 252.0 | 1.95 | 41592.2 | 0.15 | 0.11 |
| dynamic_crossing | 128 | step_mppi_smooth | 1.00 | 251.0 | 1.89 | 41177.3 | 0.13 | 0.14 |
| dynamic_crossing | 128 | csc_mppi_smooth | 1.00 | 250.0 | 1.87 | 40988.7 | 1.36 | 2.69 |
| uncertain_crossing | 128 | csc_mppi | 1.00 | 252.3 | 1.90 | 41299.8 | 1.47 | n/a |
| uncertain_crossing | 128 | csc_mppi_smooth | 1.00 | 250.0 | 1.90 | 40956.6 | 1.37 | n/a |
| uncertain_crossing | 128 | csc_mppi_strict | 1.00 | 250.7 | 1.89 | 41178.2 | 1.35 | n/a |

## Negative Result: Dynamic Pincer

CSC is not universally better. On `dynamic_pincer`, the representative update
gets close to the goal but often misses the success threshold; LP, Step, SC, and
BC are more stable here.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_pincer | 32 | lp_mppi_smooth | 1.00 | 257.7 | 1.92 | 42086.8 | 0.10 |
| dynamic_pincer | 32 | step_mppi_smooth | 1.00 | 258.0 | 1.90 | 42109.2 | 0.13 |
| dynamic_pincer | 32 | bc_mppi_smooth | 1.00 | 257.0 | 1.99 | 41916.6 | 0.73 |
| dynamic_pincer | 32 | csc_mppi_smooth | 0.00 | 260.0 | 2.33 | 42282.0 | 0.65 |
| dynamic_pincer | 128 | sc_mppi_smooth | 1.00 | 257.7 | 1.90 | 42038.2 | 0.15 |
| dynamic_pincer | 128 | csc_mppi_smooth | 0.67 | 260.0 | 1.98 | 42009.7 | 1.91 |

## Negative Result: Topology Crossing

`dynamic_crossing_with_topology` remains unsolved by every local MPPI-family
variant in this benchmark. CSC does not fix the lack of a global/topological
guide.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing_with_topology | 128 | mppi | 0.00 | 360.0 | 11.37 | 36589.3 | 0.13 |
| dynamic_crossing_with_topology | 128 | lp_mppi_smooth | 0.00 | 360.0 | 12.07 | 35674.8 | 0.13 |
| dynamic_crossing_with_topology | 128 | sc_mppi_smooth | 0.00 | 360.0 | 12.21 | 35687.4 | 0.16 |
| dynamic_crossing_with_topology | 128 | shield_mppi_smooth | 0.00 | 360.0 | 8.91 | 40967.0 | 0.12 |
| dynamic_crossing_with_topology | 128 | csc_mppi_smooth | 0.00 | 360.0 | 12.99 | 36235.2 | 2.30 |

## Takeaways

- CSC-style representative selection is the strongest result so far for
  `dynamic_bottleneck`: it solves all `K=32/64/128` smooth and strict cells.
- The cost is runtime and smoothness. Current CSC uses a single-thread cluster
  scan and commits to one sample, so `avg_control_ms` and roughness are much
  higher than LP/Step/SC.
- CSC is not a replacement for every planner. It is weak on `dynamic_pincer`
  and does not solve topology-guided planning.
- A useful ensemble policy is now visible:
  - use `csc_mppi_smooth` for timing-gate bottlenecks;
  - use `lp_mppi_smooth`, `step_mppi_smooth`, or `sc_mppi_smooth` for open
    dynamic crossing/pincer scenes;
  - use a global/topology planner for topology-constrained scenes.

## Next Faithful Steps

1. Replace the coarse `y` binning with actual DBSCAN over trajectory features.
2. Add a primal-dual projection step instead of using SC-MPPI's local safety
   controller as the constraint-shift surrogate.
3. Smooth or MPC-filter the selected representative sequence to reduce roughness.
4. Parallelize cluster scoring and margin scans.
