# SVG-MPPI Reproduction Notes

Date: 2026-06-04

## Target

Primary paper: **Stein Variational Guided Model Predictive Path Integral Control: Proposal and Experiments with Fast Maneuvering Vehicles**.

- arXiv: https://arxiv.org/abs/2309.11040
- OSS: https://github.com/kohonda/proj-svg_mppi
- ICRA 2024 implementation notes: ROS/C++ implementation, Docker/native workflows, and evaluation scripts are provided by the authors.
- Core idea: MPPI struggles when the optimal action distribution is multimodal. SVG-MPPI estimates a target mode with a modified SVGD step, then embeds that mode into MPPI as a fast mode-seeking closed-form update.

Related OSS checked during the web research phase:

- https://github.com/tud-amr/SVG-MPPI
- This is a different SVG-MPPI expansion: Semantic Visibility Graph + MPPI for Navigation Among Movable Obstacles, accepted to ICRA 2025. It uses a global semantic visibility graph to guide local MPPI rollouts around movable obstacles. It is relevant to the broader "SVG-MPPI" name, but it is not the Stein-mode method reproduced here.

## Implementation

Implemented a lightweight Stein-mode SVG-MPPI reproduction in `src/benchmark_diff_mppi.cu`:

- Added `PlannerVariant::use_svg_mode_guidance`, `svg_bandwidth`, `svg_mode_weight`, and `svg_stride`.
- Added `compute_svg_mode_weights_kernel`.
- The kernel finds the lowest-cost rollout in the current MPPI batch, treats it as a target trajectory mode, and reweights all samples by an RBF distance in trajectory space:
  - state distance uses x/y, heading, and velocity terms;
  - horizon samples are sub-sampled by `svg_stride`;
  - final sample weight is MPPI's cost weight multiplied by the trajectory-mode affinity.
- Reused the standard MPPI control update after replacing the sample weights.
- Added planners:
  - `svg_mppi`: trajectory-mode weighting without low-pass sampling.
  - `svg_mppi_smooth`: trajectory-mode weighting plus low-pass sampled control noise.
  - `svg_mppi_strong`: stronger mode weighting plus low-pass sampling.

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- It does not run a full SVGD particle update.
- It does not compute analytic Stein gradients over the action distribution.
- It uses the current best rollout as a single target mode, so it can collapse too aggressively in multimodal scenes.
- It adds an O(K*T/stride) trajectory-distance pass every MPPI update. The current kernel is intentionally simple and single-threaded, so runtime is higher than the other lightweight variants.
- It does not implement the TU Delft Semantic Visibility Graph planner or movable-object physics guidance.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Main comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios dynamic_crossing,cluttered \
  --planners mppi,lp_mppi_smooth,svg_mppi,svg_mppi_smooth,svg_mppi_strong,soppi_fast,ds_mppi,pi_mppi,cdf_lp_mppi,step_mppi_smooth \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/svg_mppi_compare.csv
```

Summary:

```bash
python3 scripts/summarize_diff_mppi.py \
  --csv build-docker-smoke/svg_mppi_compare.csv \
  --markdown-out build-docker-smoke/svg_mppi_compare_summary.md \
  --time-caps 0.5,1.0,2.0 \
  --time-targets 0.5,1.0
```

## Results

Artifacts:

- `build-docker-smoke/svg_mppi_compare.csv`
- `build-docker-smoke/svg_mppi_compare_summary.md`
- `build-docker-smoke/svg_mppi_compare_summary.tex`

Dynamic crossing, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing | 128 | mppi | 0.00 | 260.0 | 3.20 | 46034.9 | 0.13 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 252.7 | 1.89 | 41670.5 | 0.13 |
| dynamic_crossing | 128 | step_mppi_smooth | 1.00 | 250.3 | 1.93 | 41088.7 | 0.14 |
| dynamic_crossing | 128 | svg_mppi | 0.67 | 259.0 | 1.93 | 44113.7 | 0.49 |
| dynamic_crossing | 128 | svg_mppi_smooth | 1.00 | 251.3 | 1.92 | 41299.0 | 0.49 |
| dynamic_crossing | 128 | svg_mppi_strong | 1.00 | 251.3 | 1.93 | 41267.6 | 0.50 |
| dynamic_crossing | 256 | mppi | 0.00 | 260.0 | 3.14 | 45928.8 | 0.16 |
| dynamic_crossing | 256 | lp_mppi_smooth | 1.00 | 252.0 | 1.90 | 41567.5 | 0.16 |
| dynamic_crossing | 256 | step_mppi_smooth | 1.00 | 250.7 | 1.91 | 41160.2 | 0.17 |
| dynamic_crossing | 256 | svg_mppi | 0.67 | 260.0 | 1.91 | 44378.6 | 0.92 |
| dynamic_crossing | 256 | svg_mppi_smooth | 1.00 | 251.7 | 1.87 | 41329.5 | 0.91 |
| dynamic_crossing | 256 | svg_mppi_strong | 1.00 | 251.3 | 1.93 | 41212.8 | 0.92 |

Cluttered, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| cluttered | 128 | mppi | 0.00 | 220.0 | 38.54 | 49885.9 | 0.14 |
| cluttered | 128 | cdf_lp_mppi | 0.00 | 220.0 | 17.76 | 58713.7 | 0.15 |
| cluttered | 128 | pi_mppi | 0.00 | 220.0 | 38.56 | 49245.5 | 0.24 |
| cluttered | 128 | svg_mppi | 0.00 | 220.0 | 38.79 | 49567.1 | 0.51 |
| cluttered | 128 | svg_mppi_smooth | 0.00 | 220.0 | 39.76 | 48990.3 | 0.50 |
| cluttered | 128 | svg_mppi_strong | 0.00 | 220.0 | 39.99 | 49076.4 | 0.51 |
| cluttered | 256 | mppi | 0.00 | 220.0 | 38.43 | 49848.6 | 0.17 |
| cluttered | 256 | cdf_lp_mppi | 0.00 | 220.0 | 17.82 | 59303.7 | 0.15 |
| cluttered | 256 | pi_mppi | 0.00 | 220.0 | 25.87 | 44747.2 | 0.31 |
| cluttered | 256 | svg_mppi | 0.00 | 220.0 | 38.83 | 49529.1 | 0.85 |
| cluttered | 256 | svg_mppi_smooth | 0.00 | 220.0 | 39.84 | 49037.0 | 0.92 |
| cluttered | 256 | svg_mppi_strong | 0.00 | 220.0 | 39.97 | 49072.5 | 0.92 |

Observed pattern:

- `svg_mppi_smooth` and `svg_mppi_strong` are positive on `dynamic_crossing`: vanilla MPPI fails, while the low-pass SVG variants reach `1.00` success at `K=128/256`.
- The non-smooth `svg_mppi` is mixed: final distance is good, but success is only `0.67` and control is rougher.
- The current SVG pass is much slower than `lp_mppi_smooth` or `step_mppi_smooth`. It gets similar dynamic behavior, but not a better speed/performance tradeoff.
- On `cluttered`, the result is negative. Single-mode trajectory weighting does not discover the narrow/static route; `cdf_lp_mppi` and `pi_mppi` remain more useful for that scenario.
- The next faithful step would be a parallel SVGD-style target-mode estimator over action particles, then a mode-seeking MPPI update based on that estimated target distribution instead of the single best rollout.
