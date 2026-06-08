# CC-MPPI Lightweight Reproduction

## Target

- Paper: "Trajectory Distribution Control for Model Predictive Path Integral
  Control using Covariance Steering"
- Source: https://arxiv.org/abs/2109.12147
- Related paper: "Constrained Covariance Steering Based Tube-MPPI"
- Related source: https://arxiv.org/abs/2110.07744
- Public reference implementation: no dedicated CC-MPPI implementation was found
  in the web/GitHub search performed for this pass.

The target paper combines MPPI with covariance steering. The key idea is to
control the dispersion of predicted rollout trajectories, especially terminal
dispersion, so the controller is less brittle under unexpected disturbances and
uncertainty.

## Implemented Scope

This repository now has a lightweight navigation reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_covariance_control_weights` plus CC-MPPI tuning fields in
  `PlannerVariant`.
- `compute_covariance_control_weights_kernel`, which:
  1. Computes a cost-weighted terminal mode from rollout endpoints.
  2. Measures each rollout endpoint's dispersion around that mode.
  3. Adds a terminal dispersion penalty above a target radius.
  4. Recomputes normalized MPPI weights from the adjusted costs.
- Planner variants:
  - `cc_mppi`: terminal dispersion weighting with standard MPPI noise.
  - `cc_mppi_smooth`: CC weighting plus low-pass sampled controls.
  - `cc_mppi_tight`: stronger terminal dispersion penalty and stronger
    smoothing.

This is not a full covariance-steering solver. It does not solve the LTV
covariance-control subproblem, does not synthesize the paper's feedback gain
sequence, and does not enforce a hard terminal covariance constraint. It is a
small CUDA-side approximation of the terminal trajectory-distribution control
effect.

## Build And Benchmarks

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Dynamic stress benchmark:

```bash
./bin/benchmark_diff_mppi --quick \
    --scenarios dynamic_bottleneck,dynamic_pincer,dynamic_crossing,uncertain_crossing \
    --planners mppi,lp_mppi_smooth,tsallis_mppi_smooth,cc_mppi,cc_mppi_smooth,cc_mppi_tight,step_mppi_smooth,csc_mppi_smooth,dm_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/cc_mppi_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/cc_mppi_compare.csv \
      --markdown-out build-docker-smoke/cc_mppi_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5
```

Uncertainty/mismatch benchmark:

```bash
./bin/benchmark_diff_mppi --quick \
    --scenarios model_mismatch_crossing,model_mismatch_slalom,uncertain_slalom,dynamic_slalom \
    --planners mppi,lp_mppi_smooth,cc_mppi_smooth,cc_mppi_tight,tsallis_mppi_smooth,pr_mppi_smooth,step_mppi_smooth \
    --k-values 64,128,256 \
    --seed-count 3 \
    --csv build-docker-smoke/cc_mppi_uncertainty_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/cc_mppi_uncertainty_compare.csv \
      --markdown-out build-docker-smoke/cc_mppi_uncertainty_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5
```

Artifacts:

- `build-docker-smoke/cc_mppi_compare.csv`
- `build-docker-smoke/cc_mppi_compare_summary.md`
- `build-docker-smoke/cc_mppi_uncertainty_compare.csv`
- `build-docker-smoke/cc_mppi_uncertainty_compare_summary.md`

All table values below are means over 3 seeds.

## Positive Result: Dynamic Pincer

The useful result is `cc_mppi_smooth` on `dynamic_pincer`. It is lightweight and
gets full success at `K=32`, `K=64`, and `K=128`. The unsmoothed `cc_mppi`
variant fails, so low-pass sampling is part of the recipe.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_pincer | 32 | mppi | 0.00 | 260.0 | 19.70 | 52974.7 | 0.12 | 1.76 |
| dynamic_pincer | 32 | lp_mppi_smooth | 1.00 | 257.7 | 1.89 | 42171.5 | 0.14 | 0.39 |
| dynamic_pincer | 32 | tsallis_mppi_smooth | 0.33 | 260.0 | 2.08 | 41984.9 | 0.13 | 0.69 |
| dynamic_pincer | 32 | csc_mppi_smooth | 0.33 | 260.0 | 2.23 | 42124.2 | 0.67 | 3.02 |
| dynamic_pincer | 32 | dm_mppi_smooth | 0.67 | 259.3 | 2.05 | 41945.4 | 1.00 | 4.37 |
| dynamic_pincer | 32 | cc_mppi_smooth | 1.00 | 257.3 | 1.91 | 42065.7 | 0.14 | 0.37 |
| dynamic_pincer | 64 | lp_mppi_smooth | 0.67 | 259.0 | 7.82 | 43360.4 | 0.13 | 0.19 |
| dynamic_pincer | 64 | tsallis_mppi_smooth | 0.67 | 260.0 | 1.94 | 41818.3 | 0.14 | 0.30 |
| dynamic_pincer | 64 | cc_mppi_smooth | 1.00 | 257.7 | 1.91 | 41955.0 | 0.15 | 0.22 |
| dynamic_pincer | 128 | cc_mppi_smooth | 1.00 | 257.0 | 1.87 | 41768.7 | 0.18 | 0.12 |

Interpretation: the terminal dispersion penalty seems to suppress the
near-goal mode wobble that causes several variants to miss the success
threshold in pincer.

## Positive/Neutral Result: Open Crossings

On `dynamic_crossing` and `uncertain_crossing`, `cc_mppi_smooth` behaves like a
slightly more conservative LP-MPPI: full success, low roughness, but not the
lowest cost.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_crossing | 128 | mppi | 0.00 | 260.0 | 3.39 | 46137.4 | 0.13 | 0.58 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 252.0 | 1.92 | 41602.6 | 0.15 | 0.11 |
| dynamic_crossing | 128 | cc_mppi_smooth | 1.00 | 251.3 | 1.92 | 41331.5 | 0.17 | 0.12 |
| dynamic_crossing | 128 | tsallis_mppi_smooth | 1.00 | 250.0 | 1.87 | 40814.1 | 0.15 | 0.16 |
| uncertain_crossing | 128 | lp_mppi_smooth | 1.00 | 252.3 | 1.91 | 41662.5 | 0.14 | 0.11 |
| uncertain_crossing | 128 | cc_mppi_smooth | 1.00 | 251.7 | 1.90 | 41380.7 | 0.17 | 0.10 |
| uncertain_crossing | 128 | tsallis_mppi_smooth | 1.00 | 250.0 | 1.89 | 40880.0 | 0.15 | 0.15 |

## Partial Result: Model Mismatch Crossing

CC-MPPI helps over vanilla MPPI and slightly over LP at `K=64`, but it does not
beat Step/Tsallis.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| model_mismatch_crossing | 64 | mppi | 0.00 | 300.0 | 5.43 | 53807.5 | 0.13 | 0.87 |
| model_mismatch_crossing | 64 | lp_mppi_smooth | 0.33 | 300.0 | 2.05 | 48842.7 | 0.13 | 0.11 |
| model_mismatch_crossing | 64 | cc_mppi_smooth | 0.67 | 300.0 | 1.98 | 48793.5 | 0.15 | 0.11 |
| model_mismatch_crossing | 64 | pr_mppi_smooth | 0.67 | 300.0 | 1.98 | 48751.9 | 0.18 | 0.13 |
| model_mismatch_crossing | 64 | step_mppi_smooth | 1.00 | 299.3 | 1.98 | 48581.1 | 0.14 | 0.13 |
| model_mismatch_crossing | 128 | cc_mppi_smooth | 0.67 | 300.0 | 1.94 | 48707.8 | 0.17 | 0.06 |
| model_mismatch_crossing | 128 | tsallis_mppi_smooth | 1.00 | 298.0 | 1.94 | 48220.4 | 0.16 | 0.07 |

## Negative Result: Dynamic Bottleneck

The bottleneck case is a clear failure. Terminal covariance collapse does not
discover the necessary wait-through-the-gate behavior. It makes the controller
smoothly stall before the gate.

| Scenario | K | Planner | Success | Steps | Final Dist | Cum. Cost | Avg Control ms | Roughness |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | mppi | 0.33 | 307.3 | 16.71 | 43085.3 | 0.13 | 1.86 |
| dynamic_bottleneck | 32 | lp_mppi_smooth | 0.00 | 320.0 | 24.92 | 46609.9 | 0.13 | 0.49 |
| dynamic_bottleneck | 32 | tsallis_mppi_smooth | 0.67 | 302.7 | 5.59 | 39771.0 | 0.13 | 0.53 |
| dynamic_bottleneck | 32 | cc_mppi | 0.00 | 320.0 | 23.54 | 46619.2 | 0.16 | 2.02 |
| dynamic_bottleneck | 32 | cc_mppi_smooth | 0.00 | 320.0 | 24.85 | 46475.3 | 0.16 | 0.53 |
| dynamic_bottleneck | 32 | csc_mppi_smooth | 1.00 | 272.3 | 1.93 | 34627.4 | 0.64 | 2.17 |
| dynamic_bottleneck | 32 | dm_mppi_smooth | 1.00 | 274.7 | 1.91 | 34970.3 | 0.88 | 3.46 |

## Negative Result: Slalom And Hard Model Mismatch

`dynamic_slalom`, `uncertain_slalom`, and `model_mismatch_slalom` remain
unsolved by this reproduction. CC-MPPI sometimes lowers final distance versus
vanilla MPPI, but it does not reach the goal and can be worse than plain MPPI
on the hard model-mismatch slalom because it over-concentrates around a poor
terminal mode.

## Takeaways

- Keep `cc_mppi_smooth` as a lightweight pincer/open-crossing stabilizer.
- Do not use CC weighting for `dynamic_bottleneck`; CSC, DM, or Tsallis are
  better directions there.
- Do not use unsmoothed `cc_mppi`; terminal dispersion weighting without
  low-pass noise is unstable and misses most dynamic scenes.
- A faithful next step would be adding the paper's feedback-gain sampling term
  and terminal covariance target, rather than only reweighting endpoints.
