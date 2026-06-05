# DBaS-Log-MPPI Lightweight Reproduction

## Target

- Paper: "DBaS-Log-MPPI: Efficient and Safe Trajectory Optimization via
  Barrier States"
- Source: https://arxiv.org/abs/2504.06437
- HTML: https://ar5iv.labs.arxiv.org/html/2504.06437v1
- Public reference implementation: no dedicated DBaS-Log-MPPI implementation
  was found in the web/GitHub search performed for this pass.

The paper combines three ideas: embedding Discrete Barrier States (DBaS) into
the MPPI rollout cost, increasing exploration from the current barrier state,
and using a normal-lognormal sampling distribution for more feasible
high-variance perturbations. The reported paper experiments compare against
Vanilla MPPI and Log-MPPI on a 2D quadrotor, a ground vehicle, and a small
real-world ground-vehicle setup.

## Implemented Scope

This repository now has a lightweight CUDA reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_dbas_log_sampling` plus DBaS/Log-MPPI tuning fields in
  `PlannerVariant`.
- `dbas_log_noise_device`, a symmetric normal-lognormal perturbation sampler.
- `dbas_barrier_state_device`, a clipped continuous obstacle-clearance barrier
  surrogate.
- `rollout_dbas_log_kernel`, which adds barrier-state cost during rollout and
  adapts perturbation scale from the current barrier state.
- Planner variants:
  - `log_mppi`: log-sampling ablation without DBaS barrier cost.
  - `dbas_log_mppi`: moderate DBaS barrier cost.
  - `dbas_log_mppi_smooth`: DBaS plus low-pass sampled controls.
  - `dbas_log_mppi_agile`: lower barrier weight, lower margin, higher
    exploration.
  - `dbas_log_mppi_safe`: stricter margin and barrier weight ablation.

The implemented surrogate is:

```text
z_t = (min_obstacle_clearance(x_t) - safe_margin) / eps
B_t = 1 / (1 + z_t)^2      if z_t >= 0
B_t = 1 + z_t^2            otherwise
B_t = min(B_t, barrier_cap)

beta_t = (1 - gamma) * B_t + gamma * beta_{t-1}
exploration_t = sqrt(mu * log(e + beta_t)) * noise_scale

log_gain = clamp(-0.5 * sigma^2 + sigma * N(0, 1), -4, log(clip))
noise = N(0, 1) * exp(log_gain) * exploration_t

cost += barrier_weight * beta_t * dt
cost += speed_damping * barrier_weight * beta_t * v_t^2 * dt
```

This is not a full paper-faithful reproduction. It does not augment the
vehicle state vector with an exact DBaS dynamic equation, does not reproduce
the paper's 2D quadrotor or Antelope vehicle setup, does not use the same
obstacle shapes or reference tracking tasks, does not apply a Savitzky-Golay
filter, and does not provide a formal safety proof. The useful reproduced
piece is the online control hook: DBaS-like continuous obstacle-risk cost,
adaptive exploration, and normal-lognormal sampling inside the existing CUDA
MPPI benchmark.

## Build And Benchmark

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Static/clutter benchmark:

```bash
./bin/benchmark_diff_mppi --quick \
    --scenarios narrow_passage,static_s_corridor,static_u_trap,cluttered \
    --planners log_mppi,dbas_log_mppi,dbas_log_mppi_smooth,dbas_log_mppi_agile,dbas_log_mppi_safe,mppi,lp_mppi_smooth,step_mppi_smooth,sc_mppi_smooth,tsallis_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/dbas_log_mppi_tuned_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/dbas_log_mppi_tuned_compare.csv \
      --markdown-out build-docker-smoke/dbas_log_mppi_tuned_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5
```

Dynamic-obstacle benchmark:

```bash
./bin/benchmark_diff_mppi --quick \
    --scenarios dynamic_crossing,dynamic_pincer,uncertain_crossing \
    --planners log_mppi,dbas_log_mppi_smooth,dbas_log_mppi_agile,sc_mppi_smooth,step_mppi_smooth,tsallis_mppi_smooth,dra_mppi_soft \
    --k-values 64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/dbas_log_mppi_dynamic_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/dbas_log_mppi_dynamic_compare.csv \
      --markdown-out build-docker-smoke/dbas_log_mppi_dynamic_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5
```

Artifacts:

- `build-docker-smoke/dbas_log_mppi_static_compare.csv`
- `build-docker-smoke/dbas_log_mppi_static_compare_summary.md`
- `build-docker-smoke/dbas_log_mppi_tuned_compare.csv`
- `build-docker-smoke/dbas_log_mppi_tuned_compare_summary.md`
- `build-docker-smoke/dbas_log_mppi_dynamic_compare.csv`
- `build-docker-smoke/dbas_log_mppi_dynamic_compare_summary.md`

All table values below are means over 3 seeds.

## Positive Result: Narrow Passage

DBaS-style adaptive exploration is useful in the narrow-passage case. The
agile variant keeps 100% success and reduces steps versus Log-MPPI and vanilla
MPPI, while staying in the same sub-millisecond control budget.

| Planner | K | Success | Steps | Final Dist | Avg ms |
|---|---:|---:|---:|---:|---:|
| dbas_log_mppi_agile | 32 | 1.00 | 232.0 | 1.92 | 0.11 |
| dbas_log_mppi_smooth | 32 | 1.00 | 234.3 | 1.94 | 0.11 |
| log_mppi | 32 | 1.00 | 247.7 | 1.88 | 0.11 |
| mppi | 32 | 1.00 | 256.0 | 1.94 | 0.09 |
| step_mppi_smooth | 32 | 1.00 | 230.0 | 1.96 | 0.11 |
| tsallis_mppi_smooth | 32 | 1.00 | 229.0 | 1.85 | 0.09 |
| dbas_log_mppi_agile | 128 | 1.00 | 231.3 | 1.90 | 0.12 |
| dbas_log_mppi_smooth | 128 | 1.00 | 234.3 | 1.91 | 0.12 |
| log_mppi | 128 | 1.00 | 248.0 | 1.87 | 0.12 |
| mppi | 128 | 1.00 | 252.7 | 1.86 | 0.10 |

## Positive Result: Crossing Scenes

On open dynamic crossing and uncertain crossing, DBaS-Log-MPPI recovers the
success rate that Log-MPPI misses. The agile variant is also faster in steps
than the smoother DBaS setting.

| Scenario | K | Planner | Success | Steps | Final Dist |
|---|---:|---|---:|---:|---:|
| dynamic_crossing | 64 | dbas_log_mppi_agile | 1.00 | 252.0 | 1.91 |
| dynamic_crossing | 64 | dbas_log_mppi_smooth | 1.00 | 256.7 | 1.87 |
| dynamic_crossing | 64 | log_mppi | 0.33 | 260.0 | 2.39 |
| dynamic_crossing | 128 | dbas_log_mppi_agile | 1.00 | 252.3 | 1.93 |
| dynamic_crossing | 128 | log_mppi | 0.33 | 260.0 | 2.30 |
| uncertain_crossing | 64 | dbas_log_mppi_agile | 1.00 | 252.0 | 1.91 |
| uncertain_crossing | 64 | dbas_log_mppi_smooth | 1.00 | 256.0 | 1.91 |
| uncertain_crossing | 64 | log_mppi | 0.00 | 260.0 | 2.50 |
| uncertain_crossing | 128 | dbas_log_mppi_agile | 1.00 | 251.7 | 1.92 |
| uncertain_crossing | 128 | log_mppi | 0.00 | 260.0 | 2.33 |

Aggregated over `dynamic_crossing`, `dynamic_pincer`, and
`uncertain_crossing`, `dbas_log_mppi_agile` reaches 0.67 success, improving on
Log-MPPI's 0.11 success. It remains behind DRA, Step, SC, and Tsallis in this
benchmark family, which all solve the three dynamic scenarios.

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| dbas_log_mppi_agile | 64 | 0.67 | 2.40 | 0.15 |
| dbas_log_mppi_agile | 128 | 0.67 | 2.15 | 0.14 |
| dbas_log_mppi_smooth | 128 | 0.67 | 11.86 | 0.14 |
| log_mppi | 128 | 0.11 | 4.16 | 0.14 |
| dra_mppi_soft | 128 | 1.00 | 1.92 | 0.24 |
| step_mppi_smooth | 128 | 1.00 | 1.90 | 0.14 |
| tsallis_mppi_smooth | 128 | 1.00 | 1.93 | 0.14 |

## Negative Result: Pincer And Static Clutter

The DBaS variants do not solve dynamic pincer in this quick run. The agile
setting gets close to the goal but misses the success threshold; the smoother
DBaS setting is too conservative and stalls far away.

| Scenario | K | Planner | Success | Final Dist | Avg ms |
|---|---:|---|---:|---:|---:|
| dynamic_pincer | 64 | dbas_log_mppi_agile | 0.00 | 3.39 | 0.14 |
| dynamic_pincer | 128 | dbas_log_mppi_agile | 0.00 | 2.59 | 0.15 |
| dynamic_pincer | 128 | dbas_log_mppi_smooth | 0.00 | 31.83 | 0.16 |
| dynamic_pincer | 128 | log_mppi | 0.00 | 7.84 | 0.15 |
| dynamic_pincer | 128 | dra_mppi_soft | 1.00 | 1.92 | 0.34 |
| dynamic_pincer | 128 | step_mppi_smooth | 1.00 | 1.88 | 0.14 |

Across `narrow_passage`, `static_s_corridor`, `static_u_trap`, and
`cluttered`, DBaS is not the best static-obstacle method. The narrow-passage
gain is real, but cluttered and U-trap remain unsolved and the S-corridor is
better handled by Step/SC variants.

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| dbas_log_mppi | 128 | 0.25 | 16.80 | 0.13 |
| dbas_log_mppi_agile | 128 | 0.25 | 17.30 | 0.13 |
| dbas_log_mppi_smooth | 128 | 0.25 | 17.98 | 0.13 |
| log_mppi | 128 | 0.25 | 16.78 | 0.13 |
| mppi | 128 | 0.25 | 16.52 | 0.11 |
| sc_mppi_smooth | 128 | 0.42 | 14.94 | 0.14 |
| step_mppi_smooth | 128 | 0.50 | 13.97 | 0.12 |
| tsallis_mppi_smooth | 128 | 0.25 | 12.45 | 0.12 |

## Takeaways

- The normal-lognormal sampler plus DBaS-style adaptive exploration is worth
  keeping as a reproducible MPPI hook. It improves Log-MPPI on narrow passage
  and open crossing scenarios.
- The barrier weight is highly sensitive. Strict settings become conservative
  and can fail even easy passage scenarios.
- The current surrogate does not reproduce the paper's strongest cluttered
  results. To push this further, the next implementation should carry an
  explicit augmented DBaS state in the rollout dynamics, use a paper-like
  inverse/log barrier instead of the clipped surrogate, and tune against a
  scenario closer to the paper's reference-tracking missions.
- For this repository's existing dynamic pincer and static corridor tasks,
  DRA/Step/SC/Tsallis variants remain stronger baselines than this lightweight
  DBaS-Log-MPPI reproduction.
