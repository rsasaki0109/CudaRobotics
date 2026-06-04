# TD-CD-MPPI Lightweight Reproduction

## Target

- Paper: "TD-CD-MPPI: Temporal-Difference Constraint-Discounted Model
  Predictive Path Integral Control"
- Project page: https://pietronoah.github.io/td-cd-mppi/
- Paper mirror linked by the project page: https://hal.science/hal-05213269v2/document
- Public reference implementation: the project page currently links code as
  "Coming Soon"; no usable public implementation was found in the web/GitHub
  search for this pass.

The paper adds two mechanisms to MPPI:

- A terminal value function learned offline with temporal-difference learning,
  so short online rollouts can retain longer-horizon reasoning.
- Constraint-discount modulation, where constraint violation changes the
  trajectory return propagation instead of relying only on handcrafted penalty
  shaping.

## Implemented Scope

This repository now has a lightweight navigation reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_td_cd_weights` plus TD-CD tuning fields in `PlannerVariant`.
- `compute_td_cd_scores_kernel`, a per-rollout CUDA score kernel that computes:
  1. The usual navigation stage cost along each sampled trajectory.
  2. A survival discount from the minimum static/dynamic obstacle margin.
  3. A failure-mass cost when the survival discount drops.
  4. An analytic terminal value-to-go surrogate toward the goal.
  5. Standard MPPI normalization by reusing `compute_weights_kernel`.
- Planner variants:
  - `td_v_mppi_short`: short-horizon terminal-value variant with no effective
    constraint discount.
  - `td_cd_mppi_soft`: weak constraint-discount ablation.
  - `td_cd_mppi_guarded`: light constraint-discount guard.

The implemented score is:

```text
survival_0 = 1
feasibility_t = sigmoid((margin_t - safe_margin) / sigma)
survival_{t+1} = survival_t * feasibility_t ^ discount_power
score += survival_t * stage_cost_t
score += (survival_t - survival_{t+1}) * failure_cost
score += survival_T * terminal_value_surrogate(x_T)
```

This is not a faithful reproduction of the paper's full system. It does not
train a neural terminal value function, does not run TD learning, does not use a
contact simulator, and does not reproduce the legged-locomotion reward or
constraint manager. The useful part reproduced here is the online control hook:
short rollouts plus terminal value shaping, with constraint-discount scoring as
a cheap CUDA-side ablation.

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
    --planners mppi,lp_mppi_smooth,td_v_mppi_short,td_cd_mppi_soft,td_cd_mppi_guarded,tsallis_mppi_smooth,cc_mppi_smooth,step_mppi_smooth \
    --k-values 32,64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/td_cd_mppi_mild_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/td_cd_mppi_mild_compare.csv \
      --markdown-out build-docker-smoke/td_cd_mppi_mild_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Artifacts:

- `build-docker-smoke/td_cd_mppi_mild_compare.csv`
- `build-docker-smoke/td_cd_mppi_mild_compare_summary.md`
- `build-docker-smoke/td_cd_mppi_mild_compare_summary.tex`

All table values below are means over 3 seeds.

## Positive Result: Short Terminal Value Planner

The main useful result is `td_v_mppi_short`, the terminal-value-only short
horizon variant. It is consistently fast and solves open dynamic scenes with
fewer steps than the smooth MPPI baselines.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms | Collisions |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing | 32 | td_v_mppi_short | 1.00 | 240.0 | 1.93 | 0.11 | 0.00 |
| dynamic_crossing | 64 | td_v_mppi_short | 1.00 | 241.7 | 1.89 | 0.12 | 0.00 |
| dynamic_crossing | 128 | td_v_mppi_short | 1.00 | 239.3 | 1.85 | 0.13 | 0.00 |
| dynamic_pincer | 32 | td_v_mppi_short | 1.00 | 240.3 | 1.85 | 0.11 | 0.00 |
| dynamic_pincer | 64 | td_v_mppi_short | 1.00 | 239.7 | 1.89 | 0.12 | 0.00 |
| dynamic_pincer | 128 | td_v_mppi_short | 1.00 | 240.0 | 1.89 | 0.13 | 0.00 |
| uncertain_crossing | 32 | td_v_mppi_short | 1.00 | 240.0 | 1.84 | 0.13 | 0.00 |
| uncertain_crossing | 64 | td_v_mppi_short | 1.00 | 239.7 | 1.87 | 0.12 | 0.00 |
| uncertain_crossing | 128 | td_v_mppi_short | 1.00 | 240.0 | 1.84 | 0.12 | 0.00 |

Compared with `lp_mppi_smooth` and `cc_mppi_smooth`, this variant reaches the
goal about 10-18 simulation steps earlier in the crossing/pincer scenes while
staying near the same per-control runtime budget.

## Partial Result: Constraint-Discount Guard

`td_cd_mppi_guarded` shows that the constraint-discount mechanism can be useful
when tuned lightly, but it is much more sensitive than the terminal value term.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms | Collisions |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing | 32 | td_cd_mppi_guarded | 1.00 | 251.0 | 1.83 | 0.12 | 0.00 |
| dynamic_crossing | 64 | td_cd_mppi_guarded | 1.00 | 250.7 | 1.91 | 0.13 | 0.00 |
| dynamic_crossing | 128 | td_cd_mppi_guarded | 1.00 | 250.0 | 1.99 | 0.15 | 0.00 |
| dynamic_pincer | 32 | td_cd_mppi_guarded | 0.67 | 253.7 | 1.97 | 0.14 | 0.00 |
| dynamic_pincer | 64 | td_cd_mppi_guarded | 1.00 | 250.7 | 1.91 | 0.12 | 0.00 |
| dynamic_pincer | 128 | td_cd_mppi_guarded | 1.00 | 250.7 | 1.85 | 0.14 | 0.00 |

Heavier guard settings eliminated some uncertain-crossing collisions in a
follow-up sensitivity run, but pushed pincer into a conservative stopping mode.
That tradeoff suggests this reproduction needs a learned value function and
task-calibrated discount target to be robust.

## Negative Result: Bottleneck And Uncertainty

The narrow timing-gate bottleneck is not solved by the TD-CD surrogate.
`td_v_mppi_short` reaches the goal region quickly but often collides because it
does not reason far enough about the timing gate. The constraint-discounted
variants either collide or stop far from the goal.

| Scenario | K | Planner | Success | Steps | Final Dist | Avg ms | Collisions |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_bottleneck | 32 | td_v_mppi_short | 0.33 | 180.3 | 1.89 | 0.12 | 10.00 |
| dynamic_bottleneck | 64 | td_v_mppi_short | 0.33 | 178.7 | 1.82 | 0.12 | 7.33 |
| dynamic_bottleneck | 128 | td_v_mppi_short | 0.00 | 182.0 | 1.85 | 0.13 | 14.33 |
| dynamic_bottleneck | 32 | td_cd_mppi_guarded | 0.00 | 320.0 | 21.58 | 0.13 | 83.00 |
| dynamic_bottleneck | 64 | td_cd_mppi_guarded | 0.00 | 320.0 | 21.41 | 0.14 | 87.67 |
| dynamic_bottleneck | 128 | td_cd_mppi_guarded | 0.00 | 320.0 | 21.59 | 0.14 | 83.00 |

For this benchmark, `tsallis_mppi_smooth` remains the better lightweight fix
for the bottleneck timing gate.

## Takeaways

- Keep `td_v_mppi_short` as a cheap short-horizon terminal-value planner for
  open dynamic crossing and pincer-like scenes.
- Treat `td_cd_mppi_guarded` as a sensitivity/ablation variant, not as the
  default planner.
- The paper's learned terminal value function is likely the important missing
  piece. An analytic goal-distance surrogate improves speed but cannot infer
  task-specific long-horizon timing constraints.
- Constraint discounting is sign- and scale-sensitive in a cost-minimization
  MPPI implementation. Too much discounting creates conservative stopping;
  too little discounting allows near-obstacle shortcuts and collisions.
