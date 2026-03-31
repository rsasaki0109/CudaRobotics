# Diff-MPPI Results Draft

Date: 2026-04-01

This note turns the current benchmark outputs into a paper-style `Results` section draft for the `diff_mppi` line.
It is intentionally narrow. The goal is not to oversell "differentiable MPPI" as a new idea, but to state a defensible empirical claim around a lightweight CUDA hybrid controller.

Artifacts used:
- `build/benchmark_diff_mppi.csv`
- `build/benchmark_diff_mppi_summary.md`
- `build/benchmark_diff_mppi_summary.tex`
- `build/plots/diff_mppi_final_distance_vs_time.png`
- `build/plots/diff_mppi_cost_vs_time.png`
- `build/plots/diff_mppi_final_distance_vs_budget.png`

## Narrow Claim

The current evidence supports this claim:

> A minimal MPPI controller augmented with a short autodiff-based control refinement stage improves fixed-budget trajectory quality on several 2D navigation tasks, at a measurable but explicit wall-clock cost.

The current evidence does not yet support these stronger claims:
- superiority at matched wall-clock budget
- broad novelty over all prior differentiable MPPI / rollout-differentiation methods
- robust goal-reaching on all hard scenarios

## Experimental Protocol

Controller variants:
- `mppi`: vanilla sampling-only MPPI
- `diff_mppi_1`: MPPI plus 1 gradient refinement step
- `diff_mppi_3`: MPPI plus 3 gradient refinement steps

Task suite:
- `cluttered`
- `corner_turn`
- `narrow_passage`
- `slalom`

Common settings:
- sample counts `K in {1024, 2048, 4096}`
- 4 random seeds per configuration
- fixed horizon `T = 30`
- same direct-steering dynamics, cost terms, and obstacle layouts across planners

Reported metrics:
- `success`
- `steps`
- `final_distance`
- `cumulative_cost`
- `avg_control_ms`

## Main Findings

### 1. Gradient refinement improves fixed-budget quality on three of four tasks

At `K = 1024`, the best gradient-refined variant was consistently better than vanilla MPPI in `cluttered`, `corner_turn`, and `slalom`:
- `cluttered`: final distance `38.58 -> 36.29` with `diff_mppi_3` (`5.9%` lower)
- `corner_turn`: final distance `18.06 -> 2.46` with `diff_mppi_3` (`86.4%` lower)
- `slalom`: final distance `19.01 -> 4.30` with `diff_mppi_3` (`77.4%` lower)

These gains are large enough that the result is not just noise around a similar controller. In the turning and slalom tasks, the refinement stage changes the terminal behavior qualitatively, not just marginally.

### 2. In the narrow passage task, refinement mostly helps efficiency rather than final distance

All planners solved `narrow_passage`, so this task is best interpreted through step count and cumulative cost rather than terminal distance alone.

At `K = 1024`:
- `mppi`: `251.0` steps, final distance `1.88`
- `diff_mppi_1`: `238.2` steps, final distance `1.87`
- `diff_mppi_3`: `235.5` steps, final distance `1.90`

This corresponds to about `5.1%` fewer steps for `diff_mppi_1` and `6.2%` fewer steps for `diff_mppi_3`, while maintaining essentially the same terminal distance.

### 3. The 3-step variant is usually the strongest quality setting

Across scenarios at `K = 1024`, aggregate means were:
- `mppi`: final distance `19.38`, cumulative cost `47060.3`, avg control time `0.17 ms`
- `diff_mppi_1`: final distance `11.64`, cumulative cost `41132.9`, avg control time `0.44 ms`
- `diff_mppi_3`: final distance `11.24`, cumulative cost `40446.1`, avg control time `0.99 ms`

Relative to vanilla MPPI at `K = 1024`, `diff_mppi_3` reduced:
- aggregate final distance by `42.0%`
- aggregate cumulative cost by `14.1%`

The tradeoff is runtime:
- `diff_mppi_1`: about `2.6x` the control time of `mppi`
- `diff_mppi_3`: about `5.8x` the control time of `mppi`

### 4. Increasing `K` does not erase the qualitative pattern

The same ordering persists at `K = 2048` and `K = 4096`:
- `diff_mppi_3` remains clearly stronger than `mppi` on `corner_turn` and `slalom`
- `cluttered` stays modestly improved
- `narrow_passage` remains primarily a step-count advantage, not a strong terminal-distance advantage

This matters because it suggests the benefit is not only a low-sample artifact. The refinement stage keeps helping even after the nominal MPPI baseline gets more rollouts.

## Ready-To-Paste Results Section

### Results

Table 1 reports aggregate performance across four navigation scenarios, while Table 2 compares the best gradient-refined variant against vanilla MPPI at a fixed rollout budget. The main pattern is consistent: adding a short autodiff refinement stage after the MPPI sampling update improves trajectory quality on the harder geometric tasks. At `K = 1024`, the `diff_mppi_3` variant reduced mean final distance from `19.38` to `11.24` and mean cumulative cost from `47060.3` to `40446.1`, corresponding to `42.0%` and `14.1%` reductions, respectively. These gains come with a higher per-step control time, increasing from `0.17 ms` for vanilla MPPI to `0.99 ms` for `diff_mppi_3`.

The task-wise breakdown shows that the improvement is not uniform. In `corner_turn`, the refinement stage substantially changed the terminal behavior, reducing final distance from `18.06` to `2.46` at `K = 1024`. A similarly strong effect appeared in `slalom`, where final distance fell from `19.01` to `4.30`. In contrast, the `cluttered` task showed only a modest improvement (`38.58` to `36.29`), suggesting that the local gradient stage is most useful when the nominal MPPI trajectory reaches a geometrically informative region but still needs a sharper local steering correction.

The `narrow_passage` task highlights a different regime. All controllers solved this environment, so terminal distance alone is less informative. Here the main gain from refinement was efficiency: at `K = 1024`, vanilla MPPI required `251.0` steps on average, while `diff_mppi_1` and `diff_mppi_3` required `238.2` and `235.5` steps, respectively. This indicates that the refinement stage can shorten successful trajectories even when it does not materially improve the final goal distance.

Figure 1 and Figure 2 make the quality-vs-compute tradeoff explicit. The gradient-refined controllers occupy a better region of final distance and cumulative cost, but only by paying additional controller latency. Figure 3 shows that the same ranking mostly persists as the rollout budget increases from `1024` to `4096`, which suggests that the refinement stage is complementary to larger sample counts rather than merely compensating for an undersampled MPPI baseline.

Overall, the current evidence supports a narrow systems claim: a minimal CUDA implementation of MPPI plus local autodiff refinement can improve fixed-budget control quality on several navigation tasks. It does not yet establish dominance at matched wall-clock budget, and several hard tasks still fail to reach the goal within the episode horizon, so the present result should be interpreted as a compute-quality tradeoff study rather than a complete replacement for vanilla MPPI.

## Figure / Table Captions

Suggested captions:

- Table 1:
  Aggregate Diff-MPPI benchmark results across four navigation scenarios. Gradient refinement lowers mean terminal distance and cumulative cost, but increases per-step control latency.

- Table 2:
  Best gradient-refined variant at fixed rollout budget. The refinement stage yields the strongest gains in `corner_turn` and `slalom`, while `narrow_passage` primarily benefits through fewer steps.

- Figure 1:
  Final distance versus average control time. Each point denotes one planner and rollout budget. Gradient refinement shifts the controller toward lower terminal error at the cost of additional latency.

- Figure 2:
  Cumulative cost versus average control time. The gradient-refined variants reduce trajectory cost across most scenarios, but do not dominate vanilla MPPI in wall-clock terms.

- Figure 3:
  Final distance versus sample budget. Increasing the rollout count improves all methods, but the ranking between vanilla MPPI and gradient-refined variants remains largely unchanged.

## Limits To State Explicitly

These points should remain in the paper draft unless new experiments remove them:
- The evaluation is still 2D and kinematic.
- The benchmark is fixed-budget, not wall-clock matched.
- `cluttered`, `corner_turn`, and `slalom` still end without full success in the current setup.
- No dynamic-obstacle experiment is included yet.
- No direct comparison to a feedback-MPPI style method is included yet.

## Next Empirical Step

If we want this section to become a stronger paper result, the next experiment should be:

1. Add a wall-clock matched benchmark.
2. Sweep `K` and gradient steps under a fixed time cap.
3. Report the best terminal distance / success achievable under that cap.
4. Keep the current fixed-budget result as the complementary view, not the only one.
