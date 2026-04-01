# Diff-MPPI Novelty Follow-Up

Date: 2026-04-02

This note records three follow-up experiments added after the initial `diff_mppi` results draft:
- a dynamic-obstacle benchmark suite with two scenarios
- an equal-time target comparison in addition to the earlier cap-based wall-clock analysis
- a gradient-only ablation to separate local-refinement effects from the hybrid controller

Artifacts used:
- `build/benchmark_diff_mppi_dynamic_pair.csv`
- `build/benchmark_diff_mppi_dynamic_pair_summary.md`
- `build/benchmark_diff_mppi_dynamic_pair_summary.tex`
- `build/benchmark_diff_mppi_ablation.csv`
- `build/benchmark_diff_mppi_ablation_summary.md`
- `build/benchmark_diff_mppi_ablation_summary.tex`
- `build/plots_dynamic_pair/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_dynamic_pair/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_dynamic_pair/diff_mppi_final_distance_vs_budget.png`
- `build/plots_ablation/diff_mppi_final_distance_vs_budget.png`
- `build/plots_ablation/diff_mppi_final_distance_vs_equal_time.png`

## What Changed

### 1. Dynamic obstacle suite

The benchmark now includes two dynamic scenarios:
- `dynamic_crossing`: a corridor-like layout defined by four static obstacles, plus one obstacle that sweeps across the robot's path
- `dynamic_slalom`: a static slalom layout plus one descending obstacle that intersects the nominal weaving path
- time-aware collision costs inside both rollout evaluation and gradient refinement

These are still 2D kinematic studies, but together they cover two different interaction patterns: timed crossing and dynamic perturbation of a nontrivial geometric route.

### 2. Equal-time selection

The analysis scripts now support two wall-clock views:
- cap-based selection: best configuration whose average control time stays below a budget
- equal-time targets: closest configuration to a target time, planner by planner

The equal-time target view is stricter than the earlier cap-only result because it removes the advantage of simply using much less time than the opponent.

### 3. Gradient-only ablation

The benchmark now also includes `grad_only_3`, which disables the MPPI sampling update and keeps only three local gradient refinement steps.

This is not a substitute for a feedback-MPPI or rollout-differentiation baseline, but it is still useful because it asks a narrower question:

> Are the gains coming from the hybrid sampling-plus-gradient structure, or would a pure local gradient controller already explain them?

## Main Result

The dynamic-obstacle suite is now the strongest novelty-supporting result in the repository so far.

At fixed sample budgets, vanilla MPPI failed both dynamic scenarios for every tested `K`, while Diff-MPPI solved them consistently:
- `mppi K=1024`: success `0.00`, final distance `3.13`
- `diff_mppi_1 K=1024`: success `1.00`, final distance `1.95`
- `diff_mppi_3 K=1024`: success `1.00`, final distance `1.91`

For `dynamic_slalom` at the same budget:
- `mppi K=1024`: success `0.00`, final distance `14.19`
- `diff_mppi_1 K=1024`: success `1.00`, final distance `1.92`
- `diff_mppi_3 K=1024`: success `1.00`, final distance `1.90`

The same pattern survived wall-clock matching.

Under a `1.00 ms` cap:
- `dynamic_crossing` best feasible MPPI: `K=4096`, success `0.00`, final distance `2.94`
- `dynamic_crossing` best feasible Diff-MPPI: `diff_mppi_1 K=6144`, success `1.00`, final distance `1.86`
- `dynamic_slalom` best feasible MPPI: `K=4096`, success `0.00`, final distance `14.05`
- `dynamic_slalom` best feasible Diff-MPPI: `diff_mppi_3 K=256`, success `1.00`, final distance `1.84`

Under an equal-time target of `1.00 ms`:
- `dynamic_crossing` MPPI matched closest at `K=8192`, `1.00 ms`, success `0.00`, final distance `2.99`
- `dynamic_crossing` Diff-MPPI matched closest at `diff_mppi_1 K=6144`, `1.00 ms`, success `1.00`, final distance `1.86`
- `dynamic_slalom` MPPI matched closest at `K=8192`, `1.03 ms`, success `0.00`, final distance `14.14`
- `dynamic_slalom` Diff-MPPI matched closest at `diff_mppi_3 K=512`, `0.99 ms`, success `1.00`, final distance `1.91`

So the dynamic obstacle suite gives a stronger claim than the earlier static-scene benchmark:

> Across two distinct time-varying obstacle scenarios, the lightweight MPPI + autodiff refinement controller reaches successful trajectories where vanilla MPPI remains unsuccessful, even under matched per-step compute budgets.

## Hybrid vs Gradient-Only Ablation

The ablation result is also informative.

In `corner_turn`, `grad_only_3` is better than vanilla MPPI, but much weaker than the hybrid controller:
- at `K=1024`, `mppi`: final distance `18.05`
- at `K=1024`, `grad_only_3`: final distance `14.75`
- at `K=1024`, `diff_mppi_3`: final distance `2.52`

So gradients alone do help with local geometric steering, but they do not explain the full gain.

In `dynamic_crossing`, the difference is much sharper:
- at `K=1024`, `mppi`: success `0.00`, final distance `3.03`
- at `K=1024`, `grad_only_3`: success `0.00`, final distance `46.49`
- at `K=1024`, `diff_mppi_3`: success `1.00`, final distance `1.90`

The same pattern remains under matched compute views.

Under a `1.00 ms` cap:
- `dynamic_crossing` MPPI: `K=4096`, success `0.00`, final distance `2.92`
- `dynamic_crossing` grad-only: `K=4096`, success `0.00`, final distance `46.49`
- `dynamic_crossing` Diff-MPPI: `diff_mppi_3 K=512`, success `1.00`, final distance `1.84`

At a `1.00 ms` equal-time target:
- `dynamic_crossing` MPPI: `K=8192 @ 1.03 ms`, success `0.00`, final distance `2.99`
- `dynamic_crossing` grad-only: `K=8192 @ 0.78 ms`, success `0.00`, final distance `46.49`
- `dynamic_crossing` Diff-MPPI: `diff_mppi_3 K=2048 @ 1.04 ms`, success `1.00`, final distance `1.90`

This matters because it narrows the interpretation:

> The observed gain is not just "adding gradients." The useful behavior comes from combining MPPI's global stochastic search with a short local refinement stage.

## Why This Helps the Novelty Story

Before this follow-up, the strongest evidence was:
- better fixed-budget quality on static geometric tasks
- a 3-of-4 win pattern under cap-based wall-clock selection

After this follow-up, the story is stronger because:
- the dynamic suite introduces genuinely time-dependent planning challenges
- the same win pattern now appears in both a crossing task and a dynamic slalom task
- the advantage remains visible under an equal-time target, not only a loose cap
- the result is about success, not only smaller terminal distance
- the gradient-only ablation shows that hybridization, not just local gradient descent, is carrying the effect

This is still not a broad novelty claim about all differentiable MPPI methods. But it is a more defensible narrow claim for this implementation direction.

## Updated Claim

The most defensible paper-style claim now looks like this:

> We study a minimal CUDA hybrid controller that augments vanilla MPPI with a short autodiff-based control refinement stage. Across static navigation tasks it improves the quality-vs-compute tradeoff, and in a dynamic-obstacle crossing task it achieves successful trajectories under matched per-step compute budgets where vanilla MPPI does not.

The stronger current version is:

> We study a minimal CUDA hybrid controller that augments vanilla MPPI with a short autodiff-based control refinement stage. Across static navigation tasks it improves the quality-vs-compute tradeoff, and across two dynamic-obstacle tasks it achieves successful trajectories under matched per-step compute budgets where vanilla MPPI remains unsuccessful.

## What Is Still Missing

The two biggest gaps are now:
- no direct comparison to a rollout-differentiation / feedback-MPPI style baseline
- only two simple hand-designed 2D dynamic scenarios so far

The gradient-only ablation removes one weaker alternative explanation, but it does not close the stronger baseline gap.

## Next Step

If we want to keep pushing the novelty argument, the next experiment should be:

1. Add a feedback-oriented baseline beyond vanilla MPPI.
2. Add a harder dynamic scenario with interacting moving agents, not just one scripted obstacle.
3. Report exact success and final-distance comparisons at one or two fixed equal-time targets only, instead of many configurations.
