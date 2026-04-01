# Diff-MPPI Novelty Follow-Up

Date: 2026-04-02

This note records two follow-up experiments added after the initial `diff_mppi` results draft:
- a dynamic-obstacle benchmark scenario
- an equal-time target comparison in addition to the earlier cap-based wall-clock analysis

Artifacts used:
- `build/benchmark_diff_mppi_dynamic.csv`
- `build/benchmark_diff_mppi_dynamic_summary.md`
- `build/benchmark_diff_mppi_dynamic_summary.tex`
- `build/plots_dynamic/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_dynamic/diff_mppi_final_distance_vs_equal_time.png`

## What Changed

### 1. Dynamic obstacle scenario

The benchmark now includes a `dynamic_crossing` scenario with:
- a corridor-like layout defined by four static obstacles
- a moving obstacle that sweeps across the robot's path
- time-aware collision costs inside both rollout evaluation and gradient refinement

This is still a 2D kinematic study, but it is materially closer to the kind of setting where local sensitivity information should matter.

### 2. Equal-time selection

The analysis scripts now support two wall-clock views:
- cap-based selection: best configuration whose average control time stays below a budget
- equal-time targets: closest configuration to a target time, planner by planner

The equal-time target view is stricter than the earlier cap-only result because it removes the advantage of simply using much less time than the opponent.

## Main Result

The dynamic-obstacle follow-up is the strongest novelty-supporting result in the repository so far.

At fixed sample budgets, vanilla MPPI failed the `dynamic_crossing` scenario for every tested `K`, while both gradient-refined variants solved it consistently:
- `mppi K=1024`: success `0.00`, final distance `3.13`
- `diff_mppi_1 K=1024`: success `1.00`, final distance `1.95`
- `diff_mppi_3 K=1024`: success `1.00`, final distance `1.91`

The same pattern survived wall-clock matching.

Under a `1.00 ms` cap:
- best feasible MPPI: `K=4096`, success `0.00`, final distance `2.94`
- best feasible Diff-MPPI: `diff_mppi_1 K=512`, success `1.00`, final distance `1.85`

Under an equal-time target of `1.00 ms`:
- MPPI matched closest at `K=8192`, `0.98 ms`, success `0.00`, final distance `2.99`
- Diff-MPPI matched closest at `diff_mppi_3 K=2048`, `1.01 ms`, success `1.00`, final distance `1.90`

So the dynamic obstacle follow-up gives a stronger claim than the earlier static-scene benchmark:

> In a time-varying obstacle scenario, the lightweight MPPI + autodiff refinement controller reaches successful trajectories where vanilla MPPI remains unsuccessful, even under matched per-step compute budgets.

## Why This Helps the Novelty Story

Before this follow-up, the strongest evidence was:
- better fixed-budget quality on static geometric tasks
- a 3-of-4 win pattern under cap-based wall-clock selection

After this follow-up, the story is stronger because:
- the dynamic obstacle introduces a genuinely time-dependent planning challenge
- the advantage remains visible under an equal-time target, not only a loose cap
- the result is about success, not only smaller terminal distance

This is still not a broad novelty claim about all differentiable MPPI methods. But it is a more defensible narrow claim for this implementation direction.

## Updated Claim

The most defensible paper-style claim now looks like this:

> We study a minimal CUDA hybrid controller that augments vanilla MPPI with a short autodiff-based control refinement stage. Across static navigation tasks it improves the quality-vs-compute tradeoff, and in a dynamic-obstacle crossing task it achieves successful trajectories under matched per-step compute budgets where vanilla MPPI does not.

## What Is Still Missing

The two biggest gaps are now:
- no direct comparison to a rollout-differentiation / feedback-MPPI style baseline
- only a single dynamic-obstacle scenario so far

## Next Step

If we want to keep pushing the novelty argument, the next experiment should be:

1. Add a second dynamic-obstacle scenario with a different interaction pattern.
2. Add a feedback-oriented baseline beyond vanilla MPPI.
3. Report exact success and final-distance comparisons at one or two fixed equal-time targets only, instead of many configurations.
