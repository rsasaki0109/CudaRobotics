# Diff-MPPI Novelty Follow-Up

Date: 2026-04-02

This note records five follow-up experiments added after the initial `diff_mppi` results draft:
- a dynamic-obstacle benchmark suite with two scenarios
- an equal-time target comparison in addition to the earlier cap-based wall-clock analysis
- an exact matched-time tuning workflow that searches `K` directly for shared wall-clock targets
- a strengthened feedback-oriented MPPI baseline
- a gradient-only ablation to separate local-refinement effects from the hybrid controller

A later uncertainty follow-up with nominal-vs-actual obstacle mismatch is recorded separately in `paper/diff_mppi_uncertainty_followup.md`.

Artifacts used:
- `build/benchmark_diff_mppi_feedback_dynamic_pair.csv`
- `build/benchmark_diff_mppi_feedback_dynamic_pair_summary.md`
- `build/benchmark_diff_mppi_feedback_dynamic_pair_summary.tex`
- `build/benchmark_diff_mppi_exact_time.csv`
- `build/benchmark_diff_mppi_exact_time_search.csv`
- `build/benchmark_diff_mppi_exact_time_summary.md`
- `build/benchmark_diff_mppi_ablation.csv`
- `build/benchmark_diff_mppi_ablation_summary.md`
- `build/benchmark_diff_mppi_ablation_summary.tex`
- `build/plots_feedback_dynamic_pair/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_feedback_dynamic_pair/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_feedback_dynamic_pair/diff_mppi_final_distance_vs_budget.png`
- `build/plots_ablation/diff_mppi_final_distance_vs_budget.png`
- `build/plots_ablation/diff_mppi_final_distance_vs_equal_time.png`

## What Changed

### 1. Dynamic obstacle suite

The benchmark now includes two dynamic scenarios:
- `dynamic_crossing`: a corridor-like layout defined by four static obstacles, plus one obstacle that sweeps across the robot's path
- `dynamic_slalom`: a static slalom layout plus one descending obstacle that intersects the nominal weaving path
- time-aware collision costs inside both rollout evaluation and gradient refinement

These are still 2D kinematic studies, but together they cover two different interaction patterns: timed crossing and dynamic perturbation of a nontrivial geometric route.

### 2. Equal-time and exact-time selection

The analysis scripts now support two wall-clock views:
- cap-based selection: best configuration whose average control time stays below a budget
- equal-time targets: closest configuration to a target time, planner by planner
- exact-time tuning: a direct search over integer `K` values to hit the same target time without relying on a pre-sampled rollout grid

The equal-time target view is stricter than the earlier cap-only result because it removes the advantage of simply using much less time than the opponent. The exact-time tuning workflow is stricter again because it shrinks the remaining timing gap to the target instead of inheriting the fixed `K` sweep.

### 3. Gradient-only ablation

The benchmark now also includes `grad_only_3`, which disables the MPPI sampling update and keeps only three local gradient refinement steps.

This is not a substitute for a feedback-MPPI or rollout-differentiation baseline, but it is still useful because it asks a narrower question:

> Are the gains coming from the hybrid sampling-plus-gradient structure, or would a pure local gradient controller already explain them?

### 4. Strengthened feedback-oriented baseline

The benchmark now also includes `feedback_mppi`, a strengthened in-repo feedback baseline.

This baseline is not a full reproduction of recent `Feedback-MPPI` literature. Concretely, it uses:
- two open-loop MPPI update passes to seed a nominal control sequence
- a nominal-trajectory linearization and a short Riccati-style backward pass that produces time-varying feedback gains
- a closed-loop rollout pass with those gains plus residual local longitudinal, lateral, heading, and speed tracking terms
- the same weighted MPPI update machinery afterward

So this closes part of the baseline gap, but not all of it.

### 5. Closer rollout-sensitivity baseline

The benchmark now also includes `feedback_mppi_sens`, which is meant to be closer to recent sensitivity-aware MPPI papers than the Riccati-style in-repo baseline above.

This variant uses:
- one open-loop MPPI update pass to obtain sampled rollouts and weights
- a backward pass through each sampled rollout to estimate `dJ / dx_0`
- a feedback gain built from the weighted covariance between sampled controls and rollout initial-state sensitivities
- a closed-loop rollout pass around an interpolated nominal-state setpoint

This is closer to the surrounding literature because the feedback gain is derived from rollout sensitivities rather than only from a nominal local linearization.

It is still not a full literature-faithful reproduction.
In particular, the repository baseline still omits the higher-frequency controller architecture and local setpoint schedule used in recent `Feedback-MPPI` papers.

## Main Result

The dynamic-obstacle suite with the added feedback baseline is now the strongest novelty-supporting result in the repository so far.

At fixed sample budgets, vanilla MPPI failed both dynamic scenarios for every tested `K`, while Diff-MPPI solved them consistently:
- `mppi K=1024`: success `0.00`, final distance `3.13`
- `diff_mppi_1 K=1024`: success `1.00`, final distance `1.95`
- `diff_mppi_3 K=1024`: success `1.00`, final distance `1.91`

For `dynamic_slalom` at the same budget:
- `mppi K=1024`: success `0.00`, final distance `14.19`
- `diff_mppi_1 K=1024`: success `1.00`, final distance `1.92`
- `diff_mppi_3 K=1024`: success `1.00`, final distance `1.90`

The stronger feedback baseline is informative because it splits the two dynamic tasks.

At `K=1024`:
- `dynamic_crossing`, `feedback_mppi`: success `1.00`, final distance `1.88`
- `dynamic_slalom`, `feedback_mppi`: success `0.00`, final distance `11.82`

So a stronger closed-loop MPPI baseline does recover the easier crossing task, but it still does not solve the harder dynamic slalom task.

The newer rollout-sensitivity baseline sharpens that interpretation further.

At fixed rollout budgets on the dynamic pair benchmark:
- `dynamic_crossing`, `feedback_mppi_sens K=256`: success `0.75`, final distance `2.01`
- `dynamic_crossing`, `feedback_mppi_sens K=512`: success `0.75`, final distance `1.95`
- `dynamic_slalom`, `feedback_mppi_sens K=256`: success `0.00`, final distance `12.83`
- `dynamic_slalom`, `feedback_mppi_sens K=512`: success `0.00`, final distance `12.76`

So the closer sensitivity-aware baseline still improves markedly over vanilla MPPI, especially on `dynamic_crossing`, but it does not explain away the hybrid result.
It remains weaker than Diff-MPPI on both dynamic tasks, and on `dynamic_slalom` it is still weaker than the simpler nominal-linearization `feedback_mppi` baseline.

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

The feedback baseline sits between them:
- `dynamic_crossing` at `1.00 ms`: `feedback_mppi K=2048 @ 0.85 ms`, success `1.00`, final distance `1.84`
- `dynamic_slalom` at `1.00 ms`: `feedback_mppi K=2048 @ 0.84 ms`, success `0.00`, final distance `11.88`

The new exact-time tuning workflow sharpens that comparison further.

At an exact target of `1.00 ms`, the tuned configurations are:
- `dynamic_crossing`, `mppi`: `K=7827 @ 0.988 ms`, success `0.00`, final distance `2.98`
- `dynamic_crossing`, `feedback_mppi`: `K=2154 @ 1.008 ms`, success `1.00`, final distance `1.91`
- `dynamic_crossing`, best Diff-MPPI: `diff_mppi_3 K=1447 @ 0.990 ms`, success `1.00`, final distance `1.85`
- `dynamic_slalom`, `mppi`: `K=7790 @ 0.988 ms`, success `0.00`, final distance `14.21`
- `dynamic_slalom`, `feedback_mppi`: `K=2123 @ 0.987 ms`, success `0.00`, final distance `11.91`
- `dynamic_slalom`, best Diff-MPPI: `diff_mppi_3 K=453 @ 1.009 ms`, success `1.00`, final distance `1.92`

At `1.50 ms`, the same ordering remains:
- `dynamic_crossing`, `mppi`: `K=11844 @ 1.484 ms`, final distance `3.05`
- `dynamic_crossing`, `feedback_mppi`: `K=3546 @ 1.481 ms`, final distance `1.85`
- `dynamic_crossing`, best Diff-MPPI: `diff_mppi_3 K=5435 @ 1.465 ms`, final distance `1.88`
- `dynamic_slalom`, `mppi`: `K=11775 @ 1.473 ms`, final distance `14.17`
- `dynamic_slalom`, `feedback_mppi`: `K=3517 @ 1.483 ms`, final distance `11.78`
- `dynamic_slalom`, best Diff-MPPI: `diff_mppi_3 K=4176 @ 1.479 ms`, final distance `1.93`

So the dynamic obstacle suite gives a stronger claim than the earlier static-scene benchmark:

> Across two distinct time-varying obstacle scenarios, the lightweight MPPI + autodiff refinement controller reaches successful trajectories where vanilla MPPI remains unsuccessful, even under matched per-step compute budgets.

The more precise current version is:

> Across two distinct time-varying obstacle scenarios, a strengthened feedback-oriented MPPI baseline closes part of the gap to the hybrid controller, but only the hybrid MPPI + autodiff refinement controller remains successful on both tasks under cap-based, equal-time, and exact-time matched per-step compute budgets.

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

## Mechanism Analysis

The remaining reviewer-style question is whether the hybrid controller is doing something structurally different, or whether it is just another way to spend more compute.

To make that more explicit, the benchmark now supports a trace mode:
- it records the sampled nominal controls before refinement
- it records the final controls after refinement
- it records the local control gradients used by the refinement stage
- it writes per-episode-step and per-horizon-step traces to CSV

The current mechanism example uses `dynamic_slalom` at `K=1024`, which is the more diagnostic of the two dynamic tasks because vanilla MPPI and the strengthened `feedback_mppi` baseline both remain unsuccessful there while Diff-MPPI succeeds.

The resulting summary is:
- `mppi`: early-horizon correction `0.000`, late-horizon correction `0.000`
- `feedback_mppi`: early-horizon correction `0.000`, late-horizon correction `0.000`
- `diff_mppi_1`: early-horizon correction `0.018`, late-horizon correction `0.001`, mean gradient norm `3.589`, peak first-action correction `0.032`
- `diff_mppi_3`: early-horizon correction `0.025`, late-horizon correction `0.001`, mean gradient norm `2.699`, peak first-action correction `0.047`

The main qualitative signal is that the correction is strongly front-loaded in the horizon. The early-horizon correction is about `18x` larger than the late-horizon correction for `diff_mppi_1`, and about `25x` larger for `diff_mppi_3`.

That matters because it supports a narrower and more defensible mechanism claim:

> The autodiff stage does not replace the sampled MPPI plan wholesale. It mostly sharpens the near-term actions that are actually executed, which is exactly the regime where a moving-obstacle timing error is most costly.

This is still a lightweight mechanism analysis, not a full theory section. But it does answer one of the easier reviewer objections: the hybrid controller is not merely "more compute in general"; it is using the extra compute to make concentrated local corrections where the control sequence is most sensitive.

## Why This Helps the Novelty Story

Before this follow-up, the strongest evidence was:
- better fixed-budget quality on static geometric tasks
- a 3-of-4 win pattern under cap-based wall-clock selection

After this follow-up, the story is stronger because:
- the dynamic suite introduces genuinely time-dependent planning challenges
- the same win pattern now appears in both a crossing task and a dynamic slalom task
- a strengthened feedback-oriented baseline now exists inside the same harness
- a closer rollout-sensitivity feedback baseline now also exists inside the same harness
- that baseline helps on the easier dynamic crossing case, but still fails on dynamic slalom
- the advantage remains visible under an equal-time target, not only a loose cap
- the advantage also survives direct exact-time tuning, where planner-specific `K` values are chosen to hit shared timing targets within roughly `0.01-0.03 ms`
- the result is about success, not only smaller terminal distance
- the gradient-only ablation shows that hybridization, not just local gradient descent, is carrying the effect
- the new trace figures show that the refinement stage is front-loaded on the early horizon instead of rewriting the whole control sequence

This is still not a broad novelty claim about all differentiable MPPI methods. But it is a more defensible narrow claim for this implementation direction.

## Updated Claim

The most defensible paper-style claim now looks like this:

> We study a minimal CUDA hybrid controller that augments vanilla MPPI with a short autodiff-based control refinement stage. Across static navigation tasks it improves the quality-vs-compute tradeoff, and in a dynamic-obstacle crossing task it achieves successful trajectories under matched per-step compute budgets where vanilla MPPI does not.

The stronger current version is:

> We study a minimal CUDA hybrid controller that augments vanilla MPPI with a short autodiff-based control refinement stage. Across static navigation tasks it improves the quality-vs-compute tradeoff, and across two dynamic-obstacle tasks it achieves successful trajectories under matched per-step compute budgets where vanilla MPPI remains unsuccessful.

## What Is Still Missing

The two biggest gaps are now:
- the current `feedback_mppi` and `feedback_mppi_sens` comparisons are both materially stronger than the earlier fixed-gain tracker, but still not a full literature-faithful rollout-differentiation / feedback-MPPI baseline
- only two simple hand-designed 2D dynamic scenarios so far

The gradient-only ablation, the two stronger feedback baselines, and the new trace-based mechanism analysis remove weaker alternative explanations, but they still do not close the stronger literature-baseline gap.

## Next Step

If we want to keep pushing the novelty argument, the next experiment should be:

1. Strengthen the current nominal-linearization `feedback_mppi` comparison into a more literature-faithful baseline.
2. Add a harder dynamic scenario with interacting moving agents, not just one scripted obstacle.
3. Keep exact-time tuned success and final-distance comparisons at one or two fixed targets in the main paper, instead of many configurations.

A submission-oriented gap analysis for `ICRA/IROS` is recorded separately in `paper/icra_iros_gap_list.md`.
