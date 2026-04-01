# Diff-MPPI Novelty Follow-Up

Date: 2026-04-02

This note records nine follow-up experiments added after the initial `diff_mppi` results draft:
- a dynamic-obstacle benchmark suite with two scenarios
- an equal-time target comparison in addition to the earlier cap-based wall-clock analysis
- an exact matched-time tuning workflow that searches `K` directly for shared wall-clock targets
- a strengthened feedback-oriented MPPI baseline
- a covariance-regression feedback baseline that is closer to rollout-feedback control than the nominal-linearization baseline
- a fused covariance-plus-linearization feedback baseline that tightens the non-hybrid comparison again
- a lower-rate-replan high-frequency feedback-execution baseline that is closer to the surrounding controller architecture
- a release-style current-action sensitivity baseline that is closer to the public `Feedback-MPPI` gain computation
- a gradient-only ablation to separate local-refinement effects from the hybrid controller

A later uncertainty follow-up with nominal-vs-actual obstacle mismatch is recorded separately in `paper/diff_mppi_uncertainty_followup.md`.

Artifacts used:
- `build/benchmark_diff_mppi_feedback_dynamic_pair.csv`
- `build/benchmark_diff_mppi_feedback_dynamic_pair_summary.md`
- `build/benchmark_diff_mppi_feedback_dynamic_pair_summary.tex`
- `build/benchmark_diff_mppi_cov_dynamic_pair.csv`
- `build/benchmark_diff_mppi_cov_dynamic_pair_summary.md`
- `build/benchmark_diff_mppi_cov_dynamic_pair_summary.tex`
- `build/benchmark_diff_mppi_fused_dynamic_pair.csv`
- `build/benchmark_diff_mppi_fused_dynamic_pair_summary.md`
- `build/benchmark_diff_mppi_fused_dynamic_pair_summary.tex`
- `build/benchmark_diff_mppi_gap_followup.csv`
- `build/benchmark_diff_mppi_gap_followup_summary.md`
- `build/benchmark_diff_mppi_gap_followup_summary.tex`
- `build/benchmark_diff_mppi_ref_gap_followup.csv`
- `build/benchmark_diff_mppi_ref_gap_followup_summary.md`
- `build/benchmark_diff_mppi_ref_gap_followup_summary.tex`
- `build/benchmark_diff_mppi_exact_time_ref.csv`
- `build/benchmark_diff_mppi_exact_time_ref_search.csv`
- `build/benchmark_diff_mppi_exact_time_ref_summary.md`
- `build/benchmark_diff_mppi_exact_time_hf.csv`
- `build/benchmark_diff_mppi_exact_time_hf_search.csv`
- `build/benchmark_diff_mppi_exact_time_hf_summary.md`
- `build/benchmark_diff_mppi_exact_time_fused.csv`
- `build/benchmark_diff_mppi_exact_time_fused_search.csv`
- `build/benchmark_diff_mppi_exact_time_fused_summary.md`
- `build/benchmark_diff_mppi_exact_time_cov.csv`
- `build/benchmark_diff_mppi_exact_time_cov_search.csv`
- `build/benchmark_diff_mppi_exact_time_cov_summary.md`
- `build/benchmark_diff_mppi_exact_time.csv`
- `build/benchmark_diff_mppi_exact_time_search.csv`
- `build/benchmark_diff_mppi_exact_time_summary.md`
- `build/benchmark_diff_mppi_ablation.csv`
- `build/benchmark_diff_mppi_ablation_summary.md`
- `build/benchmark_diff_mppi_ablation_summary.tex`
- `build/plots_feedback_dynamic_pair/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_feedback_dynamic_pair/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_feedback_dynamic_pair/diff_mppi_final_distance_vs_budget.png`
- `build/plots_feedback_cov_dynamic_pair/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_feedback_cov_dynamic_pair/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_feedback_cov_dynamic_pair/diff_mppi_final_distance_vs_budget.png`
- `build/plots_feedback_fused_dynamic_pair/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_feedback_fused_dynamic_pair/diff_mppi_final_distance_vs_budget.png`
- `build/plots_gap_followup/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_gap_followup/diff_mppi_final_distance_vs_budget.png`
- `build/plots_ref_gap_followup/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots_ref_gap_followup/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_ref_gap_followup/diff_mppi_final_distance_vs_budget.png`
- `build/plots_exact_time_ref/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_exact_time_hf/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_exact_time_fused/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_exact_time_cov/diff_mppi_final_distance_vs_equal_time.png`
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

### 6. Covariance-regression feedback baseline

The benchmark now also includes `feedback_mppi_cov`, which is intended to tighten the direct-baseline story again without claiming a full paper reproduction.

This variant uses:
- two open-loop MPPI update passes to stabilize a nominal sequence
- a fresh rollout bundle around that updated nominal sequence
- a time-varying feedback gain obtained by regressing sampled control deviations on sampled state deviations at each horizon index
- the same closed-loop rollout pass as the other feedback baselines, with residual longitudinal, lateral, heading, and speed tracking terms

This is still not literature-faithful.
But it is a closer local-feedback controller than the earlier nominal-linearization baseline because the gain is estimated from sampled state-control covariance instead of being derived only from a single nominal Jacobian.

### 7. Fused feedback baseline

The benchmark now also includes `feedback_mppi_fused`, which tries to close the remaining in-repo baseline gap without claiming a paper-faithful reproduction.

This variant uses:
- the same stabilized nominal sequence and covariance-regression gain used by `feedback_mppi_cov`
- an auxiliary nominal-linearization / Riccati-style gain computed around that same nominal rollout
- a blended time-varying gain that combines the covariance and local-linearization views
- two closed-loop feedback passes, each followed by the same weighted MPPI update used by the other baselines

This is still not literature-faithful.
But it is the strongest current non-hybrid controller in the repo because it combines sampled state-control covariance with nominal local dynamics instead of choosing only one of those feedback constructions.

### 8. High-frequency feedback-execution baseline

The benchmark now also includes `feedback_mppi_hf`, which is meant to narrow the controller-architecture gap to recent `Feedback-MPPI` work.

This variant uses:
- the same MPPI warm-start and local gain computation machinery as the in-repo feedback baselines
- a lower-rate replan schedule, so the expensive MPPI solve is not repeated every benchmark step
- direct local feedback execution between replans instead of a closed-loop rollout bundle followed by another weighted MPPI update
- the same interpolated local setpoint idea already used inside the feedback rollout kernel

This is still not literature-faithful.
But it is a closer architecture proxy than the other in-repo baselines because it explicitly separates lower-rate replanning from per-step local feedback execution.

### 9. Release-style current-action baseline

The benchmark now also includes `feedback_mppi_ref`, which is meant to narrow the remaining gap to the released `Feedback-MPPI` gain computation rather than to the full controller stack.

This variant uses:
- one open-loop MPPI update pass to obtain sampled rollouts and weights
- rollout initial-state sensitivities `dJ / dx_0`
- the same current-action covariance form used in the released `MPPIGain.gains_computation`, namely a gain built from weighted centered sensitivities and centered first-step sampled controls
- direct local action correction around the updated first nominal action, without a full horizon feedback rollout pass

This is still not literature-faithful.
But it is materially closer to the public release than the earlier in-repo feedback proxies because the gain is computed from the same current-action weighted covariance structure rather than from a nominal Riccati pass or a full-horizon local tracker.

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

The newer rollout-sensitivity baseline sharpened that interpretation, and the newer covariance-regression baseline narrows it further.

At fixed rollout budgets on the dynamic pair benchmark:
- `dynamic_crossing`, `feedback_mppi_sens K=256`: success `0.75`, final distance `2.01`
- `dynamic_crossing`, `feedback_mppi_sens K=512`: success `0.75`, final distance `1.95`
- `dynamic_slalom`, `feedback_mppi_sens K=256`: success `0.00`, final distance `12.83`
- `dynamic_slalom`, `feedback_mppi_sens K=512`: success `0.00`, final distance `12.76`

So the closer sensitivity-aware baseline still improves markedly over vanilla MPPI, especially on `dynamic_crossing`, but it does not explain away the hybrid result.
It remains weaker than Diff-MPPI on both dynamic tasks, and on `dynamic_slalom` it is still weaker than the simpler nominal-linearization `feedback_mppi` baseline.

The new covariance-regression baseline is more useful on the hard task.

At fixed rollout budgets:
- `dynamic_crossing`, `feedback_mppi_cov K=256`: success `1.00`, final distance `1.90`
- `dynamic_crossing`, `feedback_mppi_cov K=1024`: success `1.00`, final distance `1.87`
- `dynamic_slalom`, `feedback_mppi_cov K=256`: success `0.00`, final distance `11.49`
- `dynamic_slalom`, `feedback_mppi_cov K=1024`: success `0.00`, final distance `11.44`

So `feedback_mppi_cov` still does not solve `dynamic_slalom`, but it materially narrows the gap:
- on `dynamic_crossing`, it reaches `1.00` success even at `K=256`, where `feedback_mppi_sens` is still at `0.75`
- on `dynamic_slalom`, it improves on both earlier feedback baselines, lowering terminal distance from `11.91` for `feedback_mppi` and `12.81` for `feedback_mppi_sens` to `11.49`

The newer fused baseline tightens the story again.

At fixed rollout budgets:
- `dynamic_crossing`, `feedback_mppi_fused K=256`: success `1.00`, final distance `1.86`
- `dynamic_crossing`, `feedback_mppi_fused K=512`: success `1.00`, final distance `1.91`
- `dynamic_slalom`, `feedback_mppi_fused K=256`: success `0.00`, final distance `10.28`
- `dynamic_slalom`, `feedback_mppi_fused K=512`: success `0.00`, final distance `10.26`

So `feedback_mppi_fused` still does not solve `dynamic_slalom`, but it is now the strongest non-hybrid feedback baseline in the benchmark:
- on `dynamic_crossing`, it slightly improves over `feedback_mppi_cov` while keeping `1.00` success
- on `dynamic_slalom`, it lowers terminal distance from `11.51` for `feedback_mppi_cov`, `11.91` for `feedback_mppi`, and `12.81` for `feedback_mppi_sens` to `10.28`

That is still not enough to explain away the hybrid result, but it reduces the risk that the repository is only beating a weak in-house baseline.

The newer `feedback_mppi_hf` baseline addresses a different gap.

At fixed rollout budgets:
- `dynamic_crossing`, `feedback_mppi_hf K=256`: success `0.00`, final distance `2.83`
- `dynamic_crossing`, `feedback_mppi_hf K=512`: success `0.00`, final distance `2.80`
- `dynamic_slalom`, `feedback_mppi_hf K=256`: success `0.00`, final distance `13.62`
- `dynamic_slalom`, `feedback_mppi_hf K=512`: success `0.00`, final distance `13.56`

So the closer high-frequency architecture baseline does not solve either task, but it does improve over vanilla MPPI on both:
- on `dynamic_crossing`, it lowers terminal distance from `3.07` to `2.83`
- on `dynamic_slalom`, it lowers terminal distance from `14.36` to `13.62`

That is weaker than the stronger non-hybrid feedback baselines above, but it matters because the improvement now survives even when the controller is forced into a lower-rate-replan, local-feedback-execution regime.

The newer `feedback_mppi_ref` baseline addresses the released-gain gap directly.

At fixed rollout budgets:
- `dynamic_crossing`, `feedback_mppi_ref K=256`: success `1.00`, final distance `1.91`
- `dynamic_crossing`, `feedback_mppi_ref K=512`: success `1.00`, final distance `1.87`
- `dynamic_slalom`, `feedback_mppi_ref K=256`: success `0.00`, final distance `11.89`
- `dynamic_slalom`, `feedback_mppi_ref K=512`: success `0.00`, final distance `12.08`

So the release-style current-action baseline is immediately stronger than vanilla MPPI and the lower-rate `feedback_mppi_hf` proxy:
- on `dynamic_crossing`, it reaches `1.00` success at both `K=256` and `K=512`, where `mppi` remains at `0.00` and `feedback_mppi_hf` remains unsuccessful
- on `dynamic_slalom`, it lowers terminal distance from `14.33` for `mppi` and `13.62` for `feedback_mppi_hf` to `11.89`

It still does not solve `dynamic_slalom`, and it is still weaker than the heavier `feedback_mppi_fused` baseline on the hard task.
But it matters because the repository now contains one baseline that is closer to the released `Feedback-MPPI` gain computation itself, not only to the surrounding controller architecture.

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

For the heavier covariance-regression controllers, the current repo still only has cap-based and nearest-time spot checks rather than a full refreshed exact-time sweep.
The newer architecture-gap and heavy-feedback exact-time runs are still informative:
- under a `2.00 ms` cap, the best lighter feedback row on `dynamic_slalom` is `feedback_mppi_cov K=256 @ 1.66 ms`, final distance `11.49`
- under a `1.00 ms` cap, the closer architecture row on `dynamic_slalom` is `feedback_mppi_hf K=256 @ 0.87 ms`, final distance `13.62`
- under a `1.00 ms` cap, the closer released-gain row on `dynamic_crossing` is `feedback_mppi_ref K=512 @ 0.65 ms`, success `1.00`, final distance `1.87`
- under a `1.00 ms` cap, the closer released-gain row on `dynamic_slalom` is `feedback_mppi_ref K=256 @ 0.63 ms`, final distance `11.89`
- under a `3.50 ms` cap, the best current feedback row on `dynamic_slalom` is `feedback_mppi_fused K=256 @ 3.45 ms`, final distance `10.28`
- under a `1.50 ms` equal-time target on the current fixed-`K` sweep, `feedback_mppi_cov K=256 @ 1.66 ms` is the closest lighter feedback row on `dynamic_slalom`, again with final distance `11.49`
- under exact-time tuning, `feedback_mppi_cov` now reaches `dynamic_crossing: K=219 @ 1.474 ms, dist=1.92` and `K=292 @ 1.964 ms, dist=1.91`
- under exact-time tuning, `feedback_mppi_cov` now reaches `dynamic_slalom: K=211 @ 1.490 ms, dist=11.72` and `K=293 @ 1.971 ms, dist=11.68`
- under exact-time tuning, `feedback_mppi_hf` now reaches `dynamic_crossing: K=285 @ 0.978 ms, dist=2.77`, `K=368 @ 1.486 ms, dist=2.75`, and `K=443 @ 1.989 ms, dist=2.65`
- under exact-time tuning, `feedback_mppi_hf` also reaches `dynamic_slalom: K=276 @ 0.989 ms, dist=13.63`, `K=369 @ 1.498 ms, dist=13.34`, and `K=441 @ 1.980 ms, dist=13.40`
- under exact-time tuning, `feedback_mppi_fused` reaches `dynamic_crossing: K=153 @ 1.968 ms, success=1.00, dist=1.94`
- under exact-time tuning, `feedback_mppi_fused` reaches `dynamic_slalom: K=137 @ 1.993 ms, dist=10.51`

So the newer feedback family is materially stronger than the earlier baselines on the hard task, even though the hybrid controller still remains the only successful family.
The current release-style row is especially useful for reviewer defense because it shows that the main qualitative claim survives even after moving one step closer to the public `Feedback-MPPI` gain computation.
The new covariance, architecture-gap, and heavy-feedback exact-time rows matter for the same reason: they remove the easy objection that those stronger in-repo baselines were only shown under fixed budgets or loose wall-clock caps.

It now also survives a targeted exact-time tuning pass.

At an exact target of `1.00 ms`, the tuned release-style rows are:
- `dynamic_crossing`, `feedback_mppi_ref`: `K=1263 @ 1.002 ms`, success `1.00`, final distance `1.95`
- `dynamic_slalom`, `feedback_mppi_ref`: `K=1150 @ 1.023 ms`, success `0.00`, final distance `11.89`

At `1.50 ms`, the same pattern remains:
- `dynamic_crossing`, `feedback_mppi_ref`: `K=2362 @ 1.482 ms`, success `1.00`, final distance `1.89`
- `dynamic_slalom`, `feedback_mppi_ref`: `K=2190 @ 1.472 ms`, success `0.00`, final distance `11.89`

So the release-style baseline no longer only has fixed-budget and cap-based evidence.
It remains competitive after direct time matching, although it does not displace the stronger tuned `feedback_mppi` row on the hard task.

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
- a covariance-regression baseline now also exists inside the same harness
- a fused covariance-plus-linearization baseline now also exists inside the same harness
- a lower-rate-replan high-frequency feedback-execution baseline now also exists inside the same harness
- a closer rollout-sensitivity feedback baseline now also exists inside the same harness
- all four stronger non-hybrid feedback baselines help on the easier dynamic crossing case, but none solves dynamic slalom
- the fused baseline is now the strongest non-hybrid feedback controller on `dynamic_slalom`
- the high-frequency feedback-execution baseline narrows the controller-architecture gap even though its final performance remains modest
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

The main remaining gaps are now:
- the current `feedback_mppi`, `feedback_mppi_ref`, `feedback_mppi_sens`, `feedback_mppi_cov`, and `feedback_mppi_fused` comparisons are all materially stronger than the earlier fixed-gain tracker, and all but the original `feedback_mppi_sens` now also have targeted exact-time evidence, but they still are not a full literature-faithful rollout-differentiation / feedback-MPPI baseline
- the newer `feedback_mppi_hf` comparison narrows the controller-architecture gap, while `feedback_mppi_ref` narrows the released-gain gap and `feedback_mppi_cov` narrows the covariance-gain gap, but they are still in-repo proxies rather than paper-faithful reproductions
- only two simple hand-designed 2D dynamic scenarios so far

The gradient-only ablation, the six stronger feedback baselines, and the new trace-based mechanism analysis remove weaker alternative explanations, but they still do not close the stronger literature-baseline gap.

## Next Step

If we want to keep pushing the novelty argument, the next experiment should be:

1. Strengthen the current `feedback_mppi_ref` / `feedback_mppi_sens` / `feedback_mppi_cov` / `feedback_mppi_fused` / `feedback_mppi_hf` comparisons into a more literature-faithful baseline.
2. Keep the heavier `feedback_mppi_fused` line as a narrowing study unless it also gets a broader exact-time sweep across more than one target.
3. Extend the newer `feedback_mppi_hf` line only if a more literature-faithful controller reproduction is not ready first.
4. Add a harder dynamic scenario with interacting moving agents, not just one scripted obstacle.
5. Keep exact-time tuned success and final-distance comparisons at one or two fixed targets in the main paper, instead of many configurations.

A submission-oriented gap analysis for `ICRA/IROS` is recorded separately in `paper/icra_iros_gap_list.md`.
