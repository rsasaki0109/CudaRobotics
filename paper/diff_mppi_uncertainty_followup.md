# Diff-MPPI Uncertainty Follow-Up

Date: 2026-04-02

This note records a reviewer-driven uncertainty follow-up for the `Diff-MPPI` line.
The goal is narrow:

> Show that the dynamic-obstacle result does not disappear when the controller plans against nominal obstacle motion but the executed episodes use seed-dependent obstacle mismatch.

Artifacts used:
- `build/benchmark_diff_mppi_uncertain.csv`
- `build/benchmark_diff_mppi_uncertain_summary.md`
- `build/benchmark_diff_mppi_uncertain_summary.tex`
- `build/benchmark_diff_mppi_uncertain_exact_time.csv`
- `build/benchmark_diff_mppi_uncertain_exact_time_search.csv`
- `build/benchmark_diff_mppi_uncertain_exact_time_summary.md`
- `build/plots_uncertain/`

## Benchmark Setup

The uncertainty mechanism is intentionally simple and reviewer-safe.
For the new scenarios:
- the planner still optimizes against the nominal dynamic-obstacle trajectory
- the executed episode uses a seed-dependent perturbed trajectory
- the perturbation changes obstacle time offset, speed scale, and lateral offset

This is mild model mismatch, not a full partial-observation benchmark.
It does not add perception noise, explicit belief tracking, or delayed observations.

Two scenarios are included:
- `uncertain_crossing`: nominal `dynamic_crossing` plus dynamic-obstacle mismatch
- `uncertain_slalom`: nominal `dynamic_slalom` plus dynamic-obstacle mismatch

Compared planners:
- `mppi`
- `feedback_mppi`
- `diff_mppi_1`
- `diff_mppi_3`

The fixed-budget sweep uses:
- `K in {256, 512, 1024, 2048, 4096, 6144, 8192}`
- `4` seeds per configuration

The exact-time follow-up uses:
- shared controller-time targets `{1.00, 1.50} ms`
- direct search over integer `K`

## Main Result

The uncertainty follow-up preserves the main dynamic-task pattern.

At fixed rollout budget:
- vanilla `mppi` fails both uncertain scenarios for every tested `K`
- `feedback_mppi` solves `uncertain_crossing` for every tested `K`, but still fails `uncertain_slalom`
- both `diff_mppi_1` and `diff_mppi_3` solve both uncertain scenarios for every tested `K`

Representative fixed-budget rows:
- `uncertain_crossing @ K=1024`: `mppi` success `0.00`, final distance `3.13`; `feedback_mppi` success `1.00`, final distance `1.85`; `diff_mppi_1` success `1.00`, final distance `1.92`
- `uncertain_slalom @ K=1024`: `mppi` success `0.00`, final distance `14.27`; `feedback_mppi` success `0.00`, final distance `11.82`; `diff_mppi_1` success `1.00`, final distance `1.89`

The same qualitative split remains under exact matched-time tuning.

At `1.00 ms`:
- `uncertain_crossing`: `mppi K=7584 @ 0.982 ms`, final distance `2.97`; `feedback_mppi K=2087 @ 0.990 ms`, final distance `1.87`; `diff_mppi_1 K=5457 @ 0.974 ms`, final distance `1.89`
- `uncertain_slalom`: `mppi K=7524 @ 0.988 ms`, final distance `14.17`; `feedback_mppi K=2058 @ 0.985 ms`, final distance `11.82`; `diff_mppi_3 K=346 @ 1.026 ms`, final distance `1.92`

At `1.50 ms`:
- `uncertain_crossing`: `mppi K=11176 @ 1.497 ms`, final distance `3.00`; `feedback_mppi K=3445 @ 1.493 ms`, final distance `1.87`; `diff_mppi_1 K=9208 @ 1.484 ms`, final distance `1.89`
- `uncertain_slalom`: `mppi K=11136 @ 1.468 ms`, final distance `14.20`; `feedback_mppi K=3397 @ 1.478 ms`, final distance `11.79`; `diff_mppi_3 K=4367 @ 1.498 ms`, final distance `1.91`

So the uncertainty result is not that Diff-MPPI is the only controller that remains useful.
The more precise result is:
- under uncertain timed crossing, the strengthened feedback baseline also stays strong
- under uncertain dynamic slalom, only the hybrid Diff-MPPI variants remain successful

## Why This Helps

This follow-up improves the reviewer story in three ways.

1. It is no longer only a deterministic moving-obstacle benchmark.
The planner is evaluated under seed-dependent obstacle mismatch instead of a single fixed obstacle schedule.

2. The uncertainty result still separates the methods.
`feedback_mppi` closes the gap on the easier uncertain crossing task, but it still does not solve uncertain slalom.

3. The matched-time claim survives the added mismatch.
At both `1.00 ms` and `1.50 ms`, tuned `mppi` remains unsuccessful on both uncertain tasks, while the hybrid controller remains successful on both.

That makes the narrow empirical claim stronger:

> The hybrid MPPI plus autodiff refinement stack is not only better on deterministic dynamic obstacles. It also remains effective under mild seed-dependent obstacle-motion mismatch, including under exact matched-time tuning.

## Limits

This still does not close the full uncertainty gap.

What is now addressed:
- randomized obstacle time offset
- randomized obstacle speed
- randomized obstacle lateral offset
- exact-time comparison under that mismatch

What is still missing:
- explicit observation noise
- delayed obstacle measurements
- probabilistic predictions or belief updates
- standardized uncertain-dynamics benchmarks

So this should be read as a useful uncertainty pilot, not as a full uncertainty-aware planning paper by itself.
