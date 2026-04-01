# Diff-MPPI Dynamic-Bicycle Follow-Up

Date: 2026-04-02

This note records a reviewer-driven mobile-dynamics follow-up for the `Diff-MPPI` line.
It sits between the earlier 2D kinematic navigation benchmark and a true high-fidelity robotics evaluation.

The goal is narrow:

> Show that the hybrid MPPI + autodiff refinement stack still produces useful low-budget improvements on a higher-order mobile-navigation system with steering lag and drag.

The benchmark binary is `benchmark_diff_mppi_dynamic_bicycle`.
It writes `build/benchmark_diff_mppi_dynamic_bicycle.csv`, with summaries in:
- `build/benchmark_diff_mppi_dynamic_bicycle_summary.md`
- `build/benchmark_diff_mppi_dynamic_bicycle_summary.tex`
- `build/plots_dynamic_bicycle/`

## Domain

The dynamics are a five-state dynamic bicycle model:
- planar position `x, y`
- heading `yaw`
- longitudinal speed `v`
- steering state `steer`

The controller commands:
- acceleration
- steering setpoint

Compared with the earlier kinematic benchmark, this pilot adds:
- steering actuator lag
- linear and quadratic drag
- bounded steering state dynamics
- the same style of static and moving-obstacle costs used in the main Diff-MPPI suite

Two scenarios are included:
- `dynbike_crossing`: a moving obstacle crosses the middle of a sparse field
- `dynbike_slalom`: a tighter static slalom is combined with a descending moving obstacle

Compared planners:
- `mppi`
- `diff_mppi_1`
- `diff_mppi_3`

The default sweep is intentionally low-budget:
- `K in {32, 64, 128, 256}`
- `4` seeds per configuration

This is deliberate. In this domain, low-budget behavior is more informative than high-budget saturation.

## What The Benchmark Shows

### 1. The hybrid controller transfers to higher-order mobile dynamics

The cleanest positive result is `dynbike_crossing`.

Representative fixed-budget rows:
- `K=32`: `mppi` final distance `2.14`, cumulative cost `2295.9`, steps `196.0`; `diff_mppi_3` final distance `2.09`, cumulative cost `2234.7`, steps `191.0`
- `K=64`: `mppi` final distance `2.15`; `diff_mppi_3` final distance `2.12`

These are modest gains, but they are consistent with the intended claim:
- the refinement stage still helps a higher-order mobile model
- the gain appears most clearly when the rollout budget is small

### 2. The slalom task shows a stronger low-budget win, but also exposes tuning sensitivity

At `K=32`, `dynbike_slalom` is the strongest row in the follow-up:
- `mppi`: success `0.75`, final distance `12.60`, cumulative cost `4034.9`
- `diff_mppi_3`: success `1.00`, final distance `2.24`, cumulative cost `3579.2`

That is a real qualitative difference.
Vanilla MPPI still misses the goal on one seed, while the 3-step hybrid variant reaches all four.

However, the same task also shows that higher-order dynamics make the deeper refinement more delicate:
- at `K=64`, both `mppi` and `diff_mppi_3` succeed, with near-identical terminal distance
- at `K=128`, `diff_mppi_3` drops back to `0.75` success while `diff_mppi_1` remains stable
- at `K=256`, both hybrid variants recover and all planners succeed

So the follow-up is useful partly because it is not uniformly flattering.
It shows transfer, but it also shows that refinement depth needs tuning once the dynamics get less forgiving.

### 3. The main signal is fixed-budget transfer, not matched-time dominance

The generic summary and plotting scripts can still emit cap-based and equal-time tables for this benchmark, but those views should not be the primary evidence here.

Reason:
- `mppi` operates around `0.06-0.09 ms`
- `diff_mppi_1` operates around `0.55-0.65 ms`
- `diff_mppi_3` operates around `1.5-1.8 ms`

Those timing bands are too far apart for the current generic equal-time summaries to be especially meaningful without an exact retuning loop like the one used in the main 2D dynamic-obstacle suite.

For this benchmark, the reviewer-safe interpretation should therefore stay focused on:
- fixed-budget transfer
- low-budget success behavior
- the fact that the controller stack remains functional on richer mobile dynamics

## Updated Interpretation

The current reviewer-safe interpretation is:

> We now have a second outside-domain pilot beyond CartPole: a dynamic-bicycle mobile-navigation benchmark with steering lag and drag. In that higher-order mobile setting, Diff-MPPI still produces useful low-budget improvements, including a clear `dynbike_slalom @ K=32` success gain, but the deeper refinement is tuning-sensitive and the benchmark does not yet replace a true high-fidelity robotics evaluation.

This is stronger than the CartPole-only story because:
- it returns to obstacle-avoidance planning rather than only stabilization
- it uses richer vehicle dynamics than the original kinematic benchmark
- it gives at least one clear success-rate improvement in a nontrivial moving-obstacle task

It still does not close the final venue-level gap because:
- the benchmark is still custom rather than standardized
- it is not Isaac, MuJoCo locomotion, or a manipulator benchmark
- there is no exact matched-time tuning loop for this domain yet

## Practical Value

For the current paper package, this follow-up is most useful as:
- an appendix experiment
- a reviewer rebuttal point against the "only 2D kinematic navigation" criticism
- evidence that the controller stack transfers to a richer mobile-dynamics setting

It is not yet the experiment that makes the paper main-track ready by itself.
