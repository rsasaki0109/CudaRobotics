# Diff-MPPI Planar-Manipulator Follow-Up

Date: 2026-04-02

This note records a reviewer-driven manipulation follow-up for the `Diff-MPPI` line.
It is meant to sit between the earlier navigation-only suite and a true high-fidelity manipulator study.

The goal is narrow:

> Show that the hybrid MPPI + autodiff refinement stack can improve obstacle-avoidance reaching on a simple nonlinear manipulator domain that is closer to robotics manipulation than the earlier navigation or CartPole pilots.

The benchmark binary is `benchmark_diff_mppi_manipulator`.
It writes `build/benchmark_diff_mppi_manipulator.csv`, with summaries in:
- `build/benchmark_diff_mppi_manipulator_summary.md`
- `build/benchmark_diff_mppi_manipulator_summary.tex`
- `build/plots_manipulator/`

A lightweight matched-time spot-check can also be generated with:
- `build/benchmark_diff_mppi_manipulator_exact_time.csv`
- `build/benchmark_diff_mppi_manipulator_exact_time_search.csv`
- `build/benchmark_diff_mppi_manipulator_exact_time_summary.md`

## Domain

The manipulator model is a 2-link planar arm with:
- state `q1, q2, dq1, dq2`
- torque control `tau1, tau2`
- second-order joint dynamics with damping, gravity-like terms, and coupling
- workspace obstacle penalties evaluated along the links, not only at the end effector

Two scenarios are included:
- `arm_static_shelf`: static workspace obstacles around a shelf-style reach target
- `arm_dynamic_sweep`: the same style of reach task with one moving obstacle sweeping through the approach corridor

Compared planners:
- `mppi`
- `feedback_mppi_cov`
- `diff_mppi_1`
- `diff_mppi_3`

The current fixed-budget sweep is:
- `K in {256, 512}`
- `4` seeds per configuration

## What The Benchmark Shows

### 1. The static shelf task gives a real outside-domain success split

The clearest result is `arm_static_shelf`.

At `K=256`:
- `mppi`: success `0.00`, final distance `0.23`
- `diff_mppi_1`: success `0.75`, final distance `0.15`
- `feedback_mppi_cov`: success `1.00`, final distance `0.15`
- `diff_mppi_3`: success `0.25`, final distance `0.18`

At `K=512`:
- `mppi`: success `0.00`, final distance `0.22`
- `diff_mppi_1`: success `1.00`, final distance `0.15`
- `feedback_mppi_cov`: success `1.00`, final distance `0.15`
- `diff_mppi_3`: success `0.75`, final distance `0.15`

This matters because the benchmark is no longer just "navigation plus CartPole".
The hybrid controller now has a custom manipulation-domain result where vanilla MPPI remains unsuccessful while the lighter hybrid refinement succeeds on most or all seeds.

### 2. The dynamic manipulator task is improved, but not solved

`arm_dynamic_sweep` is intentionally harder and the current result should not be oversold.

At `K=256`:
- `mppi`: final distance `0.34`
- `feedback_mppi_cov`: final distance `0.31`
- `diff_mppi_1`: final distance `0.30`
- `diff_mppi_3`: final distance `0.29`

At `K=512`:
- `mppi`: final distance `0.35`
- `feedback_mppi_cov`: final distance `0.29`
- `diff_mppi_1`: final distance `0.31`
- `diff_mppi_3`: final distance `0.30`

So the moving-obstacle manipulator task currently behaves like the earlier CartPole and dynamic-bicycle pilots:
- the hybrid and closer feedback baselines reduce terminal error materially
- none of the planners fully solve the task yet

That is still useful, because it shows that the method does not collapse outside the original navigation setting, but it is not yet a decisive manipulation success-rate result.

### 3. The matched-time view is currently a spot-check, not a main claim

The repo now includes:

```bash
python3 scripts/tune_diff_mppi_time_targets.py --preset manipulator_pilot
```

The current `3.0 ms` spot-check is best treated conservatively.
It is useful for checking that the manipulator benchmark can be passed through the same exact-time tooling, but it is not as clean as the main dynamic-navigation exact-time suite.

Representative selected rows:
- `arm_static_shelf`: `mppi K=4096 @ 0.72 ms`, `diff_mppi_1 K=4096 @ 1.71 ms`, `diff_mppi_3 K=681 @ 3.02 ms`
- `arm_dynamic_sweep`: `mppi K=4096 @ 0.74 ms`, `diff_mppi_1 K=4096 @ 1.77 ms`, `diff_mppi_3 K=128 @ 3.05 ms`

That means the exact-time story here is not yet the strong reviewer-facing point.
The cleaner evidence is still the fixed-budget shelf result and the fixed-budget dynamic-sweep quality reduction.

## Updated Interpretation

The reviewer-safe interpretation is:

> We now have a third outside-domain pilot beyond CartPole and dynamic-bicycle navigation: a custom planar manipulator obstacle-avoidance benchmark. In that manipulation pilot, vanilla MPPI remains unsuccessful on the static shelf task while one-step Diff-MPPI reaches `0.75-1.00` success and the covariance-feedback controller reaches `1.00` success. The moving-obstacle manipulator task is not yet solved, but the hybrid and covariance-feedback variants still reduce terminal error relative to vanilla MPPI.

This is stronger than the earlier outside-domain story because:
- it is an obstacle-avoidance reaching problem, not only stabilization
- it is closer to manipulation than CartPole
- it gives at least one concrete success-rate split in a manipulator-style task

It still does not close the final venue-level gap because:
- the model is still a custom planar 2-link arm
- it is not a standardized 7-DOF benchmark
- the matched-time manipulation story is weaker than the fixed-budget story

## Practical Value

For the current paper package, this follow-up is most useful as:
- an appendix or supplement experiment
- a rebuttal point against the "only navigation and toy dynamics" criticism
- a bridge toward a stronger future manipulator evaluation

It is a meaningful improvement over the earlier outside-domain story, but it is still not a replacement for a standardized higher-fidelity manipulator benchmark.
