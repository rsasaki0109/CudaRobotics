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

The exact-time tuning follow-up writes:
- `build/benchmark_diff_mppi_dynamic_bicycle_exact_time.csv`
- `build/benchmark_diff_mppi_dynamic_bicycle_exact_time_search.csv`
- `build/benchmark_diff_mppi_dynamic_bicycle_exact_time_summary.md`

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

### 3. Exact matched-time tuning partially closes the compute-matching criticism

The earlier generic cap/equal-time tables were weak for this benchmark because the planner-time bands were far apart:
- `mppi` around `0.06-0.09 ms`
- `diff_mppi_1` around `0.55-0.65 ms`
- `diff_mppi_3` around `1.5-1.8 ms`

That gap is now addressed with the same exact-time search workflow used in the main dynamic-obstacle suite:

```bash
python3 scripts/tune_diff_mppi_time_targets.py --preset dynamic_bicycle
```

The current tuned targets are `1.80 ms` and `2.00 ms`.
At those targets, all planners can be matched directly by choosing different rollout counts:
- `dynbike_crossing @ 1.80 ms`: `mppi K=14005 @ 1.782 ms`, `diff_mppi_3 K=2155 @ 1.764 ms`
- `dynbike_slalom @ 1.80 ms`: `mppi K=13982 @ 1.781 ms`, `diff_mppi_3 K=589 @ 1.784 ms`
- `dynbike_slalom @ 2.00 ms`: `mppi K=15353 @ 1.997 ms`, `diff_mppi_1 K=11408 @ 1.985 ms`

The matched-time result is more modest than the low-budget fixed-`K` story, which is exactly the right thing to report.

Representative rows:
- `dynbike_slalom @ 1.80 ms`: `mppi` final distance `2.25`, `diff_mppi_3` final distance `2.22`
- `dynbike_crossing @ 2.00 ms`: `mppi` final distance `2.14`, `diff_mppi_1` final distance `2.13`

So the exact-time view does not reveal a dramatic outside-domain win.
What it does show is still useful:
- the hybrid controllers remain competitive after controller time is matched directly
- they often achieve essentially the same terminal quality with far smaller tuned rollout counts
- the low-budget fixed-`K` gains are not just an artifact of refusing to retune compute

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
- the current exact-time result is still a medium-fidelity pilot rather than a standardized robotics-domain benchmark

## Practical Value

For the current paper package, this follow-up is most useful as:
- an appendix experiment
- a reviewer rebuttal point against the "only 2D kinematic navigation" criticism
- evidence that the controller stack transfers to a richer mobile-dynamics setting

It is not yet the experiment that makes the paper main-track ready by itself.
