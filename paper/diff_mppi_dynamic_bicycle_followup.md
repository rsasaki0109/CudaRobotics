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
- `feedback_mppi_sens`
- `diff_mppi_1`
- `diff_mppi_3`

The default sweep is intentionally low-budget:
- `K in {32, 64, 128, 256}`
- `4` seeds per configuration

This is deliberate. In this domain, low-budget behavior is more informative than high-budget saturation.

## What The Benchmark Shows

### 1. The closer feedback baseline is meaningful on the easier mobile-dynamics task

The cleanest baseline result is `dynbike_crossing`.

Representative fixed-budget rows:
- `K=32`: `mppi` final distance `2.14`, cumulative cost `2295.9`, steps `196.0`; `feedback_mppi_sens` final distance `2.15`, cumulative cost `2099.2`, steps `186.2`
- `K=128`: `mppi` final distance `2.12`, cumulative cost `2172.8`, steps `186.5`; `feedback_mppi_sens` final distance `2.16`, cumulative cost `2059.0`, steps `182.5`

This matters because the feedback baseline is no longer a straw man.
In the higher-order bicycle model, rollout-sensitivity feedback is a real efficiency baseline:
- it consistently reduces step count and cumulative cost on `dynbike_crossing`
- it does so with essentially unchanged terminal distance

So the mobile-dynamics follow-up is now more informative than a pure `mppi` versus `Diff-MPPI` comparison.

### 2. The clearest low-budget rescue is still one-step hybrid refinement

The strongest low-budget result is now `dynbike_slalom @ K=32`:
- `mppi`: success `0.75`, final distance `12.60`, cumulative cost `4034.9`
- `feedback_mppi_sens`: success `0.75`, final distance `12.67`, cumulative cost `3773.8`
- `diff_mppi_1`: success `1.00`, final distance `2.24`, cumulative cost `3656.9`

That is still a real qualitative difference.
Vanilla MPPI and the closer feedback baseline each miss one seed, while the 1-step hybrid controller reaches all four.

At `K=64`, the same ordering persists:
- `mppi`: success `1.00`, final distance `2.25`
- `feedback_mppi_sens`: success `0.75`, final distance `10.85`
- `diff_mppi_1`: success `1.00`, final distance `2.23`

At moderate budget, the picture changes again.
On `dynbike_slalom`, `feedback_mppi_sens` becomes a strong efficiency baseline at `K=128` and `K=256`:
- `K=128`: `mppi` steps `255.2`, cumulative cost `3240.0`; `feedback_mppi_sens` steps `230.8`, cumulative cost `2870.8`
- `K=256`: `mppi` steps `253.2`, cumulative cost `3224.7`; `feedback_mppi_sens` steps `236.8`, cumulative cost `2955.5`

The deeper refinement remains the most fragile setting:
- `diff_mppi_3` drops to `0.25` success at `K=32`
- it recovers at `K=128` and `K=256`, but the transfer is clearly tuning-sensitive

So this follow-up is useful partly because it is not uniformly flattering.
It shows three things at once:
- the 1-step hybrid is the clearest low-budget rescue
- the closer feedback baseline is a credible efficiency competitor once `K` is moderate
- the deeper 3-step refinement does not transfer automatically without retuning

### 3. Exact matched-time tuning now gives a conservative compute-matched spot check

The earlier generic cap/equal-time tables were weak for this benchmark because the planner-time bands were far apart:
- `mppi` around `0.06-0.09 ms`
- `diff_mppi_1` around `0.55-0.65 ms`
- `diff_mppi_3` around `1.5-1.8 ms`

That gap is now addressed with the same exact-time search workflow used in the main dynamic-obstacle suite:

```bash
python3 scripts/tune_diff_mppi_time_targets.py --preset dynamic_bicycle --time-targets 1.8
```

The current reviewer-facing spot check uses a shared `1.80 ms` controller-time target.
At that target, the three stable planners can all be matched directly by choosing very different rollout counts:
- `dynbike_crossing`: `mppi K=12953 @ 1.779 ms`, `feedback_mppi_sens K=273 @ 1.723 ms`, `diff_mppi_1 K=9343 @ 1.804 ms`
- `dynbike_slalom`: `mppi K=12855 @ 1.791 ms`, `feedback_mppi_sens K=248 @ 1.711 ms`, `diff_mppi_1 K=8905 @ 1.810 ms`

This tuned view is intentionally more conservative than the low-budget fixed-`K` story.

Representative rows:
- `dynbike_crossing @ 1.80 ms`: `mppi` final distance `2.14`, `feedback_mppi_sens` final distance `2.16`, `diff_mppi_3` final distance `2.15`
- `dynbike_slalom @ 1.80 ms`: `mppi` final distance `2.21`, `feedback_mppi_sens` final distance `2.25`, `diff_mppi_1` final distance `2.23`

So the exact-time view does not show an outside-domain hybrid win.
What it does show is still useful:
- after time matching, `mppi`, `feedback_mppi_sens`, and `diff_mppi_1` remain competitive on terminal distance
- the closer feedback baseline reaches that regime with about `K=248-273`, rather than `K≈1.29e4`
- on `dynbike_slalom`, `feedback_mppi_sens` also uses `17` fewer steps than tuned `mppi`
- the low-budget rescue story and the compute-matched story are different, and the paper should say so explicitly

## Updated Interpretation

The current reviewer-safe interpretation is:

> We now have a second outside-domain pilot beyond CartPole: a dynamic-bicycle mobile-navigation benchmark with steering lag and drag, plus a closer rollout-sensitivity baseline and a `1.80 ms` exact-time spot check. In that higher-order mobile setting, the clearest low-budget rescue is still one-step Diff-MPPI on `dynbike_slalom`, while the closer feedback baseline is a meaningful efficiency competitor once the rollout budget is moderate. This is useful reviewer evidence, but it still does not replace a true high-fidelity robotics evaluation.

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
