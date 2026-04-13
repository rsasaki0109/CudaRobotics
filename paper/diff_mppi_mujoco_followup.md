# Diff-MPPI MuJoCo Follow-Up

Date: 2026-04-14

This note records a reviewer-driven standardized-benchmark follow-up for the `Diff-MPPI` line.
It is intentionally narrow:

> Show that the current MPPI / feedback / hybrid controller stack transfers to a small public MuJoCo benchmark under the same matched-time tuning workflow used elsewhere in the repo.

The benchmark binary is `benchmark_diff_mppi_mujoco`.
It writes:
- `build/benchmark_diff_mppi_mujoco_seed5.csv`
- `build/benchmark_diff_mppi_mujoco_spot.csv`
- `build/benchmark_diff_mppi_mujoco_wide_seed3.csv`

The exact-time tuning follow-up writes:
- `build/benchmark_diff_mppi_mujoco_exact_time.csv`
- `build/benchmark_diff_mppi_mujoco_exact_time_search.csv`
- `build/benchmark_diff_mppi_mujoco_exact_time_summary.md`

The MuJoCo Reacher extension uses:
- benchmark binary: `benchmark_diff_mppi_mujoco_reacher`
- model file: `mujoco_models/reacher.xml`
- stable-seed summaries:
  - `build/benchmark_diff_mppi_mujoco_reacher_terminal_stable.csv`
  - `build/benchmark_diff_mppi_mujoco_reacher_terminal_tuned_stable.csv`

## Domain

The current MuJoCo pilot uses the public Gymnasium / MuJoCo `InvertedPendulum-v4` XML:
- model file: `mujoco_models/inverted_pendulum.xml`
- simulator: MuJoCo C API on CPU (`mj_step`)
- planner-side surrogate: a lightweight GPU cart-pole approximation inside the benchmark harness

Two reset distributions are included:
- `inverted_pendulum_v4`: near-upright reset, matching the standard task regime
- `inverted_pendulum_wide_reset`: wider initial position / angle / velocity ranges to make stabilization less forgiving

Compared planners:
- `mppi`
- `feedback_mppi_ref`
- `diff_mppi_3`

The planner-side approximation needed one MuJoCo-specific stabilization tweak:
- multi-step gradient refinement was made conservative by accepting only updates that reduce surrogate rollout cost
- the default `diff_mppi_3` step size was reduced, and exact-time multi-parameter tuning later selected stronger settings again when they remained stable

## What The Benchmark Shows

### 1. This is a real public-benchmark check, not another hand-built navigation scene

The main value of the MuJoCo pilot is not a dramatic hybrid win.
The main value is that the repo now includes:
- a public MuJoCo model
- official task-style termination behavior
- the same exact-time tuning workflow already used for the main dynamic suite

That directly weakens the simple reviewer criticism that the project only works on custom hand-authored environments.

### 2. The fixed-budget view shows transfer, but not a clean winner

Using the direct benchmark runs:

For `inverted_pendulum_v4` with 5 seeds:
- `mppi`: success `0.4 @ K=512`, `1.0 @ K=1024`, `1.0 @ K=2048`
- `feedback_mppi_ref`: success `0.6 @ K=512`, `1.0 @ K=1024`, `1.0 @ K=2048`
- `diff_mppi_3`: success `0.8 @ K=512`, `1.0 @ K=1024`, `1.0 @ K=2048`

For `inverted_pendulum_wide_reset`, the untuned spot checks were harsher:
- `diff_mppi_3` and `feedback_mppi_ref` reached `2/3` seed success at `K=1024` and `K=2048`
- `mppi` also required higher `K` before stabilizing reliably

So the transfer is real, but the story is not "Diff-MPPI uniquely solves MuJoCo."
It is closer to:
- the hybrid controller transfers
- the release-style feedback baseline also transfers
- the domain is simple enough that rollout budget alone already solves much of the task

### 3. Exact-time multi-parameter tuning makes the MuJoCo pilot clean and reproducible

The exact-time follow-up uses:

```bash
python3 scripts/tune_diff_mppi_time_targets.py --preset mujoco_pendulum --multi-param
```

This searches:
- `K`
- `feedback_gain_scale` for `feedback_mppi_ref`
- `grad_steps` and `alpha` for `diff_mppi_3`

At matched controller-time targets, all three planners solve both MuJoCo scenarios:

At `1.0 ms` aggregate across `inverted_pendulum_v4` and `inverted_pendulum_wide_reset`:
- `diff_mppi_3`: success `1.00`, final distance `0.02`, mean `K=3700`
- `feedback_mppi_ref`: success `1.00`, final distance `0.03`, mean `K=2786`
- `mppi`: success `1.00`, final distance `0.04`, mean `K=5332`

At `1.5 ms`:
- `diff_mppi_3`: success `1.00`, final distance `0.02`, mean `K=6774`
- `feedback_mppi_ref`: success `1.00`, final distance `0.02`, mean `K=4669`
- `mppi`: success `1.00`, final distance `0.03`, mean `K=8136`

Representative tuned rows:
- `inverted_pendulum_v4 @ 1.0 ms`: `mppi K=5248 @ 0.965 ms, dist=0.04`; `feedback_mppi_ref K=2816 @ 0.977 ms, dist=0.02`; `diff_mppi_3 K=3708 @ 0.952 ms, dist=0.02`
- `inverted_pendulum_wide_reset @ 1.0 ms`: `mppi K=5415 @ 0.994 ms, dist=0.03`; `feedback_mppi_ref K=2755 @ 0.959 ms, dist=0.04`; `diff_mppi_3 K=3691 @ 0.954 ms, dist=0.03`
- `inverted_pendulum_wide_reset @ 1.5 ms`: `mppi K=8203 @ 1.484 ms, dist=0.04`; `feedback_mppi_ref K=4861 @ 1.543 ms, dist=0.03`; `diff_mppi_3 K=6768 @ 1.476 ms, dist=0.03`

The tuned diff settings are informative:
- `alpha=0.012` is repeatedly selected
- `grad_steps=5` often wins on `wide_reset`
- the accepted-cost safeguard is what makes those stronger settings usable on this surrogate

## Updated Interpretation

The reviewer-safe interpretation is:

> We now have a small MuJoCo public-benchmark pilot in addition to the custom navigation and manipulator suites. On `InvertedPendulum-v4` and a wider-reset variant, vanilla MPPI, a release-style current-action feedback baseline, and Diff-MPPI can all be tuned to solve the task under shared `1.0-1.5 ms` controller budgets. Diff-MPPI is slightly better on terminal distance at the matched budgets, but the main value of this result is standardization and transfer rather than a decisive hybrid-only win.

That is useful because:
- it gives the paper one public MuJoCo protocol
- it shows the exact-time tooling is not limited to the original navigation benchmark
- it demonstrates that the hybrid controller does not collapse immediately when the environment transition comes from MuJoCo rather than the repo's hand-built simulator

It is still not the benchmark that closes the venue-level gap because:
- the task is still a simple stabilization domain
- it is not manipulation or locomotion
- all three planners solve it once compute is matched carefully

## Reacher Extension

To look for a stronger MuJoCo-side separation, the repo now also includes a small Reacher benchmark:
- ground-truth environment: MuJoCo `reacher.xml`
- planner-side surrogate: a lightweight 2-link arm approximation on GPU
- compared planners: `mppi`, `feedback_mppi_ref`, `diff_mppi_1`, `diff_mppi_3`

Three scenarios were tested:
- `reacher_v5`: standard Gymnasium-style static target
- `reacher_edge_target`: wider reset plus near-boundary target sampling
- `reacher_terminal_edge`: a harder terminal-heavy variant with wider resets and longer horizon (`T=32`)

One engineering fix mattered for the follow-up:
- episode seeds are now derived from stable hashes of scenario and planner names, so filtered sweeps are directly comparable

### What Reacher Adds

The dense static variants (`reacher_v5`, `reacher_edge_target`) do not produce a clean hybrid-only result.
They are useful mostly as another transfer check.

The more interesting case is `reacher_terminal_edge`.
With stable 5-seed runs and default controller settings:
- `mppi` reaches at most `0.8` success on the tested grid (`K=128`, `512`, `8192`)
- `feedback_mppi_ref` reaches `1.0` success at `K=128` and `K=1024`
- default `diff_mppi_3` reaches `1.0` success at `K=1024`

The tuned diff setting that helped most was:
- `grad_steps=5`, `alpha=0.012`

With that tuned hybrid setting:
- `diff_mppi_3` reaches `1.0` success at `K=128` with `1.04 ms`
- `diff_mppi_3` also reaches `1.0` success at `K=512` with `1.11 ms`

This is stronger than the pendulum pilot in one specific sense:
- it creates a real terminal-objective stress test where plain MPPI no longer saturates immediately

But it is still not a decisive paper-closing win because:
- tuned `feedback_mppi_ref` already reaches `1.0` success at `K=1024 @ 0.90 ms`
- high-`K` `mppi` can still recover to `0.8` success around `1.04 ms`
- the best Reacher result is therefore "hybrid is competitive and can dominate plain MPPI in this regime," not "hybrid uniquely solves the MuJoCo task"

So the Reacher extension is best interpreted as:
- a stronger MuJoCo stress test than pendulum
- evidence that the hybrid controller remains viable on a public manipulator model
- still appendix / follow-up material rather than a replacement for the main dynamic-obstacle story

## Practical Value

For the current paper package, this follow-up is most useful as:
- an appendix experiment
- a rebuttal point against the "custom benchmark only" criticism
- a bridge to a future MuJoCo manipulator or locomotion benchmark where tuned feedback baselines do not immediately close the gap

It should not replace the main dynamic-obstacle result, because the MuJoCo pilot is a transfer / standardization check rather than the strongest empirical separation.
