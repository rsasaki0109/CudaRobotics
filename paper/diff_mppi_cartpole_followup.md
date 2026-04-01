# Diff-MPPI CartPole Follow-Up

Date: 2026-04-02

This note records a reviewer-driven outside-domain follow-up for the `Diff-MPPI` line.
It now sits alongside the later dynamic-bicycle mobile-navigation follow-up in `paper/diff_mppi_dynamic_bicycle_followup.md`.
The goal is not to claim a full high-fidelity robotics evaluation.
The goal is narrower:

> Show that the hybrid MPPI + autodiff refinement stack can run on a nonlinear underactuated dynamics benchmark outside the original 2D kinematic navigation suite.

The benchmark binary is `benchmark_diff_mppi_cartpole`.
It writes `build/benchmark_diff_mppi_cartpole.csv`, with summaries in:
- `build/benchmark_diff_mppi_cartpole_summary.md`
- `build/benchmark_diff_mppi_cartpole_summary.tex`
- `build/plots_cartpole/`

## Domain

The benchmark reuses the repository's CartPole dynamics from the `MiniIsaacGym` line, but runs a single online planner instead of policy learning.

Two scenarios are included:
- `cartpole_recover`: moderate initial displacement and angular error, with success defined as reaching and holding a stabilized upright set for a short window
- `cartpole_large_angle`: a harder large-angle recovery setting with the same nonlinear dynamics but much larger initial angle error

Compared planners:
- `mppi`
- `diff_mppi_1`
- `diff_mppi_3`

## What The Benchmark Shows

The result is mixed, which is still useful.

### 1. The hybrid controller does transfer outside 2D navigation

The most positive case is `cartpole_recover`.

Representative fixed-budget rows:
- `K=256`: `mppi` success `0.00`, final error `0.82`; `diff_mppi_3` success `0.25`, final error `0.74`
- `K=1024`: `mppi` success `0.25`, final error `0.94`; `diff_mppi_1` success `0.25`, final error `0.85`
- `K=2048`: `mppi` success `0.00`, final error `1.34`; `diff_mppi_3` success `0.25`, final error `0.92`

So the hybrid controller is not confined to the earlier geometric navigation tasks.
It can still produce competitive or better stabilization behavior on a nonlinear underactuated system.

### 2. The harder large-angle setting remains unsolved, but the best Diff-MPPI variants reduce terminal error slightly

In `cartpole_large_angle`, no planner reaches the benchmark's stabilization success criterion.

Still, the best Diff-MPPI rows are modestly better than vanilla MPPI:
- `K=512`: `mppi` final error `1.32`, cumulative cost `2423.1`; `diff_mppi_3` final error `1.30`, cumulative cost `2353.5`
- `K=1024`: `mppi` final error `1.32`, cumulative cost `2428.2`; `diff_mppi_1` final error `1.30`, cumulative cost `2382.6`

That is not a decisive win, but it is also not a complete failure to transfer.

### 3. This is a partial reviewer answer, not a finished acceptance-level experiment

The aggregate summary stays mixed:
- at `K=512`, vanilla `mppi` still has the best aggregate success
- the equal-time and cap-based views do not show a clean outside-domain dominance result
- the harder CartPole task is still unsolved by all methods

That means this benchmark should be interpreted as:
- evidence that the implementation generalizes beyond 2D kinematic navigation
- not yet evidence that the method has a strong new outside-domain advantage

## Updated Interpretation

The current reviewer-safe interpretation is:

> We now have a pilot nonlinear underactuated-dynamics benchmark outside the original navigation suite. The hybrid controller remains competitive there and improves several fixed-budget rows, but the evidence is still mixed and does not yet replace the need for a stronger high-fidelity robotics evaluation.

This is still worth keeping, because it narrows one obvious review criticism:
- the paper is no longer `only` a 2D kinematic navigation story

But it does not close the stronger gap:
- the added domain is still CartPole, not a manipulator, mobile robot, or high-fidelity simulator

## Practical Value

For the current paper package, this follow-up is most useful as:
- a reviewer rebuttal point
- an appendix or supplement figure/table
- evidence that the controller stack is portable across dynamics classes

It is not yet the experiment that makes the paper `ICRA/IROS`-ready by itself.
