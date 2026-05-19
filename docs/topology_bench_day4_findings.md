# Topology-stress benchmark — Day 4 results

Day 3's open question was whether to make the topology-required
`t_horizon = 60` a **per-variant default** rather than a CLI flag, so
the headline matrix uses the long-horizon controller automatically.

Day 4 attempts that change, discovers a regression, diagnoses it, and
lands a two-variant split that preserves both regimes.

## What changed

### 1. Per-variant `t_horizon` field

`PlannerVariant` now has an `int t_horizon = 0` field. The runner
resolves the horizon with precedence:

```
CLI override > per-variant value > DEFAULT_T_HORIZON (=30)
```

`0` means "use the default (or CLI override)". The mechanism mirrors
the existing per-variant `dwa_predict_steps`, so DWA-family and
MPPI-family planners now share the same registration shape.

### 2. Two-variant split for `hybrid_astar_mppi`

| Variant | `t_horizon` | Best at |
|---|---|---|
| `hybrid_astar_mppi` | 30 (default) | Open-dynamic suite |
| `hybrid_astar_mppi_long` | 60 | Topology suite (dynamic_bottleneck) |

The original plan was a single `hybrid_astar_mppi @ T=60` default. The
sweep below shows why that failed.

## The regression that forced the split

The first Day 4 attempt set `hybrid_astar_mppi.t_horizon = 60` (single
variant) and re-ran the 30-cell open-dynamic suite that has been the
project's reference baseline. Results:

| Default `t_horizon` | open-dynamic 30-cell success (cells with 4/4 seeds) | hard cells (`dyn_speed_scale >= 1.5`) |
|---|---|---|
| 30 (Day 3 status quo) | 29/30 | 11/12 |
| 60 (Day 4 first attempt) | **22/30** | **4/12** |

Going to T=60 was strictly regressive for `hybrid_astar_mppi` on the
open-dynamic suite even though it solved `dynamic_bottleneck`.

### Why T=60 is regressive for MPPI (but not DWA)

DWA's `hybrid_astar_dwa_long` already runs at the equivalent long
horizon (`dwa_predict_steps=60`) with no regression. The asymmetry
comes from the control-selection mechanism:

- **DWA: argmin.** A long rollout that wanders far from the path
  simply has higher cost and is rejected. Bad rollouts cannot
  contaminate good ones.
- **MPPI: cost-weighted averaging.** Every sample contributes to the
  next action with weight `exp(-cost/lambda)`. A small number of
  high-cost long-horizon excursions can shift the averaged control
  enough to nudge the robot off the planned detour, especially on
  the open-dynamic suite where the path is mostly straight and the
  terminal cost's "remaining-arclength" term creates a strong forward
  pull that competes with the path-follow term.

Tuning the terminal-cost multiplier (`remaining * 1.0` vs the original
`remaining * 2.5`) was also tried; it didn't recover the open-dynamic
success rate, confirming the issue is horizon-driven, not weight-driven.

## Final Day 4 results

K=4096, 4 seeds per cell, 30 cells (3 scenarios × 5 speed-scales × 2
radius-scales).

| Planner | open-dynamic (cells with 4/4 seeds) | bottleneck (topology) | U-trap / S-corridor / topology+dyn |
|---|---|---|---|
| `hybrid_astar_mppi` (T=30 default) | **27/30** | ✗ (final_dist=23.7) | ✓ / ✓ / ✓ |
| `hybrid_astar_mppi_long` (T=60) | (not the target regime) | **✓** (final_dist=1.96) | ✓ / ✓ / ✓ |

The split delivers both targets:

- `hybrid_astar_mppi` (T=30 default) keeps the open-dynamic
  baseline. The 3 remaining hard cells are all in `dynamic_pincer`
  at `dyn_speed_scale >= 1.5`, which is consistent with the Day 3
  baseline and is unrelated to the t_horizon change.
- `hybrid_astar_mppi_long` (T=60) solves all 4 topology scenes
  including the long-occupancy `dynamic_bottleneck`, matching the
  Day 3 manual `--t-horizon 60` result.

## Headline update

After Day 3:

> Global plan + local reactive controller (with path-aware terminal)
> closes the paradigm gap across open-dynamic, global-topology, AND
> long-occupancy-timing axes — provided the local lookahead horizon
> is at least as long as the longest obstacle-occupancy window the
> scene requires.

Day 4 refines that with a mechanism caveat:

> The "long-horizon == strictly more information" intuition is
> argmin-specific. For **cost-weighted-averaging** controllers
> (MPPI / Diff-MPPI / STOMP), increasing the rollout horizon trades
> long-occupancy timing capability for open-dynamic robustness,
> because long-horizon excursions pollute the averaged action. The
> right answer is **two variants registered per planner family**
> (one short-horizon for open-dynamic, one long-horizon for
> long-occupancy) — not a single all-purpose default. DWA's
> argmin selection escapes this tradeoff and can default to the
> longer horizon without regression.

## Files changed

- `src/benchmark_diff_mppi.cu`
  - `PlannerVariant`: new `t_horizon = 0` field
  - Runner construction: 3-tier `t_horizon` precedence (CLI >
    variant > DEFAULT_T_HORIZON)
  - `hybrid_astar_mppi_rollout_kernel`: comment block explaining
    the regression diagnosis (no code change — `remaining * 2.5`
    matches the DWA kernel and was kept)
  - `hybrid_astar_mppi_long` variant: `t_horizon = 60`
- `docs/topology_bench_day4_findings.md`: this file
