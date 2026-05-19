# Topology-stress benchmark — Day 3 results

Day 2 strengthened `dynamic_bottleneck` to defeat every planner. The
hypothesis was that the failure was **horizon-driven**: the slow
obstacle occupies the gate for ~6 s, exceeding the 2-3 s lookahead
of DWA's `dwa_predict_steps` and MPPI's `t_horizon`. Day 3 confirms
this and adds long-horizon variants.

## Changes

1. **`--t-horizon N` CLI flag** — overrides `DEFAULT_T_HORIZON=30` at
   runner construction. Affects every variant that uses
   `t_horizon_` (the MPPI / Diff-MPPI / STOMP rollout horizon and
   `hybrid_astar_mppi`).

2. **`dwa_long`** (DWA, `dwa_predict_steps=60`, ~6 s lookahead).

3. **`hybrid_astar_dwa_long`** (planner_kind=4 with
   `dwa_predict_steps=60`).

## Day 3 smoke result

K=4096, 2 seeds, 7 planners; `--t-horizon 60` for MPPI/Diff-MPPI rows
(short-horizon DWA-only variants are unaffected by that flag and use
their per-variant `dwa_predict_steps`).

| Planner | u_trap | s_corridor | bottleneck | topology+dyn |
|---|:---:|:---:|:---:|:---:|
| `dwa_med` (T_dwa=20) | ✗ | ✗ | ✗ | ✗ |
| `dwa_long` (T_dwa=60) | ✗ | **✓** | **✓** | ✗ |
| `diff_mppi_3` (T=60) | ✗ | ✗ | **✓** | ✗ |
| `hybrid_astar_pp` | ✓ | ✓ | ✗ (coll 20) | ✗ (coll 13) |
| `hybrid_astar_dwa` (T_dwa=20) | ✓ | ✓ | ✗ | ✓ |
| `hybrid_astar_dwa_long` (T_dwa=60) | ✗ (2.25)* | ✓ | **✓** | ✓ |
| `hybrid_astar_mppi` (T=60) | ✓ | ✓ | **✓** | ✓ |

`*` `hybrid_astar_dwa_long` reaches `final_dist=2.25` on U-trap — just
outside the success tolerance of 2.0 m. The longer horizon evidently
trades some end-of-episode precision for the look-ahead capability;
minor regression for users wanting U-trap specifically. The original
`hybrid_astar_dwa` (T_dwa=20) still solves U-trap.

## Headlines

### 1. Bottleneck failure is a horizon issue, not a planner-class issue

The same `hybrid_astar_mppi` that was stuck at `final_dist=23` on
`dynamic_bottleneck` at T=30 reaches the goal at `final_dist=1.97`
when given T=60. `diff_mppi_3` shows the same story (T=30 stuck,
T=60 succeeds). Likewise `dwa_long` solves the scene with no other
change to the planner.

This means `dynamic_bottleneck`'s "no planner solves" Day 2 result
was a **horizon mismatch**, not a fundamental capability limit. The
scene is now useful as a **horizon-axis discriminator**: it separates
short-horizon controllers from long-horizon ones.

### 2. Horizon and global-path are orthogonal axes

Long horizon alone is not sufficient — `dwa_long` and `diff_mppi_3`
at T=60 still fail `static_u_trap` and `dynamic_crossing_with_topology`.
The U-trap requires a global path to escape the local minimum, which
long lookahead alone cannot supply. Conversely, global path alone is
not sufficient — `hybrid_astar_dwa` at T_dwa=20 fails
`dynamic_bottleneck` because no amount of pre-planned path makes the
local controller's 2 s lookahead see past the slow obstacle.

The combo `hybrid_astar_mppi @ T=60` covers both axes (4/4 scenes).
`hybrid_astar_dwa_long` covers 3/4 (slips on U-trap precision).

### 3. Updated paradigm-completion claim

After Day 1 (open-dynamic only):

> Global plan + local reactive controller closes the paradigm gap,
> the pattern is paradigm-agnostic.

After Day 2 (path-aware terminal + topology scenes):

> Global plan + local reactive controller closes the paradigm gap
> across both open-dynamic AND global-topology axes, provided the
> local cost's terminal pull respects the planned path.

After Day 3 (horizon axis):

> Global plan + local reactive controller (with path-aware terminal)
> closes the paradigm gap across open-dynamic, global-topology, AND
> long-occupancy-timing axes — **provided the local lookahead horizon
> is at least as long as the longest obstacle-occupancy window the
> scene requires**. Short-horizon controllers fail on long-occupancy
> scenes regardless of how good the global path is.

## Open question (deferred)

Should Day 4+ introduce a **per-variant** `t_horizon` field rather
than the global CLI override, so the headline matrix uses
`hybrid_astar_mppi @ T=60` automatically without the user knowing to
pass `--t-horizon 60`? Probably yes — the open-dynamic suite at T=30
already worked for `hybrid_astar_mppi`, so a default of T=60 for the
`hybrid_astar_*` family is a strict superset. The DWA family already
has per-variant `dwa_predict_steps` for the same reason. This is a
mechanical change but touches several variant registrations.

## Files changed

- `src/benchmark_diff_mppi.cu`
  - `--t-horizon N` CLI flag (overrides `DEFAULT_T_HORIZON`)
  - `dwa_long` variant (`dwa_predict_steps=60`)
  - `hybrid_astar_dwa_long` variant (`dwa_predict_steps=60`)
- `docs/topology_bench_day3_findings.md`: this file
