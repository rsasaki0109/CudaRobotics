# Topology-stress benchmark — Day 2 results

Day 1 surfaced two issues in the path-follow local controllers
(`hybrid_astar_dwa`, `hybrid_astar_mppi`): the `w_terminal * dist_to_goal`
term at the end of each rollout still pulled the robot toward the
abstract goal, which on local-minima scenes drags it off the planned
detour and into the trap. Day 2 fixes that and strengthens the dyn
bottleneck.

## Changes

### 1. Path-aware terminal cost

Both `hybrid_astar_dwa_grid_kernel` and `hybrid_astar_mppi_rollout_kernel`
replace direct goal-distance with a **path-aware terminal**:

```
if (path_n > 0):
    nearest_idx = arg min over path of dist(robot_end, waypoint)
    term_idx    = min(nearest_idx + lookahead_idx, path_n - 1)
    remaining   = (path_n - 1 - term_idx) * 2.5 m
    terminal    = w_terminal * (dist(robot_end, path[term_idx]) + remaining)
else:
    terminal    = w_terminal * dist(robot_end, goal)
```

The pull is now toward "a few waypoints further along the planned
detour", plus a small constant penalty per remaining waypoint. On open
scenes (where path is roughly straight to goal) this collapses to the
old behaviour; on trap scenes the terminal respects the detour
geometry rather than the straight-line goal direction.

### 2. Strengthened `dynamic_bottleneck`

Single fast obstacle (`{25, 40, 0, -2.1, 2.2}`) at Day 1 was clearable
by all planners with a small reactive slowdown. Replaced with a
slower, larger obstacle:

```
{25, 30, 0, -1.0, 2.5}  // enters gap top ~t=2s, exits bottom ~t=8s
```

The slow obstacle occupies the gate during the robot's natural
arrival window and creates a 6-second wait requirement.

## Day 2 smoke result

K=4096, 2 seeds, 5 planners.

| Planner | u_trap | s_corridor | bottleneck | topology+dyn |
|---|:---:|:---:|:---:|:---:|
| `dwa_med` | ✗ | ✗ | ✗ (stuck) | ✗ |
| `diff_mppi_3` | ✗ | ✗ | ✗ (stuck) | ✗ |
| `hybrid_astar_pp` | ✓ | ✓ | ✗ (coll 20) | ✗ (coll 13) |
| `hybrid_astar_dwa` | **✓** | ✓ | ✗ (stuck) | **✓** |
| `hybrid_astar_mppi` | **✓** | ✓ | ✗ (stuck) | **✓** |

Compared to Day 1 (bold cells changed):

| Planner | u_trap | s_corridor | bottleneck | topology+dyn |
|---|:---:|:---:|:---:|:---:|
| `dwa_med` | ✗ | ✗ | ✓→**✗** | ✗ |
| `diff_mppi_3` | ✗ | ✗ | ✓→**✗** | ✗ |
| `hybrid_astar_pp` | ✓ | ✓ | ✓→**✗** | ✗ |
| `hybrid_astar_dwa` | ✗→**✓** | ✓ | ✓→**✗** | ✗→**✓** |
| `hybrid_astar_mppi` | ✗→**✓** | ✗→**✓** | ✓→**✗** | ✗→**✓** |

## What this means

### `hybrid_astar_{dwa,mppi}` now generalise to topology

The path-aware terminal fix recovers the "global + local hybrid"
claim under topology stress. `hybrid_astar_dwa` and `hybrid_astar_mppi`
now solve `static_u_trap` and `dynamic_crossing_with_topology`
(both axes), and `hybrid_astar_mppi` additionally solves
`static_s_corridor`. The refined headline now reads:

> Global plan + local reactive controller closes the paradigm gap
> across both open-dynamic AND global-topology axes, **provided
> the local cost's terminal pull respects the planned path**.
> Direct goal-pull breaks topology; path-aware terminal recovers it.

This is materially stronger than the Day 1 statement.

### `dynamic_bottleneck` exposes a new failure mode

Strengthened to v=-1.0, the gate is blocked for ~6s during the robot's
arrival window. The slow obstacle's occupancy exceeds the reactive
controllers' lookahead horizon (`T_dwa = 20 steps = 2 s`,
`T_mppi = 30 steps = 3 s`), so DWA / MPPI never see a clear path and
get stuck before the gate, while pure pursuit ignores the dyn obs
entirely and ploughs in (20 collisions).

This is **not** something the Day 2 fix was supposed to solve, but
the result is worth keeping: it's the first scene in this benchmark
that defeats every current planner, and it motivates Day 3.

### Day 3 hooks

1. **Long-horizon timing via Diff-MPPI conditioning**: a Diff-MPPI
   local controller conditioned on the Hybrid A* path might solve
   `dynamic_bottleneck`, because the autodiff control-gradient
   refinement can extract slowdown signals from the cost landscape
   over the full T-step horizon. This is the natural Day 3 (G) item.
2. **Failure taxonomy CSV**: extend the per-episode CSV with
   `failure_type` (`stuck` / `collision` / `timeout` / `goal_miss`),
   `time_to_goal`, `min_clearance`. The current "success / final_dist /
   collisions" trio is too coarse for the new scenes — e.g.,
   `dynamic_bottleneck` distinguishes "stuck at gate" (DWA / MPPI)
   from "collided in gate" (pp), but the summary table doesn't surface
   that. Useful for the report draft.

## Files changed

- `src/benchmark_diff_mppi.cu`
  - `hybrid_astar_dwa_grid_kernel`: path-aware terminal
  - `hybrid_astar_mppi_rollout_kernel`: path-aware terminal
  - `make_dynamic_bottleneck_scene`: slower / larger dyn obs
- `docs/topology_bench_day2_findings.md`: this file
