# Hybrid A* + Pure Pursuit baseline

A "global planner blind to dynamic obstacles" baseline added to
`benchmark_diff_mppi` as `planner_kind=3`. The intent is to make the
paradigm gap between global-plan-then-track and local replanning
visible on the same scenario grid the local planners are evaluated
against.

## What it does

`include/hybrid_astar_pp.h` is a header-only forward-only Hybrid A*
search plus a pure pursuit tracker. The variant `hybrid_astar_pp`
calls the search once per episode (at first `controller_update`)
against the **static** obstacles only -- dynamic obstacles are
deliberately not consulted -- and then pure-pursuit-tracks the
returned path for the rest of the episode without replanning.

Hybrid A* parameters: 1 m x 1 m x 10-degree lattice, 7 steering
choices, 2.5 m per node expansion (so children land in distinct
cells), Euclidean heuristic on arc-length cost. Forward-only --
Reeds-Shepp is not implemented, which matches the scenario geometry
(the goal is reachable without reverse motions). Goal heading
tolerance is left at pi so the position threshold is the binding
constraint.

Pure pursuit parameters: lookahead 4 m, target speed 5 m/s, linear
speed gain 1.5. The goal_slowdown_radius brings the target speed
linearly to 0 inside 5 m of the path's final waypoint.

## Sweep result

Same grid as `docs/local_planner_comparison.md`: 3 dynamic scenarios
x 5 speed-scales x 2 radius-scales = 30 cells per planner, 4 seeds,
K=4096 (`build/sweep_with_hap_summary.csv`).

Per-planner summary across all 30 cells:

| planner            | family    | cells | solved | mean succ | mean final_d | mean coll | mean ms |
|--------------------|-----------|------:|-------:|----------:|-------------:|----------:|--------:|
| hybrid_astar_pp    | Hybrid-A* |    30 |     21 |      0.70 |         1.91 |      6.23 |    0.05 |
| dwa_fast           | DWA       |    30 |     28 |      0.93 |         1.94 |      0.60 |    0.10 |
| dwa_med            | DWA       |    30 |     30 |      1.00 |         1.91 |      0.00 |    0.11 |
| dwa_fine           | DWA       |    30 |     30 |      1.00 |         1.92 |      0.00 |    0.11 |
| stomp_3_smooth     | STOMP     |    30 |     18 |      0.60 |         2.84 |      0.00 |    1.50 |
| diff_mppi_3_early8 | Diff-MPPI |    30 |     23 |      0.77 |         2.88 |      0.56 |    0.72 |

Hard cells only (`dyn_speed_scale >= 1.5`):

| planner            | hard cells | solved | mean succ | mean final_d | mean coll |
|--------------------|-----------:|-------:|----------:|-------------:|----------:|
| hybrid_astar_pp    |         12 |      3 |      0.25 |         1.91 |     15.58 |
| dwa_med            |         12 |     12 |      1.00 |         1.91 |      0.00 |
| dwa_fine           |         12 |     12 |      1.00 |         1.94 |      0.00 |
| stomp_3_smooth     |         12 |      6 |      0.50 |         3.00 |      0.00 |
| diff_mppi_3_early8 |         12 |      5 |      0.42 |         4.38 |      1.42 |

## Reading the numbers

The Hybrid A* row is the headline:

- `mean final_d = 1.91` -- the planner **does** drive the robot to
  within 2 m of the goal on every cell. The static-obstacle path is
  geometrically fine.
- `mean coll = 6.23` overall and `15.58` on hard cells -- the robot
  collides with dynamic obstacles ~6 times per episode on average,
  16 times when the obstacles are at 1.5x+ speed. The path is fixed
  at episode start; dynamic obstacles cross it.
- `solved = 21/30` (and 3/12 hard) because the benchmark's success
  metric requires both goal reached AND collision-free. The
  collision counter is what differentiates Hybrid A* from the local
  planners here, not goal-reaching.
- `mean ms = 0.05` -- pure pursuit at runtime is essentially free
  (Hybrid A* search runs once at episode start and is excluded from
  the per-step timing).

For comparison, `dwa_med` and `dwa_fine` solve every cell collision-
free; they re-evaluate the action grid at every step against the
current dynamic-obstacle positions, so they sidestep the cross
that's about to happen.

This is not a fair fight on the dynamic grid -- Hybrid A* is the
wrong tool. It is included to anchor the lower end of "what a global
planner without re-planning can do" and to make the local-replanning
value proposition explicit.

## What would be needed to make Hybrid A* competitive

- Replan from current pose every N steps (would push `mean ms`
  toward the millisecond range).
- Include dynamic obstacles in the search by inflating predicted
  positions along the planned arc -- the time-aware Hybrid A* in
  the Karaman/Frazzoli line.
- Hybrid A* + a local replanner (MPC or DWA) inside the planned
  corridor, which is the common production split.

None of these are in scope for this PR.
