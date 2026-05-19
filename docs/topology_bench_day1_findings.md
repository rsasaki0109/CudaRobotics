# Topology-stress benchmark — Day 1 smoke findings

Per advisor roadmap, four new scenarios were added to `benchmark_diff_mppi`
to surface where global path planning actually matters. The existing
`dynamic_*` scenes are "open dynamic" — local reactive controllers
handle them — so DWA alone already solves 30/30 cells. The new scenes
target two orthogonal axes:

1. **global topology** — local minima or long-horizon detours that
   defeat any direct-goal-pull controller
2. **topology + dynamic** — both axes at once

## Scene list

| Scene                            | Axis                       | Static obs | Dyn obs |
|----------------------------------|----------------------------|-----------:|--------:|
| `static_u_trap`                  | global topology            |         11 |       0 |
| `static_s_corridor`              | global topology            |         10 |       0 |
| `dynamic_bottleneck`             | dynamic timing             |          6 |       1 |
| `dynamic_crossing_with_topology` | topology + dynamic         |         11 |       1 |

## Smoke result (K=4096, 2 seeds, default planner params)

| Planner               | u_trap | s_corridor | bottleneck | topology+dyn |
|-----------------------|:------:|:----------:|:----------:|:------------:|
| `dwa_med`             |   ✗    |     ✗      |     ✓      |      ✗       |
| `diff_mppi_3`         |   ✗    |     ✗      |     ✓      |      ✗       |
| `hybrid_astar_pp`     |   ✓    |     ✓      |     ✓      |   ✗ (coll)   |
| `hybrid_astar_dwa`    |   ✗    |     ✓      |     ✓      |      ✗       |
| `hybrid_astar_mppi`   |   ✗    |     ✗      |     ✓      |      ✗       |

`✗ (coll)` means the planner reached the goal area but collided 13 times
with the dynamic obstacle along the way (collision-free metric fails).

## Reading the result

1. **DWA / Diff-MPPI / MPPI all trapped on `static_u_trap`** — direct
   goal-pull pulls them straight into the trap; no global plan, no escape.
   `final_dist ≈ 10.22 m` for all three at `max_steps`. **Validates the
   adversarial benchmark**: the open-dynamic suite did not catch this.

2. **`hybrid_astar_pp` is the only planner solving `u_trap`** — pure
   pursuit ignores goal-distance entirely, just tracks the Hybrid A*
   path, so the detour is followed faithfully.

3. **`hybrid_astar_dwa` ALSO fails on `u_trap`** — this is the surprising
   result. The path-follow DWA kernel adds `w_terminal * dist_to_goal`
   at the end of each rollout. On open scenes this is fine; on the U
   trap, terminal pull dominates path-follow pull and drags the robot
   straight into the trap. The "global + local hybrid is paradigm-
   agnostic" claim from the existing local planner comparison only
   holds when the local cost does **not** include a direct goal-pull
   term. **Day 2 follow-up**: replace direct-goal terminal with
   path-end-waypoint terminal (or path-arc-remaining), so the terminal
   pull respects topology.

4. **`hybrid_astar_dwa` solves `s_corridor` while `hybrid_astar_mppi`
   does not** — S-corridor's detour is gradual (not a hard local minimum),
   so DWA's terminal pull is less destructive. MPPI's sampling noise
   appears to amplify the goal-pull problem (samples that veer toward
   goal get rewarded by terminal, samples that follow path get penalised
   by terminal even though they'd ultimately succeed).

5. **`dynamic_bottleneck` is too easy** — all planners solve it.
   The single dyn obstacle at `(25, 40, vy=-2.1)` reaches the gate at
   `t ≈ 7s`, the robot at `t ≈ 6s`; the reactive obstacle term lets
   every planner slow down and time the crossing. **Day 1 deferred**:
   strengthen this scene with multiple dyn obs or place the gate off
   the direct line so detour is also needed (currently this scene tests
   only timing, not topology).

6. **`dynamic_crossing_with_topology` separates pp from dwa/mppi
   differently** — pp follows the static detour blindly and hits the
   moving obstacle on the detour exit (13 collisions); dwa/mppi get
   trapped by their terminal pull as on `u_trap`. Neither solves this
   cell, motivating the dyn-aware-local + global combo. The cell will
   be useful once Day 2's path-aware terminal fix is in.

## Day 2 priority items

1. **Path-aware terminal in `hybrid_astar_{dwa,mppi}`** — replace
   `w_terminal * dist(robot_end, goal)` with a cost that respects the
   path. Candidates:
   - `w_terminal * dist(robot_end, path[nearest + look_ahead_2])`
     ("soft goal" further along the path than the heading lookahead)
   - `w_terminal * (remaining_path_arclength + dist_to_path_at_end)`
     ("how much path is left to cover")
   - Hybrid: use direct-goal terminal only when on the **last** path
     segment, path-end-waypoint terminal otherwise.

2. **Strengthen `dynamic_bottleneck`** — add a second crossing obstacle
   or offset the gate so reactive timing alone is not enough.

3. **Failure taxonomy** — extend the CSV summarizer to record:
   `success / collision / timeout / stuck / goal_miss` separately,
   plus `time_to_goal`, `min_clearance`, `path_length`,
   `oscillation_count`. This is the Day 2 metric expansion from the
   advisor roadmap.

## What this does to the existing main finding

The matrix in `docs/hybrid_astar_baseline.md` claims "global plan + local
reactive controller closes the paradigm gap, the pattern is paradigm-
agnostic". Day 1 surfaces a refinement:

> Global plan + local reactive controller closes the paradigm gap
> **only when the local cost does not double up on direct goal-pull**.
> Pure pursuit has no goal cost and works on topology. Path-follow DWA
> retains a direct goal terminal and fails on topology unless the
> terminal is made path-aware.

This is the kind of nuance the advisor was asking for. It also confirms
that the open-dynamic-only benchmark was overstating the closure result.
