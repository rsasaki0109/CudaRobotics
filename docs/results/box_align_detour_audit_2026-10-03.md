# `box_align_detour` Audit

_Why every planner scores 0/30 on `box_align_detour`: the wall never constrains
the executed motion, and the cell inherits a position gate that its
obstacle-free parent already misses._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_align_detour,box_align_detour_nowall,box_align_detour_gate \
  --planners mppi,diff_mppi_3,soppi_fast,oi_mppi \
  --k-values 256 \
  --seed-count 30 \
  --csv docs/results/box_align_detour_audit_2026-10-03.csv
```

The two companions are diagnostic cells added for this audit and run on
`box_align_detour`'s seeds, so every row below is paired by seed:

- `box_align_detour_nowall`: the same cell with the wall removed.
- `box_align_detour_gate`: the same cell (wall kept) with the position gate
  widened from 0.22 m to `box_align_strict`'s 0.28 m.

## Results (`K=256`, 30 paired seeds)

| Scenario | Planner | Success | Reached goal | Final pos err (m) | Final ang err (rad) | Wall collisions |
|---|---|---:|---:|---:|---:|---:|
| `box_align_detour` | `mppi` | 0/30 | 0/30 | 0.294 | 0.031 | 0 |
| `box_align_detour` | `diff_mppi_3` | 1/30 | 1/30 | 0.257 | 0.032 | 0 |
| `box_align_detour` | `soppi_fast` | 0/30 | 0/30 | 0.282 | 0.032 | 0 |
| `box_align_detour` | `oi_mppi` | 0/30 | 0/30 | 0.263 | 0.371 | 0 |
| `box_align_detour_nowall` | `mppi` | 0/30 | 0/30 | 0.293 | 0.031 | 0 |
| `box_align_detour_nowall` | `diff_mppi_3` | 0/30 | 0/30 | 0.255 | 0.032 | 0 |
| `box_align_detour_nowall` | `soppi_fast` | 0/30 | 0/30 | 0.281 | 0.032 | 0 |
| `box_align_detour_nowall` | `oi_mppi` | 0/30 | 0/30 | 0.264 | 0.371 | 0 |
| `box_align_detour_gate` | `mppi` | 3/30 | 3/30 | 0.295 | 0.032 | 0 |
| `box_align_detour_gate` | `diff_mppi_3` | 30/30 | 30/30 | 0.276 | 0.061 | 0 |
| `box_align_detour_gate` | `soppi_fast` | 12/30 | 12/30 | 0.286 | 0.035 | 0 |
| `box_align_detour_gate` | `oi_mppi` | 30/30 | 30/30 | 0.239 | 0.220 | 0 |

Final errors for `_gate` rows are measured when the episode ends, i.e. at the
first step inside the gate.

## Findings

1. **The wall is not what fails.** No episode in any cell touches the wall
   (0 collisions in 360 episodes), and removing it changes success by at most
   one seed per planner. The wall does enter the sampled rollouts' barrier cost
   (paired trajectories differ slightly, and the benchmark is deterministic run
   to run), but it never binds the executed motion.
2. **Why the wall is inert.** `box_obstacle_penetration_f` tests only the four
   box corners against the wall AABB. The wall (0.24 m x 0.16 m) is smaller
   than the box (0.70 m x 0.36 m), so a box sliding up the direct lane at
   heading 0 straddles the wall with all corners outside it.
3. **The position gate is.** Every planner, including those that finish the
   rotation (final heading error about 0.03 rad), stalls at 0.25-0.30 m from
   the goal, just outside the 0.22 m gate inherited from `box_align`. That is
   the same plateau the obstacle-free parent shows, and the reason
   `box_align_strict` widened its gate to 0.28 m. With that gate the wall-kept
   cell becomes 30/30 for `diff_mppi_3` and `oi_mppi`, 12/30 for `soppi_fast`,
   and 3/30 for `mppi`.

## Consequence

Published `box_align_detour` numbers (including the frozen 0/30 rows in the
contact paper evidence) remain correct as measurements, but the cell measures
the 0.22 m position plateau, not obstacle avoidance. A detour benchmark needs
an obstacle that binds the executed motion (full rectangle-rectangle overlap,
or a wall wider than the box) and a gate the obstacle-free task can meet.
