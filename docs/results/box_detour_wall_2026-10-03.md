# `box_detour_wall` First Results

_Exploratory follow-up to the [`box_align_detour` audit](box_align_detour_audit_2026-10-03.md):
a detour cell whose wall actually constrains the box, plus an object-informed
variant that follows an obstacle-aware object path._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_wall,box_detour_open \
  --planners mppi,diff_mppi_3,soppi_fast,oi_mppi,oi_path_mppi \
  --k-values 256 \
  --seed-count 30 \
  --csv docs/results/box_detour_wall_2026-10-03.csv
```

## Cells and planner

- `box_detour_wall`: carry the box from (1.5, 1.2) to (2.3, 2.8), heading 0,
  gate 0.25 m / 0.35 rad, 420 steps. A wall as wide as the box
  (x 1.15-1.85, y 1.95-2.15) sits across the lower part of the straight line.
  It uses the new rectangle-overlap wall test (`BoxParams::obs_full_overlap`),
  so the wall is a rigid obstacle: the plant pushes the box back out, and the
  box can slide along it.
- `box_detour_open`: the same task without the wall, on the same seeds.
- `oi_path_mppi`: `oi_mppi` whose object reference follows a path from
  `plan_object_path()` (A* over box-centre positions clear of the wall
  footprint, shortcut to line-of-sight waypoints) instead of the straight line
  to the goal; the nominal seed pushes along the path tangent.

## Results (`K=256`, 30 paired seeds)

| Planner | Open: success | Open: final err (m) | Wall: success | Wall: final err (m) |
|---|---:|---:|---:|---:|
| `mppi` | 0/30 | 0.503 | 3/30 | 1.096 |
| `diff_mppi_3` | 0/30 | 0.429 | 0/30 | 1.024 |
| `soppi_fast` | 0/30 | 0.482 | **20/30** | 0.507 |
| `oi_mppi` | **12/30** | 0.116 | 0/30 | 1.317 |
| `oi_path_mppi` | 5/30 | 0.335 | 0/30 | 1.250 |

Control times are omitted: another benchmark shared the GPU during this run.

## Observations

Read from per-episode box trajectories (`--traj-dir`), not only the table:

1. **Without the wall the hard part is lateral alignment.** Sampling planners
   push the box mostly upward and stall about 0.5 m left of the goal; moving
   it right needs the pusher to walk around to the box's left face, which a
   16-step horizon does not find. Only the object-informed seeds, which push
   along the diagonal from the start, succeed.
2. **The wall blocks exactly that diagonal.** `oi_mppi` drives the box into the
   wall's underside and stays there (0/30).
3. **The wall also acts as a guide.** `soppi_fast` presses the box against the
   wall's underside, slides it right along the wall, and clears the wall's right
   end close to the goal's x, which turns the lateral-alignment problem into a
   sliding contact (20/30). `mppi` finds the same route on 3/30 seeds and
   otherwise stalls under the wall.
4. **An object-level path is not enough.** `oi_path_mppi` plans the correct
   route around the wall, but the box stops where the path turns sideways: the
   missing piece is pusher face switching (repositioning the pusher to the face
   that pushes along the next path segment), not the object route.

## Status

This cell does not isolate "detour planning". It mixes the lateral-alignment
difficulty of the open task with a wall the box can slide along. Treat it as an
exploratory, contact-rich obstacle cell. The next mechanism to try is a pusher
face-switching seed: route the pusher around the box to the face opposite the
next path tangent before the object reference advances.
