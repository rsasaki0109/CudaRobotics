# Safe Face Approach and Automatic Path Choice

_Follow-up to [`box_detour_axis_path_2026-10-03.md`](box_detour_axis_path_2026-10-03.md),
which left one open question: why the axis-aligned path fails the two low-wall
cells on the smooth plant (0/30 and 2/30)._

## The bug

In the stuck episodes the pusher sits beside the box's lower corner. Replaying
`face_switch_target` at that state: the pusher is past the pushing face's
plane by half a pusher radius, so it counts as engaged, and its target becomes
the middle of the face. Moving there, it passes the box corner closer than its
own radius, clips the corner and rotates the box toward the wall. Every
rollout that does this pays the wall barrier, so MPPI keeps the pusher where
it is. Axis-aligned paths turn 90 degrees right next to the wall, so they hit
this case every time.

**Fix (`safe_slide`):** the pusher only engages once it is within the face's
span. If it is past the face plane but beyond the corner, it first slides
toward the face centre with a full radius of clearance. The flag is off by
default, so all earlier planners reproduce (8/8 spot-checked against the
axis-path CSV).

On seeds 0-7 the fix moves `oi_face_axis_mppi` on the smooth plant from 0/8 to
8/8 on both low-wall cells.

## Evaluation 1: safe slide (seeds 300-329, unseen)

| Planner | wall | open | left | far | total |
|---|---:|---:|---:|---:|---:|
| smooth: `soppi_fast` | 17 | 0 | 18 | 21 | 56/120 |
| smooth: `oi_face_track_mppi` | 18 | **28** | 21 | 23 | 90/120 |
| smooth: `oi_face_axis_mppi` | 3 | 29 | 1 | 30 | 63/120 |
| smooth: `oi_face_track_safe_mppi` | 6 | 29 | 2 | 5 | 42/120 |
| smooth: `oi_face_axis_safe_mppi` | **30** | 10 | **30** | **30** | **100/120** |
| hard: `soppi_fast` | 2 | 5 | 4 | 7 | 18/120 |
| hard: `oi_face_track_mppi` | 12 | 25 | 14 | 4 | 55/120 |
| hard: `oi_face_axis_mppi` | 24 | 9 | 25 | 26 | 84/120 |
| hard: `oi_face_track_safe_mppi` | 18 | **30** | 21 | 5 | 74/120 |
| hard: `oi_face_axis_safe_mppi` | **28** | 16 | **27** | **27** | **98/120** |

With the fix, the axis-aligned path solves the three obstacle cells on both
plants. The open cell, where the axis-aligned path is an L instead of a
straight line, is its remaining weakness.

## Automatic path choice

`oi_face_auto_mppi` uses the axis-aligned path only when the straight line to
the goal is blocked (checked against the box footprint), and the straight path
with tracking otherwise. Nothing was tuned for it; it combines two components
already evaluated.

## Evaluation 2: automatic choice (seeds 400-429, unseen)

| Planner | wall | open | left | far | total |
|---|---:|---:|---:|---:|---:|
| smooth: `soppi_fast` | 18 | 0 | 15 | 19 | 52/120 |
| smooth: `oi_face_track_mppi` | 18 | 26 | 18 | 26 | 88/120 |
| smooth: `oi_face_axis_safe_mppi` | 29 | 4 | 30 | 29 | 92/120 |
| smooth: **`oi_face_auto_mppi`** | **29** | **30** | **30** | **29** | **118/120** |
| hard: `soppi_fast` | 5 | 3 | 5 | 5 | 18/120 |
| hard: `oi_face_track_safe_mppi` | 22 | **30** | 18 | 7 | 77/120 |
| hard: `oi_face_axis_safe_mppi` | 27 | 17 | 29 | 27 | 100/120 |
| hard: **`oi_face_auto_mppi`** | **27** | **30** | **29** | **27** | **113/120** |

Every row is collision-free. Paired exact McNemar tests,
`oi_face_auto_mppi` vs `soppi_fast` (only auto / only soppi):

| Plant | wall | open | left | far |
|---|---|---|---|---|
| smooth | 12 / 1, p = 0.003 | 30 / 0, p = 1.9e-9 | 15 / 0, p = 6.1e-5 | 11 / 1, p = 0.006 |
| hard | 22 / 0, p = 4.8e-7 | 27 / 0, p = 1.5e-8 | 24 / 0, p = 1.2e-7 | 23 / 1, p = 3.0e-6 |

Against the best single-mode planner per plant, auto never loses a seed: it
equals `oi_face_axis_safe_mppi` on the obstacle cells (same path) and gains on
the open cell (smooth 26/0, hard 13/0).

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far \
  --planners soppi_fast,oi_face_track_mppi,oi_face_axis_safe_mppi,oi_face_auto_mppi \
  --k-values 256 --seed-count 30 --seed-offset 400 [--true-plant hard] \
  --csv docs/results/box_detour_auto_path_{smooth,hard}_2026-10-03.csv
```

## Summary of the detour line

| Step | Finding |
|---|---|
| [audit](box_align_detour_audit_2026-10-03.md) | `box_align_detour` measured a position plateau; its wall never bound |
| [binding wall](box_detour_wall_2026-10-03.md) | object-level paths alone fail; the pusher never changes faces |
| [face switching](box_detour_face_switch_2026-10-03.md) | 27/30 on the wall cell, smooth plant |
| [tracking](box_detour_face_track_2026-10-03.md) | route around the actual box; fixes the open cell, transfers to the hard plant |
| [held-out geometries](box_detour_generalization_2026-10-03.md) | partial transfer on the hard plant |
| [axis-aligned paths](box_detour_axis_path_2026-10-03.md) | under friction, plan paths one face can push |
| this note | a corner-clipping bug, then automatic path choice: 118/120 smooth, 113/120 hard |

## Limitations

- Four cells, one box geometry, `K=256`, one wall per cell.
- The hard plant's rigid wall is a projection with velocity removal.
- The box heading is assumed to stay near its start value; tasks that need
  large rotations are not covered by the face-switching seed.
