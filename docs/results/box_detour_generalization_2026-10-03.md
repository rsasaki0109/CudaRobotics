# Face Switching on Held-Out Geometries

_Follow-up to [`box_detour_face_track_2026-10-03.md`](box_detour_face_track_2026-10-03.md).
Every planner setting so far was chosen on `box_detour_wall` / `box_detour_open`.
These two cells were added afterwards and never used for tuning._

## Cells

- `box_detour_wall_left`: the mirror image; the goal moves to x = 0.7, so the
  straight line clips the wall's left end instead of its right end.
- `box_detour_wall_far`: the wall moves up to y 2.30-2.50 and the goal to
  (2.4, 3.2).

In both, the straight start-to-goal line is blocked by the wall (checked
against the box footprint).

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_wall_left,box_detour_wall_far \
  --planners mppi,soppi_fast,oi_mppi,oi_face_mppi,oi_face_track_mppi \
  --k-values 256 --seed-count 30 --seed-offset 100 \
  [--true-plant hard] \
  --csv docs/results/box_detour_generalization_{smooth,hard}_2026-10-03.csv
```

## Results (`K=256`, 30 paired seeds, all rows collision-free)

| Planner | left, smooth | far, smooth | left, hard | far, hard |
|---|---:|---:|---:|---:|
| `mppi` | 7/30 | 6/30 | 5/30 | 3/30 |
| `soppi_fast` | 13/30 | 24/30 | 4/30 | **8/30** |
| `oi_mppi` | 0/30 | 0/30 | 0/30 | 0/30 |
| `oi_face_mppi` | **25/30** | **28/30** | 5/30 | 0/30 |
| `oi_face_track_mppi` | 17/30 | 26/30 | **15/30** | 4/30 |

Paired exact McNemar tests for `oi_face_track_mppi` (only it succeeds / only
the other succeeds):

| Plant | Cell | vs `soppi_fast` | vs `oi_face_mppi` |
|---|---|---|---|
| smooth | left | 10 / 6, p = 0.45 | 5 / 13, p = 0.10 |
| smooth | far | 6 / 4, p = 0.75 | 1 / 3, p = 0.63 |
| hard | left | 14 / 3, p = 0.013 | 12 / 2, p = 0.013 |
| hard | far | 2 / 6, p = 0.29 | 4 / 0, p = 0.13 |

## Reading

- **On the smooth plant face switching generalizes.** Both face-switching
  planners solve most episodes on both new geometries, and the straight-line
  object reference (`oi_mppi`) fails all of them. `soppi_fast` is competitive
  on the far-wall cell (24/30), where sliding along the wall happens to lead
  toward the goal.
- **On the hard plant the transfer is partial.** Tracking carries over to the
  mirrored cell (15/30, ahead of both alternatives), but not to the far-wall
  cell, where every planner is at or below 8/30 and the differences are not
  significant.
- The earlier conclusion holds in weakened form: routing around the true box
  pose is what helps under model mismatch, but it is not sufficient for every
  geometry.

## Limitations

- Two extra geometries, one box, `K=256`.
- The hard plant's rigid wall is a projection with velocity removal.
