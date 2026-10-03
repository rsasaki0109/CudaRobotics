# Face Switching with Tracking, and the Hard Plant

_Follow-up to [`box_detour_face_switch_2026-10-03.md`](box_detour_face_switch_2026-10-03.md).
`oi_face_mppi` solved the wall cell (27/30) but not the open one (7/30), where
the failures are the final lateral alignment. This note adds a tracking
variant and evaluates both on the hard-contact plant._

## What changed

`oi_face_track_mppi` = `oi_face_mppi` plus two changes to the face-switching
seed, at the current step only:

1. **Route around the box where it is.** The pusher's detour waypoints are
   computed around the actual box pose, not its projection on the path. When
   the box has drifted sideways, the projection is a phantom box and the
   pusher walks into the real one.
2. **On the last path segment, aim from the box.** The push direction is box
   to next reference point instead of the path tangent, so lateral drift
   selects a side face. Earlier segments keep the tangent, so path corners do
   not flip the face choice.

Separately, rectangle-overlap walls (`obs_full_overlap`, i.e. only the new
`box_detour_*` cells) are now rigid in the hard plant as well: the box is
pushed out and its velocity into the wall is removed. Before this, the hard
plant had no wall physics, so a controller whose model lets the box slide along
the wall scored 0/30 on collisions (all planners did). Legacy corner-test walls
are unchanged, so published hard-plant rows reproduce.

## Search on tuning seeds (0-7)

Every variant tried, with the same blend 0.12 and reference speed 0.6 m/s:

| Variant | open | wall |
|---|---:|---:|
| `oi_face_mppi` (baseline, #244) | 2/8 | 7/8 |
| route around actual box + aim from box everywhere | 7/8 | 0/8 |
| cross-track correction, gain 1 / 2 / 4 / 8 per m | 2 / 2 / 2 / 0 of 8 | 2 / 3 / 5 / 0 of 8 |
| aim from box on last segment only | 0/8 | 7/8 |
| route around actual box only | 3/8 | 3/8 |
| **route around actual box + aim from box on last segment** | **7/8** | **5/8** |

The last row has the highest total over both cells (12/16 against 9/16 for the
baseline) and became `oi_face_track_mppi`. Evaluation below uses 30 seeds the
search never saw (`--seed-offset 100`).

## Held-out results (`K=256`, 30 paired seeds)

Smooth plant (the controller's own model):

| Planner | open | wall | wall: mean steps |
|---|---:|---:|---:|
| `soppi_fast` | 0/30 | 15/30 | 298 |
| `oi_mppi` | 6/30 | 0/30 | 420 |
| `oi_face_mppi` | 7/30 | **27/30** | 158 |
| `oi_face_track_mppi` | **29/30** | 15/30 | 325 |

Hard plant (rigid box, Coulomb friction, rigid wall):

| Planner | open | wall | wall: mean steps |
|---|---:|---:|---:|
| `mppi` | 3/30 | 5/30 | 373 |
| `soppi_fast` | 3/30 | 5/30 | 375 |
| `oi_mppi` | 3/30 | 0/30 | 420 |
| `oi_path_slow_mppi` | 3/30 | 11/30 | 330 |
| `oi_face_mppi` | 1/30 | 6/30 | 349 |
| `oi_face_track_mppi` | **21/30** | **18/30** | 255 |

No planner touches the wall on either plant (all wall rows collision-free).

Paired exact McNemar tests (only A succeeds / only B succeeds):

| Plant | Cell | A vs B | A / B | p |
|---|---|---|---:|---:|
| smooth | open | `oi_face_track_mppi` vs `oi_face_mppi` | 22 / 0 | 4.8e-7 |
| smooth | open | `oi_face_track_mppi` vs `oi_mppi` | 23 / 0 | 2.4e-7 |
| smooth | wall | `oi_face_track_mppi` vs `oi_face_mppi` | 2 / 14 | 0.004 |
| smooth | wall | `oi_face_track_mppi` vs `soppi_fast` | 7 / 7 | 1.0 |
| hard | wall | `oi_face_track_mppi` vs `soppi_fast` | 14 / 1 | 0.001 |
| hard | open | `oi_face_track_mppi` vs `oi_mppi` | 19 / 1 | 4e-5 |
| hard | wall | `oi_face_mppi` vs `soppi_fast` | 4 / 3 | 1.0 |

## Reading

- **Tracking fixes the open cell.** 29/30 against 7/30 on the smooth plant;
  the remaining open-cell failures in #244 were the pusher walking around a
  box that was no longer where the path said.
- **It costs the smooth wall cell.** 15/30 against 27/30. There is no single
  best variant on the smooth plant: `oi_face_mppi` for the wall, tracking for
  the open task.
- **Tracking is what transfers.** On the hard plant `oi_face_track_mppi` is
  best on both cells (21/30 and 18/30). `oi_face_mppi`'s wall result does not
  survive the plant change (6/30, indistinguishable from `soppi_fast`).
  Routing around the true box pose is what keeps working when the box does not
  move the way the smooth model predicts.

## Limitations

- One cell pair and one box geometry, `K=256`.
- The variant was chosen from six tried on eight seeds; held-out evaluation
  guards the selection but not the cell design.
- The hard plant's rigid wall is a simple projection with velocity removal,
  not a contact solve.
