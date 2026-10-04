# Aiming the Push From the Actual Box Removes the Stall Boost's Detour Cost

_Follow-up to [`box_detour_pusher_stall_2026-10-04.md`](box_detour_pusher_stall_2026-10-04.md).

`oi_face_rot_pstall_mppi` fixed the smooth-plant open turn but lost about 4 % of the smooth `box_detour_wall` episodes. Five gates on when to trust the seed could not separate the two stalls, because both are the same seed flip. This note fixes the flip itself._

## Cause

Logging the face-switching seed at every step of a lost wall episode (seed index 1122) shows a period-2 flip:

| Step parity | Seed's engagement test | First target |
|---|---|---:|
| even | pusher outside the push face's span (actual box) | slide to the face: (1.29, 1.65) |
| odd | pusher engaged | contact point behind the reference pose: (1.38, 1.12) |

The box (1.75, 1.74) has been pushed up against the underside of the wall, 0.54 m off the object path. Its reference pose (1.78, 1.20) stays on the path.

- **The mismatch.** The engagement test uses the actual box, but the push target is computed behind the reference pose, half a metre below the box. Each push step drags the pusher away from the face, the next step sends it back, and so on.
- **Without a boost,** MPPI's own samples eventually break the cycle.
- **With the 0.6 stall boost,** the cycle is enforced.

## Mechanism

`oi_push_actual_dist`: once the box is farther than this from its reference pose, the seed aims the first push (at the current step) from the actual box, offset by the reference's per-step advance, instead of from the reference pose. Closer in, the reference pose's offset is kept: it steers a slightly off-path box back onto the path.

`oi_face_rot_anchor_mppi` is `oi_face_rot_pstall_mppi` with `oi_push_actual_dist = 0.3`. Every earlier planner is unchanged; `--override-oi-push-actual-dist` sets the value for any face-switching planner.

## Selection

Development seeds only: the six development cells on seeds 0-29 (out of 180 per plant), and `box_detour_wall` on seeds 1100-1299 (out of 200):

| Change | smooth, 6 cells | hard, 6 cells | smooth wall | hard wall |
|---|---:|---:|---:|---:|
| `oi_face_rot_wide_mppi` | 171 | 175 | 198 | 199 |
| `oi_face_rot_pstall_mppi` | 178 | 175 | 188 | 197 |
| wide, always from the actual box | 166 | 174 | 200 | 200 |
| pstall, always from the actual box | 170 | 175 | 200 | 200 |
| wide, from the actual box beyond 0.15 m | 172 | 178 | 200 | 200 |
| pstall, from the actual box beyond 0.15 m | **180** | **179** | **200** | **200** |
| wide, from the actual box beyond 0.30 m | 172 | 179 | 200 | 200 |
| **pstall, from the actual box beyond 0.30 m** | **180** | **179** | **200** | **200** |
| wide, from the actual box beyond 0.45 m | 171 | 177 | 200 | 200 |
| pstall, from the actual box beyond 0.45 m | 178 | 178 | 200 | 200 |

- **Aiming from the actual box always** fixes the wall cell, but costs the smooth +0.9 rad turn (30 to 24/30): that turn relies on the reference pose's corrective offset.
- **With a threshold,** 0.15-0.30 m is a plateau; 0.3 m was taken.
- **The open turn still needs the stall boost.** Wide with the threshold stays at 22/30 there, so the stall boost and the push anchor are both kept.

## Evaluation (seeds 1300-1329, all eight cells)

| Plant | `oi_face_rot_wide_mppi` | `oi_face_rot_pstall_mppi` | `oi_face_rot_anchor_mppi` |
|---|---:|---:|---:|
| smooth | 229/240 | 229/240 | **240/240** |
| hard | 233/240 | 232/240 | **239/240** |

Per cell, `oi_face_rot_anchor_mppi` solves every episode on the smooth plant. Its single hard-plant miss is one `box_detour_turn` episode.

On the smooth open turn the three planners solve 21, 24 and 30 of 30.

## Detour cost (seeds 1400-1499, the four detour cells, 100 seeds each)

| Plant | wide | pstall | anchor |
|---|---:|---:|---:|
| smooth | 397/400 | 392/400 | **400/400** |
| hard | 399/400 | 400/400 | **400/400** |

## Paired tests over both fresh sets (1300-1329 and 1400-1499)

| `oi_face_rot_anchor_mppi` vs | smooth (only anchor / only other) | hard |
|---|---|---|
| `oi_face_rot_wide_mppi` | 14 / 0, p = 1e-4 | 8 / 1, p = 0.04 |
| `oi_face_rot_pstall_mppi` | 19 / 0, p = 4e-6 | 8 / 1, p = 0.04 |

All rows are collision-free.

## Reading

- **The stall boost's detour cost was a seed bug, not a property of boosting.** The push target and the engagement test disagreed about where the box is. With both on the actual box, the boost no longer reinforces a flip. The detour cells go to 400/400 on both plants, and the smooth wall cell to 200/200 on the development set that showed the 4 % loss.
- **`oi_face_rot_anchor_mppi` replaces `oi_face_rot_pstall_mppi` and `oi_face_rot_wide_mppi` as the planner to use** on the detour-and-turn tasks. On fresh seeds it solves 240/240 (smooth) and 239/240 (hard) across the eight cells. It solves strictly more episodes than either predecessor on both plants.
- **The open turn needed both pieces:** the stall boost to break the open-space stall, and the push anchor so the boost does not lock the pusher into a wall-side flip.

## Limitations

- One box, `K=256`; one new threshold, selected on development seeds that also include the cells it was meant to fix.
- The fix touches only the first step of the seed. Later steps of the seed still follow the reference pose on the path.

```bash
CELLS=box_detour_turn,box_open_turn,box_detour_turn90,box_detour_turn_neg,box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far
./bin/benchmark_diff_mppi_pushing_box --scenarios $CELLS \
  --planners oi_face_rot_wide_mppi,oi_face_rot_pstall_mppi,oi_face_rot_anchor_mppi \
  --k-values 256 --seed-count 30 --seed-offset 1300 [--true-plant hard] \
  --csv docs/results/box_detour_push_anchor_seed1300_{smooth,hard}_2026-10-04.csv
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far \
  --planners oi_face_rot_wide_mppi,oi_face_rot_pstall_mppi,oi_face_rot_anchor_mppi \
  --k-values 256 --seed-count 100 --seed-offset 1400 [--true-plant hard] \
  --csv docs/results/box_detour_push_anchor_detours_seed1400_{smooth,hard}_2026-10-04.csv
```
