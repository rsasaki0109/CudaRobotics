# The Remaining Turn Failures: a Wider Rotation Radius, and a Stall Boost That Costs Detours

_Follow-up to [`box_detour_rotation_near_2026-10-04.md`](box_detour_rotation_near_2026-10-04.md).
`oi_face_rot_near_mppi` left two turn cells short on the smooth plant: the open turn (23/30) and the reverse turn (26/30)._

## Failures (seeds 0-29)

`oi_face_rot_near_mppi` solves 22/30 of each cell on the smooth plant (hard: 30/30 each). The 16 failing episodes come in two shapes.

- **Reverse turn (`box_detour_turn_neg`).** The box stops 0.81-0.88 m from the goal, just outside the 0.8 m rotation radius. The pusher sits on the goal side of the box and keeps nudging it, so the box drifts in and out of the radius.
  - Inside the radius, the seed's target face (rotation) and its blend (0.6) differ from outside (translation, 0.12). The seed flips between the two and the box never leaves the band.
- **Open turn (`box_open_turn`).** The box stops 1.1 m from the goal and does not move for the rest of the episode. The pusher oscillates (about a 40-step period) at the front corner of the box, on the goal side.
  - This is the same stall as the quarter turn in the previous note: the seed asks the pusher to walk around to the back face, but at the 12 % blend MPPI does not follow. Here it happens outside the near-goal region.

## Mechanisms

- **Wider rotation radius:** `oi_rot_radius` 0.8 to 1.0 m. The near-goal region (rotation phase and 0.6 blend) then covers the band where the reverse turn stalled.
- **Stall boost:** `oi_stall_steps` / `oi_stall_blend`. Once the box has moved less than 0.01 m and 0.02 rad for `oi_stall_steps` control steps, the face-switching seed is blended at `oi_stall_blend`. It is off by default; `--override-oi-stall-steps` and `--override-oi-stall-blend` set it.

Two new variants; every earlier planner is unchanged:

| Planner | = `oi_face_rot_near_mppi` plus |
|---|---|
| `oi_face_rot_wide_mppi` | rotation radius 1.0 m |
| `oi_face_rot_stall_mppi` | rotation radius 1.0 m, stall boost (20 steps, 0.6) |

## Selection (seeds 0-29, six cells)

The cells are the four turn cells plus `box_detour_wall` and `box_detour_open`; the two held-out geometries were not used. Successes out of 180 per plant:

| Change to `oi_face_rot_near_mppi` | smooth | hard |
|---|---:|---:|
| none | 163 | 175 |
| stall boost (20 steps, 0.6) | 168 | 175 |
| radius 1.0 | 171 | 175 |
| radius 1.0 + stall boost | **174** | 175 |
| radius 1.2 + stall boost | 166 | 176 |

Stall thresholds of 10, 20 and 40 steps were tried on the two failing cells; 20 was best (53/60, against 44 for 10 and 51 for 40). Blends of 0.4-1.0 at 20 steps gave 49-54/60.

## First evaluation (seeds 700-729): the stall boost costs detours

The combination was evaluated on unseen seeds and then broken down into its two parts on the same seeds (both parts run as overrides of `oi_face_rot_near_mppi`):

| Planner (smooth plant) | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `oi_face_rot_near_mppi` | 30 | 23 | 30 | 21 | 104/120 | **120/120** |
| + radius 1.0 | 30 | 24 | 30 | 26 | 110/120 | **120/120** |
| + stall boost | 28 | 29 | 30 | 24 | 111/120 | 115/120 |
| + both (`oi_face_rot_stall_mppi`) | 28 | **30** | 30 | **30** | **118/120** | 115/120 |

| Planner (hard plant) | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `oi_face_rot_near_mppi` | 30 | 29 | 30 | 27 | 116/120 | **120/120** |
| + radius 1.0 | 30 | 28 | 30 | 29 | 117/120 | **120/120** |
| + stall boost | 30 | 29 | 30 | 30 | 119/120 | 117/120 |
| + both (`oi_face_rot_stall_mppi`) | 30 | 28 | 30 | 30 | 118/120 | 117/120 |

**Stall boost.** It fixed the turns it was built for: on the smooth plant the open turn goes 23 to 30 and the reverse turn 21 to 30. But it loses detour episodes on both plants and wins none: paired against `oi_face_rot_near_mppi`, 0 / 8 over the detour cells of both plants, exact p = 0.008 (the same with and without the radius change).

- On a detour the box also stands still while the pusher walks around to another face.
- Boosting the seed there takes the walk away from MPPI and costs episodes.

**Radius 1.0 on its own.** It lost no detour episode. Against `oi_face_rot_near_mppi` it solves 6 seeds the other does not, and loses none, on the smooth plant (hard: 3 / 2).

Because the radius-only planner was picked after seeing these seeds, it was confirmed on a fresh set before being kept.

## Confirmation (seeds 800-829)

The comparison was fixed in advance as `oi_face_rot_wide_mppi` vs `oi_face_rot_near_mppi`.

| Plant | Planner | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---|---:|---:|---:|---:|---:|---:|
| smooth | `oi_face_rot_mppi` | 22 | 22 | 6 | 20 | 70/120 | 119/120 |
| smooth | `oi_face_rot_near_mppi` | 30 | 22 | 28 | 23 | 103/120 | 119/120 |
| smooth | `oi_face_rot_wide_mppi` | 30 | 23 | **30** | **30** | 113/120 | **119/120** |
| smooth | `oi_face_rot_stall_mppi` | 30 | **27** | 30 | 29 | **116/120** | 114/120 |
| hard | `oi_face_rot_mppi` | 29 | 28 | 28 | 28 | 113/120 | 119/120 |
| hard | `oi_face_rot_near_mppi` | 30 | 28 | 28 | 28 | 114/120 | 119/120 |
| hard | `oi_face_rot_wide_mppi` | 30 | 28 | 28 | 29 | 115/120 | **119/120** |
| hard | `oi_face_rot_stall_mppi` | 30 | 29 | 28 | 30 | **117/120** | 116/120 |

`oi_face_rot_wide_mppi` vs `oi_face_rot_near_mppi`, paired:

| Plant | Seeds only wide solves / only near solves | Notes |
|---|---|---|
| smooth | 11 / 0 | reverse turn 7 / 0, p = 0.016 |
| hard | 2 / 1 | — |

No detour episode differs between the two on either plant. The stall boost again loses detours: 0 / 5 against `oi_face_rot_wide_mppi` on the smooth plant (wall and left-wall cells) and 0 / 3 on the hard plant.

All rows in every evaluation are collision-free.

## Reading

- **`oi_face_rot_wide_mppi` is the default to use.** Widening the near-goal region to 1.0 m removes the reverse-turn stall: smooth plant 23 to 30, on seeds that were not used to choose it. It does not cost a detour episode.
- **The smooth-plant open turn is still open** at 23/30.
  - The stall boost fixes it (27-30/30), but at the price of detour episodes on both plants.
  - A stall detector that cannot tell "stuck" from "walking to another face" is the wrong trigger. A better one would condition on the pusher, for example not making progress around the box, rather than on the box alone.
- **`oi_face_rot_stall_mppi` is kept** so this trade-off can be reproduced. It is not recommended.

## Limitations

- One box, `K=256`; two tuned parameters (radius, stall threshold), selected on 30 seeds.
- The radius-only planner was chosen after the seed-700 breakdown; the seed-800 run is the confirmation.

```bash
CELLS=box_detour_turn,box_open_turn,box_detour_turn90,box_detour_turn_neg,box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far
# first evaluation and its breakdown
./bin/benchmark_diff_mppi_pushing_box --scenarios $CELLS \
  --planners oi_face_rot_mppi,oi_face_rot_near_mppi,oi_face_rot_stall_mppi \
  --k-values 256 --seed-count 30 --seed-offset 700 [--true-plant hard] \
  --csv docs/results/box_detour_turn_residuals_seed700_{smooth,hard}_2026-10-04.csv
./bin/benchmark_diff_mppi_pushing_box --scenarios $CELLS --planners oi_face_rot_near_mppi \
  --k-values 256 --seed-count 30 --seed-offset 700 [--true-plant hard] \
  --override-oi-rot-radius 1.0 \
  --csv docs/results/box_detour_turn_residuals_seed700_radius_{smooth,hard}_2026-10-04.csv
./bin/benchmark_diff_mppi_pushing_box --scenarios $CELLS --planners oi_face_rot_near_mppi \
  --k-values 256 --seed-count 30 --seed-offset 700 [--true-plant hard] \
  --override-oi-stall-steps 20 --override-oi-stall-blend 0.6 \
  --csv docs/results/box_detour_turn_residuals_seed700_stall_{smooth,hard}_2026-10-04.csv
# confirmation
./bin/benchmark_diff_mppi_pushing_box --scenarios $CELLS \
  --planners oi_face_rot_mppi,oi_face_rot_near_mppi,oi_face_rot_wide_mppi,oi_face_rot_stall_mppi \
  --k-values 256 --seed-count 30 --seed-offset 800 [--true-plant hard] \
  --csv docs/results/box_detour_turn_residuals_seed800_{smooth,hard}_2026-10-04.csv
```
