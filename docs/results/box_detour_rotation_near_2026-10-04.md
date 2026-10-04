# A Stronger Seed Near the Goal Fixes the Smooth-Plant Quarter Turn

_Follow-up to [`box_detour_rotation_2026-10-03.md`](box_detour_rotation_2026-10-03.md),
which left the smooth-plant quarter turn as the open weakness of
`oi_face_rot_mppi` (12/30 to 5/30 against `oi_face_auto_mppi`)._

## Failure

On seeds 0-7 the failing episodes end the same way as in the earlier note:

- **Box:** turned to within 0.03 rad of the goal heading, but parked 0.4 m short of the goal position.
- **Pusher:** hovers just outside the span of the face that would push the box to the goal. It stays there for 300+ steps and never engages.
- **Planner state:** the rotation phase is over (heading error below half the gate). The face-switching seed is already asking the pusher to slide to the face centre and push.
- **MPPI:** does not follow the seed. With the default 12 % blend, each step moves the sampling mean only 12 % of the way towards the seed. That is too little to pull it out of the stalled state.

This is the second hypothesis recorded in the earlier note (seed blend too weak), and it holds. The first (the task's heading cost) was not needed.

## Mechanism

`oi_near_seed_blend` sets the face-switching seed blend used when the box is within `oi_rot_radius` (0.8 m) of the goal; elsewhere `oi_seed_blend` (0.12) still applies.

`oi_face_rot_near_mppi` is `oi_face_rot_mppi` with `oi_near_seed_blend = 0.6`; nothing else changes. `oi_face_rot_mppi` itself is unchanged, so the numbers in the earlier notes still reproduce. `--override-oi-near-seed-blend` sets the value for any face-switching planner.

## Selection (seeds 0-7, never used below)

Successes over `box_detour_turn`, `box_open_turn`, `box_detour_turn90`, `box_detour_turn_neg` and `box_detour_wall`, out of 40 per plant:

| Change to `oi_face_rot_mppi` | smooth | hard |
|---|---:|---:|
| none (blend 0.12) | 31 | 37 |
| blend 0.25 everywhere | 25 | 38 |
| blend 0.40 everywhere | 23 | 39 |
| near-goal blend 0.25 | 35 | 39 |
| near-goal blend 0.40 | 35 | 39 |
| **near-goal blend 0.60** | **39** | **39** |
| near-goal blend 0.80 | 39 | 39 |
| near-goal blend 1.00 | 39 | 39 |

- **Raising the blend everywhere:** fixes the quarter turn (2/8 to 6/8 at 0.25) but breaks the other smooth-plant turns, which need MPPI's freedom on the way to the goal.
- **Raising it only near the goal:** plateaus from 0.6 on. The smallest value on the plateau was taken.

## Results (30 unseen seeds, offset 600, `K=256`, all rows collision-free)

Smooth plant:

| Planner | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `oi_face_auto_mppi` | 5 | 18 | 9 | 0 | 32/120 | 120/120 |
| `oi_face_rot_mppi` | 20 | 22 | 3 | 23 | 68/120 | 120/120 |
| `oi_face_rot_near_mppi` | **30** | **23** | **29** | **26** | **108/120** | 120/120 |

Hard plant:

| Planner | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `oi_face_auto_mppi` | 28 | 27 | 15 | 14 | 84/120 | 112/120 |
| `oi_face_rot_mppi` | 23 | 30 | 30 | 26 | 109/120 | 120/120 |
| `oi_face_rot_near_mppi` | **30** | **30** | **30** | **26** | **116/120** | **120/120** |

The four detour cells are `box_detour_wall`, `box_detour_open`, `box_detour_wall_left` and `box_detour_wall_far`. The last two are held-out geometries that were never used for tuning.

Paired exact McNemar tests, `oi_face_rot_near_mppi` vs `oi_face_rot_mppi` (seeds only near solves / only rot solves):

| Plant | turn | open turn | turn 90 | turn neg |
|---|---|---|---|---|
| smooth | 10 / 0, p = 0.002 | 1 / 0, p = 1 | 26 / 0, p = 3e-8 | 3 / 0, p = 0.25 |
| hard | 7 / 0, p = 0.016 | 0 / 0 | 0 / 0 | 0 / 0 |

No seed is solved by `oi_face_rot_mppi` and lost by `oi_face_rot_near_mppi` on any of the 16 cell/plant pairs.

## Reading

- **The smooth-plant quarter turn is solved:** 3/30 to 29/30 against the rotation phase alone, and 9/30 for `oi_face_auto_mppi`. The regression the earlier note reported is gone.
- **The +0.9 rad turn is solved on both plants:** 30/30 each. It had the same stall, less often.
- **The remaining failures** are the open turn on the smooth plant (23/30) and the reverse turn on both plants (26/30). Neither moved much, so they are a different failure.
- **Detour cells:** stay at 120/120 on both plants.

## Limitations

- One box, `K=256`, one tuned parameter (the near-goal blend), selected on 8 seeds.
- The near-goal region reuses `oi_rot_radius`; the two were not tuned jointly.

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_turn,box_open_turn,box_detour_turn90,box_detour_turn_neg,box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far \
  --planners oi_face_auto_mppi,oi_face_rot_mppi,oi_face_rot_near_mppi \
  --k-values 256 --seed-count 30 --seed-offset 600 [--true-plant hard] \
  --csv docs/results/box_detour_rotation_near_{smooth,hard}_2026-10-04.csv
```
