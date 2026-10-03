# Rotation Phase for the Face-Switching Planner

_Follow-up to [`box_detour_safe_slide_2026-10-03.md`](box_detour_safe_slide_2026-10-03.md).
`oi_face_auto_mppi` pushes face centres, so it was built for tasks where the
box heading barely changes. This note adds detour tasks that end with a
reorientation._

## Cells

All share the `box_detour_wall` layout (box from (1.5, 1.2) to (2.3, 2.8),
wall across the straight line), with a final heading goal and a 0.25 rad gate:

| Cell | goal heading |
|---|---:|
| `box_detour_turn` | +0.9 rad |
| `box_open_turn` | +0.9 rad, no wall |
| `box_detour_turn90` | +1.571 rad |
| `box_detour_turn_neg` | -1.2 rad |

## Mechanism

`oi_face_rot_mppi` = `oi_face_auto_mppi` plus a rotation phase in the
face-switching seed (`seed_rotation_target`). Within `oi_rot_radius` of the
goal, with the heading error above half the gate, the pusher targets a point
near one end of a box face, so the push turns the box the right way. Of the
four faces it prefers the one whose push also moves the box toward the goal,
then the one nearest the pusher, and it walks to the contact point with the
same safe approach as the translation phase.

`face_switch_target` was split into `face_route_target`, which takes the face
and the contact offset explicitly; the translation behaviour is unchanged
(8/8 spot-checked against the auto-path CSV).

The radius was chosen on seeds 0-7 over the four turn cells plus
`box_detour_wall`, totalled over both plants: 0.3 m 56/80, 0.45 m 61/80,
**0.8 m 68/80**.

## Results (30 unseen seeds, offset 500, `K=256`, all rows collision-free)

Smooth plant:

| Planner | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `diff_mppi_3` | 6 | 0 | 0 | 0 | 6/120 | 11/120 |
| `soppi_fast` | 2 | 0 | 0 | 0 | 2/120 | 60/120 |
| `oi_face_auto_mppi` | 6 | 19 | **12** | 0 | 37/120 | 119/120 |
| `oi_face_rot_mppi` | **21** | **20** | 5 | **21** | **67/120** | 119/120 |

Hard plant:

| Planner | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `diff_mppi_3` | 0 | 0 | 0 | 0 | 0/120 | 33/120 |
| `soppi_fast` | 0 | 0 | 0 | 0 | 0/120 | 19/120 |
| `oi_face_auto_mppi` | **29** | 25 | 11 | 12 | 77/120 | 115/120 |
| `oi_face_rot_mppi` | 26 | **30** | **28** | **27** | **111/120** | **120/120** |

The four detour cells are `box_detour_wall`, `box_detour_open`,
`box_detour_wall_left`, `box_detour_wall_far`.

Paired exact McNemar tests, `oi_face_rot_mppi` vs `oi_face_auto_mppi` (only
rot / only auto):

| Plant | turn | open turn | turn 90 | turn neg |
|---|---|---|---|---|
| smooth | 15 / 0, p = 6e-5 | 1 / 0, p = 1 | 1 / 8, p = 0.039 | 21 / 0, p = 1e-6 |
| hard | 1 / 4, p = 0.38 | 5 / 0, p = 0.06 | 19 / 2, p = 2e-4 | 15 / 0, p = 6e-5 |

## Reading

- **On the hard plant the rotation phase nearly solves reorientation**: 111/120
  over the turn cells (auto 77, sampling baselines 0), with the quarter turn
  going from 11/30 to 28/30, and no loss on the detour cells.
- **On the smooth plant it helps two of three turns** (+0.9 and -1.2 rad) but
  hurts the quarter turn (12/30 to 5/30, p = 0.039). Under the smooth contact
  model an off-centre push near the goal also slides the box, and the 0.8 m
  radius, chosen on the totals, starts rotating early enough to push it off
  the gate on the longest turn. The quarter turn on the smooth plant is the
  open weakness.
- Sampling baselines (`diff_mppi_3`, `soppi_fast`) solve almost no turn
  episodes on either plant.

## Limitations

- Four turn cells, one box, `K=256`; one tuned parameter (radius).
- Rotation is attempted only near the goal; turns that must happen mid-path
  (for example to fit through a gap) are not covered.
