# Pusher Face Switching on `box_detour_wall`

_Follow-up to [`box_detour_wall_2026-10-03.md`](box_detour_wall_2026-10-03.md),
which found that an obstacle-aware object path alone (`oi_path_mppi`) stops
where the path turns sideways, because the pusher never changes faces._

## Mechanism

`oi_face_mppi` is `oi_path_mppi` with a face-switching nominal seed
(`seed_face_switch_nominal`, `face_switch_target`):

1. Take the box's next step along the object path and pick the box face whose
   outward normal is most opposite to that direction.
2. If the pusher is outside that face, aim it at the contact point behind the
   box's next reference pose and advance the reference.
3. Otherwise walk the pusher around the box, keeping clear of it: to the
   corner of the pushing face when it is beside the box, or first to a corner
   of the opposite face when it is behind it. The object reference is held
   while the pusher repositions, and the rollout cost delays its reference by
   the same number of steps (`ref_delay`).

## Protocol

- Settings were selected on seeds 0-7 (both cells), then frozen:

  | seed blend | reference speed (m/s) | wall | open |
  |---:|---:|---:|---:|
  | 0.12 | 1.2 | 4/8 | 1/8 |
  | **0.12** | **0.6** | **7/8** | 2/8 |
  | 0.3 | 1.2 | 2/8 | 0/8 |
  | 0.3 | 0.6 | 4/8 | 0/8 |
  | 0.6 | 1.2 | 0/8 | 0/8 |
  | 0.6 | 0.6 | 0/8 | 2/8 |

- `oi_path_slow_mppi` uses the same 0.6 m/s reference without face switching,
  to separate the two changes; it scored 0/8 on the wall during tuning.
- Evaluation uses 30 seeds the tuning never saw (`--seed-offset 100`).

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_wall,box_detour_open \
  --planners mppi,diff_mppi_3,soppi_fast,oi_mppi,oi_path_mppi,oi_path_slow_mppi,oi_face_mppi \
  --k-values 256 --seed-count 30 --seed-offset 100 \
  --csv docs/results/box_detour_face_switch_2026-10-03.csv
```

## Held-out results (`K=256`, 30 paired seeds)

| Planner | Open: success | Open: final err (m) | Wall: success | Wall: final err (m) | Wall: mean steps |
|---|---:|---:|---:|---:|---:|
| `mppi` | 0/30 | 0.510 | 3/30 | 1.064 | 394 |
| `diff_mppi_3` | 0/30 | 0.462 | 4/30 | 0.930 | 381 |
| `soppi_fast` | 0/30 | 0.486 | 15/30 | 0.587 | 298 |
| `oi_mppi` | 6/30 | 0.119 | 0/30 | 1.313 | 420 |
| `oi_path_mppi` | 3/30 | 0.348 | 0/30 | 1.250 | 420 |
| `oi_path_slow_mppi` | 4/30 | 0.319 | 0/30 | 1.205 | 420 |
| `oi_face_mppi` | 7/30 | 0.393 | **27/30** | 0.337 | **158** |

Wilson 95% intervals on the wall: `oi_face_mppi` 0.74-0.97, `soppi_fast`
0.33-0.67. Paired exact McNemar tests on the wall cell:

| Comparison | only A succeeds | only B succeeds | p |
|---|---:|---:|---:|
| `oi_face_mppi` vs `soppi_fast` | 14 | 2 | 0.004 |
| `oi_face_mppi` vs `oi_path_slow_mppi` | 27 | 0 | 1.5e-8 |
| `oi_face_mppi` vs `mppi` | 24 | 0 | 1.2e-7 |

On the open cell `oi_face_mppi` and `oi_mppi` are indistinguishable (7 vs 6
discordant seeds, p = 1.0).

## Reading

- **Face switching is the mechanism.** With the same path and the same slower
  reference, removing face switching drops the wall cell from 27/30 to 0/30.
- **It is also faster.** Successful wall episodes take 158 steps on average,
  against 298 for `soppi_fast`, which reaches the goal by sliding the box
  along the wall.
- A trajectory trace (seed 7256) shows the intended sequence: push the bottom
  face up to the wall, walk round to the left face and push the box right past
  the wall's end, then return to the bottom face and push up to the goal.
- **The open cell is still hard.** Without the wall the remaining failures are
  in the final lateral alignment near the goal, where the path is a single
  straight segment and the seed never asks for a face change.

## Limitations

- One cell pair, one box geometry, `K=256`, the smooth-contact plant only
  (no `--true-plant hard` or MuJoCo transfer yet).
- Two settings were tuned (seed blend, reference speed), on separate seeds.
- The face choice assumes the box heading stays near the planned one; large
  rotations are not handled.
