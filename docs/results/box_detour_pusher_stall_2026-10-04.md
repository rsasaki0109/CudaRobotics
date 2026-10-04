# A Stall Boost That Watches the Pusher

_Follow-up to [`box_detour_turn_residuals_2026-10-04.md`](box_detour_turn_residuals_2026-10-04.md)._

That note left the smooth-plant open turn at 23/30 with `oi_face_rot_wide_mppi`. Its stall boost fixed the open turn, but it lost detour episodes: it fired whenever the box stood still, and the box also stands still while the pusher walks to another face.

## Can the pusher tell the two apart? (seeds 0-29, offline)

On the `oi_face_rot_wide_mppi` trajectories of the six development cells (360 episodes, 14 failures), each candidate rule was checked offline. A rule boosts once the box has been still for S steps *and* the pusher has moved less than D over the last W steps.

| Rule | fires in failures | fires in successful episodes |
|---|---:|---:|
| box still 20 steps (the old boost) | 14/14 | 33/346 |
| box still 20, pusher < 0.10 m over 20 steps | 14/14 | 14/346 |
| box still 40 | 14/14 | 10/346 |
| box still 40, pusher < 0.10 m over 20 steps | 14/14 | 9/346 |

Every rule fires in every failure. Waiting longer and requiring the pusher to stay put both cut the firings in episodes that succeed without a boost.

## Selection (seeds 0-29, six cells, applied to `oi_face_rot_wide_mppi`)

Successes out of 180 per plant:

| Stall rule | smooth | hard |
|---|---:|---:|
| none (`oi_face_rot_wide_mppi`) | 171 | 175 |
| box still 20 (`oi_face_rot_stall_mppi`) | 174 | 175 |
| box still 20, pusher < 0.10 m / 20 steps | 176 | 175 |
| box still 40 | 176 | 175 |
| **box still 40, pusher < 0.10 m / 20 steps** | **178** | 175 |

`oi_face_rot_pstall_mppi` is `oi_face_rot_wide_mppi` with the last rule (blend 0.6). The pusher condition is set with `oi_stall_pusher_dist` / `oi_stall_pusher_window`. While the box is still, the pusher's world displacement is its displacement in the box frame.

## Evaluation (seeds 900-929, all eight cells)

Smooth plant:

| Planner | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `oi_face_rot_wide_mppi` | 30 | 18 | 30 | 28 | 106/120 | **120/120** |
| `oi_face_rot_stall_mppi` | 30 | **28** | 30 | 29 | **117/120** | 117/120 |
| `oi_face_rot_pstall_mppi` | 30 | 27 | 30 | 29 | 116/120 | 118/120 |

Hard plant:

| Planner | turn | open turn | turn 90 | turn neg | 4 turn cells | 4 detour cells |
|---|---:|---:|---:|---:|---:|---:|
| `oi_face_rot_wide_mppi` | 30 | 30 | 28 | 29 | 117/120 | **120/120** |
| `oi_face_rot_stall_mppi` | 30 | 30 | 29 | 30 | **119/120** | 119/120 |
| `oi_face_rot_pstall_mppi` | 30 | 30 | 29 | 30 | **119/120** | **120/120** |

`oi_face_rot_pstall_mppi` vs `oi_face_rot_wide_mppi`, paired:

| Plant | Seeds only pstall / only wide | Detail |
|---|---|---|
| smooth | 10 / 2 | open turn 9 / 0, p = 0.004 |
| hard | 2 / 0 | — |

## Detour cost (seeds 1000-1099, the four detour cells)

Thirty seeds per cell cannot resolve a cost of a few episodes, so the detour cells were rerun on 100 fresh seeds each (800 episodes per planner):

| Planner | smooth | hard | vs `oi_face_rot_wide_mppi`, both plants (only new / only wide) |
|---|---:|---:|---|
| `oi_face_rot_wide_mppi` | 400/400 | 398/400 | |
| `oi_face_rot_stall_mppi` | 386/400 | 396/400 | 0 / 16, p = 3e-5 |
| `oi_face_rot_pstall_mppi` | 396/400 | 398/400 | 1 / 5, p = 0.22 |

All rows in every evaluation are collision-free.

## Reading

- **The pusher-aware stall fixes the smooth open turn** (18 to 27/30 on unseen seeds, p = 0.004) and keeps the hard plant at 119-120/120 per group.
- **It cuts the detour cost of the boost by about two thirds**: 16 lost detour episodes in 800 become 5 (against 1 gained).
  - The remaining cost is on the smooth plant's wall cells: 4 of 400, about 1 %. It is not significant, but it is one-sided.
- **`oi_face_rot_pstall_mppi` is the planner to use** for detour-and-turn tasks. Across the turn cells it solves 116/120 (smooth) and 119/120 (hard); the detour cells stay at 99-100 %.
- **What is left:**
  - the 1 % smooth detour cost;
  - three open-turn failures in thirty.
  - A finer stall signal (the pusher's angular progress around the box, rather than its displacement) is the next thing to try.

## Limitations

- One box, `K=256`; the stall rule has three thresholds, chosen from three candidates on 30 seeds.
- The offline table counts firings, not outcomes; an episode where the boost fires may still succeed.

```bash
CELLS=box_detour_turn,box_open_turn,box_detour_turn90,box_detour_turn_neg,box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far
./bin/benchmark_diff_mppi_pushing_box --scenarios $CELLS \
  --planners oi_face_rot_wide_mppi,oi_face_rot_stall_mppi,oi_face_rot_pstall_mppi \
  --k-values 256 --seed-count 30 --seed-offset 900 [--true-plant hard] \
  --csv docs/results/box_detour_pusher_stall_seed900_{smooth,hard}_2026-10-04.csv
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far \
  --planners oi_face_rot_wide_mppi,oi_face_rot_stall_mppi,oi_face_rot_pstall_mppi \
  --k-values 256 --seed-count 100 --seed-offset 1000 [--true-plant hard] \
  --csv docs/results/box_detour_pusher_stall_detours_seed1000_{smooth,hard}_2026-10-04.csv
```
