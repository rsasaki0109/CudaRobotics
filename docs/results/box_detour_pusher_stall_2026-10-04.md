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

### A second detour run (seeds 1100-1299, wall and open cells)

A further 200 seeds on `box_detour_wall` and `box_detour_open`, run while looking for the cause, show a larger cost on the smooth wall cell. It was never used to select the rule.

Pooled over seeds 1000-1299:

| Cell | Plant | `oi_face_rot_wide_mppi` | `oi_face_rot_pstall_mppi` |
|---|---|---:|---:|
| `box_detour_wall` | smooth | 298/300 | **285/300** |
| `box_detour_wall` | hard | 298/300 | 296/300 |
| `box_detour_open` | both | 300/300 | 300/300 |
| `box_detour_wall_left` / `_far` | both | 199-200/200 | 199-200/200 |

Paired over all detour episodes:

| Plant | Only pstall / only wide | p |
|---|---|---:|
| smooth | 0 / 14 | 1e-4 |
| hard | 1 / 3 | 0.62 |

So on the smooth plant's wall cell the boost costs about 4 % of the episodes. The 1 % from seeds 1000-1099 underestimated it.

**Mechanism.** In the lost episodes the box is pressed against the underside of the wall.
- Without the boost, the pusher waits about 150 steps; MPPI's own sampling then finds a way around and the episode succeeds.
- With the boost, the seed keeps asking for the push into the wall. The 0.6 blend holds the pusher at the contact point and suppresses the exploration that would have escaped.

**Tried: boost only when the seed asks the pusher to go elsewhere.** `oi_stall_seed_gap` boosts only if the seed's next target is at least the gap away from the pusher. A blocked push has its target at the pusher, while a walk-around has it at the next corner.

| Seed gap | smooth wall (seeds 1100-1299) | hard wall | smooth open turn (seeds 0-29) |
|---|---:|---:|---:|
| none (`pstall`) | 188/200 | 197/200 | **28/30** |
| 0.1 m | 188/200 | 196/200 | 25/30 |
| 0.2 m | 189/200 | 197/200 | 23/30 |
| 0.3 m | **195/200** | **200/200** | 23/30 |
| no boost (`wide`) | 198/200 | 199/200 | 22/30 |

The gap does not separate the two cases: removing the wall cost removes the open-turn gain with it. In the open-turn stall, the seed's next target is also close to the pusher. The parameter stays in the code (off by default) as a recorded negative result.

### Further attempts to remove the wall cost (all negative)

Instrumenting a lost wall episode at the moments the boost fires changed the picture.
- The seed is not pushing into the wall. The pusher sits at the edge of the push face's span, and the seed's first target flips every step between the face contact point ("push") and the corner waypoint ("go around").
- The boost amplifies that flip. Without it, MPPI breaks out on its own.

Four more gates were tried, each off by default:

| Gate | What it tests |
|---|---|
| model rollout | the seed's controls, rolled out with and without the wall |
| `oi_stall_unblocked` | whether a short move along the seed's push direction deepens the wall overlap |
| `oi_stall_consistent_steps` | whether the seed's first target has held still for N stalled steps |
| `oi_slide_hysteresis` | engagement hysteresis on the seed itself, removing the flip at the source (no boost) |

Results on the development seeds:

| Change | smooth wall (1100-1299) | hard wall | smooth open turn (0-29) |
|---|---:|---:|---:|
| `oi_face_rot_wide_mppi` (no boost) | 198/200 | 199/200 | 22/30 |
| `oi_face_rot_pstall_mppi` | 188/200 | 197/200 | **28/30** |
| pstall + model rollout | never fires | | |
| pstall + `oi_stall_unblocked` | 188/200 | 197/200 | 28/30 |
| pstall + `oi_stall_consistent_steps` 5 | 197/200 | 200/200 | 22/30 |
| pstall + `oi_stall_consistent_steps` 10 | 198/200 | 199/200 | 22/30 |
| wide + `oi_slide_hysteresis` 0.05 | 197/200 | 199/200 | 22/30 |
| wide + `oi_slide_hysteresis` 0.10 | 196/200 | 198/200 | 22/30 |
| wide + `oi_slide_hysteresis` 0.20 | 200/200 | 199/200 | 22/30 (turn 90: 29 to 22) |

- **The two rollout/geometry gates never trigger.** The model rollout never fires: within the 16-step horizon the seed is still walking, not pushing. The push-direction check never fires either.
- **The consistency gate removes the wall cost and the open-turn gain together.** The open-turn stall is a seed flip too, and there the boosted flip is what gets the box moving.
- **Hysteresis on the seed** does not help the open turn, and at 0.2 m it costs the smooth quarter turn.

The flip that the boost exploits in the open turn is the same flip that costs it the wall cell. None of these signals separates the two.

All rows in every evaluation are collision-free.

## Reading

- **The pusher-aware stall fixes the smooth open turn** (18 to 27/30 on unseen seeds, p = 0.004). The hard plant stays at 119-120/120 per group.
- **It is not free on the smooth plant.** It loses about 4 % of `box_detour_wall` episodes (0 / 14, p = 1e-4 over seeds 1000-1299), where the box gets pinned under the wall. Elsewhere its detour cost is within noise.
- **Which planner to use:**
  - `oi_face_rot_pstall_mppi` when open-space reorientation matters more than wall-pinned detours, for example turn cells: smooth 116/120 against 106/120 on seeds 900-929.
  - `oi_face_rot_wide_mppi` stays the safer default when the task has a wall the box can get pinned against.
- **What is left.** Five gates were tried; none separates the wall stall from the open-turn stall, since both are the same seed flip. A fix would have to change what the seed asks for at the face edge, rather than when to trust it. The negative results are recorded above.

## Limitations

- One box, `K=256`; the stall rule has three thresholds, chosen from three candidates on 30 seeds.
- The offline table counts firings, not outcomes; an episode where the boost fires may still succeed.
- The seed-1100 run doubled as the search set for the seed gap, so the gap table is a development result.

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
./bin/benchmark_diff_mppi_pushing_box --scenarios box_detour_wall,box_detour_open \
  --planners oi_face_rot_wide_mppi,oi_face_rot_pstall_mppi \
  --k-values 256 --seed-count 200 --seed-offset 1100 [--true-plant hard] \
  --csv docs/results/box_detour_pusher_stall_detours_seed1100_{smooth,hard}_2026-10-04.csv
# seed-gap rule: add --override-oi-stall-seed-gap 0.1|0.2|0.3 to the pstall planner
```
