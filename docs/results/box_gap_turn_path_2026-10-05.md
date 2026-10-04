# Turning Mid-Path: a Gap the Box Only Fits Through Sideways

_Follow-up to [`box_detour_push_anchor_2026-10-04.md`](box_detour_push_anchor_2026-10-04.md). The rotation note listed as a limitation that "rotation is attempted only near the goal; turns that must happen mid-path (for example to fit through a gap) are not covered". This note adds such cells and a planner for them._

## Cells

A wall across the way at y 2.0-2.2 with a 0.55 m opening (x 1.225-1.775), built from two rectangle walls. The box is 0.70 m × 0.36 m and starts at (1.5, 1.0) with its long side across the gap, so it only fits through turned by about 90°. The goal is 1.2 m past the wall at (1.5, 3.4), beyond the 1.0 m rotation radius of the existing planners.

| Cell | goal heading | what it needs |
|---|---:|---|
| `box_gap_turn` | +1.571 rad | turn before the gap, keep the heading |
| `box_gap_return` | 0 | turn before the gap, pass, turn back |

**Supporting changes:**
- `BoxParams` gains an optional second rectangle wall (`obs2`). The overlap test, the wall cost (float and dual) and the path planner handle both walls; with one wall every function computes exactly what it did before.
- **Blocked poses.** Pushing a box that does not fit into the opening squeezes it between two walls that cannot both push it out. Both plants now treat such a pose as blocked and keep the previous pose (the hard plant also stops the box). Without this, the hard plant let the box slip through the wall at the wrong heading (164 of 360 baseline episodes collided).
- The eight earlier cells reproduce row for row.

## Baseline (seeds 0-29)

| Planner | smooth: turn | smooth: return | hard: turn | hard: return |
|---|---:|---:|---:|---:|
| `diff_mppi_3` | 4 | 0 | 0 | 3 |
| `soppi_fast` | 0 | 0 | 0 | 1 |
| `oi_face_auto_mppi` | 14 | 0 | 14 | 20 |
| `oi_face_rot_mppi` | 12 | 0 | 19 | 22 |
| `oi_face_rot_wide_mppi` | 19 | 0 | 21 | 23 |
| `oi_face_rot_anchor_mppi` | 21 | 0 | 26 | 30 |

**Smooth plant, return cell.** Every planner parks the box against the wall. The goal heading is the start heading, so the task's heading cost works against the quarter turn the gap needs; in 400 steps the box creeps from 0.2 to 0.45 rad.

**Smooth plant, turn cell.** Here the goal heading helps the turn, so 12-21 of 30 get through.

**Hard plant.** The box pivots on the wall's corner into the opening, so the contact physics does part of the turn.

## Mechanism (`oi_heading_path`)

1. **Heading-aware object path.** When no path exists at the start heading, A* runs over (cell, heading layer).
   - The layers are the start heading and a quarter turn from it.
   - A turn in place is allowed where the box can sweep between the two headings without touching a wall.
   - A final turn to the goal heading is placed as early on the last run as it fits.
   - Turns are zero-length segments of the path. On the return cell the path is: (1.5, 1.4) turn to 90°, pass, (1.5, 2.6) turn back to 0°, goal.
2. **Reference and turn subgoal.**
   - The object reference follows the path's heading and waits at a turn the box has not made yet.
   - Until the box has made the next turn, the controller plans for the turn point instead of the goal, at the current heading on the way and at the turn's heading once there. Without this, the rollouts' terminal pull toward the goal pushed the box past the turn point into the opening.
   - The controller tracks which turns are made, so a box pushed past a turn point still has to turn.
3. **Seed.**
   - The rotation phase turns toward the path's next heading at a turn point.
   - The 0.6 near blend applies at a turn point and near the real goal, but not near a turn subgoal.
4. **Replanning.** A box knocked more than 0.3 m off the path (a turning push in the hard plant can slide it well away) gets a new path from where it is, over the same two headings.

`oi_face_rot_turnpath_mppi` is `oi_face_rot_anchor_mppi` with `oi_heading_path` (`--override-oi-heading-path` sets it for any path planner). The new path is only used when the start heading has no path, so on the eight earlier cells the planner reproduces `oi_face_rot_anchor_mppi` row for row (checked on seeds 1300-1329, both plants).

## Development (seeds 0-29)

Each row adds one change:

| Change | smooth: turn | smooth: return | hard: turn | hard: return |
|---|---:|---:|---:|---:|
| `oi_face_rot_anchor_mppi` | 21 | 0 | 26 | 30 |
| heading path, reference and seed only | 18 | 0 | 24 | 18 |
| + turn subgoal | 26 | 1 | 27 | 7 |
| + hold the current heading until the turn point | 29 | 21 | 26 | 21 |
| + near blend from the real goal | 30 | 29 | 22 | 25 |
| + replan when off the path | 30 | 29 | 16 | 27 |
| + same two headings on a replan (final) | **30** | **29** | 18 | **30** |

Turn tolerance 0.15 / 0.25 rad × turn-point blend 0.3 / 0.6 gave 55-59 (smooth) and 48-53 (hard) of 60. The defaults (0.15, 0.6) were kept.

## Evaluation (seeds 1500-1599, 100 per cell)

| Planner | smooth: turn | smooth: return | hard: turn | hard: return |
|---|---:|---:|---:|---:|
| `oi_face_rot_wide_mppi` | 61 | 0 | 81 | 71 |
| `oi_face_rot_anchor_mppi` | 68 | 1 | **91** | **100** |
| `oi_face_rot_turnpath_mppi` | **99** | **95** | 85 | 94 |

`oi_face_rot_turnpath_mppi` vs `oi_face_rot_anchor_mppi`, paired (only turnpath / only anchor):

| Plant | turn | return |
|---|---|---|
| smooth | 32 / 1, p = 8e-9 | 94 / 0, p = 1e-28 |
| hard | 6 / 12, p = 0.24 | 0 / 6, p = 0.03 |

All rows are collision-free.

## Reading

- **On the plant the controller models (smooth), planned mid-path turns solve the gap.** Successes go from 69 to 194 of 200. On the return cell no earlier planner gets through (1/100), because the goal heading argues against the turn the gap needs.
- **Under the hard-contact plant the earlier planner is better.** Successes go from 191 to 179 of 200 (the return-cell loss, 0 / 6, is significant).
  - There the box pivots on the wall corner into the opening by itself.
  - The planned turn in open space is harder under friction and momentum: turning pushes slide or over-rotate the box, and some episodes end far from the path even with replanning.
- **Which to use:** `oi_face_rot_turnpath_mppi` when the model matches the plant or the passage gives nothing to pivot on; `oi_face_rot_anchor_mppi` under strong contact mismatch.
- **Open:** making the planned turn robust to the hard plant's sliding, for example by turning while the box rests against a wall, as the hard plant does on its own.

## Follow-up: making the turn robust to the hard plant (negative)

Three attempts on the development seeds; none is kept in the code. Successes out of 30:

| Change to `oi_face_rot_turnpath_mppi` | smooth: turn | smooth: return | hard: turn | hard: return |
|---|---:|---:|---:|---:|
| none | 30 | 29 | 18 | 30 |
| a replan from an off-layer heading starts by turning back to the layer | 30 | 29 | 17 | 28 |
| latch a made turn only after the box has moved 0.3 m past it | **30** | **30** | 12 | 23 |

**Falling back to the earlier planner when a turn takes too long** was ruled out before trying it. The hard-plant failures make their first turn as fast as the successes (33-53 steps against 32-53), so a timeout would not fire.

**Where the hard plant fails.** All 12 hard-plant failures are on the turn cell.
- After the turn the box keeps rotating (momentum) to 2-3 rad and slides along the wall.
- Replans from those poses often find no path, because the box cannot sweep there.
- Re-opening the turn when the heading drifts (the latch) makes the smooth plant perfect, but on the hard plant the extra turning pushes spin the box more.

The robust version probably needs to damp the box's rotation before it reaches the target heading. That is a model the controller does not have: the smooth model has no momentum.

**A fourth attempt: braking from the measured spin rate.** The spin rate is estimated from the last two control steps, and the turn aims from the heading the box will have tau seconds ahead. A box still spinning toward the target is then pushed the other way.

On the development seeds the hard plant's total over both gap cells (of 60) peaked at tau = 0.15 s:

| tau (s) | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard plant | 48 | 48 | 51 | **54** | 49 | 43 | 46 |

The smooth plant stayed at 58-60.

The peak did not survive fresh seeds (1600-1699, 100 per cell), with `oi_face_rot_turnpath_mppi` as the comparison:

| Plant | `oi_face_rot_turnpath_mppi` | with tau = 0.15 | Paired (only braked / only turnpath) |
|---|---:|---:|---|
| smooth | 192/200 | 197/200 | return 5 / 0, p = 0.06 |
| hard | 183/200 | 172/200 | return 2 / 11, p = 0.02 |

The development gain was noise on 30 seeds, and on the hard plant the braking hurts. It is not kept.

The same run reproduces the planner comparison of the main evaluation on new seeds:
- smooth plant: anchor 71, turnpath 192 of 200;
- hard plant: anchor 193, turnpath 183 of 200.

## Limitations

- One box, two gap cells, `K=256`.
- The path has two heading layers (the start heading and a quarter turn from it).
- The turn tolerance and blend were chosen on 30 development seeds.

```bash
./bin/benchmark_diff_mppi_pushing_box --scenarios box_gap_turn,box_gap_return \
  --planners diff_mppi_3,soppi_fast,oi_face_auto_mppi,oi_face_rot_mppi,oi_face_rot_wide_mppi,oi_face_rot_anchor_mppi \
  --k-values 256 --seed-count 30 --seed-offset 0 [--true-plant hard] \
  --csv docs/results/box_gap_baseline_seed0_{smooth,hard}_2026-10-05.csv
./bin/benchmark_diff_mppi_pushing_box --scenarios box_gap_turn,box_gap_return \
  --planners oi_face_rot_wide_mppi,oi_face_rot_anchor_mppi,oi_face_rot_turnpath_mppi \
  --k-values 256 --seed-count 100 --seed-offset 1500 [--true-plant hard] \
  --csv docs/results/box_gap_turn_path_seed1500_{smooth,hard}_2026-10-05.csv
```
