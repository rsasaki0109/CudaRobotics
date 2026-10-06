# Pushing box: a residual momentum model for the turn-path planner

Follow-up to [box_gap_turn_path_2026-10-05.md](box_gap_turn_path_2026-10-05.md). There, the planned mid-path turn (`oi_face_rot_turnpath_mppi`) solved the gap cells on the smooth plant. Under the hard-contact plant (impulses, Coulomb friction, momentum), though, the box kept spinning past the target heading.

The note's guess was that the controller needs a model with momentum, which its smooth quasi-static rollout model does not have. This note tests that guess two ways:
- swapping in the hard-contact model (negative);
- adding a residual term to the smooth model (kept).

## Swapping in the hard-contact model (negative)

The object-informed rollout kernel can now step the box with the hard-contact model, including the rigid rectangle walls. This is the same `push_step_box_hard_walls_f` that the episode's true plant now calls, and the plant rows are unchanged. There are two ways to start the rollouts:
- `oi_dyn = 1`: from rest;
- `oi_dyn = 2`: from the box velocity estimated from the last two observed poses.

Development seeds 0-29, K = 256 and 1024, successes of 60 per cell:

| Rollout model | smooth: turn | smooth: return | hard: turn | hard: return |
|---|---:|---:|---:|---:|
| smooth (`oi_face_rot_turnpath_mppi`) | **60** | **59** | 40 | **58** |
| hard contact, from rest | 0 | 4 | 42 | 41 |
| hard contact, from the estimated velocity | 0 | 4 | 49 | 50 |

- **On the smooth plant it collapses.** A planner that expects the box to coast pushes in short pulses, and a box that stops when the push stops reaches no goal in 500 steps.
- **On the hard plant, its own model, it only helps the turn cell.** It is worse with K = 1024 than with 256 (22 against 27 of 30 on the turn cell): the larger sample budget exploits the model harder.
- **Its failures stall.** After a turn of about 0.55 rad, the pusher sits on a box corner for the rest of the episode.

The reference, seeding and stall logic around the rollouts were all tuned with the smooth model, and they do not carry over to a different model as a whole.

## The residual model (`oi_dyn = 3`)

Keep the smooth model and add only what it misses. After each control step, the controller predicts the step with its own smooth model, from the same pose and control. The residual is the observed box pose minus that prediction, divided by dt. It is the motion the model did not explain:
- the momentum and sliding of the hard plant;
- exactly zero on a plant the model matches.

The rollouts carry the residual on as a drift of the box after each smooth step, decaying by a factor of 0.85 per step (a half-life of about 4 steps, 0.2 s). This is the disturbance estimate of offset-free MPC, applied to the box pose.

Because the residual is exactly zero on the smooth plant, the rollouts there are bit-identical to `oi_face_rot_turnpath_mppi`. All 120 development rows and all 200 test rows match in steps, final distance and cost.

Development seeds 0-29 on the hard plant, successes of 60 per cell. The baseline is 40 (turn) and 58 (return).

| Residual carried | decay 0.5 | 0.7 | 0.85 | 0.95 |
|---|---:|---:|---:|---:|
| translation and rotation | 55 / 55 | 57 / 50 | **58 / 54** | 54 / 50 |
| translation only | | 50 / 60 | 52 / 56 | |
| rotation only | | 50 / 50 | 52 / 54 | |

- **Every setting fixes most of the turn cell.** The return cell moves by about ±4 around the baseline.
- **The kept setting** is translation and rotation with decay 0.85, the best total. The other settings stay reachable through the `--override-oi-res-decay` and `--override-oi-res-parts` options.
- **The rotation alone is not enough,** although the reported failure was over-rotation: the sliding matters too.

## Test on fresh seeds (1700-1799, 100 per cell, K = 256)

No earlier step ran these seeds. The criterion was fixed in [`scripts/box_momentum_eval.py`](../../scripts/box_momentum_eval.py) before they ran:
- on the hard plant, more successes over both cells, with a pooled paired sign test at p < 0.05;
- no cell on either plant significantly worse;
- the smooth total not lower.

Report: [box_momentum_test_2026-10-06.md](box_momentum_test_2026-10-06.md).

| Plant | Cell | `oi_face_rot_anchor_mppi` | `oi_face_rot_turnpath_mppi` | `oi_face_rot_turnpath_resid_mppi` | paired (only residual / only turnpath) |
|---|---|---:|---:|---:|---|
| smooth | turn | 71 | 99 | 99 | 0 / 0 (identical) |
| smooth | return | 0 | 92 | 92 | 0 / 0 (identical) |
| hard | turn | 90 | 77 | **93** | 19 / 3, p = 0.0009 |
| hard | return | **100** | 91 | 93 | 9 / 7, p = 0.8 |
| hard | both | 190 | 168 | 186 | 28 / 10, p = 0.005 |

**The criterion is met.**
- All rows are collision-free.
- Successful hard-plant episodes take 206 steps on average, against 222 for the turn-path planner.
- The control time is unchanged (0.25 ms per step): the residual adds three multiply-adds per rollout step.

## Reading

- **What the turn-path planner gains.** On the hard plant, the turn cell goes from 77 to 93 of 100, now above the wall-pivot planner (90). The hard-plant gap to `oi_face_rot_anchor_mppi` shrinks from 22 to 4 of 200.
- **What it keeps.** On the smooth plant the planner keeps its 191 of 200, against 71 for the anchor planner.
- **What remains.** The anchor planner is still ahead on the hard return cell (100 against 93). That cell's pivot on the wall corner is something the turn-path planner never tries.
- **Why a residual and not a better model.** The smooth model stays the one every downstream heuristic was tuned with, and the residual corrects it only where the plant disagrees. A model swap changes behaviour everywhere, including where the old model was right.

## Limitations

- One box, two gap cells, K = 256 on the test.
- One hard-contact plant with default friction (mu = 0.6).
- The decay was chosen on 30 development seeds.
- The residual is a constant-velocity drift. It does not depend on the control, so it cannot predict how a push changes the sliding.

```bash
# development (seeds 0-29, K = 256 and 1024)
./bin/benchmark_diff_mppi_pushing_box --scenarios box_gap_turn,box_gap_return \
  --planners oi_face_rot_turnpath_mppi,oi_face_rot_turnpath_resid_mppi --seed-count 30 [--true-plant hard]
#   swap in the hard-contact model:     --planners oi_face_rot_turnpath_mppi --override-oi-dyn 1   (or 2)
#   residual ablations:                 --override-oi-res-decay 0.7 --override-oi-res-parts 1   (2: rotation only)
# test on fresh seeds 1700-1799 (writes the CSVs and the report)
python scripts/box_momentum_eval.py
```
