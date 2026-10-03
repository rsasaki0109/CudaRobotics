# Why the Hard Plant Fails `box_detour_wall_far`, and Axis-Aligned Paths

_Follow-up to [`box_detour_generalization_2026-10-03.md`](box_detour_generalization_2026-10-03.md),
where `oi_face_track_mppi` reached only 4/30 on `box_detour_wall_far` under
the hard-contact plant._

## Diagnosis

Per-episode trajectories (`--traj-dir`, seeds 8256-8291) show two failure
patterns:

1. **The box is pushed up into the wall's underside left of the corner.** The
   planned object path starts with a diagonal segment toward the wall's right
   corner. A single face cannot push along a diagonal: pushing the bottom face
   moves the box nearly straight up, and with Coulomb friction (hard plant)
   it does not slide sideways, so the box reaches the wall short of the corner.
2. **After a slide the pusher never comes back.** When the box glances off the
   wall it can slide almost a metre; the pusher then has to reposition for
   longer than the 16-step horizon. While it repositions the object reference
   is held, so no rollout shows box progress and nothing in the cost draws the
   pusher back.

Two interventions, tried on seeds 0-7:

- **Pusher reference cost** (the seed's planned pusher waypoints as a rollout
  cost, weight 1 / 4 / 16): `box_detour_wall_far` stays at 0-1/8. Pattern 2 is
  not the main cause. Not kept.
- **Axis-aligned object path** (`oi_face_axis_mppi`): 4-connected A* with a
  turn penalty, merging only collinear points, so every segment is pushed by
  one face. `box_detour_wall_far` goes from 0/8 to 8/8 on the hard plant.
  Wider planning margins (0.10 / 0.15 / 0.25 m) did not help any cell, so the
  default 0.05 m was kept.

## Evaluation on fresh seeds

30 seeds never used in this or any earlier search (`--seed-offset 200`),
`K=256`, all rows collision-free:

| Planner | wall | open | left | far |
|---|---:|---:|---:|---:|
| hard: `oi_face_track_mppi` | 16/30 | **23/30** | 17/30 | 7/30 |
| hard: `oi_face_axis_mppi` | **25/30** | 14/30 | 22/30 | **24/30** |
| smooth: `oi_face_track_mppi` | **20/30** | 27/30 | **20/30** | 23/30 |
| smooth: `oi_face_axis_mppi` | 0/30 | 27/30 | 2/30 | **30/30** |

Paired exact McNemar tests, axis vs track (only axis succeeds / only track
succeeds):

| Plant | wall | open | left | far |
|---|---|---|---|---|
| hard | 12 / 3, p = 0.035 | 2 / 11, p = 0.022 | 11 / 6, p = 0.33 | 19 / 2, p = 0.0002 |
| smooth | 0 / 20, p = 1.9e-6 | 3 / 3, p = 1.0 | 0 / 18, p = 7.6e-6 | 7 / 0, p = 0.016 |

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --scenarios box_detour_wall,box_detour_open,box_detour_wall_left,box_detour_wall_far \
  --planners oi_face_track_mppi,oi_face_axis_mppi \
  --k-values 256 --seed-count 30 --seed-offset 200 [--true-plant hard] \
  --csv docs/results/box_detour_axis_path_{smooth,hard}_2026-10-03.csv
```

## Reading

- **Under friction, plan paths that a single face can push.** On the hard
  plant the axis-aligned path wins all three obstacle cells and turns the
  far-wall cell from 7/30 into 24/30. The diagonal path only wins the open
  cell, where there is nothing to hit and the final tracking step handles the
  drift.
- **The smooth plant disagrees.** On the two low-wall cells the axis-aligned
  path fails almost every episode: the box reaches the end of the first
  vertical segment just below the wall, and the pusher stops beside the box's
  lower corner instead of moving onto the side face. The cause is not the
  planning margin (tested) and is not yet identified. It is plausibly the
  smooth model's soft contact, which lets the diagonal path work by sliding,
  combined with the wall barrier cost on rollouts that rotate the box near the
  wall.
- No single planner wins everywhere: the right object path depends on the
  contact model.

## Limitations

- Four cells, one box, `K=256`; the hard plant's rigid wall is a projection.
- The pusher-reference experiment was removed from the code and is reported
  here only.
