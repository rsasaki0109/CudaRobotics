# Dynamic-obstacle MPPI in CudaNav

Status: development surface on `master`, not part of a published release.

The production CUDA MPPI core (`cuda_mppi_controller::MppiGpu`) can avoid
predicted moving obstacles, not only what the costmap shows. This page covers:
- the critic;
- how CudaNav feeds it, from an oracle or from its own scan-based tracker;
- a closed-loop evaluation with pedestrians crossing the S-course.

## Critic

Callers pass tracked obstacles before planning:

```cpp
std::vector<cuda_mppi_controller::DynamicObstacle> tracks = /* x, y, vx, vy, radius */;
planner.setDynamicObstacles(tracks.data(), static_cast<int>(tracks.size()));  // <= 64
```

The tracks are given in the costmap frame and follow constant-velocity motion.
They stay in effect for every following `compute` / `computeWithDeviceCostmap`
call until they are replaced. A count of 0 clears them.

Inside the rollout kernel, at each step `tau = (t + 1) * model_dt`, the
clearance to obstacle `i` is:

```
c = |q(tau) - (p_i + v_i * tau)| - r_i - dynamic_obstacle_robot_radius - dynamic_obstacle_growth * tau
```

The cost depends on `c`:

| Clearance | Cost added |
|---|---|
| `c < 0` | `collision_cost` |
| `0 <= c < dynamic_obstacle_margin` | `dynamic_obstacle_weight * (margin - c)^2 * dt` |
| otherwise | 0 |

A predicted collision adds `collision_cost`. Such rollouts are therefore
counted as invalid, as for costmap collisions, and feed `valid_rollout_ratio`,
`all_colliding` and the retreat behaviour.

The growth term inflates the obstacle with the prediction horizon, standing in
for the uncertainty of the constant-velocity model.

| Parameter | Default | CudaNav default |
|---|---|---|
| `dynamic_obstacle_weight` | 0 (off) | 25 |
| `dynamic_obstacle_margin` | 0.4 m | 0.4 m |
| `dynamic_obstacle_robot_radius` | 0.25 m | 0.25 m |
| `dynamic_obstacle_growth` | 0.1 m/s | 0.1 m/s |
| `dynamic_obstacle_overlap_allowance` | 0 (off) | 0, or 0.1 m with the scan tracker |

**Overlap allowance.** Tracked discs carry position error. Without an
allowance, a robot that a tracking error places just inside a disc sees every
rollout collide, and it freezes. With `dynamic_obstacle_overlap_allowance = a`
greater than 0:
- the robot's current penetration `p` of the observed disc, up to `a`, is
  tolerated: a rollout collides only when it gets deeper than `min(p, a)`
  (judged without the growth term);
- the growth term then enters the soft cost only.

A value of 0 keeps the check in the table above. `CudaNavPipeline` turns the
allowance on, at `TrackerConfig.planner_overlap_allowance = 0.1 m`, only when it
runs the scan tracker.

**Back-out speed.** When every rollout collides, the core backs out by
replaying the last valid plan in reverse, scaled by `retreat_scale`. That
command is clamped to `v_min`. A forward-only robot (`v_min = 0`) therefore
only turned in place, and a robot already close to a pedestrian who was
waiting for it could not get out of the way. `retreat_reverse_speed` (m/s,
default 0) lowers the clamp for the back-out only, to
`min(v_min, -retreat_reverse_speed)`. Normal plans stay forward-only.
CudaNav's forward planner sets it to 0.2 m/s. The back-out is not checked
against the predicted obstacles.

With the weight at 0, or with no obstacles, the rollout skips the term
entirely. Plans are then bit-identical to the planner without the critic, which
`tests/mppi_dynamic_obstacle_smoke.cu` checks. The native S-course closed loop
is unchanged: 229 frames and 0.153% drift. The Nav2 plugin does not have an
obstacle input yet, so its behaviour is also unchanged.

## CudaNav integration

```python
from cudarobotics.navigation import CudaNavPipeline

nav = CudaNavPipeline()
step = nav.step(points_without_tracked_objects, path=path,
                dynamic_obstacles=tracks)  # (N, 5): x, y, vx, vy, radius
```

`dynamic_obstacles` does three things:
- it clears each track's disc, plus 0.15 m, from the occupancy grid, on both
  the device and the host stage chain;
- the ESDF and costmap are built from the cleaned grid;
- MPPI receives the tracks as predictions.

Keep the tracked objects' points out of the scan as well. Otherwise the
voxel map keeps them as obstacles, as the baseline below shows.

The planner can also be used on its own:
`cudarobotics.MppiPlanner.set_dynamic_obstacles(rows)`.

## Scan-based tracker

`CudaNavPipeline(tracker=ScanObjectTracker())` finds moving objects in the scan
itself, with no oracle (`cudarobotics.navigation.tracking`). In each frame it
does the following:

1. **Candidates.** `OccupancyAge` keeps, for every cell of the rolling
   occupancy grid, how many consecutive frames it has been occupied. A point is
   a candidate when its 3 x 3 cell block contains no long-standing structure (a
   cell occupied for 20 frames).
   - Walls therefore drop out once they are established, including the half of
     their returns that falls into the free cell in front of the wall surface.
   - Anything new, in free or never-observed space, starts as a candidate.
   - Points near a live track's prediction are added unless they touch
     structure. Points inside a confirmed track's own disc are kept even next
     to a wall.
2. **Clusters.** Candidates are linked within 0.3 m (on distinct xy returns).
   - Clusters wider than 1.0 m, or hit by fewer than 2 rays, are dropped.
   - A scan sees only an object's near side, so the centre is pushed away from
     the sensor by `2 r / pi`, with `r` half the cluster's width.
3. **Tracks.** Clusters are associated greedily within 0.8 m, and each track
   is a constant-velocity Kalman filter.
   - A track is confirmed after 3 detections, once it has moved 0.3 m from
     where it was first seen. Wall fragments jitter in place as different rays
     hit them; a person walks away.
   - A track is deleted after 5 missed frames, or after 2 s supported only by
     points touching structure.
4. **Map gating.** Every detection's points stay out of the voxel map, because
   a moving object integrated even for a few frames leaves lethal traces.
   - An object that does not move for 2 s is released to the map and ages into
     structure. Until then it goes to MPPI as a stationary disc.
   - Confirmed tracks also clear their last 10 s of positions from the
     occupancy grid, which removes traces left before they were confirmed.

The confirmed tracks then go through the same path as `dynamic_obstacles`.
`PipelineStep.tracks`, `track_ids` and `dynamic_point_count` expose the
result. The tracker is NumPy code, and its occupancy-age update copies the
grid to the host every frame. With it, the Python crossing loop runs at a
frame p95 of about 25–35 ms, against 5–7 ms with the oracle.

## Crossing evaluation

`cudarobotics-cudanav sim --scenario crossing` runs one S-course traversal
while two pedestrians (discs of radius 0.25 m) walk back and forth across the
two corridors the robot must pass, at 0.5 and 0.4 m/s.
- `--seed` sets their schedule phases.
- They do not steer around the robot, but they pause instead of stepping into
  it. A contact can therefore only come from the robot's own motion.

`--dynamic-mode` selects how the pedestrians reach the planner:

| Mode | Pedestrians in the voxel map | Planner input |
|---|---|---|
| `map` | yes, from LiDAR | none (costmap only) |
| `tracked_static` | removed | tracked positions with zero velocity |
| `predicted` | removed | tracked positions and velocities |
| `tracked_scan` | removed by the scan tracker | the scan tracker's tracks |

In `tracked_static` and `predicted` the tracker is the simulator itself (an
oracle). The `--obs-latency-frames`, `--obs-position-noise-m`,
`--obs-velocity-noise-mps` and `--obs-dropout` options degrade it.
`tracked_scan` uses no simulator knowledge. It also reports tracking accuracy
against the true pedestrians:
- recall: the share of visible pedestrian-frames with a confirmed track within
  0.5 m;
- position and velocity RMSE;
- false tracks: confirmed tracks more than 0.7 m from every pedestrian.

Results from 6 seeds per condition, with the back-out speed of 0.2 m/s
([`results/cudanav_dynamic_obstacles_2026-11-22.md`](results/cudanav_dynamic_obstacles_2026-11-22.md),
generated by `scripts/render_cudanav_dynamic_obstacles.py`):

| Condition | Goal reached | Robot-caused collisions | Median time to goal | Median pedestrian yields | Median retreats |
|---|---|---|---|---|---|
| Map only | 0/6 | 0 | - | 10 | 4 |
| Tracked, zero velocity (oracle) | 2/6 | 0 | 50.7 s | 24 | 174 |
| Tracked + prediction (oracle) | 6/6 | 0 | 31.0 s | 13 | 46 |
| Tracked + prediction (oracle, 2-frame delay, 0.05 m / 0.1 m/s noise) | 6/6 | 0 | 31.9 s | 33 | 47 |
| Scan tracker + prediction (no oracle) | 6/6 | 0 | 33.5 s | 30 | 56 |

Scan tracker accuracy over its 6 runs: 90.2% recall, a median position RMSE of
0.11 m, a median velocity RMSE of 0.29 m/s, and 1.10 false tracks per frame.

![Map-only baseline, oracle prediction and scan tracker, seed 0](results/cudanav_dynamic_obstacles_2026-11-22.png)

What the runs show:
- **Map only.** The robot never collides, but it never arrives.
  - Each pedestrian's sweep leaves lethal cells across the corridor: up to
    106 occupied cells away from the walls. The voxel map clears them only
    slowly, so the robot circles in front of the first corridor.
  - The planar scan's ±0.45 m returns rarely pass back through those voxels,
    which is why they persist.
- **Tracked, zero velocity.** Keeping the pedestrians out of the map removes
  the trails. However, a planner that treats them as static keeps steering
  into their path, and 4 of 6 runs timed out.
- **Tracked + prediction (oracle).** All 6 runs reached the goal (median
  31.0 s) with no robot-caused collision.
- **Delay and noise.** All 6 runs reached the goal (median 31.9 s) with no
  robot-caused collision.
- **Scan tracker.** With no oracle, all 6 runs reached the goal (median
  33.5 s), with no robot-caused collision.
  - Recall is 90.2%, against perfect tracks for the oracle.
  - Up to 46 occupied cells away from the walls remain where tracking briefly
    lost a pedestrian.
  - Without the overlap allowance, the same tracker reached the goal in 0 of 6
    runs (measured before the back-out change): the robot froze whenever a
    position error placed it inside a disc.

**What the back-out speed changed.** The previous results
([`results/cudanav_dynamic_obstacles_2026-11-21.md`](results/cudanav_dynamic_obstacles_2026-11-21.md))
had a back-out clamped to `v = 0`.

| Condition | Goal reached | Robot-caused collisions |
|---|---|---|
| Oracle prediction | 5/6 → 6/6 | 0 → 0 |
| Delay and noise | 4/6 → 6/6 | 12 → 0 |
| Scan tracker | 4/6 → 6/6 | 0 → 0 |

Every failure and every contact came from the same situation:
1. A pedestrian stopped close to the robot, waiting for it to clear the
   pedestrian's path.
2. The robot saw every rollout collide:
   - with noisy tracks, because the observed disc overlapped it;
   - even with exact tracks, because the growth term inflates a stopped
     pedestrian over the horizon until standing still also collides.
3. Its back-out could only turn in place, so neither side moved.
4. The contacts happened when a noisy track briefly looked farther away: the
   robot drove forward from that pose and hit the pedestrian.

Before the change, the back-out was the leading suspect for the contacts,
because it is not checked against the predicted obstacles. In fact, none of
the 12 contacts came from a back-out command. All 12 came from normal plans
that still had valid rollouts.

Other changes tried on the delay-and-noise condition:

| Change | Goal reached | Robot-caused collisions |
|---|---|---|
| Overlap allowance 0.1 m | 5/6 | 19 |
| Growth term in the soft cost only | 5/6 | 37 |
| Robot radius 0.35 m plus allowance 0.15 m | 4/6 | 0 |
| `v_min = -0.2` for every plan, with radius 0.30–0.35 m and the allowance | 3/6–4/6 | 0 |

Allowing reverse in every plan broke path following, so the reverse motion
is limited to the back-out.

The cost is time on runs that already succeeded, because the robot now backs
off where it used to squeeze past:
- oracle seeds 1 and 2 took 34.9 s and 31.5 s, against 26.4 s and 24.3 s
  before;
- the oracle median rose from 25.9 s (over 5 runs) to 31.0 s (over 6).

## Limits

- **Synthetic.** The world, LiDAR and plant are synthetic. The scan tracker
  has only been evaluated on this simulator's planar three-layer scan and
  disc-shaped pedestrians.
- **Tracker assumptions.** It is a 2-D, model-free tracker:
  - a person who stands still for 2 s before ever moving becomes map
    structure;
  - one who stays in contact with a wall can merge into it;
  - its NumPy implementation and per-frame grid copy cost about 20–30 ms per
    frame.
- **Back-out.** The back-out replays the last valid plan in reverse and is not
  checked against the predicted obstacles or the costmap. In this scenario
  nothing is behind the robot when it backs out, so it could still back into
  a person in other settings. The standoffs are resolved by the robot giving
  way, which costs time; nothing reasons about who should go first.
- **Simple motion model.** The prediction is constant velocity, and the
  pedestrians move in straight lines with turnarounds.
- **Small clearances.** Minimum clearances stay around 0.05 m in every
  condition, partly because the pedestrians pause at 0.05 m.
- **No Nav2 input yet.** The Nav2 controller plugin has no input path for
  predicted obstacles yet (a topic and message type).
