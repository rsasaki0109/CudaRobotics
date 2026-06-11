# CUDA MPPI Path-Angle Critic (2026-06-11)

Closed-loop GPU-only comparison of the default costmap critic with and
without the path-angle critic. The path-angle critic adds a light stage cost on
rollout yaw error relative to the local path tangent near the follow point.

Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.
Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,
K = 8192, T = 56, dt = 0.05.

CSV: [`cuda_mppi_path_angle_2026-06-11.csv`](cuda_mppi_path_angle_2026-06-11.csv)

## Results

| scenario | path angle weight | result | sim time | mean solve | p95 | max | exceptions |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall_gap | 0.00 | success | 16.5s | 1.13 ms | 1.31 ms | 1.44 ms | 0 |
| wall_gap | 0.25 | success | 16.4s | 1.07 ms | 1.14 ms | 1.21 ms | 0 |
| narrow_corridor | 0.00 | success | 17.4s | 1.15 ms | 1.35 ms | 1.57 ms | 0 |
| narrow_corridor | 0.25 | success | 17.4s | 1.08 ms | 1.17 ms | 1.27 ms | 0 |
| u_turn | 0.00 | success | 44.8s | 1.06 ms | 1.14 ms | 1.39 ms | 0 |
| u_turn | 0.25 | success | 39.9s | 1.11 ms | 1.19 ms | 1.48 ms | 0 |

## Readout

- A strong `path_angle_weight=2.0` matched Nav2's nominal PathAngleCritic
  weight but over-selected stationary straight-heading rollouts in this CUDA
  MPPI sampler. The checked-in default is therefore a light stabilizer:
  `path_angle_weight=0.25`.
- The corrected `u_turn` cell benefits most: K8192 time-to-goal drops from
  44.75 s to 39.9 s while keeping mean solve time close to 1 ms.
- `wall_gap` and `narrow_corridor` remain successful with similar time-to-goal,
  so the critic does not trade off the straight and corridor smoke cells.
- Reverse samples compare against `path_tangent + pi`; the existing backward
  motion critic still decides whether reverse motion is worth using.

## Reproduce

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_path_angle_bench path_angle
```
