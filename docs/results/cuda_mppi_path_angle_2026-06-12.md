# CUDA MPPI Path-Angle Critic Metrics Refresh (2026-06-12)

Closed-loop GPU-only comparison of the default costmap/path critics with and
without the path-angle critic. This refresh uses the trajectory-quality metrics
added to `controller_benchmark` after the original 2026-06-11 report.

Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.
Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,
K = 8192, T = 56, dt = 0.05.

CSV: [`cuda_mppi_path_angle_2026-06-12.csv`](cuda_mppi_path_angle_2026-06-12.csv)

## Results

| scenario | path angle weight | result | sim time | mean solve | distance | mean speed | mean \|w\| | mean \|curv\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wall_gap | 0.00 | success | 16.5s | 1.12 ms | 7.88 m | 0.48 m/s | 0.36 rad/s | 0.85 |
| wall_gap | 0.25 | success | 16.4s | 1.06 ms | 7.80 m | 0.48 m/s | 0.25 rad/s | 0.75 |
| narrow_corridor | 0.00 | success | 17.4s | 1.12 ms | 7.88 m | 0.45 m/s | 0.37 rad/s | 1.38 |
| narrow_corridor | 0.25 | success | 17.4s | 1.06 ms | 7.90 m | 0.45 m/s | 0.28 rad/s | 0.89 |
| u_turn | 0.00 | success | 44.8s | 1.04 ms | 19.81 m | 0.44 m/s | 0.30 rad/s | 3.08 |
| u_turn | 0.25 | success | 39.9s | 1.09 ms | 19.26 m | 0.48 m/s | 0.28 rad/s | 0.69 |

## Readout

- All rows succeed with no exceptions.
- The corrected `u_turn` cell benefits most: time-to-goal drops from 44.75 s to
  39.9 s and distance drops from 19.81 m to 19.26 m.
- The new trajectory metrics show why the critic is useful: `u_turn` mean
  absolute curvature drops from 3.08 to 0.69, and `narrow_corridor` drops from
  1.38 to 0.89.
- `wall_gap` and `narrow_corridor` keep similar time-to-goal while reducing
  mean absolute yaw rate.

## Reproduce

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_metrics_path_angle path_angle
```
