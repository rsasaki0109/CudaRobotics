# CUDA MPPI Curvature Speed Critic Metrics Refresh (2026-06-12)

Closed-loop GPU-only comparison of the default costmap/path critics with and
without the optional curvature speed critic. This refresh uses the
trajectory-quality metrics added to `controller_benchmark` after the original
2026-06-11 report.

Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.
Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,
K = 8192, T = 56, dt = 0.05, `path_angle_weight=0.25`.

CSV: [`cuda_mppi_curvature_speed_2026-06-12.csv`](cuda_mppi_curvature_speed_2026-06-12.csv)

## Results

| scenario | curvature speed weight | result | sim time | mean solve | distance | mean speed | mean \|w\| | mean \|curv\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wall_gap | 0.0 | success | 16.4s | 1.06 ms | 7.80 m | 0.48 m/s | 0.25 rad/s | 0.75 |
| wall_gap | 8.0 | success | 16.4s | 1.11 ms | 7.80 m | 0.48 m/s | 0.25 rad/s | 0.75 |
| narrow_corridor | 0.0 | success | 17.4s | 1.20 ms | 7.90 m | 0.45 m/s | 0.28 rad/s | 0.89 |
| narrow_corridor | 8.0 | success | 17.4s | 1.11 ms | 7.90 m | 0.45 m/s | 0.28 rad/s | 0.89 |
| u_turn | 0.0 | success | 39.9s | 1.09 ms | 19.26 m | 0.48 m/s | 0.28 rad/s | 0.69 |
| u_turn | 8.0 | success | 41.4s | 1.14 ms | 19.90 m | 0.48 m/s | 0.33 rad/s | 0.79 |

## Readout

- All rows succeed with no exceptions.
- Straight-path cells are unchanged because local path curvature is zero.
- The aggregate trajectory metrics do not show a smoothness improvement from
  the enabled row in this fixed-seed run. In `u_turn`, the enabled row takes
  longer, travels farther, and increases mean absolute curvature.
- The checked-in default therefore remains `curvature_speed_weight=0.0`; this
  critic is an optional safety/smoothness knob for users to tune against their
  own vehicle model and limits.

## Reproduce

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_metrics_curvature_speed curvature_speed
```
