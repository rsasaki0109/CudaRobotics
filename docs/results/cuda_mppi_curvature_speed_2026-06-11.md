# CUDA MPPI Curvature Speed Critic (2026-06-11)

Closed-loop GPU-only comparison of the default costmap/path critics with and
without the optional curvature speed critic. The critic estimates path curvature
around the follow point and penalizes forward speed above a curvature-limited
target. It is disabled by default (`curvature_speed_weight=0.0`) because it is a
speed-vs-smoothness tuning knob rather than a universal time-to-goal win.

Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.
Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,
K = 8192, T = 56, dt = 0.05, `path_angle_weight=0.25`.

CSV: [`cuda_mppi_curvature_speed_2026-06-11.csv`](cuda_mppi_curvature_speed_2026-06-11.csv)

## Results

| scenario | curvature speed weight | result | sim time | mean solve | p95 | max | exceptions |
|---|---:|---:|---:|---:|---:|---:|---:|
| wall_gap | 0.0 | success | 16.4s | 1.35 ms | 1.50 ms | 10.20 ms | 0 |
| wall_gap | 8.0 | success | 16.4s | 1.35 ms | 1.39 ms | 10.10 ms | 0 |
| narrow_corridor | 0.0 | success | 17.4s | 1.38 ms | 1.47 ms | 10.51 ms | 0 |
| narrow_corridor | 8.0 | success | 17.4s | 1.39 ms | 1.42 ms | 10.12 ms | 0 |
| u_turn | 0.0 | success | 39.9s | 1.30 ms | 1.54 ms | 2.01 ms | 0 |
| u_turn | 8.0 | success | 41.4s | 1.24 ms | 1.48 ms | 2.10 ms | 0 |

## Readout

- `wall_gap` and `narrow_corridor` are unchanged because the followed path is
  straight through those cells.
- In `u_turn`, the enabled critic slows near the two 90-degree bends. The
  average speed within 1 m of the bend points changes from 0.497 m/s to
  0.481 m/s in this run.
- The time-to-goal trade-off is explicit: the enabled row takes 41.4 s vs
  39.9 s, so the checked-in default leaves the critic disabled.
- Use `curvature_speed_weight > 0` when bend-entry smoothness or model safety is
  more important than the fastest possible closed-loop completion time.

## Reproduce

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_curvature_speed_bench curvature_speed
```
