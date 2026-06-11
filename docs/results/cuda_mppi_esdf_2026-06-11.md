# CUDA MPPI ESDF Clearance Critic (2026-06-11)

Closed-loop GPU-only comparison of the default costmap critic against
the optional ESDF-style distance-field clearance critic added to
`cuda_mppi_controller`.

Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.
Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,
K = 8192, T = 56, dt = 0.05. ESDF row uses
`distance_field_weight=12.0` and `distance_field_cutoff=0.8`; both rows
use the default path-angle critic (`path_angle_weight=0.25`).

CSV: [`cuda_mppi_esdf_2026-06-11.csv`](cuda_mppi_esdf_2026-06-11.csv)

## Results

| scenario | critic | result | sim time | mean solve | p95 | max | exceptions |
|---|---|---:|---:|---:|---:|---:|---:|
| wall_gap | costmap | success | 16.4s | 1.16 ms | 1.38 ms | 1.62 ms | 0 |
| wall_gap | costmap + ESDF | success | 17.6s | 1.21 ms | 1.29 ms | 1.50 ms | 0 |
| narrow_corridor | costmap | success | 17.4s | 1.17 ms | 1.37 ms | 1.61 ms | 0 |
| narrow_corridor | costmap + ESDF | success | 16.8s | 1.22 ms | 1.29 ms | 1.38 ms | 0 |
| u_turn | costmap | success | 39.9s | 1.09 ms | 1.15 ms | 1.29 ms | 0 |
| u_turn | costmap + ESDF | success | 39.9s | 1.23 ms | 1.30 ms | 1.47 ms | 0 |

## Readout

- The ESDF critic is disabled by default; this benchmark enables it only
  for the `gpu_esdf_K8192` rows.
- All three corrected scenarios succeed with both the default costmap
  critic and the ESDF clearance critic at K=8192.
- ESDF keeps similar solve latency and time-to-goal on the corrected
  scenarios in this run; it is a clearance smoother rather than a speed
  optimization.
- The corrected `u_turn` path goes around the obstacle endpoint; the
  previous benchmark path crossed a lethal wall cell and was therefore
  not a valid planner-tracking test.
- The distance-field cost is a clearance smoother, not a replacement for
  lethal-cell collision rejection or footprint checking.

## Reproduce

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_esdf_bench esdf
cd ..
python3 scripts/render_cuda_mppi_esdf_benchmark.py /tmp/mppi_esdf_bench 2026-06-11
```
