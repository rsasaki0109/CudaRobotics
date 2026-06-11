# CUDA MPPI ESDF Clearance Critic (2026-06-11)

Closed-loop GPU-only comparison of the default costmap critic against
the optional ESDF-style distance-field clearance critic added to
`cuda_mppi_controller`.

Hardware: local CUDA-capable benchmark machine, ROS 2 Jazzy, Release build.
Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,
K = 8192, T = 56, dt = 0.05. ESDF row uses
`distance_field_weight=12.0` and `distance_field_cutoff=0.8`.

CSV: [`cuda_mppi_esdf_2026-06-11.csv`](cuda_mppi_esdf_2026-06-11.csv)

## Results

| scenario | critic | result | sim time | mean solve | p95 | max | exceptions |
|---|---|---:|---:|---:|---:|---:|---:|
| wall_gap | costmap | success | 16.5s | 1.16 ms | 1.29 ms | 3.46 ms | 0 |
| wall_gap | costmap + ESDF | success | 16.6s | 1.18 ms | 1.23 ms | 1.37 ms | 0 |
| narrow_corridor | costmap | success | 17.4s | 1.16 ms | 1.32 ms | 1.38 ms | 0 |
| narrow_corridor | costmap + ESDF | success | 17.8s | 1.17 ms | 1.22 ms | 1.25 ms | 0 |
| u_turn | costmap | timeout | 60.0s | 1.07 ms | 1.12 ms | 1.17 ms | 0 |
| u_turn | costmap + ESDF | timeout | 60.0s | 1.20 ms | 1.23 ms | 1.43 ms | 0 |

## Readout

- The ESDF critic is disabled by default; this benchmark enables it only
  for the `gpu_esdf_K8192` rows.
- On `wall_gap` and `narrow_corridor`, ESDF keeps the same success result
  with slightly more conservative time-to-goal and similar mean solve
  time.
- In this run ESDF lowers the p95/max solve-time spikes on the two
  successful corridor cells, but it is not a speed optimization.
- The `u_turn` cell remains unsolved at K=8192 for both critics; ESDF
  does not replace the need for better global-plan tracking or a richer
  scenario-specific critic there.
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
