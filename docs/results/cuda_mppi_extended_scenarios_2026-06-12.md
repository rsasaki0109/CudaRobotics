# CUDA MPPI Extended Controller Scenarios (2026-06-12)

Closed-loop `cuda_mppi_controller` benchmark summary for scenarios beyond
the original wall-gap / narrow-corridor / U-turn smoke set.

Hardware: local CUDA-capable benchmark machine, ROS 2 workspace, Release build.
Scenario setup: 10 m x 10 m synthetic costmap, 20 Hz closed loop,
T = 56, dt = 0.05. The exact controller rows are preserved in the CSV.

CSV: [`cuda_mppi_extended_scenarios_2026-06-12.csv`](cuda_mppi_extended_scenarios_2026-06-12.csv)

## Scenario Intent

- `double_gap` exercises path-following through two separated wall gaps with a deliberately bent global path.
- `moving_crossing` repaints a crossing obstacle into the costmap during closed-loop control, so it is a dynamic-map smoke test rather than a static obstacle benchmark.

## Results

| scenario | label | result | K | sim time | mean solve | p95 | distance | mean speed | mean \|w\| | mean \|curv\| | exceptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| double_gap | cpu_mppi_K2000 | success | 2000 | 29.4s | 6.83 ms | 8.25 ms | 12.04 m | 0.41 m/s | 0.19 rad/s | 0.57 | 0 |
| double_gap | gpu_mppi_K8192 | success | 8192 | 23.4s | 1.13 ms | 1.32 ms | 10.75 m | 0.46 m/s | 0.38 rad/s | 1.01 | 0 |
| moving_crossing | gpu_mppi_K2048 | success | 2048 | 20.1s | 0.42 ms | 0.45 ms | 8.88 m | 0.44 m/s | 0.35 rad/s | 1.12 | 0 |
| moving_crossing | gpu_mppi_K8192 | success | 8192 | 19.6s | 1.07 ms | 1.11 ms | 8.84 m | 0.45 m/s | 0.35 rad/s | 1.05 | 0 |

## Readout

- Treat this as extended controller coverage, not a universal navigation
  benchmark. The scenarios are synthetic and intentionally small enough
  to run during local plugin development.
- `double_gap` is useful for spotting path-window, path-angle, and
  smoothing regressions that a straight wall-gap benchmark can miss.
- `moving_crossing` is useful for costmap-refresh and diagnostics checks;
  it does not replace a real perception or tracking pipeline.
- Inspect per-cycle diagnostics with
  `scripts/render_cuda_mppi_diagnostics.py` when a row times out,
  retreats, or shows low valid-rollout ratios.

## Reproduce

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_extended_scenarios double_gap cpu_gpu
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_extended_scenarios moving_crossing quick
cd ..
python3 scripts/render_cuda_mppi_extended_scenarios.py /tmp/mppi_extended_scenarios 2026-06-12
```
