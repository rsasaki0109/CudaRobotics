# cuda_mppi_controller motion models (2026-06-10)

Closed-loop verification of **DiffDrive**, **Ackermann**, and **Omni** motion
models in `cuda_mppi_controller`, using the same synthetic wall-gap scenario as
the CPU-vs-GPU head-to-head benchmark.

## Synthetic benchmark (`controller_benchmark`)

Hardware: local CUDA-capable benchmark machine, ROS 2 Jazzy, Release build.
Scenario: 10 m × 10 m costmap, lethal wall at x = 5 m with a 2 m gap,
plan from (1, 5) to (9, 5), 20 Hz closed loop, K = 8192, T = 56, dt = 0.05.

| motion model | result | sim time to goal | mean solve | p95 | max |
|---|---|---:|---:|---:|---:|
| DiffDrive | success | 16.5 s | 1.40 ms | 4.91 ms | 5.52 ms |
| Ackermann (`min_turning_r=0.2`) | success | 16.7 s | 1.13 ms | 2.88 ms | 5.68 ms |
| Omni (`vy_max=0.5`) | success | 14.2 s | 1.08 ms | 2.66 ms | 4.13 ms |

Side-by-side rollout GIF:
[`gif/cuda_mppi_motion_models_wall_gap.gif`](../../gif/cuda_mppi_motion_models_wall_gap.gif)

Omni reaches the goal faster in this straight-ish corridor because lateral
`vy` can trim cornering time; Ackermann matches DiffDrive quality with the
curvature gate enforced inside the rollout kernel.

## Nav2 loopback sim (full stack)

Full-stack two-waypoint missions on the tb3 sandbox map are supported via:

```bash
# terminal 1 — pick a quiet ROS_DOMAIN_ID and motion-model params file
export ROS_DOMAIN_ID=107 PYTHONNOUSERSITE=1
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ros2 launch nav2_bringup tb3_loopback_simulation.launch.py use_rviz:=False \
  params_file:=$(ros2 pkg prefix cuda_mppi_controller)/share/cuda_mppi_controller/config/nav2_loopback_demo_ackermann.yaml

# terminal 2
python3 scripts/run_nav2_loopback_demo.py /tmp/nav2_ackermann
python3 scripts/render_nav2_loopback_demo.py /tmp/nav2_ackermann \
  cuda_mppi_nav2_loopback_ackermann.gif "loopback sim, GPU MPPI Ackermann K=8,192 @ 20 Hz"
```

Or run all three motion models end-to-end:

```bash
./scripts/run_nav2_motion_model_demos.sh /tmp/nav2_motion_models 107
```

Configs: `config/nav2_loopback_demo_{ackermann,omni}.yaml` (full Nav2 param
files derived from the DiffDrive baseline).

## Reproduce benchmark numbers

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
mkdir -p /tmp/mppi_motion_bench
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_motion_bench wall_gap
python3 scripts/render_cuda_mppi_motion_models.py /tmp/mppi_motion_bench
```
