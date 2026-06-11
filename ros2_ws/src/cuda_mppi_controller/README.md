# cuda_mppi_controller

GPU-accelerated MPPI controller plugin for [Nav2](https://docs.nav2.org/).
A drop-in alternative to `nav2_mppi_controller` that runs every sampled
trajectory rollout on the GPU — **1 CUDA thread = 1 trajectory**, the same
parallel pattern used across [CudaRobotics](https://github.com/rsasaki0109/CudaRobotics).

Because rollouts are embarrassingly parallel, sample counts that are
impractical on CPU stay comfortably inside a 20 Hz control budget:

| batch_size (K) | mean solve time | max | control budget @ 20 Hz |
|---:|---:|---:|---:|
| 2,048  | 2.1 ms | 10.4 ms | 50 ms |
| 8,192  | 2.6 ms | 13.7 ms | 50 ms |
| 16,384 | 3.7 ms | 9.8 ms | 50 ms |
| 65,536 | 10.4 ms | 23.0 ms | 50 ms |

Measured with `mppi_gpu_standalone` (T=56, dt=0.05, 200×200 costmap upload
included) on a benchmark GPU, ROS 2 Jazzy, CUDA 12.0. For reference,
the stock CPU MPPI controller typically runs K≈2,000; here K=65,536 still
fits the cycle with room to spare.

## Head-to-head vs nav2_mppi_controller (CPU)

`controller_benchmark` loads both controllers through pluginlib and drives
the same plant through the same costmap and plan at 20 Hz
(benchmark CPU vs benchmark GPU):

| | K=1–2k | K=5k | K=10k | K=16k | K=65k |
|---|---:|---:|---:|---:|---:|
| nav2 MPPI (CPU) mean | 3.6–5.2 ms | 13.2 ms | 27.4 ms | — | — |
| CUDA MPPI (GPU) mean | 2.6 ms | — | — | 3.9 ms | 10.6 ms |

Time-to-goal matches the CPU baseline, and improves monotonically with K on
the GPU (16.8 s @ 2k → 16.0 s @ 65k) — more samples buy better trajectories.
Full setup, tuning notes, and reproduction steps:
[`docs/results/cuda_mppi_vs_nav2_2026-06-10.md`](../../../docs/results/cuda_mppi_vs_nav2_2026-06-10.md).

![side-by-side rollout](../../../gif/cuda_mppi_vs_nav2_cpu.gif)

## Status

Experimental, but verified end-to-end in the full Nav2 stack — bt_navigator →
planner_server → controller_server (this plugin) → velocity_smoother — both in
the nav2 loopback simulation and in the **Gazebo physics simulation**
(TurtleBot3 dynamics + lidar + AMCL localization), two-waypoint mission each:

<img src="../../../gif/cuda_mppi_nav2_gazebo.gif" alt="nav2 gazebo demo" width="420"/>

```bash
# terminal 1 — Nav2 + Gazebo (or tb3_loopback_simulation.launch.py for the lightweight sim)
ROS_DOMAIN_ID=101 PYTHONNOUSERSITE=1 ros2 launch nav2_bringup \
  tb3_simulation_launch.py headless:=True use_rviz:=False \
  params_file:=$(ros2 pkg prefix cuda_mppi_controller)/share/cuda_mppi_controller/config/nav2_loopback_demo.yaml

# terminal 2 — waypoint mission + trajectory recording + GIF
ROS_DOMAIN_ID=101 PYTHONNOUSERSITE=1 python3 scripts/run_nav2_loopback_demo.py /tmp/nav2_gz_demo amcl
python3 scripts/render_nav2_loopback_demo.py /tmp/nav2_gz_demo cuda_mppi_nav2_gazebo.gif

# (pick any quiet ROS_DOMAIN_ID; PYTHONNOUSERSITE avoids user-site numpy clashes)
```

Motion models: **DiffDrive** (`vx`, `ωz`), **Ackermann** (curvature limit
`|ωz| ≤ |vx| / min_turning_r`), **Omni** (adds `vy`). Costs implemented:

- **Path align** — squared lateral distance to the global plan window
- **Path follow** — distance to a point `follow_lookahead` ahead on the plan
  (pulls rollouts forward, like nav2's PathFollowCritic)
- **Path angle** — yaw error to the local path tangent near the follow point
  (like nav2's PathAngleCritic)
- **Curvature speed** — optional overspeed penalty near sharp bends in the
  followed path; disabled by default so straight-line cruise tuning is unchanged
- **Goal** — linear terminal distance to the window end, yaw activates near
  the final goal
- **Costmap** — per-step lookup in the local costmap; lethal/inscribed cells
  add a collision penalty, inflated cells add a graded cost
- **Distance field** (optional, `distance_field_weight`) — builds a truncated
  GPU distance-to-obstacle field from the same local costmap each control
  cycle and adds a smooth clearance cost inside `distance_field_cutoff`
- **Footprint** (optional, `consider_footprint`) — the robot's polygon
  footprint is swept along each rollout; edge cells are sampled at costmap
  resolution at intermediate SE(2) poses, so rotation-in-place can catch
  corner clips between rollout samples. Gated on non-zero inflated cost, so
  it requires an inflation layer and stays cheap in free space
- **Retreat** — when every sampled rollout collides but a previous valid
  sequence exists, the controller returns a scaled reverse command from that
  sequence instead of throwing `NoValidControl` every cycle
- **Smoothness / backward motion / control limits**

Motion-model verification (2026-06-10): DiffDrive, Ackermann, and Omni all
succeed on the wall-gap benchmark and in the nav2 loopback stack.
See [`docs/results/cuda_mppi_motion_models_2026-06-10.md`](../../../docs/results/cuda_mppi_motion_models_2026-06-10.md)
and run `./scripts/run_nav2_motion_model_demos.sh` for loopback GIFs.

Still future work: broader scenario coverage beyond the wall-gap / tb3 sandbox
cells.

## Build

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Requires ROS 2 Jazzy (or any distro shipping `nav2_core`), CUDA Toolkit >= 12,
and an NVIDIA GPU.

## Verify without a robot

```bash
# pluginlib discovery, exactly how controller_server loads it
ros2 run cuda_mppi_controller plugin_load_test

# invalid parameter rejection, no CUDA driver required
ros2 run cuda_mppi_controller parameter_validation_test

# closed-loop synthetic scenario (wall with a gap) + solve-time report
ros2 run cuda_mppi_controller mppi_gpu_standalone           # default K=2048
ros2 run cuda_mppi_controller mppi_gpu_standalone 16384     # K sweep
ros2 run cuda_mppi_controller mppi_gpu_standalone 2048 ackermann   # or omni / footprint
ros2 run cuda_mppi_controller mppi_gpu_standalone 2048 esdf        # distance-field critic
```

## Use with Nav2

Point `controller_server` at the plugin (see
[`config/cuda_mppi_params.example.yaml`](config/cuda_mppi_params.example.yaml)):

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "cuda_mppi_controller::CudaMppiController"
      batch_size: 8192
      time_steps: 56
      model_dt: 0.05
```

### Parameters

| name | default | description |
|---|---:|---|
| `batch_size` | 2048 | sampled trajectories per cycle (1 CUDA thread each) |
| `time_steps` | 56 | horizon length |
| `model_dt` | 0.05 | [s] integration step |
| `iteration_count` | 1 | optimizer iterations per control cycle |
| `motion_model` | DiffDrive | DiffDrive / Ackermann / Omni |
| `v_max` / `v_min` / `w_max` | 0.5 / -0.35 / 1.9 | control limits |
| `vy_max` | 0.5 | lateral velocity limit (Omni) |
| `min_turning_r` | 0.2 | [m] minimum turning radius (Ackermann) |
| `v_std` / `w_std` | 0.2 / 0.4 | sampling noise std |
| `vy_std` | 0.2 | lateral noise std (Omni) |
| `consider_footprint` | false | polygon footprint collision check (needs inflation layer) |
| `enable_retreat` | true | back out from the last valid sequence on all-colliding rollouts |
| `retreat_scale` | 0.5 | scale applied when reversing the last valid controls |
| `temperature` | 0.12 | MPPI softmin λ |
| `goal_weight` | 20.0 | terminal local-goal distance (linear) |
| `goal_yaw_weight` | 3.0 | terminal yaw error near the final goal |
| `path_weight` | 10.0 | lateral deviation² from the plan |
| `path_follow_weight` | 5.0 | pull toward a point ahead on the plan |
| `path_angle_weight` | 0.25 | heading error to the local path tangent |
| `curvature_speed_weight` | 0.0 | optional penalty when forward speed exceeds the curvature target |
| `curvature_speed_min` | 0.18 | [m/s] floor for curvature-limited target speed |
| `follow_lookahead` | 1.0 | [m] how far ahead that point is |
| `costmap_weight` | 3.0 | graded cost for inflated cells |
| `distance_field_weight` | 0.0 | optional ESDF-style clearance cost; disabled by default |
| `distance_field_cutoff` | 0.75 | [m] distance-field penalty radius |
| `smoothness_weight` | 0.2 | (Δu)² between consecutive steps |
| `backward_weight` | 0.5 | penalty on v < 0 |
| `speed_weight` | 3.0 | penalty on (v_max − v): cruise at the limit |
| `angular_weight` | 0.5 | penalty on wz²: damps heading random walk |
| `yaw_goal_activation_dist` | 0.5 | [m] range to enable the yaw goal cost |
| `lookahead_dist` | 3.0 | [m] global plan window fed to the GPU |
| `transform_tolerance` | 0.1 | [s] TF lookup tolerance |
| `diagnostics_log_period` | 0.0 | [s] periodic one-line solve/valid-rollout logging; 0 disables |
| `diagnostics_csv_path` | `""` | optional per-cycle diagnostics CSV path |

Parameters above are validated at configure time and during live ROS parameter
updates. Invalid values, such as zero horizon length, non-positive model step,
unknown motion models, or negative cost weights, are rejected before the GPU
optimizer is rebuilt.

Set `diagnostics_log_period` to a positive value for throttled controller logs,
or set `diagnostics_csv_path` to capture one row per control cycle. The CSV
includes solve time, best/mean rollout cost, valid rollout count and ratio,
all-colliding/retreat flags, path window size, costmap size, and the selected
command.

Render a diagnostics CSV into a compact plot and Markdown summary:

```bash
python3 scripts/render_cuda_mppi_diagnostics.py \
  /tmp/cuda_mppi_diagnostics.csv \
  --output-stem docs/results/cuda_mppi_diagnostics_run
```

For bag replay or live-stack evaluation, use
[`docs/cuda_mppi_bag_eval.md`](../../../docs/cuda_mppi_bag_eval.md) and
`scripts/run_cuda_mppi_bag_eval.py` to keep launch logs, recorded topics,
diagnostics, and command metadata under one output directory.

## Benchmark scenarios

`controller_benchmark` runs closed-loop CPU vs GPU comparisons on synthetic maps:

```bash
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench wall_gap
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench narrow_corridor
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench u_turn
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench double_gap
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench moving_crossing quick
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench all
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench double_gap quick
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench double_gap cpu_gpu
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench esdf
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench path_angle
ros2 run cuda_mppi_controller controller_benchmark /tmp/bench curvature_speed
```

The optional preset is `full` by default. Use `quick` for GPU K=2,048/8,192
smoke runs, or `cpu_gpu` for CPU K=2,000 vs GPU K=8,192. The `esdf`,
`path_angle`, and `curvature_speed` benchmark families keep their fixed
comparison sets.

`all` also runs Ackermann/Omni GPU configs (`gpu_ackermann_K8192`, `gpu_omni_K8192`).
`esdf` runs a GPU-only comparison of the default costmap critic against the
optional distance-field clearance critic. Results:
[`docs/results/cuda_mppi_esdf_2026-06-11.md`](../../../docs/results/cuda_mppi_esdf_2026-06-11.md).
`path_angle` runs a GPU-only comparison with and without the path-angle critic.
Results:
[`docs/results/cuda_mppi_path_angle_2026-06-12.md`](../../../docs/results/cuda_mppi_path_angle_2026-06-12.md).
`curvature_speed` runs a GPU-only comparison with and without the optional
curvature speed critic. Results:
[`docs/results/cuda_mppi_curvature_speed_2026-06-12.md`](../../../docs/results/cuda_mppi_curvature_speed_2026-06-12.md).
Extended `double_gap` and `moving_crossing` scenario results:
[`docs/results/cuda_mppi_extended_scenarios_2026-06-12.md`](../../../docs/results/cuda_mppi_extended_scenarios_2026-06-12.md).
Each scenario writes `summary.csv` with outcome, solve-time, and trajectory
quality columns: distance traveled, mean/max command speed, mean/max absolute
yaw rate, and mean absolute curvature.

Loopback motion-model configs: `config/nav2_loopback_demo_{ackermann,omni}.yaml`.

## Architecture

```
cuda_mppi_controller.cpp   nav2_core::Controller (ROS layer, no CUDA)
        │  PIMPL boundary
mppi_gpu.cu                rollout_kernel        1 thread = 1 trajectory
                           update_controls_kernel softmin-weighted average
```

The local costmap (raw `unsigned char` grid) is uploaded to the GPU each
cycle — at typical local costmap sizes this is tens of microseconds. The
nominal control sequence stays on the GPU between cycles (warm start).
