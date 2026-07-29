# CudaNav Autonomy Evidence Suite

`scripts/run_autonomy_suite.py` is the project-level entry point for CudaNav
release evidence. It orchestrates three deliberately distinct modes:

| Mode | Meaning | Child runner |
|---|---|---|
| `closed_loop` | Commands affect subsequent simulated robot state | `run_cudanav_closed_loop.py` |
| `real_rosbag_shadow` | CUDA controller is evaluated against recorded motion; not closed loop | `run_cudanav_rosbag_replay.py` |
| `multi_gpu` | The closed-loop smoke reproduces across physical GPU models | `run_cudanav_multi_gpu.py` |

The aggregate gate never relabels recorded/shadow evidence as closed-loop
success. It independently reruns every child validator and requires identical
full git commit and controller-config SHA-256 values across all three modes.

## Release run

```bash
python3 scripts/run_autonomy_suite.py \
  --output-dir build/cudanav_autonomy_release \
  --profile release \
  --bag /data/pointcloud_nav_run \
  --derived-path-bag build/cudanav_real_dataset/path_sidecar \
  --dataset-materialization \
    build/cudanav_real_dataset/materialization.json \
  --evaluation-db /data/pointcloud_nav_run/rosbag2_0.db3 \
  --controller-config ros2_ws/src/cuda_nav_bringup/config/controller.yaml \
  --controller-command \
    "ros2 launch cuda_nav_bringup cudanav_recorded_shadow.launch.py \
     params_file:={controller_config} \
     diagnostics_csv:={diagnostics_csv} \
     points_topic:=/pandar_points \
     path_topic:=/cuda_nav/derived_plan \
     sensor_frame:=" \
  --multi-gpu-run /evidence/other_gpu/cudanav_smoke
```

The native recorded-shadow launch expects compatible PointCloud2, TF, and Path
streams. LaserScan-only datasets require an explicitly recorded conversion or
a platform-specific launch; the suite does not silently reinterpret LaserScan
as the KISS-ICP input.

The selected real-sensor source and its stricter derived-Path provenance
contract are documented in
[`cudanav_real_dataset.md`](cudanav_real_dataset.md). The sidecar generator and
rosbag2 multi-input replay integration are implemented; the checked-in
selection remains `valid: true, ready: false` until the public bag is
downloaded and materialized. A derived Path always uses the distinct
`real_sensor_shadow_with_derived_path` label; it cannot be relabelled as a
recorded Path or closed-loop execution.

The runner selects the PointCloud2 quality evaluator when a dataset
materialization is supplied. It pairs recorded cloud header stamps with actual
CUDA MPPI diagnostics commands and retains the result as
`real_sensor_shadow_with_derived_path`. The older Twist/Odometry/LaserScan
path remains available for compatible DB3 bags.

The local release closed-loop directory is automatically included in the
cross-machine GPU aggregate. Repeat `--multi-gpu-run` for more imported
machines. Alternatively, use `--multi-gpu-devices 0,1` when two distinct GPU
models are installed in one host.

Interrupted suites retain attempt-numbered child directories and driver logs:

```bash
python3 scripts/run_autonomy_suite.py \
  ...same arguments... \
  --resume
```

Resume is refused if the commit, inputs, profile, controller command, or
hardware-collection plan changes. A previously valid child attempt is
revalidated and reused; an invalid attempt remains visible and a new numbered
attempt is created.

Validate the aggregate independently:

```bash
python3 scripts/validate_autonomy_suite.py \
  build/cudanav_autonomy_release
```

After the release suite and the commit-matched ROS Jazzy workflow artifact
both pass, freeze the portable systems-paper artifacts:

```bash
python3 scripts/publish_cudanav_systems_evidence.py \
  --suite-dir build/cudanav_autonomy_release \
  --ros-ci build/paper/ros_jazzy_ci/ros_jazzy_ci_evidence.json \
  --output-dir docs/results \
  --prefix cudanav_systems_YYYY-MM-DD

python3 scripts/publish_cudanav_systems_evidence.py \
  --suite-dir build/cudanav_autonomy_release \
  --ros-ci build/paper/ros_jazzy_ci/ros_jazzy_ci_evidence.json \
  --output-dir docs/results \
  --prefix cudanav_systems_YYYY-MM-DD \
  --check
```

The publisher independently reruns the aggregate and ROS CI validators. It
refuses smoke profiles, dirty autonomy suites, failed child modes, or differing
commits. Its summary preserves the semantic split between closed-loop
simulation and recorded-motion shadow evaluation.

The release suite passes only when:

- the 10-minute closed-loop release policy passes with retained bag and video;
- the closed-loop MCAP is content-addressed and has positive message counts
  for every required sensor, state, control, collision, and diagnostic topic;
- the content-addressed real rosbag release policy passes and remains labelled
  `shadow_controller_with_recorded_motion`;
- the shadow output MCAP contains positive message counts for CudaNav commands,
  odometry, occupancy, and typed ESDF;
- the multi-GPU matrix passes with at least two physical UUIDs and two model
  names;
- all child manifests and their content hashes are valid;
- one clean full commit and one controller configuration bind every mode.

## Development smoke

A closed-loop-only development smoke is available:

```bash
python3 scripts/run_autonomy_suite.py \
  --output-dir build/cudanav_autonomy_smoke \
  --profile smoke
```

This validates orchestration but does not satisfy the release or systems-paper
gate because real-data and multi-GPU modes are absent.
