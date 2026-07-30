# CudaNav Closed-Loop Bringup

This package supplies the first deterministic end-to-end CudaNav loop:

```text
CUDA MPPI cmd_vel
  -> command-driven simulator state
  -> synthetic PointCloud2
  -> GPU KISS-ICP odom/TF
  -> rolling GPU voxel occupancy
  -> typed GPU ESDF + Nav2 costmap layer
  -> next CUDA MPPI command
```

The simulator publishes ground truth only on `ground_truth` for evaluation. It
does not publish odometry or TF, and no production component subscribes to the
ground-truth topic.

```bash
ros2 launch cuda_nav_bringup cudanav_closed_loop.launch.py \
  output_path:=/tmp/cudanav_closed_loop.json \
  controller_config:=/path/to/controller.yaml
```

The automatic mission follows a fixed S-course and writes one JSON artifact
containing action outcome, collision count, true distance, final true goal
distance, odometry position error/drift percentage, and command deadline-miss
rate. This short mission is an integration smoke, not the v1.0 10-minute
release gate.

Use `scripts/run_cudanav_closed_loop.py` from the repository root to retain the
summary together with the exact commit, configuration hash, GPU identity, and
launch log. The runner launches with the retained configuration copy itself,
so the recorded hash cannot accidentally describe a stale colcon install-space
copy. `scripts/validate_cudanav_closed_loop.py` rechecks either the
`smoke` or strict `release` policy without rerunning the stack.
Release acquisition alternates the S-course 30 times, retains an MCAP rosbag,
and renders the recorded truth/odometry trajectory as a GIF.

The development Docker image exposes the same short integration path:

```bash
docker build --pull --no-cache -f docker/Dockerfile -t cudarobotics .
docker run --rm --gpus all -v "$PWD/out:/out" cudarobotics cudanav
```

It writes `/out/cudanav_closed_loop.json` and returns non-zero unless the
machine-readable smoke gate passes.

Lifecycle transitions use the standard `change_state` and `get_state` services
in dependency order. This avoids pretending that the custom lifecycle nodes
implement Nav2 bond semantics.

## Recorded-data shadow replay

For a rosbag containing compatible `sensor_msgs/PointCloud2`, TF, and
`nav_msgs/Path` streams:

```bash
ros2 launch cuda_nav_bringup cudanav_recorded_shadow.launch.py \
  params_file:="$(ros2 pkg prefix cuda_nav_bringup)/share/cuda_nav_bringup/config/controller_recorded_shadow.yaml" \
  diagnostics_csv:=/tmp/cudanav_diagnostics.csv \
  points_topic:=/points \
  path_topic:=/plan \
  readiness_timeout_sec:=30.0 \
  use_sim_time:=true
```

The launch runs GPU KISS-ICP, voxel mapping, ESDF, the Nav2 CUDA MPPI
controller, and a small adapter that forwards the newest recorded Path to the
FollowPath action. The adapter never synthesizes or transforms the recorded
path, and it retains that Path across transient lifecycle rejection or an
aborted shadow goal so startup races do not discard the only recorded plan.
CUDA MPPI commands are shadow outputs and do not modify subsequent bag
messages, so this launch must not be described as closed-loop success.

When replaying a bag with `/tf_static`, preserve its transient-local
durability so late-joining transform consumers receive the recorded static
tree:

```bash
ros2 bag play /path/to/bag \
  --qos-profile-overrides-path \
  "$(ros2 pkg prefix cuda_nav_bringup)/share/cuda_nav_bringup/config/rosbag_qos_overrides.yaml"
```

The 30-second readiness default leaves room for controller startup, runner
settling, and bags whose first point cloud follows their initial static TF.
This offline-only launch also disables wall/`/clock` age rejection because
some immutable datasets record sensor header stamps in a different epoch.
Zero/non-monotonic stamps are still rejected; live CudaNav launches retain
their bounded scan-age checks. Its dedicated controller config treats
never-observed cells as traversable for shadow scoring; recorded occupied
cells remain lethal and raw-cloud clearance is evaluated independently.
