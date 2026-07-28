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
  output_path:=/tmp/cudanav_closed_loop.json
```

The automatic mission follows a fixed S-course and writes one JSON artifact
containing action outcome, collision count, true distance, final true goal
distance, odometry position error/drift percentage, and command deadline-miss
rate. This short mission is an integration smoke, not the v1.0 10-minute
release gate.

Use `scripts/run_cudanav_closed_loop.py` from the repository root to retain the
summary together with the exact commit, configuration hash, GPU identity, and
launch log. `scripts/validate_cudanav_closed_loop.py` rechecks either the
`smoke` or strict `release` policy without rerunning the stack.
Release acquisition alternates the S-course 30 times, retains an MCAP rosbag,
and renders the recorded truth/odometry trajectory as a GIF.

Lifecycle transitions use the standard `change_state` and `get_state` services
in dependency order. This avoids pretending that the custom lifecycle nodes
implement Nav2 bond semantics.
