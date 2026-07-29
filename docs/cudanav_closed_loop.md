# CudaNav Deterministic Closed Loop

`cuda_nav_bringup` is the first integration harness in which CUDA MPPI commands
change the next sensor observation consumed by the autonomy stack.

The simulator integrates `TwistStamped`, ray-casts a fixed S-course, and emits
a schema-correct XYZ `PointCloud2` with three vertical return levels. It does
not emit odometry or TF. Ground truth is published in the isolated
`world_truth` frame only for the evidence recorder.

The production-side loop is:

```text
PointCloud2 -> GPU KISS-ICP odom/TF -> GPU voxel occupancy
            -> typed GPU ESDF
            -> Nav2 voxel costmap layer -> CUDA MPPI -> TwistStamped
            -> simulator motion -> next PointCloud2
```

Run:

```bash
ros2 launch cuda_nav_bringup cudanav_closed_loop.launch.py \
  output_path:=/tmp/cudanav_closed_loop.json
```

The short automatic mission writes:

- Nav2 `FollowPath` outcome;
- latched collision state and collision count;
- ground-truth travel and final goal distance;
- final KISS-ICP position error and drift percentage;
- command interval count, deadline misses, and miss rate;
- diagnostic component coverage, ERROR count, and every observed
  failure/dropped counter;
- a conservative `smoke_pass` boolean.

For a retained run directory, use:

```bash
python scripts/run_cudanav_closed_loop.py \
  --output-dir build/cudanav_runs/smoke_001 \
  --profile smoke
python scripts/validate_cudanav_closed_loop.py \
  build/cudanav_runs/smoke_001 --profile smoke
```

The harness refuses a non-empty output directory and records the full git
commit, dirty-worktree state, controller-config SHA-256, GPU UUID/driver,
selected environment, exact launch command, launch log, mission summary, and
machine-readable gate results. A passing manifest requires a clean worktree and
the retained config bytes must match the recorded hash. The runner passes that
retained config copy back into the launch through `controller_config:=...`;
independent validation checks its retained filename and, while the original
command path remains available, its bytes. Cross-machine copies retain the
filename plus recorded SHA-256 binding. This keeps the executed Nav2 parameters
bound even when the colcon install space contains an older package copy.
Artifact paths are constrained to the run directory when revalidated.

The `release` profile is intentionally stricter: at least 600 seconds,
collision count zero, drift below 1%, command deadline misses below 1%, and
retained rosbag and video artifacts. It defaults to 30 alternating S-course
traversals, a 1,200-second mission timeout, full MCAP recording, and a rendered
truth-vs-odometry trajectory GIF:

```bash
python scripts/run_cudanav_closed_loop.py \
  --output-dir build/cudanav_runs/release_001 \
  --profile release
```

The renderer uses Pillow (`python3-pil` on Ubuntu). The bag contains sensor,
odometry/TF, occupancy, typed ESDF, commands, ground truth, collision state,
and per-component diagnostics. A missing MCAP metadata file or GIF keeps the
release manifest red; smoke evidence cannot be relabelled as the v1.0 gate.
Both profiles also require diagnostics from odometry, mapping, and ESDF, zero
ERROR status, and zero reported transform/schema/capacity/dropped counters.

Ubuntu/Jazzy artifact dependencies:

```bash
sudo apt install python3-pil ros-jazzy-ros2bag \
  ros-jazzy-rosbag2-storage-mcap
```

For hardware-matrix acquisition and cross-run invariants, continue with
[cudanav_multi_gpu.md](cudanav_multi_gpu.md).

The S-course geometry and path clearance have a no-ROS deterministic unit test.
The ROS Jazzy workflow compiles the complete stack, loads both plugins, and runs
all no-GPU tests. A real GPU run of this launch is still required before its
JSON is accepted as benchmark evidence. The short smoke also does not replace
the 10-minute v1.0 closed-loop gate.
