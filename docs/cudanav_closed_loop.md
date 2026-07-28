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
- a conservative `smoke_pass` boolean.

The S-course geometry and path clearance have a no-ROS deterministic unit test.
The ROS Jazzy workflow compiles the complete stack, loads both plugins, and runs
all no-GPU tests. A real GPU run of this launch is still required before its
JSON is accepted as benchmark evidence. The short smoke also does not replace
the 10-minute v1.0 closed-loop gate.
