# CudaNav native all-GPU closed loop

This harness closes the control loop around the deterministic CudaNav S-course
without requiring a ROS 2 installation:

```text
command-driven plant -> synthetic LiDAR -> GPU KISS-ICP
  -> GPU voxel map -> GPU ESDF costmap -> CUDA MPPI -> plant
```

The controller receives the KISS-ICP estimate, not ground truth. Ground truth is
used only to generate the next scan and to score collision, progress, and
odometry drift. The geometry, LiDAR shape, robot radius, and mission path match
the ROS 2 loopback simulator.

Build and run:

```bash
cmake --build build --target cudanav_gpu_closed_loop_s_course -j
python scripts/run_cudanav_gpu_closed_loop.py \
  --output-dir build/cudanav_gpu_closed_loop
```

The gate requires a reached goal within 0.30 m, zero collisions, less than 5%
final odometry drift, fewer than 5% 150 ms frame-deadline misses, at least 5 m
of command-caused motion, finite commands, and healthy localization/mapping
signals.

This is native GPU-core closed-loop evidence. It is neither a ROS 2 runtime
result nor real-world recorded-data evidence; those gates remain separate.
