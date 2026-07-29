# CudaNav native all-GPU closed-loop S-course

Date: 2026-07-29

Source commit: `3f65377c3cf38b3874c31d5efcb193720f245cca`

Result: **PASS**

The deterministic S-course plant generates LiDAR from ground truth, but the
controller only receives the GPU KISS-ICP estimate. Each CUDA MPPI command is
applied to the plant before the next scan. GPU voxel mapping and GPU ESDF build
the controller costmap in the same process.

## Result

- Goal reached: true
- Final ground-truth goal distance: 0.282 m
- Collision count: 0
- Ground-truth distance: 10.387 m
- Command-effect distance: 10.387 m
- KISS-ICP ATE RMSE: 0.012 m
- KISS-ICP final drift: 0.202%
- Minimum ICP inliers: 243
- Final observed voxels: 43278
- Peak occupied 2D cells: 668
- MPPI solve p95: 0.466 ms
- Full frame p95: 4.284 ms
- Command deadline miss rate: 0.000%
- All-colliding evaluations: 0
- Minimum nonzero valid-rollout ratio: 0.010

## Scope

Native deterministic S-course simulation. CUDA MPPI commands are applied to the plant and affect later LiDAR scans. This is not a ROS 2 runtime result and does not use recorded real-world data.
