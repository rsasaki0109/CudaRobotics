# CudaNav native all-GPU closed-loop S-course

Date: 2026-07-29

Source commit: `9746686845ff6fbaba51ebb139dc1d6c741a7e2b`

Result: **PASS**

The deterministic S-course plant generates LiDAR from ground truth, but the
controller only receives the GPU KISS-ICP estimate. Each CUDA MPPI command is
applied to the plant before the next scan. GPU voxel mapping and GPU ESDF build
the controller costmap in the same process.

## Result

- Goal reached: true
- Final ground-truth goal distance: 0.277 m
- Collision count: 0
- Ground-truth distance: 10.922 m
- Command-effect distance: 10.922 m
- KISS-ICP ATE RMSE: 0.013 m
- KISS-ICP final drift: 0.221%
- Minimum ICP inliers: 225
- Final observed voxels: 36409
- Peak occupied 2D cells: 612
- MPPI solve p95: 0.622 ms
- Full frame p95: 4.717 ms
- Command deadline miss rate: 0.000%
- All-colliding evaluations: 0
- Minimum nonzero valid-rollout ratio: 0.421

## Scope

Native deterministic S-course simulation. CUDA MPPI commands are applied to the plant and affect later LiDAR scans. This is not a ROS 2 runtime result and does not use recorded real-world data.
