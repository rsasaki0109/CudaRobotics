# CudaNav native all-GPU 10-minute closed-loop release

Date: 2026-07-29

Source commit: `5bc4446a812097118dca830ae74c421d9fd24b13`

Profile: `release`

Result: **PASS**

The deterministic S-course plant generates LiDAR from ground truth, but the
controller only receives the GPU KISS-ICP estimate. Each CUDA MPPI command is
applied to the plant before the next scan. GPU voxel mapping and GPU ESDF build
the controller costmap in the same process.

## Result

- Goal reached: true
- Traversals: 30/30
- Simulated duration: 1005.0 s
- Final ground-truth goal distance: 0.259 m
- Collision count: 0
- Ground-truth distance: 352.211 m
- Command-effect distance: 352.211 m
- KISS-ICP ATE RMSE: 0.013 m
- KISS-ICP final drift: 0.002%
- Minimum ICP inliers: 216
- Final observed voxels: 53384
- Peak occupied 2D cells: 723
- MPPI solve p95: 0.617 ms
- Full frame p95: 6.417 ms
- Command deadline miss rate: 0.000%
- All-colliding evaluations: 0
- Minimum nonzero valid-rollout ratio: 0.250

## Scope

Native deterministic S-course simulation. CUDA MPPI commands are applied to the plant and affect later LiDAR scans. This is not a ROS 2 runtime result and does not use recorded real-world data.
