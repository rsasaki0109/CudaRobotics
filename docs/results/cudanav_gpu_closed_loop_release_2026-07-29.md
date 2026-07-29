# CudaNav native all-GPU 10-minute closed-loop release

Date: 2026-07-29

Source commit: `b2c490799ad098c5cc0f26127c18c8d8d186ccf2`

Profile: `release`

Result: **PASS**

![CudaNav native all-GPU closed-loop release](../../gif/cudanav_gpu_closed_loop_release.gif)

The deterministic S-course plant generates LiDAR from ground truth, but the
controller only receives the GPU KISS-ICP estimate. Each CUDA MPPI command is
applied to the plant before the next scan. GPU voxel mapping and GPU ESDF build
the controller costmap in the same process.

## Result

- Goal reached: true
- Traversals: 30/30
- Simulated duration: 1059.4 s
- Final ground-truth goal distance: 0.296 m
- Collision count: 0
- Ground-truth distance: 352.748 m
- Command-effect distance: 352.748 m
- KISS-ICP ATE RMSE: 0.012 m
- KISS-ICP final drift: 0.003%
- Minimum ICP inliers: 213
- Final observed voxels: 53012
- Peak occupied 2D cells: 760
- MPPI solve p95: 0.455 ms
- Full frame p95: 5.237 ms
- Command deadline miss rate: 0.000%
- All-colliding evaluations: 0
- Minimum nonzero valid-rollout ratio: 0.195

## Visual evidence

The 180-frame GIF retains the start and end of every traversal and is bound to
the 10,594-row trajectory by SHA-256. Its machine-readable sidecar records the
source evidence, trajectory, renderer, dimensions, frame count, and GIF hash:
[`cudanav_gpu_closed_loop_release.json`](../../gif/cudanav_gpu_closed_loop_release.json).

```bash
python scripts/render_cudanav_gpu_closed_loop.py \
  --evidence docs/results/cudanav_gpu_closed_loop_release_2026-07-29.json \
  --trajectory build/cudanav_gpu_closed_loop_release_b2c4907/trajectory.csv \
  --output gif/cudanav_gpu_closed_loop_release.gif \
  --manifest gif/cudanav_gpu_closed_loop_release.json \
  --check-only
```

## Scope

Native deterministic S-course simulation. CUDA MPPI commands are applied to the plant and affect later LiDAR scans. This is not a ROS 2 runtime result and does not use recorded real-world data.
