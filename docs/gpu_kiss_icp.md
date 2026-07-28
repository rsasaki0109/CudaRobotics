# GPU KISS-ICP LiDAR Odometry

`src/gpu_kiss_icp.cu`

This demo estimates an SE(3) sensor trajectory from a stream of LiDAR scans
without IMU, wheel odometry, or loop closure. It follows the practical
KISS-ICP recipe: voxel downsampling, a motion-model initial guess, an adaptive
correspondence threshold, robust ICP, and a rolling voxel map.

The expensive correspondence stage runs on CUDA. One thread transforms and
queries one scan point against the local map, while a second kernel accumulates
the robust point-to-plane normal equations. The host only solves the resulting
6x6 system.

![GPU KISS-ICP trajectory](https://rsasaki0109.github.io/CudaRobotics/gpu_kiss_icp.gif)

## Pipeline

1. Downsample the incoming scan to one centroid per voxel.
2. Predict the next pose from the previous estimate.
3. Set the correspondence gate from an exponential moving average of recent
   prediction error.
4. Find nearest local-map correspondences in parallel on the GPU.
5. Accumulate Geman-McClure-weighted point-to-plane normal equations on CUDA
   and solve the 6x6 pose update on the host.
6. Insert the registered scan into a first-observation voxel map and remove
   voxels outside the local window.

The synthetic benchmark builds a structured world, generates noisy
range-limited scans along a closed 3D trajectory, and gives the odometry only
the scans. Ground truth is used solely for the initial anchor and final
accuracy report.

## Reproduce

Build and run the full animated demo:

```bash
cmake --build build --target gpu_kiss_icp -j$(nproc)
./bin/gpu_kiss_icp
```

Run the deterministic, video-free accuracy gate and emit JSON:

```bash
./bin/gpu_kiss_icp \
  --check \
  --no-video \
  --frames 64 \
  --json build/gpu_kiss_icp_check.json
```

The same gate is registered with CTest:

```bash
ctest --test-dir build -R gpu_kiss_icp_gate --output-on-failure
```

`--check` returns non-zero unless translation ATE is below `0.5 m`, final
drift is below `1.0 m`, and the mean accepted-correspondence count is at least
10. Use `--frames N` to select 12–2000 scans and `--no-video` for benchmark or
CI runs.

## Measured smoke result

GTX 1660 Ti, CUDA 12.8, 64 scans, fixed seeds:

| Metric | Result |
|---|---:|
| Trajectory length | 43.62 m |
| Translation ATE | 0.0127 m |
| Maximum translation error | 0.0162 m |
| Final drift | 0.0139 m (0.032%) |
| Mean accepted correspondences | 7293.9 / registered scan |
| Mean ICP iterations | 8.2 / registered scan |
| GPU nearest-neighbour time | 11.76 ms / registered scan |
| End-to-end odometry time | 52.11 ms / scan |

Timing is hardware-specific. The deterministic accuracy and correspondence
gates are the portable regression signal.

## Scope and limitations

- The bundled benchmark uses synthetic, already deskewed scans. Real spinning
  LiDAR input needs timestamp-aware deskewing and a dataset adapter.
- Nearest-neighbour search is GPU-parallel brute force against the bounded
  local voxel map. It is exact within the adaptive distance gate, but a GPU
  voxel-hash or 3D KD-tree index is the next scaling step for much larger maps.
- The previous-pose predictor is deliberate for this scan spacing. The
  constant-velocity predictor was less stable on the coarse synthetic scans.
- This is odometry only: it does not perform loop closure or global map
  optimization.
