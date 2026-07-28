# GPU KISS-ICP LiDAR Odometry

`src/gpu_kiss_icp.cu`

This demo estimates an SE(3) sensor trajectory from a stream of LiDAR scans
without IMU, wheel odometry, or loop closure. It follows the practical
KISS-ICP recipe: voxel downsampling, a motion-model initial guess, an adaptive
correspondence threshold, robust ICP, and a rolling voxel map.

The expensive correspondence stage runs on CUDA. The default backend builds an
open-addressed voxel hash directly on the GPU, then one thread queries the
neighbouring hash cells for each scan point. A second kernel accumulates the
robust point-to-plane normal equations. The host only solves the resulting 6x6
system.

![GPU KISS-ICP trajectory](https://rsasaki0109.github.io/CudaRobotics/gpu_kiss_icp.gif)

## Pipeline

1. Downsample the incoming scan to one centroid per voxel.
2. Predict the next pose from the previous estimate.
3. Set the correspondence gate from an exponential moving average of recent
   prediction error.
4. Build a GPU voxel hash for the local map and find exact, radius-gated
   nearest-neighbour correspondences in parallel.
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
  --nn voxel \
  --json build/gpu_kiss_icp_check.json
```

Run the exhaustive GPU reference with `--nn brute`. Both backends apply the
same adaptive distance gate and emit the same JSON schema.

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
| Final drift | 0.0141 m (0.032%) |
| Mean accepted correspondences | 7293.9 / registered scan |
| Mean ICP iterations | 8.3 / registered scan |

Correspondence/backend comparison:

| Metric | GPU voxel hash | GPU brute force |
|---|---:|---:|
| GPU nearest-neighbour time | 6.58 ms / registered scan | 11.38 ms / registered scan |
| Index build time | 0.072 ms / registered scan | n/a |
| End-to-end odometry time | 24.13 ms / scan | 28.49 ms / scan |

On this run, voxel hashing made correspondence search `1.73x` faster and the
complete odometry pipeline `1.18x` faster. Translation ATE differed by less
than `0.04 mm`; tiny trajectory differences come from atomic insertion and
floating-point accumulation order.

Timing is hardware-specific. The deterministic accuracy and correspondence
gates are the portable regression signal.

## Scope and limitations

- The bundled benchmark uses synthetic, already deskewed scans. Real spinning
  LiDAR input needs timestamp-aware deskewing and a dataset adapter.
- The voxel hash uses a fixed-capacity open-addressed table sized for the
  bounded 200k-point local map. Larger production maps should resize the table
  or expose overflow diagnostics.
- Map-normal estimation is still an exhaustive GPU kNN-PCA pass and now
  dominates runtime. Reusing normals stored in the voxel map is the next
  scaling step.
- The previous-pose predictor is deliberate for this scan spacing. The
  constant-velocity predictor was less stable on the coarse synthetic scans.
- This is odometry only: it does not perform loop closure or global map
  optimization.
