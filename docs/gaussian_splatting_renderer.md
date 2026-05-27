# Gaussian Splatting Map Renderer

`comparison_gaussian_splatting` is a forward-only Gaussian splatting demo for
robotics map visualization. It represents a synthetic 3D map as colored Gaussian
surfels and renders it from a moving camera.

The goal is not to reproduce a full 3D Gaussian Splatting training pipeline.
Instead, this PR-sized version keeps the useful CUDA pattern:

- one Gaussian per CUDA thread
- project each 3D Gaussian to screen space
- splat a small Gaussian footprint into image accumulators
- use atomic additive blending for dense surfel maps

This pairs naturally with the 3D LiDAR simulator and point-cloud modules: later
work can turn LiDAR hits or registered point clouds into Gaussian surfel maps.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target comparison_gaussian_splatting -j$(nproc)
./bin/comparison_gaussian_splatting
```

Generated files:

- `gif/comparison_gaussian_splatting.avi`
- `gif/comparison_gaussian_splatting.gif`

## Output

The GIF has three panels:

- CPU sparse Gaussian map
- CUDA dense Gaussian map
- CUDA accumulated opacity / splat density

The executable also runs a deterministic CPU/GPU accumulator check on a small
Gaussian set before rendering the animation.

Latest local run:

- Correctness check on `2,048` Gaussians: CPU `35.07 ms`, CUDA `29.624 ms`,
  accumulator MAE `0.000000`.
- Animated average: CPU `66.84 ms/frame` for `4,096` Gaussians, CUDA
  `0.77 ms/frame` for `65,536` Gaussians.
- Per-Gaussian throughput: CUDA `0.0118 us` vs CPU `16.319 us`, about
  `1381x` faster.
