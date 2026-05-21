# 3D LiDAR Simulator

`comparison_lidar3d_sim` is a CUDA multi-ring LiDAR raycasting demo. It extends
the existing 2D massive LiDAR simulator into a procedural 3D scene with analytic
geometry, a sparse CPU scan, a dense CUDA scan, and a range-image view.

## Scope

- Scene: ground plane, walls, building-like boxes, vehicle-sized boxes, and
  pillar cylinders.
- Sensor: spinning multi-ring LiDAR with configurable channel and azimuth
  counts.
- CUDA mapping: one ray per thread.
- Outputs: nearest hit range, xyz point, and primitive label.
- Visualization: CPU sparse point cloud, CUDA dense point cloud, and CUDA range
  image in one GIF.

The first version deliberately avoids mesh BVHs, OptiX, ROS bags, and learned
rendering. Those can build on the same ray/primitive interface later.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target comparison_lidar3d_sim -j$(nproc)
./bin/comparison_lidar3d_sim
```

Generated files:

- `gif/comparison_lidar3d_sim.avi`
- `gif/comparison_lidar3d_sim.gif`

## Checks

The executable prints a same-ray CPU/GPU sweep for `16x512`, `32x1024`,
`64x2048`, and `128x4096` scans. It also reports a deterministic correctness
check for the `16x512` case:

- max absolute CPU/GPU range error
- label match rate

Latest local run:

| Scan | Rays | CPU ms | CUDA ms | Speedup |
|---|---:|---:|---:|---:|
| `16x512` | 8,192 | 3.85 | 0.070 | 54.8x |
| `32x1024` | 32,768 | 15.18 | 0.079 | 191.2x |
| `64x2048` | 131,072 | 63.00 | 0.088 | 715.7x |
| `128x4096` | 524,288 | 248.19 | 0.116 | 2144.9x |

Correctness on `16x512`: max absolute range error `0.000381 m`, label match
rate `100.00%`.

Animated comparison average: CPU `3.82 ms` for `8,192` rays, CUDA `0.09 ms`
for `131,072` rays, or about `651x` faster per ray.
