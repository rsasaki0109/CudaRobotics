# GPU RGB-D Gaussian-Splatting SLAM

`src/gpu_gaussian_splatting_slam.cu`

The repo already had a forward Gaussian-splatting *renderer*
(`gpu_gaussian_splatting.cu`). This turns that map representation into an
**online SLAM system**: a camera scans an unknown room and, frame by frame, the
GPU both **tracks its own pose** and **grows a 3D Gaussian map** — SplaTAM in
spirit, everything heavy on the device.

![demo](https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif)

Left: the RGB-D sensor frame. Middle: the current Gaussian map splatted from the
*estimated* pose (what the SLAM thinks it sees). Right: the global Gaussian map
filling in as the scan proceeds.

## Pipeline (per frame)

1. **Sensor** — the ground-truth world (an analytic room: 5 planes + 4 spheres,
   checker-textured) is **ray-cast** on the GPU from the *true* pose into a
   sharp RGB-D frame. This is the only thing the SLAM system is allowed to see.
   (A real depth sensor returns crisp depth; splat "expected depth" is too
   blurry to track against, so the sensor is an honest ray-caster.)
2. **Tracking** — the observed depth is back-projected to a point cloud and
   aligned to the current Gaussian map with **frame-to-model point-to-plane
   ICP**. One GPU thread = one observed point: it brute-force finds its nearest
   map point, forms the 1×6 point-to-plane Jacobian `[p×n | n]`, and a block
   reduction accumulates the 6×6 normal equations (double precision). The host
   solves the 6-DoF update; ~14 iterations / frame, per-step clamped.
3. **Mapping** — observed points are back-projected with the *estimated* pose
   and fused into the global map through a **voxel hash**: only still-empty
   cells spawn a new Gaussian (with a depth-image normal), so the map grows
   without unbounded duplication.
4. **Render** — the growing map is splatted from the estimated pose and from a
   slow overview camera (the panels above).

## Why a scanning sweep (and not a full orbit)

Frame-to-model SLAM with no loop closure drifts once the camera turns entirely
into territory it has only ever mapped from its own (slightly wrong) poses — the
freshly-fused points give *false confirmation* and the weakly-observable depth
axis floats away (measured: a full 360° orbit diverged past ~frame 14). Real
systems fight this with bundle adjustment / loop closure. Here the camera
instead performs a **back-and-forth scanning sweep** that always keeps the
central sphere cluster and the same walls in view, so tracking always has a
strongly-anchored, full-rank overlap and the revisits act as implicit loop
closure — **no unbounded drift**.

## Result (this machine)

| metric | value |
|---|---|
| ATE (RMSE) over 120 frames | **`0.018 m`** (1.8 cm) |
| final map | `14064` Gaussians |
| tracking | `~5.0 ms` / frame (GPU point-to-plane ICP, 14 iters) |
| rendering | `~15.6 ms` / frame (2 splat renders) |
| sensor | `384×288` ray-cast RGB-D |

Honesty: frame 0 is anchored to the true pose (SLAM trajectories are only
defined up to a global gauge); the ATE is the RMSE of the estimated camera
positions against ground truth — **measured, not assumed**. Tracking holds at
the centimetre level for the whole run (worst single-frame error ≈ 3.6 cm during
the widest swing, and it recovers).

## Reproduce

```bash
cd build && cmake .. && make gpu_gaussian_splatting_slam -j$(nproc)
cd .. && ./bin/gpu_gaussian_splatting_slam
```

Prints the ATE / timing summary and writes
`gif/gpu_gaussian_splatting_slam.gif`.

## Notes

- One demo = one `.cu`; reuses `include/cuda_check.cuh` and
  `include/cuda_video.h`, and the projection/alpha-composite splat kernels from
  the renderer demo.
- The point-to-plane normal equations are summed in double precision with a
  CAS-based `atomicAdd(double)` so the demo runs on compute capability < 6.0.
- `--expt-relaxed-constexpr`; GIF served from gh-pages (not committed to the
  repo).
