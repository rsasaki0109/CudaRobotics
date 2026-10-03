# 3D ESDF-MPPI

`gpu_esdf_mppi_3d` flies a 3D double integrator through a window in a wall,
over a suspended slab, to a goal 17.7 m away. Two GPU fields drive MPPI:

- a **3D Euclidean distance field** (ESDF) for clearance, built with 3D Jump
  Flooding on 128 x 128 x 64 voxels (0.125 m);
- a **cost-to-go field** for progress: the geodesic distance to the goal
  around obstacles, from a parallel wavefront (Bellman-Ford relaxation over 26
  neighbours) on a 64 x 64 x 32 grid, with voxels closer than the vehicle
  radius to an obstacle blocked.

<img src="https://rsasaki0109.github.io/CudaRobotics/gpu_esdf_mppi_3d.gif" width="720"/>

Left: top view, ESDF slice at the vehicle's altitude. Right: side view, slice
at the vehicle's y. Grey: sampled rollouts; yellow: the MPPI plan; white: the
flown path.

## Why the cost-to-go field

With straight-line distance to the goal as the progress cost, MPPI parks under
the wall at the point closest to the goal and never finds the window: every
rollout that heads for the window first moves away from the goal. The
wavefront field measures distance along the free space instead, so rollouts
toward the window are the cheap ones. In the default scene the geodesic
start-to-goal distance is 19.97 m against a 17.68 m straight line.

## Pipeline

1. Occupancy grid: ground, a wall across `y = 8` with one 2 m x 2 m window,
   seven pillars, a slab just below the goal altitude.
2. 3D JFA (`jfa3d_*` kernels, as in `comparison_esdf_3d.cu`): distance to the
   nearest occupied voxel.
3. Wavefront (`ctg_*` kernels): relaxation sweeps until no voxel changes,
   checked every 16 sweeps.
4. MPPI, one thread per sampled trajectory: acceleration noise around the
   nominal sequence, trilinear lookups of both fields, softmin weights and the
   weighted update from `mppi_reduction.cuh`, two iterations per control step,
   warm-started by shifting the sequence.

The rollout cost is `__host__ __device__`. At start-up the demo evaluates the
first sampled batch on the CPU with the same function and the same controls,
and prints the timing and the largest relative cost difference (about 3e-5,
float rounding).

## Default run

```bash
./bin/gpu_esdf_mppi_3d             # writes gif/gpu_esdf_mppi_3d.gif
./bin/gpu_esdf_mppi_3d --help      # --samples, --steps, --seed, --no-video, --headless
```

| quantity | value |
|---|---|
| 3D ESDF (1,048,576 voxels) | about 2 ms |
| Cost-to-go (131,072 voxels, 96 sweeps) | about 3 ms |
| Rollout batch, K=4096, T=40 | about 0.2 ms GPU vs about 11 ms CPU (single thread) |
| MPPI per control step (2 iterations) | about 0.5 ms |
| Result | goal in 69 steps (6.9 s), path 19.0 m, min clearance 0.41 m |

Timings are from single runs on the development machine and vary between runs.
Seeds 1, 7, 42 and `--samples 1024` / `16384` also reach the goal without
collision (70-87 steps). The program exits non-zero if the goal is not reached
or the vehicle collides; `demo_headless_gpu_esdf_mppi_3d` runs it under CTest.

## Limitations

- Point-mass vehicle: no attitude dynamics or thrust limits.
- The maps are static and built once; the cost-to-go is computed for a single
  goal.
- The wavefront runs on a 0.25 m grid, so passages narrower than about two
  coarse voxels plus the vehicle diameter disappear from it.
