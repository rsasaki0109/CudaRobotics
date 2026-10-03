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
./bin/gpu_esdf_mppi_3d --help      # --samples, --steps, --seed, --no-video, --headless,
                                   # --movers, --mode, --trials, --mover-speed
```

| quantity | value |
|---|---|
| 3D ESDF (1,048,576 voxels) | about 2 ms |
| Cost-to-go (131,072 voxels, 96 sweeps) | about 3 ms |
| Rollout batch, K=4096, T=40 | about 0.2 ms GPU vs about 11 ms CPU (single thread) |
| MPPI per control step (2 iterations) | about 0.5 ms |
| Result | goal in 68 steps (6.8 s), path 18.8 m, min clearance 0.43 m |

Timings are from single runs on the development machine and vary between runs.
Seeds 1, 7, 42 and `--samples 1024` / `16384` also reach the goal without
collision (70-87 steps). The program exits non-zero if the goal is not reached
or the vehicle collides; `demo_headless_gpu_esdf_mppi_3d` runs it under CTest.

## Moving obstacles

`--movers N` adds up to eight spheres (0.45 m radius) that move at constant
velocity beyond the wall and bounce off the walls of their region
(x 1-15 m, y 9-13.5 m, z 1.5-6 m). The vehicle has to cross their region to
reach the goal. `--mode` sets how the planner sees them:

| mode | what the rollouts see | per control step |
|---|---|---|
| `0` static | the static ESDF only (movers ignored) | the default MPPI work |
| `1` rebuild | the movers stamped into the occupancy grid at their current positions, ESDF rebuilt by JFA every step | + a full 3D JFA |
| `2` predict | the static ESDF plus the analytic distance to each mover at its constant-velocity prediction for that rollout step | + 40 x N sphere distances per rollout |

`--trials 30` runs the same 30 mover scenarios for all three modes and prints
a table ([raw tables](results/gpu_esdf_mppi_3d_dynamic_2026-10-03.md)). Success
means reaching the goal with no contact against the static map or the movers'
true positions.

| movers | static | rebuild | predict |
|---|---:|---:|---:|
| 4, 1x speed | 26/30 (4 collisions) | 29/30 | **30/30** |
| 6, 1x speed | 23/30 (7 collisions) | **30/30** | 29/30 |
| 8, 1x speed | 22/30 (8 collisions) | **30/30** | 29/30 |
| 6, 2x speed | 21/30 (9 collisions) | 25/30 (5 collisions) | **30/30** |
| 6, 3x speed | 23/30 (7 collisions) | 27/30 (3 collisions) | **29/30** (1 collision) |

- Ignoring the movers collides in a quarter of the episodes; both
  mover-aware modes are collision-free at the base speed.
- Rebuilding the ESDF costs about 2.4 ms per control step against 0.46 ms for
  prediction (the full JFA dominates).
- At 2-3x speed, reacting to current positions is no longer enough: rebuild
  collides 8 times in 60 paired episodes, prediction once (paired exact
  McNemar on success, 8 vs 1, p = 0.039).
- Prediction assumes constant velocity, so it does not anticipate bounces;
  its one fast-speed collision and the occasional timeout come from that.

<img src="https://rsasaki0109.github.io/CudaRobotics/gpu_esdf_mppi_3d_dynamic.gif" width="720"/>

`--movers 6 --mover-speed 2 --mode 2`: red discs are the movers, red lines
their predicted positions at the end of the 4 s horizon.

## Limitations

- Point-mass vehicle: no attitude dynamics or thrust limits.
- The static map and the cost-to-go are built once; the cost-to-go is computed
  for a single goal and ignores the movers.
- Movers are predicted at constant velocity with known state (no estimation).
- The wavefront runs on a 0.25 m grid, so passages narrower than about two
  coarse voxels plus the vehicle diameter disappear from it.
