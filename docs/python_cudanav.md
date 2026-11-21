# Python CudaNav (`cudarobotics.navigation`)

Status: development surface on `master`, not yet in a published release. It
is milestone B-1 of Epic 4 in the
[1.0 roadmap](cudarobotics_1_0_roadmap.md#epic-4-v04-python-cudanav).

`cudarobotics.navigation` lets Python call the same tested CUDA cores that the
ROS 2 CudaNav packages use. Nothing is re-implemented in Python:

| Python | CUDA core | ROS 2 package |
|---|---|---|
| `KissIcpOdometry` | `cudarobotics::KissIcpOdometry` (`src/gpu_kiss_icp.cu`) | `cuda_kiss_icp` |
| `VoxelMap` | `cudarobotics::VoxelMapperGpu` (`src/voxel_mapping_gpu.cu`) | `cuda_voxel_mapping` |
| `Esdf2D` | `cudarobotics::Esdf2DGpu` (`src/esdf_2d_gpu.cu`) | `cuda_esdf` |
| `cudarobotics.MppiPlanner` | `cuda_mppi_controller::MppiGpu` (`src/mppi_gpu.cu`) | `cuda_mppi_controller` |

`CudaNavPipeline` connects these stages. The order and the default parameters
are the same as the native closed loop in
`tools/cudanav_gpu_closed_loop_s_course.cu`. Only the glue between the stages
is NumPy:
- footprint clearing;
- the occupancy plus ESDF to uint8 costmap mapping;
- the local path window.

## Install

The navigation cores are compiled into the same `_cudarobotics` extension as
MPPI and registration. Build from a checkout:

```bash
./scripts/sync_python_core.sh
pip install 'python/[examples]'
```

## API

```python
import numpy as np
from cudarobotics.navigation import CudaNavPipeline
from cudarobotics.navigation.simulation import Pose2D, course_segments, interpolate_path, simulate_scan

nav = CudaNavPipeline()
path = interpolate_path()  # (M, 2) float32 mission path
step = nav.step(simulate_scan(Pose2D(), course_segments()), path=path)

step.odometry.translation, step.odometry.yaw   # GPU KISS-ICP pose
step.occupancy.data                             # int8 (H, W): -1 unknown, 0 free, 100 occupied
step.distance_field.distances                   # float32 (H, W) metres, clipped at 2.0
step.costmap                                    # uint8 (H, W) MPPI costmap
step.command                                    # (v, w) from CUDA MPPI
step.timings_ms                                 # odometry/mapping/esdf/costmap/mppi/total
```

The stages can also be used on their own:
- `KissIcpOdometry.register(points, point_times=None)` returns an
  `OdometryFrame`. Passing per-point times enables GPU deskew.
- `VoxelMap.integrate(points_world, sensor_origin)` followed by
  `VoxelMap.occupancy()` returns an `OccupancyGrid`.
- `Esdf2D.compute(grid)` returns a `DistanceField`.
  `esdf_2d_cpu_reference(grid, ...)` is the exhaustive CPU oracle and does not
  need a GPU.
- `costmap_from_esdf(grid, field)` builds the costmap with these costs:

  | Cell | Cost |
  |---|---|
  | unknown | 255 |
  | occupied | 254 |
  | within 0.25 m | 253 |
  | out to 1.0 m | `252·exp(-3(d-0.25))` |
  | otherwise | 0 |

- `CudaNavPipeline(enable_planner=False).step(points)` runs odometry, mapping
  and ESDF without MPPI, for recorded data.

### Stage chain

`CudaNavPipeline(stage_chain="device")` is the default. In this mode the
occupancy, ESDF and costmap never leave the GPU between the voxel projection
and MPPI:

1. `VoxelMapperGpu::occupancy_projection_device`
2. GPU footprint clearing (`CostmapBuilderGpu`)
3. `Esdf2DGpu::compute_device`
4. the GPU costmap kernel
5. `MppiGpu::computeWithDeviceCostmap`

Only the occupied-cell count comes back to the host each frame. The step's
`occupancy`, `distance_field` and `costmap` are copied to NumPy the first time
they are read, and they stay readable only until the next `step()` call, which
overwrites the GPU buffers. Call `step.materialize()` to keep a step's grids.
`step.occupied_cells` is always available.

`stage_chain="host"` is the original path. It copies each product to NumPy and
builds the costmap with `costmap_from_esdf`. The two chains produce the same
occupancy and distances. Costmaps can differ by one step in the exponential
band, because device and host `expf` round differently.

The CLI selects the mode with `--stage-chain device|host`, and records it in
`summary.json` and `manifest.json`.

Measured by running `cudarobotics-cudanav sim` three times per mode on an
NVIDIA consumer GPU (CUDA 13.3 build). Values are the median p50 / p95 in
milliseconds:

| Stage | host | device |
|---|---|---|
| mapping (integrate + projection + footprint) | 0.82 / 1.35 | 0.51 / 0.85 |
| ESDF | 0.81 / 1.01 | 0.66 / 0.78 |
| costmap | 0.15 / 0.21 | 0.01 / 0.03 |
| MPPI | 0.43 / 0.77 | 0.36 / 0.60 |
| whole frame | 3.95 / 7.09 | 3.33 / 5.61 |

Both modes passed the smoke gate in every run. The frame and drift values were
the same in both modes: 227–230 frames and 0.22–0.24% drift. The native tool
`cudanav_gpu_closed_loop_s_course --stage-chain device|host` shows the same
direction with a smaller gain, because its host chain does no NumPy work. Its
mapping, ESDF, costmap and MPPI stages together take 1.72 ms at p50 on the
host chain and 1.53 ms on the device chain. Odometry dominates the frame time,
and the KISS-ICP map still lives on the host.

No Jetson measurement exists. On Jetson, the CPU and GPU share memory and the
CPU is weaker, so avoiding host round trips and NumPy work should matter more
there. That is an expectation, not a result.

### Moving obstacles

Pass tracked moving objects as `step(..., dynamic_obstacles=rows)`, with rows
of `(x, y, vx, vy, radius)` in the map frame. Their cells are cleared from the
occupancy grid, and CUDA MPPI avoids their constant-velocity predictions.
Remove their points from the scan as well. Alternatively,
`CudaNavPipeline(tracker=ScanObjectTracker())` finds and tracks moving objects
in the scan itself (`step.tracks`), with no external tracker. The critic, the
tracker and an evaluation are described in
[`cudanav_dynamic_obstacles.md`](cudanav_dynamic_obstacles.md).

Configuration objects (`KissIcpConfig`, `VoxelMappingConfig`, `Esdf2DConfig`,
`MppiParams`) are accepted by every constructor. Keyword overrides are checked
against the field names, and invalid values are rejected by the C++
validators with `ValueError`.

## CLI

```bash
cudarobotics-cudanav sim --output-dir build/cudanav_sim [--traversals N] [--check]
cudarobotics-cudanav sequence recording.crkicp --output-dir build/cudanav_seq [--check]
```

**`sim`** drives the S-course mission in closed loop. It is the same world,
LiDAR model, unicycle plant and quality gate as the native tool:
- the smoke gate for one traversal: drift < 5% and deadline misses < 5%;
- the release-style gate for several traversals: drift < 1% and deadline
  misses < 1%;
- in both cases: zero collisions, a goal within 0.30 m, a causal command
  effect, and the native bounds.

**`sequence`** runs odometry, mapping and ESDF over a recorded `CRKICP1` file.
- `scripts/export_cudanav_kiss_icp_sequence.py` writes this format from a ROS 2
  bag without needing a ROS installation.
- The reference poses in the file give ATE and final drift.
- The gate defaults are the native ones: ATE ≤ 5 m, drift ≤ 10%, and at least
  30 inliers.

**`sim --scenario crossing`** runs one S-course traversal while two
pedestrians cross the corridors.
- `--dynamic-mode map|tracked_static|predicted` selects how they reach the
  planner.
- `--seed` sets their schedules.
- The `--obs-*` options degrade the oracle tracker.
- The gate requires the goal within 60 s, no wall contact, and no
  robot-caused pedestrian contact.

See [`cudanav_dynamic_obstacles.md`](cudanav_dynamic_obstacles.md).

Both commands write the following files to `--output-dir`:

| File | Contents |
|---|---|
| `summary.json` | `schema_version`, `success`, `smoke_pass`, `quality_pass`, `checks`, odometry/collision/deadline metrics, `stage_latency_ms` (p50/p95/max per stage) |
| `manifest.json` | package version, source-provenance digest, CUDA runtime/driver/device/memory, command line, parameters, seeds, input SHA-256, output names |
| `trajectory.csv` | per-frame reference and estimated pose, error, inliers, map statistics, timings |
| `occupancy.npy`, `esdf.npy`, `costmap.npy` | final grids |
| `overview.png` | occupancy, ESDF and both trajectories (only when matplotlib is installed and `--no-plot` is not given) |

When a run fails, `summary.json` still satisfies the Docker `cudanav` summary
contract and adds `failure.stage`, `failure.type` and `failure.message`, and
the command exits 1. A run with no visible GPU is reported this way, at the
`initialize` stage. `--check` turns a completed run that fails its gate into
exit 2.

## Determinism

- The simulator uses no randomness.
- CUDA MPPI uses its fixed cuRAND seed (42), which is recorded in the manifest.
- Stages given the same input are repeatable.
- A full closed loop is not bit-identical from run to run. The voxel map
  combines log-odds updates to the same voxel with clamped float atomics, so
  their order can change an occupancy cell. That changes the costmap, and the
  mission can then differ by a few frames. The ROS 2 and native stacks share
  this core, and the tests therefore check the gate outcome rather than
  bitwise trajectories.

## Reference result

The Python `sim` closed loop on an NVIDIA consumer GPU (CUDA 13.3 build), next
to the retained native result in
`docs/results/cudanav_gpu_closed_loop_2026-07-29.json`:

| Run | Frames | Drift | ATE | Goal distance | Collisions | Frame p95 |
|---|---|---|---|---|---|---|
| Python `cudarobotics-cudanav sim` | 227–230 | 0.22–0.24% | 0.013–0.014 m | 0.28 m | 0 | 6.1 ms |
| Native `cudanav_gpu_closed_loop_s_course` (retained) | 225 | 0.22% | 0.013 m | 0.28 m | 0 | 4.7 ms |

These are synthetic smoke results, not release evidence.

## Limits and next steps

- `sim` uses synthetic LiDAR and a command-driven plant, so it is not
  real-sensor evidence. `sequence` provides the recorded-data path, but no
  public quickstart dataset has been selected yet.
- Still to do for Epic 4 (milestone B-2):
  - a documented public-dataset quickstart of manageable size;
  - fixed-seed benchmark evidence with at least three repetitions;
  - a release that ships this API in the CPython 3.10/3.12 manylinux wheels.
