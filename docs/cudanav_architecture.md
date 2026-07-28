# CudaNav Architecture Contract

Status: post-v0.2 development contract for the v1.0 end-to-end GPU autonomy
stack.

This document defines the integration boundary before the existing CUDA demos
are connected. It deliberately does not treat the current `voxel_node` and
`esdf_node` topics as a stable API: those nodes are demonstrations whose frame
handling and output semantics must be migrated to this contract.

## End-to-End Graph

```text
sensor_msgs/PointCloud2
          |
          v
cuda_kiss_icp_odometry -----> nav_msgs/Odometry + odom -> base_link TF
          |
          v
cuda_voxel_mapper ----------> local-map PointCloud2 + OccupancyGrid projection
          |                                      |
          v                                      v
cuda_esdf_node                         cuda_voxel_costmap_layer
          |                                      |
          +---- DistanceField2D -----------------+
                                                 |
                                                 v
                                      cuda_mppi_controller
                                                 |
                                                 v
                                      geometry_msgs/TwistStamped
```

The controller remains a Nav2 plugin. The voxel costmap layer is the supported
bridge into the Nav2 layered costmap; the controller may construct or consume
the GPU distance field behind that boundary. Separate-process CUDA pointer
sharing is not part of the first contract.

## Components and Topics

All topic names in node source must be relative so the complete stack can be
placed under a namespace. Names below show the default `cuda_nav` namespace.

| Component | Subscribes | Publishes |
|---|---|---|
| `cuda_kiss_icp_odometry` | `points` (`sensor_msgs/PointCloud2`) | `odom` (`nav_msgs/Odometry`), `/tf`, `diagnostics` |
| `cuda_voxel_mapper` | `points`, `odom` or TF | `local_map` (`sensor_msgs/PointCloud2`), `occupancy` (`nav_msgs/OccupancyGrid`), `diagnostics` |
| `cuda_esdf_node` | `occupancy` | `esdf` (`cuda_robotics_msgs/DistanceField2D`), `diagnostics` |
| `cuda_voxel_costmap_layer` | `occupancy` | Nav2 layered-costmap updates |
| `cuda_mppi_controller` | Nav2 costmap, plan, pose | Nav2 command output and `diagnostics` |

Default resolved topics:

- `/cuda_nav/points`
- `/cuda_nav/odom`
- `/cuda_nav/local_map`
- `/cuda_nav/occupancy`
- `/cuda_nav/esdf`
- `/cuda_nav/diagnostics`

The system input may be remapped from a sensor driver topic. No implementation
may hard-code an absolute input or output topic other than `/tf`.

## Frame and Time Invariants

- Input point coordinates are interpreted in `PointCloud2.header.frame_id`.
- Every point cloud must be transformed with the complete SE(3) transform;
  translation-only or yaw-only transforms are invalid.
- `odom` is the continuous local frame produced by KISS-ICP.
- `base_link` is the robot body frame. The odometry component owns the
  `odom -> base_link` transform.
- A future loop-closure component may own `map -> odom`; no odometry or mapping
  node may publish that transform.
- Output stamps use the input sensor stamp, not callback wall-clock time.
- A transform lookup failure drops that sensor update, increments a diagnostic
  counter, and never republishes stale data with a new timestamp.
- Deskew state is explicit in diagnostics. An undeskewed scan is never labelled
  as deskewed evidence.

## QoS Contract

| Stream | QoS |
|---|---|
| Point clouds | sensor-data profile, best effort, volatile, depth 5 |
| Odometry and TF | reliable, volatile, depth 10 |
| Local map | reliable, volatile, depth 1 |
| Occupancy and ESDF | reliable, transient local, depth 1 |
| Diagnostics | reliable, volatile, depth 10 |

Slow consumers may lose point clouds but must receive the newest map and
distance field.

## DistanceField2D Invariants

`cuda_robotics_msgs/DistanceField2D` replaces the demonstration encoding that
stored distance in `nav_msgs/OccupancyGrid.data`. Occupancy value `-1` means
unknown in the standard message and must not mean occupied or zero distance.

- `resolution` is finite and strictly positive.
- `width * height == len(distances)`.
- `origin` is the pose of cell `(0, 0)` in `header.frame_id`.
- Data is row-major: index `y * width + x`.
- Each distance is finite and lies in `[0, max_distance]`.
- Occupied cells have distance `0`.
- Cells with no obstacle inside the truncation radius have `max_distance`.
- Unknown-space policy is a node parameter and is reported in diagnostics.

## Lifecycle and Failure Behavior

The production nodes use ROS 2 lifecycle semantics:

1. `configure`: validate parameters and CUDA device capability.
2. `activate`: allocate GPU buffers and activate publishers.
3. `deactivate`: stop subscriptions and synchronize active CUDA work.
4. `cleanup`: release GPU memory.

CUDA allocation, launch, and synchronization errors transition the component
to an error state. Logging an error and continuing with partially updated data
is not allowed. Capacity overflow, invalid fields, and dropped scans are
observable counters.

## Integration Sequence

1. Introduce the typed message and contract tests.
2. Port GPU KISS-ICP into a reusable core and ROS 2 lifecycle component.
3. Correct voxel mapping to use full SE(3), field-name lookup, relative topics,
   and explicit unknown-space semantics.
4. Publish typed ESDF data and add CPU-reference comparison tests.
5. Implement `cuda_voxel_costmap_layer`.
6. Bring up the complete graph in simulation.
7. Record deterministic closed-loop evidence and then add real sensor input.

## v1.0 Closed-Loop Gate

The gate is satisfied only when all components run together and controller commands affect subsequent robot state:

- at least 10 minutes of closed-loop simulation;
- zero collisions in the release scenario;
- controller deadline-miss rate below 1%;
- odometry drift below 1% of travelled distance;
- no silent transform, capacity, CUDA, or schema failures;
- bag, parameters, seed, manifest, metrics, logs, and video retained from the
  same git commit.

Recorded or shadow-controller bags remain valid negative or offline evidence,
but do not satisfy this closed-loop gate.
