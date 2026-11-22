# cuda_nav_tracking

`scan_tracker_node` finds and tracks moving objects in LiDAR scans and
publishes them as `cuda_robotics_msgs/TrackedObstacleArray`. The
`cuda_mppi_controller` dynamic-obstacle input consumes that message
(`dynamic_obstacles_topic`).

It wraps `cudarobotics.navigation.tracking.ScanTrackingFrontend`, the
tracker evaluated in
[`docs/cudanav_dynamic_obstacles.md`](../../../docs/cudanav_dynamic_obstacles.md).
That frontend:
- finds points in cells of the occupancy grid that are not long-standing
  structure;
- clusters them;
- tracks the clusters with a constant-velocity Kalman filter.

The `cudarobotics` Python package must be importable in the node's
environment.

| Parameter | Default | Meaning |
|---|---|---|
| `input_topic` | `points` | `sensor_msgs/PointCloud2` scans, in any frame TF relates to `tracking_frame` |
| `occupancy_topic` | `occupancy` | `nav_msgs/OccupancyGrid` in `tracking_frame` (for example from `cuda_voxel_mapping`) |
| `output_topic` | `tracked_obstacles` | `TrackedObstacleArray` in `tracking_frame`, stamped with the scan time |
| `static_points_topic` | `points_static` | the scan without the tracked objects' points; empty disables |
| `tracking_frame` | `odom` | frame the tracking runs in |
| `transform_timeout_sec` | 0.1 | TF lookup timeout for each scan |

Wiring:
- Feed `points_static` to the mapper, not the raw scan. Otherwise moving
  objects enter the occupancy grid and the costmap, and they leave trails.
- Still objects that are not yet released to the map are reported with id
  `0xFFFFFFFF` and zero velocity.
- A cell counts as structure after `TrackerConfig.static_frames` (20) grid
  updates, so the grid should be published at roughly the scan rate.

Limits:
- Synthetic evaluation only.
- The node does not clear the tracks' past positions from the mapper's grid,
  as the Python `CudaNavPipeline` does.
- It accepts only unrotated grids in `tracking_frame`.
