# cuda_robotics_msgs

Typed ROS 2 interfaces for the post-v0.2 CudaNav integration.

## DistanceField2D

`DistanceField2D` represents metric obstacle clearance without overloading
`nav_msgs/OccupancyGrid` values:

- `header.frame_id` identifies the grid frame;
- `origin` is the pose of cell `(0, 0)`;
- `resolution` is metres per cell;
- `distances[y * width + x]` is metres to the nearest occupied cell;
- values are finite and clamped to `[0, max_distance]`;
- occupied cells have value `0`.

The complete producer/consumer, frame, QoS, and failure contract is documented
in [`docs/cudanav_architecture.md`](../../../docs/cudanav_architecture.md).

## TrackedObstacle / TrackedObstacleArray

Moving objects from a tracker, for the `cuda_mppi_controller` dynamic-obstacle
critic (`dynamic_obstacles_topic`):

- `header.stamp` is when the objects were observed and `header.frame_id`
  the frame of every `position` and `velocity`;
- each object is a disc of `radius` metres moving at constant `velocity`
  (m/s); z components are ignored;
- `id` is the tracker's identifier and is informational only.

The controller transforms the objects into its costmap frame, extrapolates
them to the planning time, and drops a message older than
`dynamic_obstacles_timeout`.
