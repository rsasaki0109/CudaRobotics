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
