# cuda_esdf

Managed exact 2D GPU Euclidean distance field component.

Input is standard `nav_msgs/OccupancyGrid`; output is
`cuda_robotics_msgs/DistanceField2D` with one finite metric distance per cell.
The input origin, resolution, frame, dimensions, and sensor timestamp are
preserved.

`unknown_policy` is explicit:

- `occupied` (default): unknown cells are zero-distance obstacle seeds;
- `free`: unknown cells do not seed the distance transform.

The core uses a separable exact squared Euclidean distance transform and is
checked cell-by-cell against a brute-force CPU reference.
