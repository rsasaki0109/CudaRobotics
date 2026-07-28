# Metric distance to the nearest occupied cell on a row-major 2D grid.
# A finite value in [0, max_distance] is required for every cell.

std_msgs/Header header
geometry_msgs/Pose origin
float32 resolution
uint32 width
uint32 height
float32 max_distance
float32[] distances
