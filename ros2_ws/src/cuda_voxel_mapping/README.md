# cuda_voxel_mapping

Managed rolling 3D voxel mapper for CudaNav.

- Decodes PointCloud2 fields through `cuda_robotics_common`.
- Looks up `odom <- sensor` at the original scan timestamp.
- Applies the complete SE(3) transform before GPU ray integration.
- Maintains observed state separately from zero log-odds.
- Publishes standard `OccupancyGrid` values: `-1` unknown, `0` free,
  `100` occupied.
- Publishes occupied 3D voxel centres as `local_map`.

GPU buffers exist only while active. Runtime CUDA/capacity failures publish an
ERROR diagnostic and force the lifecycle node inactive.
