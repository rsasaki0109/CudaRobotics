# Rolling GPU Voxel Mapping Core

`cudarobotics::VoxelMapperGpu` is the persistent 3D occupancy core used by the
CudaNav mapper.

Each finite world-frame LiDAR endpoint is integrated by a CUDA DDA ray:

- traversed voxels receive the configured free log-odds update;
- an in-range endpoint receives the occupied update;
- log odds are atomically clamped;
- observed state is stored separately, so zero log odds never means unknown.

The XY window shifts by integer voxel cells when the sensor enters its rolling
margin. Existing overlapping cells are copied on the GPU and newly exposed
cells become unknown. Z remains fixed in the odometry frame.

The 2D projection has deliberately strict standard semantics:

- `-1`: no observed voxel in the column;
- `0`: observed and no occupied voxel;
- `100`: at least one occupied voxel.

## Verification

```bash
cmake --build build --target voxel_mapping_gpu_smoke -j"$(nproc)"
ctest --test-dir build -R voxel_mapping_gpu_smoke --output-on-failure
```

The smoke test checks free-space ray traversal, endpoint occupancy, unknown
preservation, rolling-window shifting, retained obstacles, snapshot shape, and
malformed input rejection.
