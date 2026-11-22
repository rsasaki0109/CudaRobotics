# Rolling GPU Voxel Mapping Core

`cudarobotics::VoxelMapperGpu` is the persistent 3D occupancy core used by the
CudaNav mapper.

Each finite world-frame LiDAR endpoint is integrated by a CUDA DDA ray:

- traversed voxels receive the configured free log-odds update;
- an in-range endpoint receives the occupied update;
- the updates are applied in two passes, so the result does not depend on the
  order in which rays reach a voxel. The rays count each voxel's free and
  occupied updates in one 64-bit integer (`atomicAdd`). One thread per voxel
  then adds that scan's updates at once and clamps the log odds;
- observed state is stored separately, so zero log odds never means unknown.

Clamping once per scan, rather than after every ray, can differ from a
sequential per-ray update when a voxel's log odds reach a bound within one
scan. The per-ray version gave an order-dependent value there, so reruns of
the closed loop diverged.

The XY window shifts by integer voxel cells when the sensor enters its rolling
margin. Existing overlapping cells are copied on the GPU and newly exposed
cells become unknown. Z remains fixed in the odometry frame.

The 2D projection has deliberately strict standard semantics:

- `-1`: no observed voxel in the column;
- `0`: observed and no occupied voxel;
- `100`: at least one occupied voxel.

`projection_min_z` and `projection_max_z` select the half-open world-Z band
used by this 2D projection without discarding the rest of the 3D map. Their
wide defaults preserve the full-column behavior. CudaNav bringup narrows the
band to exclude ground and overhead returns from the navigation costmap.

## Verification

```bash
cmake --build build --target voxel_mapping_gpu_smoke -j"$(nproc)"
ctest --test-dir build -R voxel_mapping_gpu_smoke --output-on-failure
```

The smoke test checks free-space ray traversal, endpoint occupancy, unknown
preservation, rolling-window shifting, retained obstacles, projection height
filtering, snapshot shape, and malformed input rejection.
