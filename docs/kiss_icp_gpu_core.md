# Reusable GPU KISS-ICP Core

`cudarobotics_kiss_icp_gpu` is the streaming LiDAR-odometry core used by the
`gpu_kiss_icp` demo and intended for the CudaNav ROS 2 component. It has no
OpenCV dependency and never consumes ground truth.

## API

Include:

```cpp
#include <cudarobotics/kiss_icp_gpu.hpp>
```

Create one persistent odometry instance, choose the coordinate-system anchor
explicitly, and submit tightly packed `x, y, z` points:

```cpp
cudarobotics::KissIcpConfig config;
config.map_voxel_size = 0.5f;
config.scan_voxel_size = 0.5f;
config.nn_backend = cudarobotics::KissIcpNnBackend::Voxel;

cudarobotics::KissIcpOdometry odometry(config);
cudarobotics::KissIcpPose initial_pose;  // identity by default
odometry.reset(initial_pose);

for (const std::vector<float>& xyz : scans) {
    auto frame = odometry.register_scan(xyz);
    publish_odometry(frame.pose);
}
```

The first scan initializes the voxel map at `initial_pose`. Every later call
uses the previous estimate as the prediction, computes adaptive-threshold
point-to-plane ICP against the local map, and inserts the registered scan.

## Contract

- Input is finite sensor-frame XYZ in metres, with exactly three floats per
  point.
- Output pose is `T_world_sensor`; the caller chooses `world` in `reset()`.
- A `KissIcpOdometry` instance owns persistent CUDA buffers and is not copyable.
- Scan and local-map capacities are explicit. Overflow raises an exception;
  points are never silently truncated.
- The voxel NN hash capacity must be a power of two and at least the configured
  map capacity.
- `map_snapshot()` returns the current first-observation voxel map in world
  coordinates.
- `KissIcpFrameResult` exposes input/sample/map counts and per-frame ICP
  diagnostics. `timing()` exposes accumulated map upload, normal estimation,
  and index-build time.

Configuration can be checked before CUDA allocation with
`validate_kiss_icp_config()`.

## Verification

```bash
cmake --build build --target cudarobotics_kiss_icp_gpu \
  kiss_icp_gpu_streaming_smoke gpu_kiss_icp -j"$(nproc)"
ctest --test-dir build -R 'kiss_icp_gpu_streaming_smoke|gpu_kiss_icp_gate' \
  --output-on-failure
```

The streaming smoke verifies explicit-pose initialization, stationary-scan
registration, malformed input rejection, and reset semantics. The existing
64-frame benchmark remains the accuracy and correspondence-performance gate.

## CudaNav boundary

The ROS 2 lifecycle component will own this class. PointCloud2 decoding,
timestamp checks, TF publication, diagnostics, and lifecycle recovery remain
ROS responsibilities and are intentionally outside this CUDA core.
