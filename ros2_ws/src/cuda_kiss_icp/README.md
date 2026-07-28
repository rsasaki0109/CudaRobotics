# cuda_kiss_icp

Managed ROS 2 component for the reusable CudaRobotics GPU KISS-ICP core.

The node subscribes to relative topic `points`, decodes named XYZ fields, uses
TF to apply the complete sensor-to-`base_link` transform, and publishes
`nav_msgs/Odometry`, `odom -> base_link` TF, and diagnostics at the original
sensor timestamp.

```bash
ros2 launch cuda_kiss_icp cuda_kiss_icp.launch.py
ros2 lifecycle set /cuda_nav/cuda_kiss_icp_odometry configure
ros2 lifecycle set /cuda_nav/cuda_kiss_icp_odometry activate
```

GPU memory is allocated on activation and released on deactivation. A fatal
CUDA/core/capacity failure publishes an ERROR diagnostic and forces the node
inactive; cleanup and reconfiguration are required before resuming.
