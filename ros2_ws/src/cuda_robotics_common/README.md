# cuda_robotics_common

Shared CudaNav ROS 2 host utilities:

- schema-aware PointCloud2 XYZ decoding by field name;
- FLOAT32/FLOAT64, organized-row padding, and endianness handling;
- complete quaternion-rotation plus translation SE(3) point transforms.

The utilities are independent of CUDA so schema and frame correctness can be
tested on no-GPU CI runners.
