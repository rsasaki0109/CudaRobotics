# End-to-end CudaNav + GPU MPPI demo image. The default keeps the existing
# Nav2 CPU-vs-GPU benchmark; `cudanav` runs the short closed-loop stack:
#
#   docker build -f docker/Dockerfile -t cuda-mppi-demo .
#   docker run --rm --gpus all cuda-mppi-demo                  # wall_gap benchmark
#   docker run --rm --gpus all -v "$PWD/out:/out" cuda-mppi-demo cudanav
#   docker run --rm --gpus all cuda-mppi-demo benchmark all    # all scenarios
#   docker run --rm --gpus all -v "$PWD/out:/out" cuda-mppi-demo
#
# Requires the NVIDIA Container Toolkit and a CUDA 12 capable driver (>= 525).

ARG ROS_DISTRO=jazzy

FROM ros:${ROS_DISTRO} AS build
ARG ROS_DISTRO
ARG CUDA_VER=12-6

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake python3-pil python3-colcon-common-extensions \
      cuda-nvcc-${CUDA_VER} cuda-cudart-dev-${CUDA_VER} libcurand-dev-${CUDA_VER} \
      ros-${ROS_DISTRO}-nav2-core ros-${ROS_DISTRO}-nav2-controller \
      ros-${ROS_DISTRO}-nav2-costmap-2d ros-${ROS_DISTRO}-nav2-msgs \
      ros-${ROS_DISTRO}-nav2-util ros-${ROS_DISTRO}-nav2-mppi-controller \
      ros-${ROS_DISTRO}-ros2bag ros-${ROS_DISTRO}-rosbag2-storage-mcap \
      ros-${ROS_DISTRO}-ament-cmake-gtest \
      ros-${ROS_DISTRO}-rosidl-default-generators \
      ros-${ROS_DISTRO}-tf2-geometry-msgs && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:${PATH}

# The ROS packages read the shared GPU core from the repo root (../../.. from
# each package), so mirror that layout: /ws/{src,include,ros2_ws}.
WORKDIR /ws
COPY src/mppi_gpu.cu src/gpu_kiss_icp.cu src/voxel_mapping_gpu.cu src/esdf_2d_gpu.cu src/
COPY include/cuda_check.cuh include/cuda_check.cuh
COPY include/cudarobotics include/cudarobotics
COPY include/cuda_mppi_controller include/cuda_mppi_controller
COPY ros2_ws/src/cuda_robotics_msgs ros2_ws/src/cuda_robotics_msgs
COPY ros2_ws/src/cuda_robotics_common ros2_ws/src/cuda_robotics_common
COPY ros2_ws/src/cuda_kiss_icp ros2_ws/src/cuda_kiss_icp
COPY ros2_ws/src/cuda_voxel_mapping ros2_ws/src/cuda_voxel_mapping
COPY ros2_ws/src/cuda_esdf ros2_ws/src/cuda_esdf
COPY ros2_ws/src/cuda_voxel_costmap_layer ros2_ws/src/cuda_voxel_costmap_layer
COPY ros2_ws/src/cuda_mppi_controller ros2_ws/src/cuda_mppi_controller
COPY ros2_ws/src/cuda_nav_bringup ros2_ws/src/cuda_nav_bringup

# SASS for Turing/Ampere/Ada (+ Hopper PTX so newer GPUs JIT).
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && cd ros2_ws && \
    colcon build \
      --packages-select \
        cuda_robotics_msgs cuda_robotics_common cuda_kiss_icp \
        cuda_voxel_mapping cuda_esdf cuda_voxel_costmap_layer \
        cuda_mppi_controller cuda_nav_bringup \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
        "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90"

FROM ros:${ROS_DISTRO}-ros-core
ARG ROS_DISTRO
ARG CUDA_VER=12-6

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
      cuda-cudart-${CUDA_VER} libcurand-${CUDA_VER} bsdextrautils python3-pil \
      ros-${ROS_DISTRO}-nav2-core ros-${ROS_DISTRO}-nav2-controller \
      ros-${ROS_DISTRO}-nav2-costmap-2d ros-${ROS_DISTRO}-nav2-msgs \
      ros-${ROS_DISTRO}-nav2-util ros-${ROS_DISTRO}-nav2-mppi-controller \
      ros-${ROS_DISTRO}-ros2bag ros-${ROS_DISTRO}-rosbag2-storage-mcap \
      ros-${ROS_DISTRO}-tf2-geometry-msgs && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /ws/ros2_ws/install /ws/install
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["/entrypoint.sh"]
CMD ["benchmark"]
