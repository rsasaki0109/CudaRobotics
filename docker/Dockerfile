# GPU MPPI controller demo image — runs the Nav2 CPU-vs-GPU head-to-head
# benchmark from ros2_ws/src/cuda_mppi_controller without any local setup:
#
#   docker build -f docker/Dockerfile -t cuda-mppi-demo .
#   docker run --rm --gpus all cuda-mppi-demo                  # wall_gap benchmark
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
      build-essential cmake python3-colcon-common-extensions \
      cuda-nvcc-${CUDA_VER} cuda-cudart-dev-${CUDA_VER} libcurand-dev-${CUDA_VER} \
      ros-${ROS_DISTRO}-nav2-core ros-${ROS_DISTRO}-nav2-costmap-2d \
      ros-${ROS_DISTRO}-nav2-util ros-${ROS_DISTRO}-nav2-mppi-controller \
      ros-${ROS_DISTRO}-tf2-geometry-msgs && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:${PATH}

# The package reads the shared GPU core from the repo root (../../.. from the
# package dir), so mirror that layout: /ws/{src,include,ros2_ws}.
WORKDIR /ws
COPY src/mppi_gpu.cu src/
COPY include/cuda_mppi_controller include/cuda_mppi_controller
COPY ros2_ws/src/cuda_mppi_controller ros2_ws/src/cuda_mppi_controller

# SASS for Turing/Ampere/Ada (+ Hopper PTX so newer GPUs JIT).
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && cd ros2_ws && \
    colcon build --packages-select cuda_mppi_controller \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
        "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90"

FROM ros:${ROS_DISTRO}-ros-core
ARG ROS_DISTRO
ARG CUDA_VER=12-6

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
      cuda-cudart-${CUDA_VER} libcurand-${CUDA_VER} bsdextrautils \
      ros-${ROS_DISTRO}-nav2-core ros-${ROS_DISTRO}-nav2-costmap-2d \
      ros-${ROS_DISTRO}-nav2-util ros-${ROS_DISTRO}-nav2-mppi-controller \
      ros-${ROS_DISTRO}-tf2-geometry-msgs && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /ws/ros2_ws/install /ws/install
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["/entrypoint.sh"]
CMD ["benchmark"]
