# Docker demo: end-to-end CudaNav and Nav2 GPU MPPI

One-command demo of [`cuda_mppi_controller`](../ros2_ws/src/cuda_mppi_controller/)
— the GPU-accelerated MPPI controller plugin for Nav2 (1 CUDA thread =
1 sampled rollout). No ROS or CUDA toolkit needed on the host, only an
NVIDIA driver (>= 525, CUDA 12 capable) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
# from a published image
docker run --rm --gpus all ghcr.io/rsasaki0109/cuda-mppi-controller-demo

# or build locally from the repo root
docker build -f docker/Dockerfile -t cuda-mppi-demo .
docker run --rm --gpus all cuda-mppi-demo
```

Run the end-to-end short closed loop:

```bash
docker build -f docker/Dockerfile -t cudarobotics .
docker run --rm --gpus all \
  -v "$PWD/out:/out" \
  cudarobotics cudanav
```

This launches GPU KISS-ICP, rolling voxel mapping, typed ESDF, the Nav2
costmap layer, CUDA MPPI, and the command-driven simulator. It exits non-zero
unless the generated `/out/cudanav_closed_loop.json` has `smoke_pass: true`;
the full ROS launch log is retained alongside it. This is the short
integration demo, not the 10-minute v1.0 release evidence run.
The source-build command is authoritative until a release tag containing this
mode has passed the Docker/GPU gate and updated the GHCR image.

The default command remains backward compatible and loads the controller
plugin through pluginlib exactly as Nav2's
`controller_server` does, then runs the closed-loop head-to-head benchmark —
stock `nav2_mppi_controller` (CPU) vs `cuda_mppi_controller` (GPU) on the
same costmap, plan, and limits — and prints the summary table
(success / steps / mean solve ms per configuration).

More modes:

```bash
# all three scenarios: wall_gap, narrow_corridor, u_turn
docker run --rm --gpus all cuda-mppi-demo benchmark all

# keep the CSV + per-run trajectories
docker run --rm --gpus all -v "$PWD/out:/out" cuda-mppi-demo

# standalone GPU optimizer (no nav2 plugins): K rollouts, motion model
docker run --rm --gpus all cuda-mppi-demo standalone 65536 ackermann

# poke around
docker run --rm -it --gpus all cuda-mppi-demo bash
```

The image is built by `.github/workflows/docker-image.yml` on version tags
(`v*`) and pushed to GHCR.
