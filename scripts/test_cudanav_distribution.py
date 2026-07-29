#!/usr/bin/env python3
"""Static distribution contract for the end-to-end CudaNav Docker path."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = {
    "cuda_robotics_msgs",
    "cuda_robotics_common",
    "cuda_kiss_icp",
    "cuda_voxel_mapping",
    "cuda_esdf",
    "cuda_voxel_costmap_layer",
    "cuda_mppi_controller",
    "cuda_nav_bringup",
}


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    dockerfile = read("docker/Dockerfile")
    entrypoint = read("docker/entrypoint.sh")
    for package in REQUIRED_PACKAGES:
        assert f"ros2_ws/src/{package}" in dockerfile
        assert package in dockerfile
    for dependency in (
        "nav2-controller",
        "nav2-costmap-2d",
        "nav2-mppi-controller",
        "ros2bag",
        "rosbag2-storage-mcap",
        "cuda-cudart-${CUDA_VER}",
        "libcurand-${CUDA_VER}",
    ):
        assert dependency in dockerfile
    for term in (
        "cudanav)",
        "cudanav_closed_loop.launch.py",
        "cudanav_closed_loop.json",
        "smoke_pass",
        "CudaNav closed-loop smoke failed",
        'CMD ["benchmark"]',
    ):
        assert term in entrypoint or term in dockerfile

    command = 'cudarobotics cudanav'
    for relative in (
        "README.md",
        "docker/README.md",
        "docs/site/install.html",
        "docs/site/nav2.html",
        "ros2_ws/src/cuda_nav_bringup/README.md",
    ):
        assert command in read(relative), relative
    assert "10-minute v1.0" in read("docker/README.md")
    assert "10-minute" in read("docs/site/nav2.html")

    workflow = read(".github/workflows/docker-image.yml")
    assert "workflow_dispatch:" in workflow
    assert "file: docker/Dockerfile" in workflow
    assert "Verify end-to-end CudaNav packages in pushed image" in workflow
    for package in REQUIRED_PACKAGES:
        assert package in workflow
    print("CudaNav distribution contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
