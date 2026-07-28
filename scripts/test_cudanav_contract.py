#!/usr/bin/env python3
"""Static contract checks for the CudaNav ROS 2 interface package."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_robotics_msgs"
ODOMETRY_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_kiss_icp"


def message_fields(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    expected_fields = [
        "std_msgs/Header header",
        "geometry_msgs/Pose origin",
        "float32 resolution",
        "uint32 width",
        "uint32 height",
        "float32 max_distance",
        "float32[] distances",
    ]
    assert message_fields(PACKAGE / "msg" / "DistanceField2D.msg") == (
        expected_fields
    )

    package_root = ET.parse(PACKAGE / "package.xml").getroot()
    assert package_root.findtext("name") == "cuda_robotics_msgs"
    assert package_root.findtext("version") == "0.3.0"
    dependencies = {
        element.text
        for tag in ("buildtool_depend", "depend", "exec_depend")
        for element in package_root.findall(tag)
    }
    assert {
        "ament_cmake",
        "geometry_msgs",
        "rosidl_default_generators",
        "rosidl_default_runtime",
        "std_msgs",
    } <= dependencies
    assert package_root.findtext("member_of_group") == (
        "rosidl_interface_packages"
    )

    cmake = (PACKAGE / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "rosidl_generate_interfaces(${PROJECT_NAME}" in cmake
    assert '"msg/DistanceField2D.msg"' in cmake
    assert "install(FILES README.md" in cmake
    assert "ament_export_dependencies(rosidl_default_runtime)" in cmake

    odometry_root = ET.parse(ODOMETRY_PACKAGE / "package.xml").getroot()
    assert odometry_root.findtext("name") == "cuda_kiss_icp"
    assert odometry_root.findtext("version") == "0.3.0"
    odometry_dependencies = {
        element.text
        for tag in ("buildtool_depend", "depend", "test_depend")
        for element in odometry_root.findall(tag)
    }
    assert {
        "ament_cmake",
        "diagnostic_msgs",
        "lifecycle_msgs",
        "nav_msgs",
        "rclcpp_components",
        "rclcpp_lifecycle",
        "sensor_msgs",
        "tf2_ros",
    } <= odometry_dependencies

    odometry_cmake = (
        ODOMETRY_PACKAGE / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    for term in (
        "CUDAROBOTICS_KISS_ICP_CORE_ONLY",
        "gpu_kiss_icp.cu",
        "rclcpp_components_register_nodes",
        "pointcloud_decoder_test",
        "pointcloud_transform_test",
        "lifecycle_configuration_test",
    ):
        assert term in odometry_cmake

    odometry_source = (
        ODOMETRY_PACKAGE / "src" / "cuda_kiss_icp_node.cpp"
    ).read_text(encoding="utf-8")
    for term in (
        'declare_parameter("input_topic", "points")',
        'declare_parameter("odom_topic", "odom")',
        "decode_xyz(*message)",
        "transform_to_base",
        "lookupTransform",
        "odom.header.stamp = stamp",
        "transform.header = odom.header",
        'key_value("deskewed", "false")',
        "TRANSITION_DEACTIVATE",
    ):
        assert term in odometry_source, f"missing odometry contract term: {term}"
    for forbidden in ('"/points"', '"/odom"', '"/diagnostics"'):
        assert forbidden not in odometry_source

    decoder_source = (
        ODOMETRY_PACKAGE / "src" / "pointcloud_decoder.cpp"
    ).read_text(encoding="utf-8")
    for term in (
        'find_field(message, "x")',
        'find_field(message, "y")',
        'find_field(message, "z")',
        "message.row_step",
        "message.is_bigendian",
        "std::isfinite",
    ):
        assert term in decoder_source

    transform_test = (
        ODOMETRY_PACKAGE / "test" / "pointcloud_transform_test.cpp"
    ).read_text(encoding="utf-8")
    for term in (
        "AppliesCompleteSe3",
        "transform.rotation.z",
        "transform.translation.x",
        "EXPECT_NEAR",
    ):
        assert term in transform_test

    architecture = (
        ROOT / "docs" / "cudanav_architecture.md"
    ).read_text(encoding="utf-8")
    required_contract_terms = (
        "cuda_kiss_icp_odometry",
        "cuda_voxel_mapper",
        "cuda_esdf_node",
        "cuda_voxel_costmap_layer",
        "cuda_mppi_controller",
        "/cuda_nav/esdf",
        "width * height == len(distances)",
        "complete SE(3)",
        "controller commands affect subsequent robot state",
        "safe inactive state",
    )
    for term in required_contract_terms:
        assert term in architecture, f"missing CudaNav contract term: {term}"

    print("CudaNav ROS 2 interface contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
