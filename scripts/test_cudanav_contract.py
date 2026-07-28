#!/usr/bin/env python3
"""Static contract checks for the CudaNav ROS 2 interface package."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_robotics_msgs"
ODOMETRY_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_kiss_icp"
COMMON_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_robotics_common"
MAPPING_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_voxel_mapping"
ESDF_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_esdf"


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

    common_root = ET.parse(COMMON_PACKAGE / "package.xml").getroot()
    assert common_root.findtext("name") == "cuda_robotics_common"
    assert common_root.findtext("version") == "0.3.0"

    odometry_cmake = (
        ODOMETRY_PACKAGE / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    for term in (
        "CUDAROBOTICS_KISS_ICP_CORE_ONLY",
        "gpu_kiss_icp.cu",
        "rclcpp_components_register_nodes",
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
        COMMON_PACKAGE / "src" / "pointcloud_decoder.cpp"
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
        COMMON_PACKAGE / "test" / "pointcloud_transform_test.cpp"
    ).read_text(encoding="utf-8")
    for term in (
        "AppliesCompleteSe3",
        "transform.rotation.z",
        "transform.translation.x",
        "EXPECT_NEAR",
    ):
        assert term in transform_test

    mapping_root = ET.parse(MAPPING_PACKAGE / "package.xml").getroot()
    assert mapping_root.findtext("name") == "cuda_voxel_mapping"
    assert mapping_root.findtext("version") == "0.3.0"
    mapping_dependencies = {
        element.text
        for tag in ("buildtool_depend", "depend", "test_depend")
        for element in mapping_root.findall(tag)
    }
    assert {
        "ament_cmake",
        "cuda_robotics_common",
        "diagnostic_msgs",
        "nav_msgs",
        "rclcpp_components",
        "rclcpp_lifecycle",
        "sensor_msgs",
        "tf2_ros",
    } <= mapping_dependencies
    mapping_cmake = (
        MAPPING_PACKAGE / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    for term in (
        "voxel_mapping_gpu.cu",
        "rclcpp_components_register_nodes",
        "lifecycle_configuration_test",
    ):
        assert term in mapping_cmake
    mapping_source = (
        MAPPING_PACKAGE / "src" / "cuda_voxel_mapper_node.cpp"
    ).read_text(encoding="utf-8")
    for term in (
        'declare_parameter("input_topic", "points")',
        'declare_parameter("occupancy_topic", "occupancy")',
        'declare_parameter("local_map_topic", "local_map")',
        "cuda_robotics_common::decode_xyz",
        "cuda_robotics_common::transform_xyz",
        "lookupTransform",
        "message.data = projection.data",
        'key_value("unknown_value", "-1")',
        'key_value("free_value", "0")',
        'key_value("occupied_value", "100")',
        "TRANSITION_DEACTIVATE",
    ):
        assert term in mapping_source, f"missing mapping contract term: {term}"
    for forbidden in ('"/points"', '"/occupancy"', '"/local_map"', '"/diagnostics"'):
        assert forbidden not in mapping_source

    mapping_core = (
        ROOT / "src" / "voxel_mapping_gpu.cu"
    ).read_text(encoding="utf-8")
    for term in (
        "raycast_kernel",
        "shift_grid_kernel",
        "project_occupancy_kernel",
        "!any_observed ? -1 : (occupied ? 100 : 0)",
        "atomic_add_clamped",
    ):
        assert term in mapping_core

    esdf_root = ET.parse(ESDF_PACKAGE / "package.xml").getroot()
    assert esdf_root.findtext("name") == "cuda_esdf"
    assert esdf_root.findtext("version") == "0.3.0"
    esdf_dependencies = {
        element.text
        for tag in ("buildtool_depend", "depend", "test_depend")
        for element in esdf_root.findall(tag)
    }
    assert {
        "ament_cmake",
        "cuda_robotics_msgs",
        "diagnostic_msgs",
        "nav_msgs",
        "rclcpp_components",
        "rclcpp_lifecycle",
    } <= esdf_dependencies
    esdf_cmake = (ESDF_PACKAGE / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    for term in (
        "esdf_2d_gpu.cu",
        "rclcpp_components_register_nodes",
        "lifecycle_configuration_test",
    ):
        assert term in esdf_cmake
    esdf_source = (
        ESDF_PACKAGE / "src" / "cuda_esdf_node.cpp"
    ).read_text(encoding="utf-8")
    for term in (
        'declare_parameter("occupancy_topic", "occupancy")',
        'declare_parameter("esdf_topic", "esdf")',
        'declare_parameter("unknown_policy", "occupied")',
        "cuda_robotics_msgs::msg::DistanceField2D",
        "output.header = message->header",
        "output.origin = message->info.origin",
        "output.distances = result.distances",
        "occupancy values must lie in [-1, 100]",
        "TRANSITION_DEACTIVATE",
    ):
        assert term in esdf_source, f"missing ESDF contract term: {term}"
    for forbidden in ('"/occupancy"', '"/esdf"', '"/diagnostics"'):
        assert forbidden not in esdf_source
    esdf_core = (ROOT / "src" / "esdf_2d_gpu.cu").read_text(
        encoding="utf-8"
    )
    for term in (
        "distance_transform_1d",
        "row_distance_kernel",
        "column_distance_kernel",
        "compute_esdf_2d_cpu_reference",
        "unknown_space_policy_name",
    ):
        assert term in esdf_core

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
