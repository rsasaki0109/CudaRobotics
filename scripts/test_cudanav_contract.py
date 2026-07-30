#!/usr/bin/env python3
"""Static contract checks for the CudaNav ROS 2 interface package."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_robotics_msgs"
ODOMETRY_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_kiss_icp"
COMMON_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_robotics_common"
MAPPING_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_voxel_mapping"
ESDF_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_esdf"
COSTMAP_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_voxel_costmap_layer"
BRINGUP_PACKAGE = ROOT / "ros2_ws" / "src" / "cuda_nav_bringup"
SOURCE_VERSION = json.loads(
    (ROOT / "docs" / "v1_support_matrix.json").read_text(encoding="utf-8")
)["source_version"]


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
    assert package_root.findtext("version") == SOURCE_VERSION
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
    assert odometry_root.findtext("version") == SOURCE_VERSION
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
    assert common_root.findtext("version") == SOURCE_VERSION

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
    assert mapping_root.findtext("version") == SOURCE_VERSION
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
    assert esdf_root.findtext("version") == SOURCE_VERSION
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

    costmap_root = ET.parse(COSTMAP_PACKAGE / "package.xml").getroot()
    assert costmap_root.findtext("name") == "cuda_voxel_costmap_layer"
    assert costmap_root.findtext("version") == SOURCE_VERSION
    costmap_dependencies = {
        element.text
        for tag in ("buildtool_depend", "depend", "test_depend")
        for element in costmap_root.findall(tag)
    }
    assert {
        "ament_cmake",
        "geometry_msgs",
        "nav2_costmap_2d",
        "nav_msgs",
        "pluginlib",
        "rclcpp",
    } <= costmap_dependencies
    costmap_cmake = (COSTMAP_PACKAGE / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    for term in (
        "pluginlib_export_plugin_description_file",
        "occupancy_bridge_test",
        "plugin_load_test",
    ):
        assert term in costmap_cmake
    costmap_plugin = ET.parse(
        COSTMAP_PACKAGE / "cuda_voxel_costmap_plugin.xml"
    ).getroot()
    assert costmap_plugin.find("class").get("base_class_type") == (
        "nav2_costmap_2d::Layer"
    )
    costmap_source = (
        COSTMAP_PACKAGE / "src" / "cuda_voxel_costmap_layer.cpp"
    ).read_text(encoding="utf-8")
    for term in (
        'declareParameter("occupancy_topic"',
        '"occupancy_topic must be non-empty"',
        "validate_occupancy_grid",
        "sample_occupancy_cost",
        "mapToWorld",
        "NO_INFORMATION",
        "PLUGINLIB_EXPORT_CLASS",
    ):
        assert term in costmap_source, (
            f"missing voxel costmap contract term: {term}"
        )
    assert 'rclcpp::ParameterValue("/occupancy")' not in costmap_source

    bringup_root = ET.parse(BRINGUP_PACKAGE / "package.xml").getroot()
    assert bringup_root.findtext("name") == "cuda_nav_bringup"
    assert bringup_root.findtext("version") == SOURCE_VERSION
    bringup_dependencies = {
        element.text
        for tag in ("buildtool_depend", "exec_depend", "test_depend")
        for element in bringup_root.findall(tag)
    }
    assert {
        "ament_python",
        "cuda_esdf",
        "cuda_kiss_icp",
        "cuda_mppi_controller",
        "cuda_voxel_costmap_layer",
        "cuda_voxel_mapping",
        "diagnostic_msgs",
        "lifecycle_msgs",
        "nav2_controller",
        "nav2_msgs",
        "rclpy",
        "ros2bag",
        "rosbag2_storage_mcap",
        "sensor_msgs",
    } <= bringup_dependencies
    bringup_launch = (
        BRINGUP_PACKAGE / "launch" / "cudanav_closed_loop.launch.py"
    ).read_text(encoding="utf-8")
    for term in (
        'package="cuda_kiss_icp"',
        'package="cuda_voxel_mapping"',
        'package="cuda_esdf"',
        'package="nav2_controller"',
        'executable="cudanav_loopback_simulator"',
        'executable="lifecycle_orchestrator"',
        'executable="follow_path_mission"',
        'DeclareLaunchArgument(',
        '"controller_config"',
        '"transform_timeout_sec": 0.4',
    ):
        assert term in bringup_launch
    controller_config = (
        BRINGUP_PACKAGE / "config" / "controller.yaml"
    ).read_text(encoding="utf-8")
    for term in (
        "/cuda_nav/controller_server:",
        "/cuda_nav/local_costmap/local_costmap:",
        "    controller_frequency: 10.0",
        'plugin: "cuda_mppi_controller::CudaMppiController"',
        'plugin: "nav2_controller::PoseProgressChecker"',
        "      required_movement_angle: 0.35",
        'plugin: "cuda_voxel_costmap_layer::CudaVoxelCostmapLayer"',
        "    width: 12\n",
        "    height: 6\n",
    ):
        assert term in controller_config, (
            f"missing namespaced controller config term: {term}"
        )
    assert "    width: 12.0\n" not in controller_config
    assert "    height: 6.0\n" not in controller_config
    assert "\ncontroller_server:" not in controller_config
    assert "\nlocal_costmap:" not in controller_config
    shadow_launch = (
        BRINGUP_PACKAGE / "launch" / "cudanav_recorded_shadow.launch.py"
    ).read_text(encoding="utf-8")
    for term in (
        'package="cuda_kiss_icp"',
        'package="cuda_voxel_mapping"',
        'package="cuda_esdf"',
        'package="nav2_controller"',
        '"FollowPath.diagnostics_csv_path"',
        'executable="recorded_path_shadow"',
        'DeclareLaunchArgument("points_topic"',
        'DeclareLaunchArgument("path_topic"',
        'DeclareLaunchArgument("readiness_timeout_sec"',
        'DeclareLaunchArgument("use_sim_time"',
        '"readiness_timeout_sec": readiness_timeout_sec',
        '"max_scan_age_sec": 0.0',
        '"max_input_age_sec": 0.0',
    ):
        assert term in shadow_launch
    rosbag_qos = (
        BRINGUP_PACKAGE / "config" / "rosbag_qos_overrides.yaml"
    ).read_text(encoding="utf-8")
    for term in ("/tf_static:", "reliable", "transient_local"):
        assert term in rosbag_qos
    setup_source = (BRINGUP_PACKAGE / "setup.py").read_text(encoding="utf-8")
    assert '"config/rosbag_qos_overrides.yaml"' in setup_source
    assert '"config/controller_recorded_shadow.yaml"' in setup_source
    shadow_controller_config = (
        BRINGUP_PACKAGE / "config" / "controller_recorded_shadow.yaml"
    ).read_text(encoding="utf-8")
    assert "      max_map_age_sec: 0.0" in shadow_controller_config
    assert "      unknown_is_free: true" in shadow_controller_config
    mppi_controller_source = (
        ROOT
        / "ros2_ws"
        / "src"
        / "cuda_mppi_controller"
        / "src"
        / "cuda_mppi_controller.cpp"
    ).read_text(encoding="utf-8")
    assert "std::setprecision(17)" in mppi_controller_source
    shadow_source = (
        BRINGUP_PACKAGE
        / "cuda_nav_bringup"
        / "recorded_path_shadow.py"
    ).read_text(encoding="utf-8")
    for term in (
        "FollowPath",
        "Path",
        "minimum_path_poses",
        "commands do not alter",
        "self._retain_inflight()",
        "retaining recorded path for retry",
    ):
        assert term in shadow_source
    simulator_source = (
        BRINGUP_PACKAGE / "cuda_nav_bringup" / "loopback_simulator.py"
    ).read_text(encoding="utf-8")
    for term in (
        "TwistStamped",
        "PointCloud2",
        "ground_truth_topic",
        "collision_count_topic",
        "raycast(",
    ):
        assert term in simulator_source
    for forbidden in (
        "TransformBroadcaster",
        "Odometry()",
        "create_publisher(Odometry",
    ):
        assert forbidden not in simulator_source
    orchestrator_source = (
        BRINGUP_PACKAGE
        / "cuda_nav_bringup"
        / "lifecycle_orchestrator.py"
    ).read_text(encoding="utf-8")
    assert "ChangeState" in orchestrator_source
    assert "GetState" in orchestrator_source
    assert "contains_transform" in orchestrator_source
    assert "_wait_for_odometry_transform" in orchestrator_source
    assert 'TFMessage, "/tf"' in orchestrator_source
    mission_source = (
        BRINGUP_PACKAGE / "cuda_nav_bringup" / "follow_path_mission.py"
    ).read_text(encoding="utf-8")
    for term in (
        "FollowPath",
        '"schema_version": 1',
        '"collision_count"',
        '"odometry_drift_percent"',
        '"command_deadline_miss_rate"',
        '"smoke_pass"',
        '"traversals_requested"',
        '"traversals_completed"',
        '"trajectory_csv"',
        '"diagnostic_error_count"',
        '"diagnostic_components"',
        '"failure_counters"',
        'self.declare_parameter("controller_frequency", 10.0)',
        "while rclpy.ok() and not node.finished",
        "action server is not active yet; retrying first goal",
        "except KeyboardInterrupt:",
    ):
        assert term in mission_source
    evidence_source = (ROOT / "scripts" / "cudanav_evidence.py").read_text(
        encoding="utf-8"
    )
    for term in (
        '"smoke": GatePolicy',
        '"release": GatePolicy',
        "min_elapsed_sec=600.0",
        "max_drift_percent=1.0",
        "max_deadline_miss_rate=0.01",
        "require_bag=True",
        "require_video=True",
        "is_relative_to(root)",
        '"all_traversals_completed"',
        "REQUIRED_CLOSED_LOOP_BAG_TOPICS",
        '"rosbag_content_unchanged"',
        '"required_bag_topic_messages"',
        '"bag_command_bound"',
    ):
        assert term in evidence_source
    rosbag_evidence_source = (
        ROOT / "scripts" / "cudanav_rosbag_evidence.py"
    ).read_text(encoding="utf-8")
    for term in (
        "REQUIRED_CUDANAV_OUTPUT_TOPICS",
        "rosbag_topic_counts",
        '"recording_content_unchanged"',
        '"required_output_topic_messages"',
        '"record_command_bound"',
    ):
        assert term in rosbag_evidence_source
    harness_source = (
        ROOT / "scripts" / "run_cudanav_closed_loop.py"
    ).read_text(encoding="utf-8")
    for term in (
        '"git_commit"',
        '"git_dirty"',
        '"config_sha256"',
        '"artifact_sha256"',
        'f"controller_config:={config_copy}"',
        '"gpu"',
        '"launch_log"',
        '"rosbag"',
        '"rosbag_identity"',
        "REQUIRED_CLOSED_LOOP_BAG_TOPICS",
        '"video"',
        '"traversal_count"',
        "refusing non-empty output directory",
    ):
        assert term in harness_source
    renderer_source = (
        ROOT / "scripts" / "render_cudanav_trajectory.py"
    ).read_text(encoding="utf-8")
    for term in (
        "default_segments",
        "Image.new",
        'format="GIF"',
        "temporary.replace(output)",
    ):
        assert term in renderer_source
    multi_gpu_source = (
        ROOT / "scripts" / "cudanav_multi_gpu.py"
    ).read_text(encoding="utf-8")
    for term in (
        '"same_git_commit"',
        '"same_config"',
        '"gpu_device_coverage"',
        '"gpu_model_coverage"',
        '"device_binding"',
        '"manifest_binding"',
        "is_relative_to(root)",
    ):
        assert term in multi_gpu_source
    multi_gpu_runner = (
        ROOT / "scripts" / "run_cudanav_multi_gpu.py"
    ).read_text(encoding="utf-8")
    for term in (
        '"CUDA_VISIBLE_DEVICES"',
        '"minimum_gpu_devices"',
        '"minimum_gpu_models"',
        "output inside the repository must be git-ignored",
    ):
        assert term in multi_gpu_runner

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
