from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    namespace = "cuda_nav"
    output_path = LaunchConfiguration("output_path")
    traversal_count = LaunchConfiguration("traversal_count")
    mission_timeout_sec = LaunchConfiguration("mission_timeout_sec")
    default_controller_config = str(
        Path(get_package_share_directory("cuda_nav_bringup"))
        / "config"
        / "controller.yaml"
    )
    controller_config = LaunchConfiguration("controller_config")
    lifecycle_nodes = [
        LifecycleNode(
            package="cuda_kiss_icp",
            executable="cuda_kiss_icp_node",
            name="cuda_kiss_icp_odometry",
            namespace=namespace,
            output="screen",
            parameters=[
                {
                    "input_topic": "points",
                    "odom_topic": "odom",
                    "diagnostics_topic": "odometry_diagnostics",
                    "odom_frame": "odom",
                    "base_frame": "base_link",
                    "expected_sensor_frame": "base_link",
                    "map_voxel_size": 0.20,
                    "scan_voxel_size": 0.15,
                    "map_radius": 20.0,
                    "max_scan_age_sec": 0.5,
                }
            ],
        ),
        LifecycleNode(
            package="cuda_voxel_mapping",
            executable="cuda_voxel_mapper_node",
            name="cuda_voxel_mapper",
            namespace=namespace,
            output="screen",
            parameters=[
                {
                    "input_topic": "points",
                    "occupancy_topic": "occupancy",
                    "local_map_topic": "local_map",
                    "diagnostics_topic": "mapping_diagnostics",
                    "odom_frame": "odom",
                    "expected_sensor_frame": "base_link",
                    "width": 160,
                    "height": 100,
                    "depth": 24,
                    "resolution": 0.10,
                    "origin_z": -1.2,
                    "max_range": 12.0,
                    "rolling_margin_cells": 30,
                    "max_scan_age_sec": 0.5,
                }
            ],
        ),
        LifecycleNode(
            package="cuda_esdf",
            executable="cuda_esdf_node",
            name="cuda_esdf_node",
            namespace=namespace,
            output="screen",
            parameters=[
                {
                    "occupancy_topic": "occupancy",
                    "esdf_topic": "esdf",
                    "diagnostics_topic": "esdf_diagnostics",
                    "expected_frame": "odom",
                    "unknown_policy": "occupied",
                    "max_distance": 3.0,
                    "max_width": 160,
                    "max_height": 100,
                    "max_input_age_sec": 0.5,
                }
            ],
        ),
        LifecycleNode(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            namespace=namespace,
            output="screen",
            parameters=[controller_config],
        ),
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_path",
                default_value="/tmp/cudanav_closed_loop.json",
            ),
            DeclareLaunchArgument("traversal_count", default_value="1"),
            DeclareLaunchArgument(
                "mission_timeout_sec", default_value="90.0"
            ),
            DeclareLaunchArgument(
                "controller_config",
                default_value=default_controller_config,
            ),
            Node(
                package="cuda_nav_bringup",
                executable="cudanav_loopback_simulator",
                name="cudanav_loopback_simulator",
                namespace=namespace,
                output="screen",
            ),
            *lifecycle_nodes,
            Node(
                package="cuda_nav_bringup",
                executable="lifecycle_orchestrator",
                name="cudanav_lifecycle_orchestrator",
                namespace=namespace,
                output="screen",
            ),
            Node(
                package="cuda_nav_bringup",
                executable="follow_path_mission",
                name="cudanav_follow_path_mission",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "output_path": output_path,
                        "traversal_count": ParameterValue(
                            traversal_count, value_type=int
                        ),
                        "mission_timeout_sec": ParameterValue(
                            mission_timeout_sec, value_type=float
                        ),
                    }
                ],
            ),
        ]
    )
