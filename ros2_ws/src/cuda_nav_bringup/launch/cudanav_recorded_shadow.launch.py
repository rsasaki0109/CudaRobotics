from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    params_file = LaunchConfiguration("params_file")
    diagnostics_csv = LaunchConfiguration("diagnostics_csv")
    points_topic = LaunchConfiguration("points_topic")
    path_topic = LaunchConfiguration("path_topic")
    sensor_frame = LaunchConfiguration("sensor_frame")
    use_sim_time = ParameterValue(
        LaunchConfiguration("use_sim_time"), value_type=bool
    )
    common = {"use_sim_time": use_sim_time}
    lifecycle_nodes = [
        LifecycleNode(
            package="cuda_kiss_icp",
            executable="cuda_kiss_icp_node",
            name="cuda_kiss_icp_odometry",
            namespace=namespace,
            output="screen",
            parameters=[
                {
                    **common,
                    "input_topic": "points",
                    "odom_topic": "odom",
                    "diagnostics_topic": "odometry_diagnostics",
                    "odom_frame": "odom",
                    "base_frame": "base_link",
                    "expected_sensor_frame": sensor_frame,
                    "map_voxel_size": 0.20,
                    "scan_voxel_size": 0.15,
                    "map_radius": 20.0,
                    "max_scan_age_sec": 0.5,
                }
            ],
            remappings=[("points", points_topic)],
        ),
        LifecycleNode(
            package="cuda_voxel_mapping",
            executable="cuda_voxel_mapper_node",
            name="cuda_voxel_mapper",
            namespace=namespace,
            output="screen",
            parameters=[
                {
                    **common,
                    "input_topic": "points",
                    "occupancy_topic": "occupancy",
                    "local_map_topic": "local_map",
                    "diagnostics_topic": "mapping_diagnostics",
                    "odom_frame": "odom",
                    "expected_sensor_frame": sensor_frame,
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
            remappings=[("points", points_topic)],
        ),
        LifecycleNode(
            package="cuda_esdf",
            executable="cuda_esdf_node",
            name="cuda_esdf_node",
            namespace=namespace,
            output="screen",
            parameters=[
                {
                    **common,
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
            parameters=[
                params_file,
                {
                    "use_sim_time": use_sim_time,
                    "FollowPath.diagnostics_csv_path": diagnostics_csv,
                },
            ],
        ),
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="cuda_nav"),
            DeclareLaunchArgument("params_file"),
            DeclareLaunchArgument("diagnostics_csv"),
            DeclareLaunchArgument("points_topic", default_value="/points"),
            DeclareLaunchArgument("path_topic", default_value="/plan"),
            DeclareLaunchArgument("sensor_frame", default_value="base_link"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            *lifecycle_nodes,
            Node(
                package="cuda_nav_bringup",
                executable="lifecycle_orchestrator",
                name="cudanav_lifecycle_orchestrator",
                namespace=namespace,
                output="screen",
                parameters=[common],
            ),
            Node(
                package="cuda_nav_bringup",
                executable="recorded_path_shadow",
                name="cudanav_recorded_path_shadow",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        **common,
                        "path_topic": "recorded_path",
                        "action_name": "follow_path",
                    }
                ],
                remappings=[("recorded_path", path_topic)],
            ),
        ]
    )
