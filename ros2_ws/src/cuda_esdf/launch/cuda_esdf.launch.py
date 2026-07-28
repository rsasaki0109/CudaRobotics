from launch import LaunchDescription
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    return LaunchDescription(
        [
            LifecycleNode(
                package="cuda_esdf",
                executable="cuda_esdf_node",
                name="cuda_esdf_node",
                namespace="cuda_nav",
                output="screen",
                parameters=[
                    {
                        "occupancy_topic": "occupancy",
                        "esdf_topic": "esdf",
                        "diagnostics_topic": "diagnostics",
                        "expected_frame": "odom",
                        "unknown_policy": "occupied",
                    }
                ],
            )
        ]
    )
