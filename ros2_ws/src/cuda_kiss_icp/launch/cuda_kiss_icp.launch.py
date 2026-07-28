from launch import LaunchDescription
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    return LaunchDescription(
        [
            LifecycleNode(
                package="cuda_kiss_icp",
                executable="cuda_kiss_icp_node",
                name="cuda_kiss_icp_odometry",
                namespace="cuda_nav",
                output="screen",
                parameters=[
                    {
                        "input_topic": "points",
                        "odom_topic": "odom",
                        "diagnostics_topic": "diagnostics",
                        "odom_frame": "odom",
                        "base_frame": "base_link",
                    }
                ],
            )
        ]
    )
