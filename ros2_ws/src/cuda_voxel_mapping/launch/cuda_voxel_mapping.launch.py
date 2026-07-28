from launch import LaunchDescription
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    return LaunchDescription(
        [
            LifecycleNode(
                package="cuda_voxel_mapping",
                executable="cuda_voxel_mapper_node",
                name="cuda_voxel_mapper",
                namespace="cuda_nav",
                output="screen",
                parameters=[
                    {
                        "input_topic": "points",
                        "occupancy_topic": "occupancy",
                        "local_map_topic": "local_map",
                        "diagnostics_topic": "diagnostics",
                        "odom_frame": "odom",
                    }
                ],
            )
        ]
    )
