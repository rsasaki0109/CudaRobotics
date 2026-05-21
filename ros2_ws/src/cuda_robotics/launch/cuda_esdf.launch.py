from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_distance', default_value='10.0'),
        DeclareLaunchArgument('occupancy_threshold', default_value='50'),

        Node(
            package='cuda_robotics',
            executable='esdf_node',
            name='esdf_node',
            output='screen',
            parameters=[{
                'max_distance':        LaunchConfiguration('max_distance'),
                'occupancy_threshold': LaunchConfiguration('occupancy_threshold'),
            }],
            remappings=[
                ('/map',  '/map'),
                ('/esdf', '/esdf'),
            ],
        ),
    ])
