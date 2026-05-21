from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('nx',        default_value='128'),
        DeclareLaunchArgument('ny',        default_value='128'),
        DeclareLaunchArgument('nz',        default_value='16'),
        DeclareLaunchArgument('world_x',   default_value='20.0'),
        DeclareLaunchArgument('world_y',   default_value='20.0'),
        DeclareLaunchArgument('world_z',   default_value='5.0'),
        DeclareLaunchArgument('max_range', default_value='20.0'),
        DeclareLaunchArgument('origin_x',  default_value='-10.0'),
        DeclareLaunchArgument('origin_y',  default_value='-10.0'),
        DeclareLaunchArgument('origin_z',  default_value='0.0'),

        Node(
            package='cuda_robotics',
            executable='voxel_node',
            name='voxel_node',
            output='screen',
            parameters=[{
                'nx':        LaunchConfiguration('nx'),
                'ny':        LaunchConfiguration('ny'),
                'nz':        LaunchConfiguration('nz'),
                'world_x':   LaunchConfiguration('world_x'),
                'world_y':   LaunchConfiguration('world_y'),
                'world_z':   LaunchConfiguration('world_z'),
                'max_range': LaunchConfiguration('max_range'),
                'origin_x':  LaunchConfiguration('origin_x'),
                'origin_y':  LaunchConfiguration('origin_y'),
                'origin_z':  LaunchConfiguration('origin_z'),
            }],
            remappings=[
                ('/points',    '/points'),
                ('/odom',      '/odom'),
                ('/voxel_map', '/voxel_map'),
            ],
        ),
    ])
