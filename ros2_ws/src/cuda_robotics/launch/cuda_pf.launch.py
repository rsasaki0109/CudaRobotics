from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('num_particles', default_value='1000'),
        DeclareLaunchArgument('dt', default_value='0.1'),
        DeclareLaunchArgument('measurement_noise_q', default_value='0.01'),
        DeclareLaunchArgument('motion_noise_v', default_value='1.0'),
        DeclareLaunchArgument('motion_noise_yaw', default_value='0.5236'),
        DeclareLaunchArgument('max_range', default_value='20.0'),

        Node(
            package='cuda_robotics',
            executable='particle_filter_node',
            name='particle_filter_node',
            output='screen',
            parameters=[{
                'num_particles':      LaunchConfiguration('num_particles'),
                'dt':                 LaunchConfiguration('dt'),
                'measurement_noise_q': LaunchConfiguration('measurement_noise_q'),
                'motion_noise_v':     LaunchConfiguration('motion_noise_v'),
                'motion_noise_yaw':   LaunchConfiguration('motion_noise_yaw'),
                'max_range':          LaunchConfiguration('max_range'),
            }],
            remappings=[
                ('/odom',      '/odom'),
                ('/landmarks', '/landmarks'),
            ],
        ),
    ])
