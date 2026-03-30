from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_speed', default_value='1.0'),
        DeclareLaunchArgument('min_speed', default_value='-0.5'),
        DeclareLaunchArgument('max_yawrate', default_value='40.0'),
        DeclareLaunchArgument('max_accel', default_value='0.2'),
        DeclareLaunchArgument('max_dyawrate', default_value='40.0'),
        DeclareLaunchArgument('v_reso', default_value='0.01'),
        DeclareLaunchArgument('yawrate_reso', default_value='0.1'),
        DeclareLaunchArgument('dt', default_value='0.1'),
        DeclareLaunchArgument('predict_time', default_value='3.0'),
        DeclareLaunchArgument('to_goal_cost_gain', default_value='1.0'),
        DeclareLaunchArgument('speed_cost_gain', default_value='1.0'),
        DeclareLaunchArgument('robot_radius', default_value='1.0'),
        DeclareLaunchArgument('goal_tolerance', default_value='0.5'),

        Node(
            package='cuda_robotics',
            executable='dwa_node',
            name='dwa_node',
            output='screen',
            parameters=[{
                'max_speed':         LaunchConfiguration('max_speed'),
                'min_speed':         LaunchConfiguration('min_speed'),
                'max_yawrate':       LaunchConfiguration('max_yawrate'),
                'max_accel':         LaunchConfiguration('max_accel'),
                'max_dyawrate':      LaunchConfiguration('max_dyawrate'),
                'v_reso':            LaunchConfiguration('v_reso'),
                'yawrate_reso':      LaunchConfiguration('yawrate_reso'),
                'dt':                LaunchConfiguration('dt'),
                'predict_time':      LaunchConfiguration('predict_time'),
                'to_goal_cost_gain': LaunchConfiguration('to_goal_cost_gain'),
                'speed_cost_gain':   LaunchConfiguration('speed_cost_gain'),
                'robot_radius':      LaunchConfiguration('robot_radius'),
                'goal_tolerance':    LaunchConfiguration('goal_tolerance'),
            }],
            remappings=[
                ('/odom',      '/odom'),
                ('/goal',      '/goal'),
                ('/obstacles', '/obstacles'),
            ],
        ),
    ])
