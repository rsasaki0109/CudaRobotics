"""In-process end-to-end check of scan_tracker_node with synthetic inputs.

Needs rclpy and the cudarobotics Python package; skipped without them.
"""

import math
import time

import numpy as np
import pytest


def test_tracks_a_walker_beside_a_wall():
    pytest.importorskip("cudarobotics.navigation.tracking")
    rclpy = pytest.importorskip("rclpy")
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

    from cuda_nav_tracking.scan_tracker_node import ScanTrackerNode
    from cuda_robotics_msgs.msg import TrackedObstacleArray

    rclpy.init()
    try:
        driver = Node("scan_tracker_test_driver")
        tracker = ScanTrackerNode()
        executor = SingleThreadedExecutor()
        executor.add_node(driver)
        executor.add_node(tracker)

        transform = TransformStamped()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "lidar"
        transform.transform.translation.x = 0.5  # sensor 0.5 m ahead of odom
        transform.transform.rotation.w = 1.0
        broadcaster = StaticTransformBroadcaster(driver)
        broadcaster.sendTransform(transform)

        grid_pub = driver.create_publisher(OccupancyGrid, "occupancy", 5)
        cloud_pub = driver.create_publisher(PointCloud2, "points", qos_profile_sensor_data)
        tracks, statics = [], []
        driver.create_subscription(TrackedObstacleArray, "tracked_obstacles", tracks.append, 10)
        driver.create_subscription(PointCloud2, "points_static", statics.append, qos_profile_sensor_data)

        grid = OccupancyGrid()
        grid.header.frame_id = "odom"
        grid.info.resolution = 0.05
        grid.info.width = grid.info.height = 80
        grid.info.origin.orientation.w = 1.0
        cells = np.zeros((80, 80), np.int8)
        cells[60, :] = 100  # wall at y = 3.025
        grid.data = cells.reshape(-1).tolist()
        wall = np.stack([np.linspace(0.2, 3.8, 40), np.full(40, 3.025)], axis=1)

        start = driver.get_clock().now()
        for frame in range(40):
            stamp = (start + rclpy.duration.Duration(seconds=0.1 * frame)).to_msg()
            grid.header.stamp = stamp
            grid_pub.publish(grid)
            cx, cy = 1.0 + 0.05 * frame, 1.0  # walking +x at 0.5 m/s
            facing = math.atan2(-cy, 0.5 - cx)
            angles = facing + np.linspace(-1.2, 1.2, 9)
            walker = np.stack([cx + 0.25 * np.cos(angles), cy + 0.25 * np.sin(angles)], axis=1)
            world = np.concatenate([wall, walker])
            points = np.zeros((world.shape[0], 3), np.float32)
            points[:, 0] = world[:, 0] - 0.5
            points[:, 1] = world[:, 1]
            cloud_pub.publish(point_cloud2.create_cloud_xyz32(Header(stamp=stamp, frame_id="lidar"), points))
            deadline = time.time() + 0.1
            while time.time() < deadline:
                executor.spin_once(timeout_sec=0.01)
        for _ in range(50):
            executor.spin_once(timeout_sec=0.02)

        assert tracks, "no TrackedObstacleArray received"
        last = tracks[-1]
        assert last.header.frame_id == "odom" and len(last.obstacles) == 1
        walker_track = last.obstacles[0]
        assert walker_track.position.x == pytest.approx(1.0 + 0.05 * 39, abs=0.15)
        assert walker_track.position.y == pytest.approx(1.0, abs=0.15)
        assert walker_track.velocity.x == pytest.approx(0.5, abs=0.15)
        assert abs(walker_track.velocity.y) < 0.15
        static = point_cloud2.read_points_numpy(statics[-1], field_names=("x", "y", "z"))
        assert statics[-1].header.frame_id == "lidar"
        assert static.shape[0] == wall.shape[0]  # the walker's points are removed
        tracker.destroy_node()
        driver.destroy_node()
    finally:
        rclpy.shutdown()
