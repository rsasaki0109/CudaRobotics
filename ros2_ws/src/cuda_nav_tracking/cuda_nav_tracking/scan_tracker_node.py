"""Track moving objects in LiDAR scans and publish TrackedObstacleArray.

Inputs:
- ``input_topic`` (sensor_msgs/PointCloud2), in any frame TF can relate to
  ``tracking_frame``;
- ``occupancy_topic`` (nav_msgs/OccupancyGrid) in ``tracking_frame``, for
  example from ``cuda_voxel_mapping``. Long-occupied cells are structure;
  points elsewhere may belong to moving objects.

Outputs:
- ``output_topic`` (cuda_robotics_msgs/TrackedObstacleArray) in
  ``tracking_frame``, stamped with the scan time: moving tracks with their
  velocity, plus still objects not yet released to the map (id 0xFFFFFFFF,
  zero velocity);
- ``static_points_topic`` (PointCloud2, input frame; empty disables): the scan
  without the tracked objects' points. Feed this to the mapper so moving
  objects do not leave trails in the occupancy grid.

The tracking itself is ``cudarobotics.navigation.tracking.ScanTrackingFrontend``
(NumPy); the cudarobotics Python package must be importable.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from cuda_robotics_msgs.msg import TrackedObstacle, TrackedObstacleArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from .frames import grid_from_message, quaternion_matrix, transform_points

STILL_OBJECT_ID = 0xFFFFFFFF


class ScanTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("scan_tracker")
        from cudarobotics.navigation.tracking import ScanTrackingFrontend

        self.input_topic = self.declare_parameter("input_topic", "points").value
        self.occupancy_topic = self.declare_parameter("occupancy_topic", "occupancy").value
        self.output_topic = self.declare_parameter("output_topic", "tracked_obstacles").value
        self.static_points_topic = self.declare_parameter("static_points_topic", "points_static").value
        self.tracking_frame = self.declare_parameter("tracking_frame", "odom").value
        self.transform_timeout = float(self.declare_parameter("transform_timeout_sec", 0.1).value)
        self.frontend = ScanTrackingFrontend()
        self._have_grid = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(TrackedObstacleArray, self.output_topic, 10)
        self.static_publisher = (
            self.create_publisher(PointCloud2, self.static_points_topic, qos_profile_sensor_data)
            if self.static_points_topic
            else None
        )
        self.create_subscription(OccupancyGrid, self.occupancy_topic, self.on_grid, 5)
        self.create_subscription(PointCloud2, self.input_topic, self.on_scan, qos_profile_sensor_data)
        self.get_logger().info(
            f"tracking {self.input_topic} against {self.occupancy_topic} in {self.tracking_frame}"
            f" -> {self.output_topic}"
        )

    def on_grid(self, msg: OccupancyGrid) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.tracking_frame:
            self.get_logger().warn(
                f"occupancy frame {msg.header.frame_id!r} is not {self.tracking_frame!r}; ignored",
                throttle_duration_sec=5.0,
            )
            return
        q = msg.info.origin.orientation
        if abs(2.0 * math.atan2(q.z, q.w)) > 1.0e-3 or abs(q.x) > 1.0e-6 or abs(q.y) > 1.0e-6:
            self.get_logger().warn("rotated occupancy grids are not supported; ignored",
                                   throttle_duration_sec=5.0)
            return
        grid = grid_from_message(msg.data, msg.info.width, msg.info.height)
        self.frontend.update_grid(
            grid, msg.info.resolution, (msg.info.origin.position.x, msg.info.origin.position.y)
        )
        self._have_grid = True

    def on_scan(self, msg: PointCloud2) -> None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.tracking_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self.transform_timeout),
            )
        except TransformException as error:
            self.get_logger().warn(f"no transform for the scan: {error}", throttle_duration_sec=5.0)
            return
        rotation = quaternion_matrix(
            tf.transform.rotation.x, tf.transform.rotation.y,
            tf.transform.rotation.z, tf.transform.rotation.w,
        )
        t = tf.transform.translation
        sensor = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        sensor = np.asarray(sensor, dtype=np.float64).reshape(-1, 3)
        world = transform_points(sensor, rotation, (t.x, t.y, t.z))
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1.0e-9
        try:
            obstacles, ids, static = self.frontend.process(world[:, :2], stamp, sensor_xy=(t.x, t.y))
        except ValueError:
            # Time went backwards (for example a looping bag): start over.
            self.get_logger().warn("scan time went backwards; resetting the tracker")
            self.frontend.tracker.reset()
            obstacles, ids, static = self.frontend.process(world[:, :2], stamp, sensor_xy=(t.x, t.y))
        if not self._have_grid:
            # Without a grid every point is a candidate; publish nothing yet.
            obstacles = obstacles[:0]

        out = TrackedObstacleArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.tracking_frame
        for (x, y, vx, vy, radius), identifier in zip(obstacles, ids):
            item = TrackedObstacle()
            item.id = int(identifier) if identifier >= 0 else STILL_OBJECT_ID
            item.position.x, item.position.y = float(x), float(y)
            item.velocity.x, item.velocity.y = float(vx), float(vy)
            item.radius = float(radius)
            out.obstacles.append(item)
        self.publisher.publish(out)

        if self.static_publisher is not None:
            self.static_publisher.publish(
                point_cloud2.create_cloud_xyz32(msg.header, sensor[static].astype(np.float32))
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanTrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
