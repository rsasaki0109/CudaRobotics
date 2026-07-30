"""Command-driven deterministic point-cloud simulator for CudaNav."""

from __future__ import annotations

import math
import struct
from threading import Lock

from geometry_msgs.msg import PoseStamped, TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, UInt32

from .simulation_geometry import collides, default_segments, raycast


class CudaNavLoopbackSimulator(Node):
    def __init__(self) -> None:
        super().__init__("cudanav_loopback_simulator")
        self.declare_parameter("point_topic", "points")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("ground_truth_topic", "ground_truth")
        self.declare_parameter("collision_topic", "collision")
        self.declare_parameter("collision_count_topic", "collision_count")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("truth_frame", "world_truth")
        self.declare_parameter("integration_rate_hz", 100.0)
        self.declare_parameter("scan_rate_hz", 10.0)
        self.declare_parameter("ray_count", 240)
        self.declare_parameter("max_range", 12.0)
        self.declare_parameter("robot_radius", 0.24)
        self.declare_parameter("command_timeout_sec", 0.25)

        integration_rate = float(self.get_parameter("integration_rate_hz").value)
        scan_rate = float(self.get_parameter("scan_rate_hz").value)
        self._ray_count = int(self.get_parameter("ray_count").value)
        self._max_range = float(self.get_parameter("max_range").value)
        self._robot_radius = float(self.get_parameter("robot_radius").value)
        self._command_timeout = float(
            self.get_parameter("command_timeout_sec").value
        )
        if (
            integration_rate <= 0.0
            or scan_rate <= 0.0
            or self._ray_count < 32
            or self._max_range <= 0.0
            or self._robot_radius <= 0.0
            or self._command_timeout <= 0.0
        ):
            raise ValueError("simulator rates, ranges, and capacities must be positive")

        self._base_frame = str(self.get_parameter("base_frame").value)
        self._truth_frame = str(self.get_parameter("truth_frame").value)
        self._segments = default_segments()
        self._lock = Lock()
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._linear = 0.0
        self._angular = 0.0
        self._last_command_ns = 0
        self._last_integrate_ns = self.get_clock().now().nanoseconds
        self._collisions = 0
        self._collision_latched = False

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._point_publisher = self.create_publisher(
            PointCloud2, str(self.get_parameter("point_topic").value), sensor_qos
        )
        self._truth_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("ground_truth_topic").value),
            10,
        )
        self._collision_publisher = self.create_publisher(
            Bool, str(self.get_parameter("collision_topic").value), map_qos
        )
        self._collision_count_publisher = self.create_publisher(
            UInt32, str(self.get_parameter("collision_count_topic").value), map_qos
        )
        self._command_subscription = self.create_subscription(
            TwistStamped,
            str(self.get_parameter("cmd_vel_topic").value),
            self._command_callback,
            10,
        )
        self.create_timer(1.0 / integration_rate, self._integrate)
        self.create_timer(1.0 / scan_rate, self._publish_scan)
        self.create_timer(0.02, self._publish_truth)
        self._publish_collision_state()

    def _command_callback(self, message: TwistStamped) -> None:
        linear = float(message.twist.linear.x)
        angular = float(message.twist.angular.z)
        if not math.isfinite(linear) or not math.isfinite(angular):
            self.get_logger().error("rejected non-finite velocity command")
            return
        with self._lock:
            self._linear = max(-0.8, min(0.8, linear))
            self._angular = max(-2.5, min(2.5, angular))
            self._last_command_ns = self.get_clock().now().nanoseconds

    def _integrate(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            dt = min(0.05, max(0.0, (now_ns - self._last_integrate_ns) * 1.0e-9))
            self._last_integrate_ns = now_ns
            if (
                self._last_command_ns == 0
                or (now_ns - self._last_command_ns) * 1.0e-9
                > self._command_timeout
            ):
                linear = angular = 0.0
            else:
                linear = self._linear
                angular = self._angular
            middle_yaw = self._yaw + 0.5 * angular * dt
            candidate_x = self._x + linear * math.cos(middle_yaw) * dt
            candidate_y = self._y + linear * math.sin(middle_yaw) * dt
            candidate_yaw = math.atan2(
                math.sin(self._yaw + angular * dt),
                math.cos(self._yaw + angular * dt),
            )
            if collides(
                candidate_x,
                candidate_y,
                self._robot_radius,
                self._segments,
            ):
                self._linear = 0.0
                self._angular = 0.0
                self._collisions += 1
                self._collision_latched = True
                publish_collision = True
            else:
                self._x = candidate_x
                self._y = candidate_y
                self._yaw = candidate_yaw
                publish_collision = False
        if publish_collision:
            self._publish_collision_state()

    def _publish_collision_state(self) -> None:
        with self._lock:
            collision = self._collision_latched
            count = self._collisions
        self._collision_publisher.publish(Bool(data=collision))
        self._collision_count_publisher.publish(UInt32(data=count))

    def _publish_truth(self) -> None:
        with self._lock:
            x, y, yaw = self._x, self._y, self._yaw
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._truth_frame
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        self._truth_publisher.publish(message)

    def _publish_scan(self) -> None:
        with self._lock:
            x, y, yaw = self._x, self._y, self._yaw
        points: list[tuple[float, float, float]] = []
        z_levels = (-0.45, 0.0, 0.45)
        for ray_index in range(self._ray_count):
            local_angle = (
                -math.pi + 2.0 * math.pi * ray_index / self._ray_count
            )
            distance = raycast(
                x,
                y,
                yaw + local_angle,
                self._segments,
                self._max_range,
            )
            if distance >= self._max_range:
                continue
            local_x = distance * math.cos(local_angle)
            local_y = distance * math.sin(local_angle)
            points.extend((local_x, local_y, z) for z in z_levels)
        message = PointCloud2()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._base_frame
        message.height = 1
        message.width = len(points)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = message.point_step * message.width
        message.data = b"".join(struct.pack("<fff", *point) for point in points)
        message.is_dense = True
        self._point_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CudaNavLoopbackSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
