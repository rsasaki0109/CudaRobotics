"""Send a deterministic FollowPath goal and retain closed-loop evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry, Path as PathMessage
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, UInt32

from .simulation_geometry import interpolate_polyline, mission_waypoints


class FollowPathMission(Node):
    def __init__(self) -> None:
        super().__init__("cudanav_follow_path_mission")
        self.declare_parameter("action_name", "follow_path")
        self.declare_parameter("output_path", "/tmp/cudanav_closed_loop.json")
        self.declare_parameter("startup_delay_sec", 8.0)
        self.declare_parameter("mission_timeout_sec", 90.0)
        self.declare_parameter("controller_frequency", 20.0)
        self._client = ActionClient(
            self,
            FollowPath,
            str(self.get_parameter("action_name").value),
        )
        self._output_path = Path(str(self.get_parameter("output_path").value))
        self._startup_time = time.monotonic()
        self._send_time = 0.0
        self._sent = False
        self._finished = False
        self._truth: tuple[float, float] | None = None
        self._last_truth: tuple[float, float] | None = None
        self._truth_distance = 0.0
        self._odom: tuple[float, float] | None = None
        self._collision = False
        self._collision_count = 0
        self._last_command_time: float | None = None
        self._command_intervals: list[float] = []
        self._shutdown_timer = None
        self.create_subscription(PoseStamped, "ground_truth", self._truth_cb, 10)
        self.create_subscription(Odometry, "odom", self._odom_cb, 10)
        self.create_subscription(Bool, "collision", self._collision_cb, 1)
        self.create_subscription(
            UInt32, "collision_count", self._collision_count_cb, 1
        )
        self.create_subscription(
            TwistStamped, "cmd_vel", self._command_cb, 10
        )
        self.create_timer(0.2, self._tick)

    def _truth_cb(self, message: PoseStamped) -> None:
        current = (message.pose.position.x, message.pose.position.y)
        if self._last_truth is not None:
            step = math.dist(current, self._last_truth)
            if step < 0.1:
                self._truth_distance += step
        self._last_truth = current
        self._truth = current

    def _odom_cb(self, message: Odometry) -> None:
        self._odom = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
        )

    def _collision_cb(self, message: Bool) -> None:
        self._collision = self._collision or message.data

    def _collision_count_cb(self, message: UInt32) -> None:
        self._collision_count = max(self._collision_count, int(message.data))

    def _command_cb(self, _: TwistStamped) -> None:
        now = time.monotonic()
        if self._last_command_time is not None:
            self._command_intervals.append(now - self._last_command_time)
        self._last_command_time = now

    def _tick(self) -> None:
        if self._finished:
            return
        elapsed = time.monotonic() - self._startup_time
        timeout = float(self.get_parameter("mission_timeout_sec").value)
        delay = float(self.get_parameter("startup_delay_sec").value)
        if not self._sent and elapsed > delay + timeout:
            self._finish(False, "action server startup timeout", None)
            return
        if self._sent and time.monotonic() - self._send_time > timeout:
            self._finish(False, "mission timeout", None)
            return
        if self._sent or elapsed < delay or not self._client.server_is_ready():
            return
        goal = FollowPath.Goal()
        goal.path = self._make_path()
        goal.controller_id = "FollowPath"
        goal.goal_checker_id = "general_goal_checker"
        goal.progress_checker_id = "progress_checker"
        self._sent = True
        self._send_time = time.monotonic()
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)

    def _make_path(self) -> PathMessage:
        now = self.get_clock().now().to_msg()
        path = PathMessage()
        path.header.stamp = now
        path.header.frame_id = "odom"
        for x, y, yaw in interpolate_polyline(mission_waypoints(), 0.12):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            path.poses.append(pose)
        return path

    def _goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exception:
            self._finish(False, f"goal request failed: {exception}", None)
            return
        if goal_handle is None or not goal_handle.accepted:
            self._finish(False, "goal rejected", None)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result)

    def _result(self, future) -> None:
        try:
            wrapped = future.result()
        except Exception as exception:
            self._finish(False, f"action result failed: {exception}", None)
            return
        success = (
            wrapped is not None
            and wrapped.status == GoalStatus.STATUS_SUCCEEDED
        )
        status = wrapped.status if wrapped is not None else "missing"
        self._finish(success, f"action status {status}", wrapped)

    def _finish(self, success: bool, reason: str, _) -> None:
        if self._finished:
            return
        self._finished = True
        goal = mission_waypoints()[-1]
        goal_distance = (
            math.dist(self._truth, goal) if self._truth is not None else None
        )
        drift = (
            math.dist(self._odom, self._truth)
            if self._odom is not None and self._truth is not None
            else None
        )
        frequency = float(self.get_parameter("controller_frequency").value)
        deadline = 1.5 / frequency
        misses = sum(
            interval > deadline for interval in self._command_intervals
        )
        drift_percent = (
            100.0 * drift / self._truth_distance
            if drift is not None and self._truth_distance > 1.0e-6
            else None
        )
        deadline_miss_rate = (
            misses / len(self._command_intervals)
            if self._command_intervals
            else None
        )
        smoke_pass = (
            success
            and not self._collision
            and goal_distance is not None
            and goal_distance <= 0.30
            and drift_percent is not None
            and drift_percent < 5.0
            and deadline_miss_rate is not None
            and deadline_miss_rate < 0.05
        )
        summary = {
            "success": bool(success),
            "smoke_pass": smoke_pass,
            "reason": reason,
            "elapsed_sec": time.monotonic() - self._send_time
            if self._send_time
            else 0.0,
            "collision": self._collision,
            "collision_count": self._collision_count,
            "ground_truth_distance_m": self._truth_distance,
            "ground_truth_goal_distance_m": goal_distance,
            "odometry_position_error_m": drift,
            "odometry_drift_percent": drift_percent,
            "command_intervals": len(self._command_intervals),
            "command_deadline_misses": misses,
            "command_deadline_miss_rate": deadline_miss_rate,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(
            f"mission complete: {reason}; evidence={self._output_path}"
        )
        self._shutdown_timer = self.create_timer(0.2, self._shutdown)

    def _shutdown(self) -> None:
        if rclpy.ok():
            rclpy.shutdown()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FollowPathMission()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
