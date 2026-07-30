"""Send a deterministic FollowPath goal and retain closed-loop evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

from action_msgs.msg import GoalStatus
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
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
        self.declare_parameter("traversal_count", 1)
        self._client = ActionClient(
            self,
            FollowPath,
            str(self.get_parameter("action_name").value),
        )
        self._output_path = Path(str(self.get_parameter("output_path").value))
        self._trajectory_path = self._output_path.with_name("trajectory.csv")
        self._traversal_count = int(
            self.get_parameter("traversal_count").value
        )
        if self._traversal_count <= 0:
            raise ValueError("traversal_count must be positive")
        self._startup_time = time.monotonic()
        self._send_time = 0.0
        self._mission_started = False
        self._goal_in_flight = False
        self._traversals_completed = 0
        self._current_goal = mission_waypoints()[-1]
        self._finished = False
        self._truth: tuple[float, float] | None = None
        self._last_truth: tuple[float, float] | None = None
        self._truth_distance = 0.0
        self._odom: tuple[float, float] | None = None
        self._collision = False
        self._collision_count = 0
        self._last_command_time: float | None = None
        self._command_intervals: list[float] = []
        self._diagnostic_error_count = 0
        self._diagnostic_warn_count = 0
        self._diagnostic_status_samples = 0
        self._diagnostic_components: set[str] = set()
        self._failure_counters: dict[str, int] = {}
        self._trajectory: list[
            tuple[float, float, float, float | None, float | None]
        ] = []
        self.create_subscription(PoseStamped, "ground_truth", self._truth_cb, 10)
        self.create_subscription(Odometry, "odom", self._odom_cb, 10)
        self.create_subscription(Bool, "collision", self._collision_cb, 1)
        self.create_subscription(
            UInt32, "collision_count", self._collision_count_cb, 1
        )
        self.create_subscription(
            TwistStamped, "cmd_vel", self._command_cb, 10
        )
        for topic in (
            "odometry_diagnostics",
            "mapping_diagnostics",
            "esdf_diagnostics",
        ):
            self.create_subscription(
                DiagnosticArray, topic, self._diagnostic_cb, 10
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
        odom_x = self._odom[0] if self._odom is not None else None
        odom_y = self._odom[1] if self._odom is not None else None
        self._trajectory.append(
            (
                time.monotonic() - self._startup_time,
                current[0],
                current[1],
                odom_x,
                odom_y,
            )
        )

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

    def _diagnostic_cb(self, message: DiagnosticArray) -> None:
        for status in message.status:
            self._diagnostic_status_samples += 1
            self._diagnostic_components.add(status.name)
            if status.level >= DiagnosticStatus.ERROR:
                self._diagnostic_error_count += 1
            elif status.level == DiagnosticStatus.WARN:
                self._diagnostic_warn_count += 1
            for item in status.values:
                normalized = item.key.lower()
                if "failure" not in normalized and "dropped" not in normalized:
                    continue
                try:
                    value = int(item.value)
                except ValueError:
                    continue
                key = f"{status.name}:{item.key}"
                self._failure_counters[key] = max(
                    value, self._failure_counters.get(key, 0)
                )

    def _tick(self) -> None:
        if self._finished:
            return
        elapsed = time.monotonic() - self._startup_time
        timeout = float(self.get_parameter("mission_timeout_sec").value)
        delay = float(self.get_parameter("startup_delay_sec").value)
        if not self._mission_started and elapsed > delay + timeout:
            self._finish(False, "action server startup timeout", None)
            return
        if (
            self._mission_started
            and time.monotonic() - self._send_time > timeout
        ):
            self._finish(False, "mission timeout", None)
            return
        if (
            self._goal_in_flight
            or elapsed < delay
            or not self._client.server_is_ready()
        ):
            return
        self._send_next_goal()

    def _send_next_goal(self) -> None:
        goal = FollowPath.Goal()
        reverse = self._traversals_completed % 2 == 1
        self._current_goal = (
            mission_waypoints()[0] if reverse else mission_waypoints()[-1]
        )
        goal.path = self._make_path(reverse)
        goal.controller_id = "FollowPath"
        goal.goal_checker_id = "general_goal_checker"
        goal.progress_checker_id = "progress_checker"
        self._goal_in_flight = True
        if not self._mission_started:
            self._mission_started = True
            self._send_time = time.monotonic()
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)

    def _make_path(self, reverse: bool) -> PathMessage:
        now = self.get_clock().now().to_msg()
        path = PathMessage()
        path.header.stamp = now
        path.header.frame_id = "odom"
        waypoints = mission_waypoints()
        if reverse:
            waypoints = tuple(reversed(waypoints))
        for x, y, yaw in interpolate_polyline(waypoints, 0.12):
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
            if self._traversals_completed == 0:
                self._goal_in_flight = False
                self._mission_started = False
                self._send_time = 0.0
                self.get_logger().info(
                    "action server is not active yet; retrying first goal"
                )
                return
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
        if not success:
            self._finish(False, f"action status {status}", wrapped)
            return
        self._traversals_completed += 1
        self._goal_in_flight = False
        if self._traversals_completed >= self._traversal_count:
            self._finish(True, f"action status {status}", wrapped)

    def _finish(self, success: bool, reason: str, _) -> None:
        if self._finished:
            return
        self._finished = True
        goal = self._current_goal
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
            "schema_version": 1,
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
            "traversals_requested": self._traversal_count,
            "traversals_completed": self._traversals_completed,
            "trajectory_csv": self._trajectory_path.name,
            "diagnostic_error_count": self._diagnostic_error_count,
            "diagnostic_warn_count": self._diagnostic_warn_count,
            "diagnostic_status_samples": self._diagnostic_status_samples,
            "diagnostic_components": sorted(self._diagnostic_components),
            "failure_counters": dict(sorted(self._failure_counters.items())),
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_temporary = self._trajectory_path.with_suffix(".csv.tmp")
        with trajectory_temporary.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["elapsed_sec", "truth_x", "truth_y", "odom_x", "odom_y"]
            )
            writer.writerows(self._trajectory)
        trajectory_temporary.replace(self._trajectory_path)
        temporary_path = self._output_path.with_suffix(
            self._output_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(
                summary, indent=2, sort_keys=True, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._output_path)
        self.get_logger().info(
            f"mission complete: {reason}; evidence={self._output_path}"
        )

    @property
    def finished(self) -> bool:
        return self._finished


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FollowPathMission()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
