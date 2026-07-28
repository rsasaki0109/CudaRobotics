"""Forward a recorded Path to Nav2 without claiming closed-loop execution."""

from __future__ import annotations

from action_msgs.msg import GoalStatus
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node


class RecordedPathShadow(Node):
    def __init__(self) -> None:
        super().__init__("cudanav_recorded_path_shadow")
        self.declare_parameter("path_topic", "recorded_path")
        self.declare_parameter("action_name", "follow_path")
        self.declare_parameter("controller_id", "FollowPath")
        self.declare_parameter("goal_checker_id", "general_goal_checker")
        self.declare_parameter("progress_checker_id", "progress_checker")
        self.declare_parameter("minimum_path_poses", 2)
        self._minimum_path_poses = int(
            self.get_parameter("minimum_path_poses").value
        )
        if self._minimum_path_poses < 2:
            raise ValueError("minimum_path_poses must be at least two")
        self._client = ActionClient(
            self,
            FollowPath,
            str(self.get_parameter("action_name").value),
        )
        self._pending: Path | None = None
        self._goal_active = False
        self._goals_sent = 0
        self._goals_completed = 0
        self.create_subscription(
            Path,
            str(self.get_parameter("path_topic").value),
            self._path_callback,
            10,
        )
        self.create_timer(0.2, self._tick)

    def _path_callback(self, message: Path) -> None:
        if (
            len(message.poses) < self._minimum_path_poses
            or not message.header.frame_id
        ):
            self.get_logger().warning(
                "ignoring recorded path without frame or enough poses"
            )
            return
        # Keep only the newest recorded plan. It is intentionally not transformed
        # or synthesized: the bag must contain a path compatible with its TF data.
        self._pending = message

    def _tick(self) -> None:
        if (
            self._goal_active
            or self._pending is None
            or not self._client.server_is_ready()
        ):
            return
        path = self._pending
        self._pending = None
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = str(self.get_parameter("controller_id").value)
        goal.goal_checker_id = str(
            self.get_parameter("goal_checker_id").value
        )
        goal.progress_checker_id = str(
            self.get_parameter("progress_checker_id").value
        )
        self._goal_active = True
        self._goals_sent += 1
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_response)

    def _goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exception:
            self.get_logger().error(f"shadow goal request failed: {exception}")
            self._goal_active = False
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("shadow FollowPath goal was rejected")
            self._goal_active = False
            return
        self.get_logger().info(
            "recorded path accepted for shadow control; commands do not alter "
            "the recorded robot state"
        )
        result = goal_handle.get_result_async()
        result.add_done_callback(self._goal_result)

    def _goal_result(self, future) -> None:
        try:
            wrapped = future.result()
            status = wrapped.status if wrapped is not None else None
        except Exception as exception:
            self.get_logger().error(f"shadow goal result failed: {exception}")
            status = None
        self._goals_completed += 1
        self._goal_active = False
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("recorded-path shadow goal completed")
        else:
            self.get_logger().warning(
                f"recorded-path shadow goal ended with status {status}"
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RecordedPathShadow()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
