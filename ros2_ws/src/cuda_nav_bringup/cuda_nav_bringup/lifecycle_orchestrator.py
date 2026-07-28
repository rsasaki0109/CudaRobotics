"""Sequential standard lifecycle transitions without Nav2 bond assumptions."""

from __future__ import annotations

import sys

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
import rclpy
from rclpy.node import Node


class LifecycleOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__("cudanav_lifecycle_orchestrator")
        self.declare_parameter(
            "node_names",
            [
                "cuda_kiss_icp_odometry",
                "cuda_voxel_mapper",
                "cuda_esdf_node",
                "controller_server",
            ],
        )
        self.declare_parameter("service_timeout_sec", 60.0)

    def run(self) -> bool:
        timeout = float(self.get_parameter("service_timeout_sec").value)
        for name in self.get_parameter("node_names").value:
            target = str(name).strip("/")
            if not self._transition(
                target,
                Transition.TRANSITION_CONFIGURE,
                State.PRIMARY_STATE_INACTIVE,
                timeout,
            ):
                return False
            if not self._transition(
                target,
                Transition.TRANSITION_ACTIVATE,
                State.PRIMARY_STATE_ACTIVE,
                timeout,
            ):
                return False
        self.get_logger().info("CudaNav lifecycle sequence is active")
        return True

    def _transition(
        self,
        node_name: str,
        transition_id: int,
        expected_state: int,
        timeout: float,
    ) -> bool:
        change_client = self.create_client(
            ChangeState, f"{node_name}/change_state"
        )
        state_client = self.create_client(GetState, f"{node_name}/get_state")
        if not change_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"lifecycle service unavailable for {node_name}"
            )
            return False
        request = ChangeState.Request()
        request.transition.id = transition_id
        future = change_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if (
            not future.done()
            or future.result() is None
            or not future.result().success
        ):
            self.get_logger().error(
                f"transition {transition_id} failed for {node_name}"
            )
            return False
        if not state_client.wait_for_service(timeout_sec=timeout):
            return False
        state_future = state_client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(self, state_future, timeout_sec=timeout)
        if (
            not state_future.done()
            or state_future.result() is None
            or state_future.result().current_state.id != expected_state
        ):
            self.get_logger().error(
                f"{node_name} did not reach expected state {expected_state}"
            )
            return False
        self.get_logger().info(
            f"{node_name}: transition {transition_id} succeeded"
        )
        return True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LifecycleOrchestrator()
    success = False
    try:
        success = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not success:
        sys.exit(1)
