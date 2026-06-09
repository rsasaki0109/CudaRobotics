#!/usr/bin/env python3
"""Drive the nav2 loopback simulation with cuda_mppi_controller and record
the run for rendering.

Prereq (separate terminal, isolated domain):
  ROS_DOMAIN_ID=42 PYTHONNOUSERSITE=1 ros2 launch nav2_bringup \
    tb3_loopback_simulation.launch.py use_rviz:=False \
    params_file:=.../cuda_mppi_controller/config/nav2_loopback_demo.yaml

Run:
  ROS_DOMAIN_ID=42 PYTHONNOUSERSITE=1 python3 scripts/run_nav2_loopback_demo.py /tmp/nav2_demo

Writes <out>/robot_path.csv, <out>/plan_<i>.csv, exits 0 on success.
"""
import csv
import math
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

START = (-2.0, -0.5, 0.0)
WAYPOINTS = [(1.8, 1.4, 0.0), (-0.3, -1.8, math.pi / 2)]


def make_pose(navigator, x, y, yaw):
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = navigator.get_clock().now().to_msg()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.orientation.z = math.sin(yaw / 2.0)
    p.pose.orientation.w = math.cos(yaw / 2.0)
    return p


class Recorder(Node):
    def __init__(self):
        # the whole loopback stack runs on sim time; mismatched clocks make
        # tf2 resolve the chain at its oldest common stamp (frozen pose)
        super().__init__(
            "demo_recorder",
            parameter_overrides=[
                rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.trail = []
        self.plans = []
        self.create_subscription(PathMsg, "/plan", self.on_plan, 10)
        self.create_timer(0.1, self.sample)

    def on_plan(self, msg):
        self.plans.append(
            [(p.pose.position.x, p.pose.position.y) for p in msg.poses])

    def sample(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            self.trail.append(
                (time.monotonic(), t.transform.translation.x,
                 t.transform.translation.y, yaw))
        except Exception:
            pass


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/nav2_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    navigator = BasicNavigator()
    # no AMCL in the loopback launch -> only wait for nav2 servers
    navigator.waitUntilNav2Active(localizer="smoother_server")

    recorder = Recorder()

    # republish the initial pose until the loopback sim teleports the robot
    # there (a single publish can be lost to discovery timing, and without
    # AMCL BasicNavigator has no feedback loop for this)
    placed = False
    for _ in range(30):
        navigator.setInitialPose(make_pose(navigator, *START))
        for _ in range(10):
            rclpy.spin_once(recorder, timeout_sec=0.1)
        try:
            t = recorder.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())
            dx = t.transform.translation.x - START[0]
            dy = t.transform.translation.y - START[1]
            if math.hypot(dx, dy) < 0.3:
                placed = True
                break
        except Exception:
            continue
    if not placed:
        print("FAIL: initial pose was never applied")
        sys.exit(1)

    # record map->base_link in a separate process so its TF subscription is
    # not starved by BasicNavigator spinning the shared global executor
    path_csv = out_dir / "robot_path.csv"
    rec_proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "record_nav2_path.py"),
         str(path_csv)])
    time.sleep(1.0)

    ok = True
    for i, wp in enumerate(WAYPOINTS):
        navigator.goToPose(make_pose(navigator, *wp))
        while not navigator.isTaskComplete():
            rclpy.spin_once(recorder, timeout_sec=0.05)
        result = navigator.getResult()
        print(f"waypoint {i}: {result}")
        if result != TaskResult.SUCCEEDED:
            ok = False
            break

    rec_proc.terminate()
    rec_proc.wait(timeout=10)

    with open(path_csv, newline="") as f:
        trail = list(csv.DictReader(f))
    xs = {round(float(r["x"]), 3) for r in trail}
    if ok and len(xs) < 10:
        print("FAIL: recorded trail did not move (TF recording broken)")
        ok = False
    print(f"recorded {len(trail)} poses")

    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
