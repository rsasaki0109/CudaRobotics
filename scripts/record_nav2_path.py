#!/usr/bin/env python3
"""Standalone map->base_link recorder (10 Hz) for the nav2 loopback demo.

Runs in its own process so its TF subscriptions are never starved by other
spinning nodes. Appends to the CSV until killed.

Usage: record_nav2_path.py <out_csv>
"""
import csv
import math
import sys
import time

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def main():
    out_csv = sys.argv[1]
    rclpy.init()
    node = Node(
        "demo_path_recorder",
        parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
    buf = Buffer()
    TransformListener(buf, node)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "yaw"])
        last_sample = 0.0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now - last_sample < 0.1:
                continue
            last_sample = now
            try:
                t = buf.lookup_transform("map", "base_link", rclpy.time.Time())
            except Exception:
                continue
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            writer.writerow(
                [f"{now:.3f}", t.transform.translation.x,
                 t.transform.translation.y, yaw])
            f.flush()


if __name__ == "__main__":
    main()
