#!/usr/bin/env python3
"""CPU-only checks for the public navigation bag inspector."""

import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_erl_navigation_bags.py")
SPEC = importlib.util.spec_from_file_location("prepare_erl_navigation_bags", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    fixture = """
rosbag2_bagfile_information:
  topics_with_message_count:
    - topic_metadata:
        name: /front/scan
        type: sensor_msgs/msg/LaserScan
    - topic_metadata:
        name: /wheel/odom
        type: nav_msgs/msg/Odometry
    - topic_metadata:
        name: /tf
        type: tf2_msgs/msg/TFMessage
    - topic_metadata:
        name: /robot/cmd_vel
        type: geometry_msgs/msg/Twist
"""
    with tempfile.TemporaryDirectory() as directory:
        metadata = Path(directory) / "bag" / "metadata.yaml"
        metadata.parent.mkdir()
        metadata.write_text(fixture)
        result = MODULE.inspect_bag(metadata)
    assert result["readiness"] == "shadow_ready"
    assert result["scan_topics"] == ["/front/scan"]
    assert result["odom_topics"] == ["/wheel/odom"]
    assert result["command_topics"] == ["/robot/cmd_vel"]
    print("ERL navigation bag inspector checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
