#!/usr/bin/env python3
"""Run a lightweight CUDA MPPI bag/real-data evaluation session.

This script intentionally does not encode a specific robot launch file. It
orchestrates a controller command, optional rosbag playback, optional mission
command, topic recording, and diagnostics rendering around one output
directory. Command strings may use:

  {out_dir}          evaluation output directory
  {diagnostics_csv}  desired cuda_mppi_controller diagnostics CSV path

Example:

  python3 scripts/run_cuda_mppi_bag_eval.py \
    --bag /data/site.bag \
    --controller-command 'ros2 launch my_nav stack.launch.py params_file:=nav.yaml' \
    --mission-command 'python3 scripts/send_waypoints.py' \
    --duration 120
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS = (
    "/tf",
    "/tf_static",
    "/odom",
    "/cmd_vel",
    "/plan",
    "/local_costmap/costmap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, help="Optional rosbag2 directory or file to play.")
    parser.add_argument("--output-dir", type=Path, default=Path("build/cuda_mppi_bag_eval"))
    parser.add_argument("--diagnostics-csv", type=Path)
    parser.add_argument("--controller-command", help="Shell command that launches Nav2/controller stack.")
    parser.add_argument("--mission-command", help="Optional shell command that sends goals/waypoints.")
    parser.add_argument("--bag-play-args", default="", help="Extra arguments appended to ros2 bag play.")
    parser.add_argument("--use-sim-time", action="store_true", help="Pass --clock to ros2 bag play.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds; 0 waits for bag playback.")
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--mission-delay", type=float, default=2.0)
    parser.add_argument("--ros-domain-id", type=int)
    parser.add_argument("--record-topics", nargs="*", default=list(DEFAULT_TOPICS))
    parser.add_argument("--no-record", action="store_true")
    parser.set_defaults(render_diagnostics=True)
    parser.add_argument("--render-diagnostics", action="store_true")
    parser.add_argument("--no-render-diagnostics", dest="render_diagnostics", action="store_false")
    return parser.parse_args()


def env_with_domain(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.ros_domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env


def format_command(command: str, out_dir: Path, diagnostics_csv: Path) -> str:
    return command.format(out_dir=str(out_dir), diagnostics_csv=str(diagnostics_csv))


def start_shell(command: str, env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log = log_path.open("w")
    return subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )


def stop_process(proc: subprocess.Popen | None, timeout: float = 8.0) -> int | None:
    if proc is None:
        return None
    if proc.poll() is not None:
        return proc.returncode
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return proc.wait(timeout=timeout)


def maybe_start_recorder(args: argparse.Namespace, env: dict[str, str]) -> subprocess.Popen | None:
    if args.no_record or not args.record_topics:
        return None
    record_dir = args.output_dir / "topics"
    cmd = "ros2 bag record -o {record_dir} {topics}".format(
        record_dir=record_dir,
        topics=" ".join(args.record_topics),
    )
    return start_shell(cmd, env, args.output_dir / "rosbag_record.log")


def maybe_start_bag(args: argparse.Namespace, env: dict[str, str]) -> subprocess.Popen | None:
    if args.bag is None:
        return None
    clock = " --clock" if args.use_sim_time else ""
    cmd = f"ros2 bag play {args.bag}{clock} {args.bag_play_args}".strip()
    return start_shell(cmd, env, args.output_dir / "rosbag_play.log")


def render_diagnostics(args: argparse.Namespace, diagnostics_csv: Path, env: dict[str, str]) -> int | None:
    if not args.render_diagnostics or not diagnostics_csv.exists():
        return None
    stem = args.output_dir / "diagnostics"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "render_cuda_mppi_diagnostics.py"),
        str(diagnostics_csv),
        "--output-stem",
        str(stem),
        "--title",
        "CUDA MPPI bag evaluation diagnostics",
    ]
    completed = subprocess.run(cmd, cwd=REPO, env=env, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_csv = args.diagnostics_csv or (args.output_dir / "diagnostics.csv")
    env = env_with_domain(args)

    processes: dict[str, subprocess.Popen | None] = {
        "controller": None,
        "recorder": None,
        "bag": None,
        "mission": None,
    }
    commands: dict[str, str] = {}

    try:
        if args.controller_command:
            commands["controller"] = format_command(
                args.controller_command, args.output_dir, diagnostics_csv)
            processes["controller"] = start_shell(
                commands["controller"], env, args.output_dir / "controller.log")
            time.sleep(max(0.0, args.settle_seconds))

        processes["recorder"] = maybe_start_recorder(args, env)
        if processes["recorder"] is not None:
            commands["recorder"] = "ros2 bag record"
            time.sleep(1.0)

        processes["bag"] = maybe_start_bag(args, env)
        if processes["bag"] is not None:
            commands["bag"] = f"ros2 bag play {args.bag}"

        if args.mission_command:
            time.sleep(max(0.0, args.mission_delay))
            commands["mission"] = format_command(
                args.mission_command, args.output_dir, diagnostics_csv)
            processes["mission"] = start_shell(
                commands["mission"], env, args.output_dir / "mission.log")

        if args.duration > 0.0:
            time.sleep(args.duration)
        elif processes["bag"] is not None:
            processes["bag"].wait()
        elif processes["mission"] is not None:
            processes["mission"].wait()
        else:
            raise RuntimeError("--duration is required when no bag or mission command is provided")
    finally:
        returncodes = {
            name: stop_process(proc)
            for name, proc in reversed(list(processes.items()))
        }
        render_rc = render_diagnostics(args, diagnostics_csv, env)
        manifest = {
            "output_dir": str(args.output_dir),
            "diagnostics_csv": str(diagnostics_csv),
            "ros_domain_id": env.get("ROS_DOMAIN_ID", ""),
            "commands": commands,
            "returncodes": returncodes,
            "diagnostics_render_returncode": render_rc,
            "recorded_topics": [] if args.no_record else args.record_topics,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {args.output_dir / 'manifest.json'}")

    expected_stop_codes = {
        None,
        0,
        -signal.SIGINT,
        128 + signal.SIGINT,
        -signal.SIGTERM,
        128 + signal.SIGTERM,
    }
    failed = [
        (name, rc)
        for name, rc in returncodes.items()
        if rc not in expected_stop_codes
        and name not in {"recorder"}
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
