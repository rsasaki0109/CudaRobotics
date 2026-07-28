#!/usr/bin/env python3
"""Run CudaNav launch and retain a self-describing smoke evidence directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any

from cudanav_evidence import evaluate_manifest, evaluate_summary


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_CONFIG = (
    ROOT
    / "ros2_ws"
    / "src"
    / "cuda_nav_bringup"
    / "config"
    / "controller.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "release"), default="smoke"
    )
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--mission-timeout-sec", type=float)
    parser.add_argument("--traversal-count", type=int)
    parser.add_argument(
        "--record-bag",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_identity() -> list[dict[str, str]]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            gpus.append(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": fields[3],
                }
            )
    return gpus


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15.0,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=15.0)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait(timeout=10.0)


def start_process(
    command: list[str], log_handle
) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )


def main() -> int:
    args = parse_args()
    release = args.profile == "release"
    timeout_sec = (
        args.timeout_sec
        if args.timeout_sec is not None
        else (1500.0 if release else 180.0)
    )
    mission_timeout_sec = (
        args.mission_timeout_sec
        if args.mission_timeout_sec is not None
        else (1200.0 if release else 90.0)
    )
    traversal_count = (
        args.traversal_count
        if args.traversal_count is not None
        else (30 if release else 1)
    )
    record_bag = args.record_bag if args.record_bag is not None else release
    render_video = (
        args.render_video if args.render_video is not None else release
    )
    if timeout_sec <= 0.0 or mission_timeout_sec <= 0.0:
        raise SystemExit("--timeout-sec must be positive")
    if timeout_sec <= mission_timeout_sec + 15.0:
        raise SystemExit("--timeout-sec must exceed mission timeout by 15 seconds")
    if traversal_count <= 0:
        raise SystemExit("--traversal-count must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_dir}")
    commit = command_output(["git", "rev-parse", "HEAD"])
    dirty = git_dirty()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "mission_summary.json"
    trajectory_path = output_dir / "trajectory.csv"
    log_path = output_dir / "launch.log"
    bag_log_path = output_dir / "rosbag.log"
    render_log_path = output_dir / "render.log"
    bag_path = output_dir / "rosbag"
    video_path = output_dir / "trajectory.gif"
    config_copy = output_dir / "controller.yaml"
    shutil.copy2(CONTROLLER_CONFIG, config_copy)

    command = [
        "ros2",
        "launch",
        "cuda_nav_bringup",
        "cudanav_closed_loop.launch.py",
        f"output_path:={summary_path}",
        f"traversal_count:={traversal_count}",
        f"mission_timeout_sec:={mission_timeout_sec}",
    ]
    bag_command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(bag_path),
        "/cuda_nav/points",
        "/cuda_nav/odom",
        "/cuda_nav/occupancy",
        "/cuda_nav/esdf",
        "/cuda_nav/cmd_vel",
        "/cuda_nav/ground_truth",
        "/cuda_nav/collision",
        "/cuda_nav/collision_count",
        "/cuda_nav/odometry_diagnostics",
        "/cuda_nav/mapping_diagnostics",
        "/cuda_nav/esdf_diagnostics",
        "/tf",
        "/tf_static",
    ]
    render_command = [
        os.sys.executable,
        str(ROOT / "scripts" / "render_cudanav_trajectory.py"),
        "--csv",
        str(trajectory_path),
        "--output",
        str(video_path),
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    process: subprocess.Popen[Any] | None = None
    timed_out = False
    launch_error = ""
    bag_error = ""
    bag_process: subprocess.Popen[Any] | None = None
    bag_log = None
    if record_bag:
        bag_log = bag_log_path.open("w", encoding="utf-8")
        try:
            bag_process = start_process(bag_command, bag_log)
            time.sleep(1.0)
            if bag_process.poll() is not None:
                bag_error = (
                    f"rosbag exited early with code {bag_process.returncode}"
                )
        except OSError as exception:
            bag_error = str(exception)
            bag_log.write(f"rosbag launch failed: {exception}\n")
    with log_path.open("w", encoding="utf-8") as log:
        try:
            try:
                process = start_process(command, log)
            except OSError as exception:
                launch_error = str(exception)
                log.write(f"launch failed: {exception}\n")
            if process is not None:
                deadline = time.monotonic() + timeout_sec
                while time.monotonic() < deadline:
                    if summary_path.is_file():
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.25)
                else:
                    timed_out = True
        finally:
            if process is not None:
                stop_process(process)
            if bag_process is not None:
                stop_process(bag_process)
            if bag_log is not None:
                bag_log.close()

    summary: dict[str, Any] = {}
    summary_error = ""
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            summary_error = str(exception)
    else:
        summary_error = "mission summary was not created"
    if launch_error:
        summary_error = f"{summary_error}; launch error: {launch_error}"
    render_error = ""
    if render_video and trajectory_path.is_file():
        with render_log_path.open("w", encoding="utf-8") as render_log:
            try:
                result = subprocess.run(
                    render_command,
                    cwd=ROOT,
                    stdout=render_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=120.0,
                )
                if result.returncode != 0:
                    render_error = (
                        f"renderer exited with code {result.returncode}"
                    )
            except (OSError, subprocess.SubprocessError) as exception:
                render_error = str(exception)
    summary_gate = evaluate_summary(summary, args.profile)
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "bag_command": bag_command if record_bag else None,
        "render_command": render_command if render_video else None,
        "timeout_sec": timeout_sec,
        "mission_timeout_sec": mission_timeout_sec,
        "traversal_count": traversal_count,
        "timed_out": timed_out,
        "launch_returncode": process.returncode if process else None,
        "launch_error": launch_error,
        "bag_error": bag_error,
        "render_error": render_error,
        "summary_error": summary_error,
        "git_commit": commit,
        "git_dirty": dirty,
        "config_sha256": sha256(config_copy),
        "gpu": gpu_identity(),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "ROS_DISTRO",
                "ROS_DOMAIN_ID",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "artifacts": {
            "summary": summary_path.name,
            "trajectory": trajectory_path.name,
            "launch_log": log_path.name,
            "controller_config": config_copy.name,
            "rosbag": bag_path.name
            if record_bag and (bag_path / "metadata.yaml").is_file()
            else None,
            "rosbag_log": bag_log_path.name if record_bag else None,
            "video": video_path.name if video_path.is_file() else None,
            "render_log": render_log_path.name
            if render_video and render_log_path.is_file()
            else None,
        },
        "summary_gate": summary_gate,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_gate = evaluate_manifest(manifest, output_dir, args.profile)
    artifact_binding = {
        "trajectory_matches_summary": (
            summary.get("trajectory_csv") == trajectory_path.name
        ),
        "traversal_count_matches": (
            summary.get("traversals_requested") == traversal_count
        ),
    }
    passed = (
        summary_gate["passed"]
        and manifest_gate["passed"]
        and all(artifact_binding.values())
    )
    manifest["manifest_gate"] = manifest_gate
    manifest["artifact_binding"] = artifact_binding
    manifest["passed"] = passed
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "manifest": str(manifest_path),
                "summary_gate": summary_gate,
                "manifest_gate": manifest_gate,
                "artifact_binding": artifact_binding,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
