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
    parser.add_argument("--timeout-sec", type=float, default=180.0)
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


def main() -> int:
    args = parse_args()
    if args.timeout_sec <= 0.0:
        raise SystemExit("--timeout-sec must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_dir}")
    commit = command_output(["git", "rev-parse", "HEAD"])
    dirty = git_dirty()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "mission_summary.json"
    log_path = output_dir / "launch.log"
    config_copy = output_dir / "controller.yaml"
    shutil.copy2(CONTROLLER_CONFIG, config_copy)

    command = [
        "ros2",
        "launch",
        "cuda_nav_bringup",
        "cudanav_closed_loop.launch.py",
        f"output_path:={summary_path}",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    process: subprocess.Popen[Any] | None = None
    timed_out = False
    launch_error = ""
    with log_path.open("w", encoding="utf-8") as log:
        try:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=os.name != "nt",
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if os.name == "nt"
                        else 0
                    ),
                )
            except OSError as exception:
                launch_error = str(exception)
                log.write(f"launch failed: {exception}\n")
            if process is not None:
                deadline = time.monotonic() + args.timeout_sec
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
    summary_gate = evaluate_summary(summary, args.profile)
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "timeout_sec": args.timeout_sec,
        "timed_out": timed_out,
        "launch_returncode": process.returncode if process else None,
        "launch_error": launch_error,
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
            "launch_log": log_path.name,
            "controller_config": config_copy.name,
            "rosbag": None,
            "video": None,
        },
        "summary_gate": summary_gate,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_gate = evaluate_manifest(manifest, output_dir, args.profile)
    passed = summary_gate["passed"] and manifest_gate["passed"]
    manifest["manifest_gate"] = manifest_gate
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
