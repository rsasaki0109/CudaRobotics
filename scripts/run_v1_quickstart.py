#!/usr/bin/env python3
"""Measure a fresh-clone, no-cache Docker CudaNav quickstart."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

from v1_quickstart_evidence import (
    REQUIRED_ARTIFACTS,
    describe_artifacts,
    evaluate_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "https://github.com/rsasaki0109/CudaRobotics.git"
ROS_PACKAGES = (
    "cuda_robotics_msgs",
    "cuda_robotics_common",
    "cuda_kiss_icp",
    "cuda_voxel_mapping",
    "cuda_esdf",
    "cuda_voxel_costmap_layer",
    "cuda_mppi_controller",
    "cuda_nav_bringup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument(
        "--profile",
        choices=("development", "release"),
        default="development",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--keep-image", action="store_true")
    return parser.parse_args()


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path | None,
    timeout: float,
) -> tuple[int, float]:
    began = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(timeout, 0.1),
            )
            return result.returncode, time.perf_counter() - began
        except subprocess.TimeoutExpired:
            log.write("\nCOMMAND TIMED OUT\n")
            return 124, time.perf_counter() - began
        except OSError as error:
            log.write(f"\nCOMMAND FAILED TO START: {error}\n")
            return 127, time.perf_counter() - began


def remaining(deadline: float) -> float:
    return max(deadline - time.perf_counter(), 0.1)


def docker_output(*args: str) -> str:
    return subprocess.check_output(
        ["docker", *args],
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    ).strip()


def gpu_identity() -> list[dict[str, str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    )
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            devices.append(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "driver_version": fields[2],
                }
            )
    return devices


def component_versions(checkout: Path) -> dict[str, Any]:
    pyproject = (checkout / "python" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    if match is None:
        raise ValueError("Python version is missing from cloned checkout")
    ros = {}
    for package in ROS_PACKAGES:
        package_xml = (
            checkout / "ros2_ws" / "src" / package / "package.xml"
        )
        version = ET.parse(package_xml).getroot().findtext("version")
        if version is None:
            raise ValueError(f"ROS version is missing: {package}")
        ros[package] = version
    return {
        "python_version": match.group(1),
        "ros_package_versions": ros,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0 or args.timeout_seconds > 900:
        raise SystemExit("timeout must be in (0, 900] seconds")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")

    started_at = datetime.now(timezone.utc).isoformat()
    began = time.perf_counter()
    deadline = began + args.timeout_seconds
    try:
        engine_version = docker_output(
            "version", "--format", "{{.Server.Version}}"
        )
        preexisting_image = (
            subprocess.run(
                ["docker", "image", "inspect", "cudarobotics"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        preexisting_container = (
            subprocess.run(
                ["docker", "container", "inspect", "cudarobotics-quickstart"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        devices = gpu_identity()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Docker/GPU environment check failed: {error}") from error
    if preexisting_image:
        raise SystemExit(
            "refusing pre-existing cudarobotics image; remove it for a fresh run"
        )
    if preexisting_container:
        raise SystemExit(
            "refusing pre-existing cudarobotics-quickstart container"
        )
    if not devices:
        raise SystemExit("no NVIDIA GPU reported by nvidia-smi")

    output.mkdir(parents=True, exist_ok=True)
    result_dir = output / "result"
    result_dir.mkdir()
    clone_log = output / "clone.log"
    build_log = output / "docker_build.log"
    run_log = output / "docker_run.log"
    returncodes = {"clone": 127, "build": 127, "run": 127}
    phases = {"clone": 0.0, "build": 0.0, "run": 0.0}
    image_id = ""
    commit = ""
    dirty = True
    matrix: dict[str, Any] = {}
    versions: dict[str, Any] = {}
    clone_command: list[str] = []
    build_command = [
        "docker",
        "build",
        "--pull",
        "--no-cache",
        "-f",
        "docker/Dockerfile",
        "-t",
        "cudarobotics",
        ".",
    ]
    run_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--name",
        "cudarobotics-quickstart",
        "-v",
        f"{result_dir}:/out",
        "cudarobotics",
        "cudanav",
    ]

    with tempfile.TemporaryDirectory(prefix="cudarobotics-v1-") as temporary:
        checkout = Path(temporary) / "CudaRobotics"
        clone_command = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            args.ref,
            args.repository,
            str(checkout),
        ]
        returncodes["clone"], phases["clone"] = run_logged(
            clone_command,
            clone_log,
            cwd=None,
            timeout=remaining(deadline),
        )
        if returncodes["clone"] == 0:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                text=True,
                encoding="utf-8",
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=checkout,
                    text=True,
                    encoding="utf-8",
                ).strip()
            )
            source_matrix = checkout / "docs" / "v1_support_matrix.json"
            shutil.copyfile(source_matrix, output / "support_matrix.json")
            matrix = json.loads(source_matrix.read_text(encoding="utf-8"))
            versions = component_versions(checkout)
            returncodes["build"], phases["build"] = run_logged(
                build_command,
                build_log,
                cwd=checkout,
                timeout=remaining(deadline),
            )
        else:
            build_log.write_text("build skipped: clone failed\n", encoding="utf-8")
        if returncodes["build"] == 0:
            try:
                image_id = docker_output(
                    "image", "inspect", "cudarobotics", "--format", "{{.Id}}"
                )
            except (OSError, subprocess.CalledProcessError):
                image_id = ""
            returncodes["run"], phases["run"] = run_logged(
                run_command,
                run_log,
                cwd=checkout,
                timeout=remaining(deadline),
            )
        else:
            run_log.write_text("run skipped: build failed\n", encoding="utf-8")

    duration = time.perf_counter() - began
    subprocess.run(
        ["docker", "rm", "-f", "cudarobotics-quickstart"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    summary: dict[str, Any] = {}
    summary_path = result_dir / "cudanav_closed_loop.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    raw_passed = (
        all(code == 0 for code in returncodes.values())
        and duration <= args.timeout_seconds
        and summary.get("schema_version") == 1
        and summary.get("smoke_pass") is True
        and summary.get("success") is True
    )
    matrix_main = matrix.get("main_demo", {})
    artifact_paths = set(REQUIRED_ARTIFACTS)
    manifest = {
        "schema_version": 1,
        "evidence_mode": "v1_quickstart",
        "profile": args.profile,
        "status": "passed" if raw_passed else "failed",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 3),
        "phase_seconds": {
            name: round(value, 3) for name, value in phases.items()
        },
        "time_budget_seconds": args.timeout_seconds,
        "target_version": matrix.get("target_version"),
        "source_ref": args.ref,
        "repository": args.repository,
        "git_commit": commit,
        "git_dirty": dirty,
        "component_versions": versions,
        "preexisting_image": preexisting_image,
        "preexisting_container": preexisting_container,
        "commands": {
            "clone": clone_command,
            "build": build_command,
            "run": run_command,
        },
        "build_command_contract": matrix_main.get("build_command"),
        "run_command_contract": matrix_main.get("run_command"),
        "returncodes": returncodes,
        "docker": {
            "engine_version": engine_version,
            "image_id": image_id,
        },
        "gpu": devices,
        "platform": platform.platform(),
        "result": matrix_main.get("result"),
        "support_matrix_sha256": (
            sha256_file(output / "support_matrix.json")
            if (output / "support_matrix.json").is_file()
            else None
        ),
        "artifacts": describe_artifacts(output, artifact_paths),
    }
    gate = evaluate_manifest(
        manifest,
        output,
        expected_profile=args.profile,
        expected_commit=commit if commit else None,
    )
    manifest["gate"] = gate
    manifest["status"] = "passed" if gate["passed"] else "failed"
    write_json(output / "manifest.json", manifest)

    if not args.keep_image and image_id:
        subprocess.run(
            ["docker", "image", "rm", "cudarobotics"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(output / "manifest.json")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
