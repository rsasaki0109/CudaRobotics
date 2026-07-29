#!/usr/bin/env python3
"""Create and validate a portable ROS 2 Jazzy CudaNav CI attestation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any


REQUIRED_PACKAGES = {
    "cuda_robotics_msgs",
    "cuda_robotics_common",
    "cuda_kiss_icp",
    "cuda_voxel_mapping",
    "cuda_esdf",
    "cuda_voxel_costmap_layer",
    "cuda_mppi_controller",
    "cuda_nav_bringup",
}
REQUIRED_CHECKS = {
    "clean_checkout",
    "colcon_build",
    "python_evidence_contracts",
    "costmap_plugin_load",
    "controller_plugin_load",
    "controller_parameter_validation",
    "colcon_tests",
}


def evaluate(
    payload: dict[str, Any], *, expected_commit: str | None = None
) -> dict[str, Any]:
    checks = payload.get("checks")
    packages = payload.get("packages")
    commit = payload.get("git_commit")
    run = payload.get("github", {})
    platform = payload.get("platform", {})
    ros = payload.get("ros", {})
    cuda = payload.get("cuda", {})
    results = {
        "schema_version": payload.get("schema_version") == 1,
        "evidence_mode": payload.get("evidence_mode") == "ros_jazzy_ci",
        "status": payload.get("status") == "passed",
        "git_commit": isinstance(commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", commit))
        and (expected_commit is None or commit == expected_commit),
        "clean_checkout": payload.get("git_dirty") is False,
        "github_repository": isinstance(run.get("repository"), str)
        and "/" in run["repository"],
        "github_run_id": isinstance(run.get("run_id"), int)
        and run["run_id"] > 0,
        "github_run_attempt": isinstance(run.get("run_attempt"), int)
        and run["run_attempt"] > 0,
        "github_run_url": run.get("run_url")
        == (
            f"https://github.com/{run.get('repository')}/actions/runs/"
            f"{run.get('run_id')}"
        ),
        "github_workflow": run.get("workflow") == "ROS2 CUDA MPPI",
        "github_event": run.get("event") in {"push", "workflow_dispatch"},
        "github_ref": isinstance(run.get("ref"), str)
        and run["ref"].startswith("refs/"),
        "runner": platform.get("os") == "Linux"
        and platform.get("image") == "ubuntu-24.04"
        and platform.get("arch") in {"X64", "ARM64"},
        "ros_distro": ros.get("distro") == "jazzy",
        "cuda_compiler": isinstance(cuda.get("compiler"), str)
        and "release" in cuda["compiler"].lower()
        and isinstance(cuda.get("toolkit"), str)
        and bool(re.fullmatch(r"12\.[0-9]+", cuda["toolkit"])),
        "packages": isinstance(packages, list)
        and len(packages) == len(set(packages))
        and REQUIRED_PACKAGES <= set(packages),
        "checks": isinstance(checks, dict)
        and REQUIRED_CHECKS <= set(checks)
        and all(checks.get(name) == "passed" for name in REQUIRED_CHECKS),
    }
    return {"passed": all(results.values()), "checks": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", "")
    )
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument(
        "--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    )
    parser.add_argument(
        "--workflow", default=os.environ.get("GITHUB_WORKFLOW", "")
    )
    parser.add_argument(
        "--event", default=os.environ.get("GITHUB_EVENT_NAME", "")
    )
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--runner-os", default=os.environ.get("RUNNER_OS", ""))
    parser.add_argument(
        "--runner-arch", default=os.environ.get("RUNNER_ARCH", "")
    )
    parser.add_argument("--runner-image", required=True)
    parser.add_argument("--ros-distro", required=True)
    parser.add_argument("--cuda-toolkit", required=True)
    parser.add_argument("--cuda-compiler", required=True)
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--check", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_id = int(args.run_id)
        run_attempt = int(args.run_attempt)
    except ValueError as error:
        raise SystemExit("GitHub run identifiers must be integers") from error
    repository = args.repository
    payload = {
        "schema_version": 1,
        "evidence_mode": "ros_jazzy_ci",
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "git_dirty": False,
        "github": {
            "repository": repository,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
            "workflow": args.workflow,
            "event": args.event,
            "ref": args.ref,
        },
        "platform": {
            "os": args.runner_os,
            "arch": args.runner_arch,
            "image": args.runner_image,
        },
        "ros": {"distro": args.ros_distro},
        "cuda": {
            "toolkit": args.cuda_toolkit,
            "compiler": args.cuda_compiler,
        },
        "packages": sorted(set(args.package)),
        "checks": {name: "passed" for name in sorted(set(args.check))},
    }
    validation = evaluate(payload)
    if not validation["passed"]:
        failed = sorted(
            name for name, passed in validation["checks"].items() if not passed
        )
        raise SystemExit("invalid ROS CI evidence: " + ", ".join(failed))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
