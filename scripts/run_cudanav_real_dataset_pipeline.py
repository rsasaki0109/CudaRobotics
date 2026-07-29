#!/usr/bin/env python3
"""Materialize and optionally run the selected CudaNav real-data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from cudanav_real_dataset import DEFAULT_SPEC, read_json
from cudanav_rosbag_evidence import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER = (
    ROOT
    / "ros2_ws"
    / "src"
    / "cuda_nav_bringup"
    / "config"
    / "controller.yaml"
)
DEFAULT_CONTROLLER_COMMAND = (
    "ros2 launch cuda_nav_bringup cudanav_recorded_shadow.launch.py "
    "params_file:={controller_config} "
    "diagnostics_csv:={diagnostics_csv} "
    "points_topic:=/pandar_points "
    "path_topic:=/cuda_nav/derived_plan "
    "sensor_frame:="
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("build/datasets/cudanav_istanbul"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("build/cudanav_real_dataset"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-backend",
        choices=("curl", "gdown"),
        default="curl",
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--generate-metadata", action="store_true")
    parser.add_argument(
        "--sidecar-storage",
        choices=("mcap", "sqlite3"),
        default="mcap",
    )
    parser.add_argument("--run-autonomy", action="store_true")
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument(
        "--autonomy-output-dir",
        type=Path,
        default=Path("build/cudanav_autonomy_real"),
    )
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument(
        "--controller-command", default=DEFAULT_CONTROLLER_COMMAND
    )
    parser.add_argument("--rosbag-duration-sec", type=float, default=0.0)
    parser.add_argument("--multi-gpu-run", type=Path, action="append", default=[])
    parser.add_argument("--multi-gpu-devices")
    parser.add_argument("--multi-gpu-repetitions", type=int, default=1)
    parser.add_argument("--closed-loop-timeout-sec", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.profile == "release" and not (
        args.multi_gpu_run or args.multi_gpu_devices
    ):
        errors.append(
            "release profile requires --multi-gpu-run or --multi-gpu-devices"
        )
    if args.multi_gpu_run and args.multi_gpu_devices:
        errors.append("--multi-gpu-run and --multi-gpu-devices are exclusive")
    if args.rosbag_duration_sec < 0.0:
        errors.append("--rosbag-duration-sec must be non-negative")
    if args.multi_gpu_repetitions <= 0:
        errors.append("--multi-gpu-repetitions must be positive")
    if (
        args.closed_loop_timeout_sec is not None
        and args.closed_loop_timeout_sec <= 0.0
    ):
        errors.append("--closed-loop-timeout-sec must be positive")
    if errors:
        raise SystemExit("; ".join(errors))


def command_plan(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = args.spec.resolve()
    spec = read_json(spec_path)
    dataset = args.dataset_dir.resolve()
    work = args.work_dir.resolve()
    database = dataset / spec["acquisition"]["expected_database"]
    inspection = dataset / "inspection.json"
    sidecar = work / "path_sidecar"
    generator = work / "path_generator.json"
    materialization = work / "materialization.json"

    prepare = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_cudanav_istanbul_dataset.py"),
        "--spec",
        str(spec_path),
        "--output-dir",
        str(dataset),
        "--report",
        str(inspection),
    ]
    if args.download:
        prepare.extend(
            ["--download", "--download-backend", args.download_backend]
        )
    if args.probe:
        prepare.append("--probe")
    if args.reindex:
        prepare.append("--reindex")
    if args.generate_metadata:
        prepare.append("--generate-metadata")
    derive = [
        sys.executable,
        str(ROOT / "scripts" / "derive_cudanav_path_sidecar.py"),
        "--spec",
        str(spec_path),
        "--source-bag",
        str(dataset),
        "--database",
        str(database),
        "--output-bag",
        str(sidecar),
        "--report",
        str(generator),
        "--acquisition-report",
        str(inspection),
        "--materialization",
        str(materialization),
        "--storage",
        args.sidecar_storage,
    ]
    validate = [
        sys.executable,
        str(ROOT / "scripts" / "validate_cudanav_real_dataset.py"),
        "--spec",
        str(spec_path),
        "--materialization",
        str(materialization),
    ]
    autonomy = [
        sys.executable,
        str(ROOT / "scripts" / "run_autonomy_suite.py"),
        "--output-dir",
        str(args.autonomy_output_dir.resolve()),
        "--profile",
        args.profile,
        "--bag",
        str(dataset),
        "--derived-path-bag",
        str(sidecar),
        "--dataset-materialization",
        str(materialization),
        "--evaluation-db",
        str(database),
        "--controller-config",
        str(args.controller_config.resolve()),
        "--controller-command",
        args.controller_command,
        "--rosbag-duration-sec",
        str(args.rosbag_duration_sec),
        "--multi-gpu-repetitions",
        str(args.multi_gpu_repetitions),
    ]
    for run in args.multi_gpu_run:
        autonomy.extend(["--multi-gpu-run", str(run.resolve())])
    if args.multi_gpu_devices:
        autonomy.extend(["--multi-gpu-devices", args.multi_gpu_devices])
    if args.closed_loop_timeout_sec is not None:
        autonomy.extend(
            ["--closed-loop-timeout-sec", str(args.closed_loop_timeout_sec)]
        )
    if args.resume:
        autonomy.append("--resume")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    stages = {
        "prepare": prepare,
        "derive_path": derive,
        "validate_materialization": validate,
    }
    if args.run_autonomy:
        stages["run_autonomy"] = autonomy
    return {
        "schema_version": 1,
        "git_commit": commit,
        "dataset_id": spec["dataset_id"],
        "dataset_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
        },
        "profile": args.profile,
        "run_autonomy": args.run_autonomy,
        "paths": {
            "dataset": str(dataset),
            "database": str(database),
            "inspection": str(inspection),
            "sidecar": str(sidecar),
            "generator_report": str(generator),
            "materialization": str(materialization),
        },
        "stages": stages,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    plan = command_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "pipeline_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, command in plan["stages"].items():
        print(f"[{name}] {subprocess.list2cmdline(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"pipeline plan: {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
