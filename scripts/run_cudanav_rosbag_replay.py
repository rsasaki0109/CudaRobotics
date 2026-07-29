#!/usr/bin/env python3
"""Run and quality-gate a reproducible CUDA CudaNav real-rosbag replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from cudanav_rosbag_evidence import (
    REQUIRED_CUDANAV_OUTPUT_TOPICS,
    describe_input,
    evaluate_manifest,
    sha256_file,
)
from run_cudanav_closed_loop import command_output, git_dirty, gpu_identity
from validate_cudanav_real_dataset import evaluate as evaluate_real_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS = (
    "/tf",
    "/tf_static",
    "/points",
    "/plan",
    "/cuda_nav/odom",
    "/cuda_nav/cmd_vel",
    "/cuda_nav/occupancy",
    "/cuda_nav/local_map",
    "/cuda_nav/esdf",
    "/cuda_nav/local_costmap/costmap",
    "/cuda_nav/odometry_diagnostics",
    "/cuda_nav/mapping_diagnostics",
    "/cuda_nav/esdf_diagnostics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--derived-path-bag", type=Path)
    parser.add_argument("--dataset-materialization", type=Path)
    parser.add_argument(
        "--evaluation-db",
        type=Path,
        required=True,
        help="DB3 file inside the input bag used by the offline quality evaluator.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-config", type=Path, required=True)
    parser.add_argument(
        "--controller-command",
        required=True,
        help=(
            "Controller launch command. It is parsed as argv, not by a shell; "
            "{out_dir}, {diagnostics_csv}, and {controller_config} are expanded."
        ),
    )
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--settle-sec", type=float, default=5.0)
    parser.add_argument("--ros-domain-id", type=int)
    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--bag-play-arg", action="append", default=[])
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--record-topic", action="append", default=[])
    parser.add_argument("--command-topic", default="/mobile_base_controller/cmd_vel")
    parser.add_argument("--odometry-topic", default="/mobile_base_controller/odom")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--minimum-clearance-m", type=float, default=0.10)
    parser.add_argument("--maximum-solve-p95-ms", type=float, default=50.0)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.50)
    return parser.parse_args()


def start(command: list[str], log, env: dict[str, str]) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def stop(process: subprocess.Popen[Any] | None) -> int | None:
    if process is None:
        return None
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGINT)
        return process.wait(timeout=15.0)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        return process.wait(timeout=10.0)


def controller_argv(template: str, replacements: dict[str, str]) -> list[str]:
    expanded = template.format(**replacements)
    command = shlex.split(expanded)
    if not command:
        raise ValueError("controller command is empty")
    return command


def play_argv(
    source: Path,
    derived_path_bag: Path | None,
    use_sim_time: bool,
    extra_args: list[str],
) -> list[str]:
    command = ["ros2", "bag", "play"]
    if derived_path_bag is not None:
        command.extend(["-i", str(source), "-i", str(derived_path_bag)])
    else:
        command.append(str(source))
    if use_sim_time:
        command.append("--clock")
    command.extend(extra_args)
    return command


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.duration_sec < 0.0 or args.settle_sec < 0.0:
        raise SystemExit("duration and settle times must be non-negative")
    source = args.bag.resolve()
    derived_path_bag = (
        args.derived_path_bag.resolve() if args.derived_path_bag else None
    )
    dataset_materialization = (
        args.dataset_materialization.resolve()
        if args.dataset_materialization
        else None
    )
    if (derived_path_bag is None) != (dataset_materialization is None):
        raise SystemExit(
            "--derived-path-bag and --dataset-materialization must be used together"
        )
    dataset_gate = None
    materialization_payload = None
    if dataset_materialization is not None:
        dataset_gate = evaluate_real_dataset(
            ROOT / "docs" / "cudanav_real_dataset.json",
            dataset_materialization,
        )
        if not dataset_gate["ready"]:
            raise SystemExit("real-dataset materialization gate did not pass")
    evaluation_db = args.evaluation_db.resolve()
    config = args.controller_config.resolve()
    if not evaluation_db.is_file():
        raise SystemExit(f"evaluation DB does not exist: {evaluation_db}")
    if not evaluation_db.is_relative_to(source if source.is_dir() else source.parent):
        raise SystemExit("--evaluation-db must be contained in --bag")
    if not config.is_file():
        raise SystemExit(f"controller config does not exist: {config}")
    record = args.record if args.record is not None else args.profile == "release"
    if args.profile == "release" and not record:
        raise SystemExit("release profile requires output rosbag recording")
    topics = args.record_topic or list(DEFAULT_TOPICS)
    missing_output_topics = sorted(
        set(REQUIRED_CUDANAV_OUTPUT_TOPICS) - set(topics)
    )
    if args.profile == "release" and missing_output_topics:
        raise SystemExit(
            "release recording omits required CudaNav outputs: "
            + ", ".join(missing_output_topics)
        )
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    diagnostics = output / "diagnostics.csv"
    config_copy = output / "controller.yaml"
    shutil.copy2(config, config_copy)
    materialization_copy = output / "dataset_materialization.json"
    if dataset_materialization is not None:
        shutil.copy2(dataset_materialization, materialization_copy)
    input_identity = describe_input(source)
    derived_identity = (
        describe_input(derived_path_bag) if derived_path_bag is not None else None
    )
    if dataset_gate is not None:
        materialization_payload = json.loads(
            materialization_copy.read_text(encoding="utf-8")
        )
        if (
            materialization_payload["source_bag"]["tree_sha256"]
            != input_identity["tree_sha256"]
            or materialization_payload["derived_path_bag"]["tree_sha256"]
            != derived_identity["tree_sha256"]
        ):
            raise SystemExit("runner bags do not match dataset materialization")
    input_root = source if source.is_dir() else source.parent
    replacements = {
        "out_dir": str(output),
        "diagnostics_csv": str(diagnostics),
        "controller_config": str(config_copy),
    }
    controller_command = controller_argv(args.controller_command, replacements)
    play_command = play_argv(
        source, derived_path_bag, args.use_sim_time, args.bag_play_arg
    )
    recording = output / "recording"
    record_command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(recording),
        *topics,
    ]
    evaluation_dir = output / "evaluation"
    evaluate_command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_mppi_rosbag.py"),
        str(evaluation_db),
        "--diagnostics-csv",
        str(diagnostics),
        "--output-dir",
        str(evaluation_dir),
        "--command-topic",
        args.command_topic,
        "--odometry-topic",
        args.odometry_topic,
        "--scan-topic",
        args.scan_topic,
        "--minimum-clearance-m",
        str(args.minimum_clearance_m),
        "--maximum-solve-p95-ms",
        str(args.maximum_solve_p95_ms),
        "--minimum-valid-ratio",
        str(args.minimum_valid_ratio),
    ]
    if materialization_payload is not None:
        dataset_spec = json.loads(
            (ROOT / "docs" / "cudanav_real_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        quality_filter = dataset_spec["quality_evaluation"]["filter"]
        evaluate_command.extend(
            [
                "--pointcloud-topic",
                dataset_spec["recorded_inputs"]["pointcloud"]["topic"],
                "--odometry-topic",
                dataset_spec["recorded_inputs"]["odometry"]["topic"],
                "--pointcloud-half-angle-rad",
                str(quality_filter["half_angle_rad"]),
                "--pointcloud-minimum-z-m",
                str(quality_filter["minimum_z_m"]),
                "--pointcloud-maximum-z-m",
                str(quality_filter["maximum_z_m"]),
                "--pointcloud-minimum-range-m",
                str(quality_filter["minimum_range_m"]),
                "--pointcloud-maximum-range-m",
                str(quality_filter["maximum_range_m"]),
                "--pointcloud-maximum-command-age-ms",
                str(quality_filter["maximum_command_age_ms"]),
            ]
        )
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    if args.ros_domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    processes: dict[str, subprocess.Popen[Any] | None] = {
        "controller": None,
        "record": None,
        "play": None,
    }
    launch_errors: dict[str, str] = {}
    started_at = datetime.now(timezone.utc).isoformat()
    timed_out = False
    with (
        (output / "controller.log").open("w", encoding="utf-8") as controller_log,
        (output / "play.log").open("w", encoding="utf-8") as play_log,
        (output / "record.log").open("w", encoding="utf-8") as record_log,
    ):
        try:
            try:
                processes["controller"] = start(controller_command, controller_log, env)
                time.sleep(args.settle_sec)
                if record:
                    processes["record"] = start(record_command, record_log, env)
                    time.sleep(1.0)
                processes["play"] = start(play_command, play_log, env)
                if args.duration_sec > 0.0:
                    try:
                        processes["play"].wait(timeout=args.duration_sec)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                else:
                    processes["play"].wait()
            except OSError as exception:
                launch_errors["runtime"] = str(exception)
        finally:
            returncodes = {
                name: stop(process)
                for name, process in reversed(list(processes.items()))
            }

    evaluation_returncode: int | None = None
    with (output / "evaluation.log").open("w", encoding="utf-8") as evaluation_log:
        if diagnostics.is_file():
            try:
                evaluation_returncode = subprocess.run(
                    evaluate_command,
                    cwd=ROOT,
                    env=env,
                    stdout=evaluation_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300.0,
                    check=False,
                ).returncode
            except (OSError, subprocess.SubprocessError) as exception:
                launch_errors["evaluation"] = str(exception)
        else:
            evaluation_log.write("diagnostics CSV was not created\n")

    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "evidence_mode": (
            "real_sensor_shadow_with_derived_path"
            if derived_path_bag is not None
            else "shadow_controller_with_recorded_motion"
        ),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "gpu": gpu_identity(),
        "environment": {
            key: env.get(key, "")
            for key in ("ROS_DISTRO", "ROS_DOMAIN_ID", "CUDA_VISIBLE_DEVICES")
        },
        "input_bag": input_identity,
        "derived_path_bag": derived_identity,
        "dataset_materialization_sha256": (
            sha256_file(materialization_copy)
            if dataset_materialization is not None
            else ""
        ),
        "evaluation_database": {
            "source": str(evaluation_db),
            "relative_path": evaluation_db.relative_to(input_root).as_posix(),
            "sha256": sha256_file(evaluation_db),
        },
        "controller_config_sha256": sha256_file(config_copy),
        "record_topics": topics if record else [],
        "required_output_topics": list(REQUIRED_CUDANAV_OUTPUT_TOPICS),
        "recording_identity": (
            describe_input(recording)
            if record and (recording / "metadata.yaml").is_file()
            else None
        ),
        "diagnostics_sha256": (
            sha256_file(diagnostics) if diagnostics.is_file() else ""
        ),
        "evaluation_sha256": (
            sha256_file(evaluation_dir / "evaluation.json")
            if (evaluation_dir / "evaluation.json").is_file()
            else ""
        ),
        "timed_out": timed_out,
        "launch_errors": launch_errors,
        "returncodes": {
            **returncodes,
            "evaluate": evaluation_returncode,
        },
        "commands": {
            "controller": controller_command,
            "play": play_command,
            "record": record_command if record else None,
            "evaluate": evaluate_command,
        },
        "artifacts": {
            "evaluation": (
                "evaluation/evaluation.json"
                if (evaluation_dir / "evaluation.json").is_file()
                else None
            ),
            "diagnostics": diagnostics.name if diagnostics.is_file() else None,
            "controller_config": config_copy.name,
            "dataset_materialization": (
                materialization_copy.name
                if dataset_materialization is not None
                else None
            ),
            "controller_log": "controller.log",
            "play_log": "play.log",
            "record_log": "record.log",
            "evaluation_log": "evaluation.log",
            "recording": (
                recording.name
                if record and (recording / "metadata.yaml").is_file()
                else None
            ),
        },
    }
    gate = evaluate_manifest(manifest, output, args.profile)
    manifest["gate"] = gate
    manifest["passed"] = gate["passed"]
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
