#!/usr/bin/env python3
"""Run the distinct CudaNav release evidence modes from one entry point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from cudanav_autonomy_suite import evaluate_suite, sha256_file
from cudanav_evidence import (
    evaluate_manifest as evaluate_closed_loop_manifest,
    evaluate_summary,
)
from cudanav_multi_gpu import evaluate_multi_gpu_suite
from cudanav_rosbag_evidence import evaluate_manifest as evaluate_rosbag_manifest
from run_cudanav_closed_loop import command_output, git_dirty
from run_cudanav_multi_gpu import output_is_ignored

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument("--bag", type=Path)
    parser.add_argument("--derived-path-bag", type=Path)
    parser.add_argument("--dataset-materialization", type=Path)
    parser.add_argument("--evaluation-db", type=Path)
    parser.add_argument("--controller-config", type=Path)
    parser.add_argument("--controller-command")
    parser.add_argument("--rosbag-duration-sec", type=float, default=0.0)
    parser.add_argument("--multi-gpu-run", type=Path, action="append", default=[])
    parser.add_argument("--multi-gpu-devices")
    parser.add_argument("--multi-gpu-repetitions", type=int, default=1)
    parser.add_argument("--closed-loop-timeout-sec", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result.returncode


def closed_loop_command(
    directory: Path, profile: str, timeout_sec: float | None
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_cudanav_closed_loop.py"),
        "--output-dir",
        str(directory),
        "--profile",
        profile,
    ]
    if timeout_sec is not None:
        command.extend(["--timeout-sec", str(timeout_sec)])
    return command


def rosbag_command(directory: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_cudanav_rosbag_replay.py"),
        "--bag",
        str(args.bag.resolve()),
        "--evaluation-db",
        str(args.evaluation_db.resolve()),
        "--output-dir",
        str(directory),
        "--controller-config",
        str(args.controller_config.resolve()),
        "--controller-command",
        args.controller_command,
        "--profile",
        args.profile,
    ]
    if args.rosbag_duration_sec > 0.0:
        command.extend(["--duration-sec", str(args.rosbag_duration_sec)])
    if args.derived_path_bag is not None:
        command.extend(["--derived-path-bag", str(args.derived_path_bag.resolve())])
    if args.dataset_materialization is not None:
        command.extend(
            [
                "--dataset-materialization",
                str(args.dataset_materialization.resolve()),
            ]
        )
    return command


def multi_gpu_command(
    directory: Path,
    args: argparse.Namespace,
    closed_loop_directory: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_cudanav_multi_gpu.py"),
        "--output-dir",
        str(directory),
        "--repetitions",
        str(args.multi_gpu_repetitions),
    ]
    if args.multi_gpu_run:
        for source in [closed_loop_directory, *args.multi_gpu_run]:
            command.extend(["--import-run", str(source.resolve())])
    elif args.multi_gpu_devices:
        command.extend(["--devices", args.multi_gpu_devices])
    return command


def validate_closed_loop(directory: Path, profile: str) -> dict[str, Any]:
    try:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
        summary_path = (directory / artifacts["summary"]).resolve()
        summary_path.relative_to(directory.resolve())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_gate = evaluate_summary(summary, profile)
        manifest_gate = evaluate_closed_loop_manifest(manifest, directory, profile)
        binding = artifacts.get("trajectory") == summary.get(
            "trajectory_csv"
        ) and manifest.get("traversal_count") == summary.get("traversals_requested")
        return {
            "passed": (summary_gate["passed"] and manifest_gate["passed"] and binding),
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "summary_gate": summary_gate,
            "manifest_gate": manifest_gate,
            "binding": binding,
        }
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {"passed": False, "error": str(error)}


def validate_rosbag(directory: Path, profile: str) -> dict[str, Any]:
    try:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gate = evaluate_rosbag_manifest(manifest, directory, profile)
        return {
            "passed": gate["passed"],
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "manifest_gate": gate,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error": str(error)}


def validate_multi_gpu(directory: Path) -> dict[str, Any]:
    try:
        manifest_path = directory / "multi_gpu_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gate = evaluate_multi_gpu_suite(manifest, directory)
        return {
            "passed": gate["passed"],
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "manifest_gate": gate,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error": str(error)}


def find_valid_attempt(
    mode_root: Path,
    validator,
) -> tuple[Path, dict[str, Any]] | None:
    if not mode_root.is_dir():
        return None
    for directory in sorted(mode_root.glob("attempt_*"), reverse=True):
        if directory.is_dir():
            result = validator(directory)
            if result["passed"]:
                return directory, result
    return None


def run_mode(
    *,
    mode: str,
    output: Path,
    command_builder,
    validator,
) -> tuple[Path | None, dict[str, Any]]:
    mode_root = output / mode
    previous = find_valid_attempt(mode_root, validator)
    if previous is not None:
        return previous
    mode_root.mkdir(parents=True, exist_ok=True)
    attempt_number = len(list(mode_root.glob("attempt_*"))) + 1
    directory = mode_root / f"attempt_{attempt_number:03d}"
    log_path = output / "logs" / f"{mode}_attempt_{attempt_number:03d}.log"
    command = command_builder(directory)
    returncode = run_logged(command, log_path)
    result = validator(directory)
    result["command"] = command
    result["returncode"] = returncode
    result["driver_log"] = str(log_path.relative_to(output))
    result["passed"] = result["passed"] and returncode == 0
    return (directory if result["passed"] else None), result


def validate_arguments(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    bag_values = (
        args.bag,
        args.evaluation_db,
        args.controller_config,
        args.controller_command,
    )
    bag_count = sum(value is not None for value in bag_values)
    if bag_count not in (0, len(bag_values)):
        errors.append(
            "--bag, --evaluation-db, --controller-config, and "
            "--controller-command must be supplied together"
        )
    if args.profile == "release" and bag_count != len(bag_values):
        errors.append("release suite requires real-rosbag inputs")
    if (args.derived_path_bag is None) != (args.dataset_materialization is None):
        errors.append(
            "--derived-path-bag and --dataset-materialization "
            "must be supplied together"
        )
    if args.multi_gpu_run and args.multi_gpu_devices:
        errors.append("--multi-gpu-run and --multi-gpu-devices are exclusive")
    if args.multi_gpu_repetitions <= 0:
        errors.append("--multi-gpu-repetitions must be positive")
    if args.rosbag_duration_sec < 0.0:
        errors.append("--rosbag-duration-sec must be non-negative")
    if args.closed_loop_timeout_sec is not None and args.closed_loop_timeout_sec <= 0:
        errors.append("--closed-loop-timeout-sec must be positive")
    return errors


def main() -> int:
    args = parse_args()
    errors = validate_arguments(args)
    if errors:
        raise SystemExit("; ".join(errors))
    output = args.output_dir.resolve()
    bag_enabled = args.bag is not None
    multi_enabled = bool(args.multi_gpu_run or args.multi_gpu_devices)
    required_modes = ["closed_loop"]
    if bag_enabled:
        required_modes.append("real_rosbag_shadow")
    if multi_enabled:
        required_modes.append("multi_gpu")
    commit = command_output(["git", "rev-parse", "HEAD"])
    plan = {
        "schema_version": 1,
        "profile": args.profile,
        "git_commit": commit,
        "required_modes": required_modes,
        "bag": str(args.bag.resolve()) if args.bag else None,
        "derived_path_bag": (
            str(args.derived_path_bag.resolve()) if args.derived_path_bag else None
        ),
        "dataset_materialization": (
            str(args.dataset_materialization.resolve())
            if args.dataset_materialization
            else None
        ),
        "evaluation_db": (
            str(args.evaluation_db.resolve()) if args.evaluation_db else None
        ),
        "controller_config": (
            str(args.controller_config.resolve()) if args.controller_config else None
        ),
        "controller_command": args.controller_command,
        "rosbag_duration_sec": args.rosbag_duration_sec,
        "multi_gpu_runs": [str(path.resolve()) for path in args.multi_gpu_run],
        "multi_gpu_devices": args.multi_gpu_devices,
        "multi_gpu_repetitions": args.multi_gpu_repetitions,
        "closed_loop_timeout_sec": args.closed_loop_timeout_sec,
    }
    preview_closed = output / "closed_loop" / "attempt_001"
    if args.dry_run:
        commands: dict[str, list[str]] = {
            "closed_loop": closed_loop_command(
                preview_closed, args.profile, args.closed_loop_timeout_sec
            )
        }
        if bag_enabled:
            commands["real_rosbag_shadow"] = rosbag_command(
                output / "real_rosbag_shadow" / "attempt_001", args
            )
        if multi_enabled:
            commands["multi_gpu"] = multi_gpu_command(
                output / "multi_gpu" / "attempt_001",
                args,
                preview_closed,
            )
        print(json.dumps({"plan": plan, "commands": commands}, indent=2))
        return 0
    if not output_is_ignored(output):
        raise SystemExit("suite output inside the repository must be git-ignored")
    if args.profile == "release" and git_dirty() is not False:
        raise SystemExit("release suite requires a clean worktree")
    plan_path = output / "plan.json"
    if args.resume:
        if not plan_path.is_file():
            raise SystemExit("--resume requires an existing plan.json")
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise SystemExit("refusing resume: suite plan changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"refusing non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "logs").mkdir()
        write_json_atomic(plan_path, plan)

    mode_entries: dict[str, dict[str, str]] = {}
    execution: dict[str, dict[str, Any]] = {}
    closed_directory, closed_result = run_mode(
        mode="closed_loop",
        output=output,
        command_builder=lambda directory: closed_loop_command(
            directory, args.profile, args.closed_loop_timeout_sec
        ),
        validator=lambda directory: validate_closed_loop(directory, args.profile),
    )
    execution["closed_loop"] = closed_result
    if closed_directory is not None:
        mode_entries["closed_loop"] = {
            "directory": closed_directory.relative_to(output).as_posix(),
            "manifest_sha256": sha256_file(closed_directory / "manifest.json"),
        }

    if bag_enabled:
        rosbag_directory, rosbag_result = run_mode(
            mode="real_rosbag_shadow",
            output=output,
            command_builder=lambda directory: rosbag_command(directory, args),
            validator=lambda directory: validate_rosbag(directory, args.profile),
        )
        execution["real_rosbag_shadow"] = rosbag_result
        if rosbag_directory is not None:
            mode_entries["real_rosbag_shadow"] = {
                "directory": rosbag_directory.relative_to(output).as_posix(),
                "manifest_sha256": sha256_file(rosbag_directory / "manifest.json"),
            }

    if multi_enabled:
        if args.multi_gpu_run and closed_directory is None:
            execution["multi_gpu"] = {
                "passed": False,
                "error": "cross-machine aggregation requires valid local closed loop",
            }
        else:
            multi_directory, multi_result = run_mode(
                mode="multi_gpu",
                output=output,
                command_builder=lambda directory: multi_gpu_command(
                    directory,
                    args,
                    closed_directory or preview_closed,
                ),
                validator=validate_multi_gpu,
            )
            execution["multi_gpu"] = multi_result
            if multi_directory is not None:
                mode_entries["multi_gpu"] = {
                    "directory": multi_directory.relative_to(output).as_posix(),
                    "manifest_sha256": sha256_file(
                        multi_directory / "multi_gpu_manifest.json"
                    ),
                }

    suite = {
        "schema_version": 1,
        "evidence_mode": "cudanav_autonomy_suite",
        "profile": args.profile,
        "git_commit": commit,
        "git_dirty": git_dirty(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "required_modes": required_modes,
        "modes": mode_entries,
        "execution": execution,
        "passed": set(mode_entries) == set(required_modes),
    }
    preliminary_path = output / "manifest.json"
    write_json_atomic(preliminary_path, suite)
    gate = evaluate_suite(suite, output)
    suite["gate"] = gate
    suite["passed"] = gate["passed"]
    suite["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(preliminary_path, suite)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if suite["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
