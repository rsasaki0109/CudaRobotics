#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    target: str
    binary: str
    args: tuple[str, ...]
    time_caps: str
    time_targets: str
    optional: bool = False


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "dynamic_nav_smoke": BenchmarkSpec(
        name="dynamic_nav_smoke",
        target="benchmark_diff_mppi",
        binary="benchmark_diff_mppi",
        args=(
            "--quick",
            "--scenarios",
            "dynamic_crossing",
            "--planners",
            "mppi,diff_mppi_1",
            "--k-values",
            "256",
            "--seed-count",
            "1",
        ),
        time_caps="1.0,1.5",
        time_targets="1.0",
    ),
    "dynamic_nav_quick": BenchmarkSpec(
        name="dynamic_nav_quick",
        target="benchmark_diff_mppi",
        binary="benchmark_diff_mppi",
        args=("--quick",),
        time_caps="1.0,1.5",
        time_targets="1.0,1.5",
    ),
    "cartpole_quick": BenchmarkSpec(
        name="cartpole_quick",
        target="benchmark_diff_mppi_cartpole",
        binary="benchmark_diff_mppi_cartpole",
        args=("--quick",),
        time_caps="0.25,0.5,0.75",
        time_targets="0.25,0.5",
    ),
    "dynamic_bicycle_quick": BenchmarkSpec(
        name="dynamic_bicycle_quick",
        target="benchmark_diff_mppi_dynamic_bicycle",
        binary="benchmark_diff_mppi_dynamic_bicycle",
        args=("--quick",),
        time_caps="0.1,0.7,1.8",
        time_targets="0.1,0.7,1.8",
    ),
    "manipulator_quick": BenchmarkSpec(
        name="manipulator_quick",
        target="benchmark_diff_mppi_manipulator",
        binary="benchmark_diff_mppi_manipulator",
        args=("--quick",),
        time_caps="3.0,6.5,10.5",
        time_targets="3.0",
    ),
    "manipulator_7dof_quick": BenchmarkSpec(
        name="manipulator_7dof_quick",
        target="benchmark_diff_mppi_manipulator_7dof",
        binary="benchmark_diff_mppi_manipulator_7dof",
        args=("--quick",),
        time_caps="3.0,5.0",
        time_targets="3.0,5.0",
    ),
    "mujoco_pendulum_quick": BenchmarkSpec(
        name="mujoco_pendulum_quick",
        target="benchmark_diff_mppi_mujoco",
        binary="benchmark_diff_mppi_mujoco",
        args=("--quick",),
        time_caps="0.25,0.5,0.75",
        time_targets="0.25,0.5",
        optional=True,
    ),
    "mujoco_reacher_quick": BenchmarkSpec(
        name="mujoco_reacher_quick",
        target="benchmark_diff_mppi_mujoco_reacher",
        binary="benchmark_diff_mppi_mujoco_reacher",
        args=("--quick",),
        time_caps="1.0,2.0,4.0",
        time_targets="1.0,2.0",
        optional=True,
    ),
}


SUITES: dict[str, tuple[str, ...]] = {
    "smoke": ("dynamic_nav_smoke",),
    "diff-mppi": (
        "dynamic_nav_quick",
        "cartpole_quick",
        "dynamic_bicycle_quick",
        "manipulator_quick",
        "manipulator_7dof_quick",
    ),
    "standard": ("mujoco_pendulum_quick", "mujoco_reacher_quick"),
    "all": (
        "dynamic_nav_quick",
        "cartpole_quick",
        "dynamic_bicycle_quick",
        "manipulator_quick",
        "manipulator_7dof_quick",
        "mujoco_pendulum_quick",
        "mujoco_reacher_quick",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible CudaRobotics benchmark suites and write a machine-readable manifest."
    )
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke", help="Benchmark suite to run.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(BENCHMARKS),
        help="Run explicit benchmark task names instead of the selected suite.",
    )
    parser.add_argument(
        "--output-dir",
        default="build/repro_suite",
        help="Directory for CSVs, summaries, logs, and manifest.",
    )
    parser.add_argument("--build-dir", default="build", help="CMake build directory used when --build is set.")
    parser.add_argument("--bin-dir", default="bin", help="Directory containing benchmark binaries.")
    parser.add_argument("--build", action="store_true", help="Build each selected benchmark target before running it.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel build jobs for --build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest without building or running commands.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not run summarize_diff_mppi.py after benchmarks.",
    )
    parser.add_argument("--plots", action="store_true", help="Also run plot_diff_mppi.py for each benchmark CSV.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining tasks after a failed command.",
    )
    parser.add_argument("--strict-optional", action="store_true", help="Treat missing optional benchmarks as failures.")
    parser.add_argument("--list", action="store_true", help="List suites and task names, then exit.")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def display_command(command: list[str]) -> str:
    return shlex.join(command)


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def selected_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    names = args.only if args.only else list(SUITES[args.suite])
    return [BENCHMARKS[name] for name in names]


def benchmark_command(spec: BenchmarkSpec, bin_dir: Path, csv_path: Path) -> list[str]:
    return [str(bin_dir / spec.binary), *spec.args, "--csv", str(csv_path)]


def summary_command(spec: BenchmarkSpec, csv_path: Path, markdown_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "summarize_diff_mppi.py"),
        "--csv",
        str(csv_path),
        "--markdown-out",
        str(markdown_path),
        "--time-caps",
        spec.time_caps,
        "--time-targets",
        spec.time_targets,
    ]


def plot_command(csv_path: Path, plot_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "plot_diff_mppi.py"),
        "--csv",
        str(csv_path),
        "--out-dir",
        str(plot_dir),
    ]


def build_command(spec: BenchmarkSpec, build_dir: Path, jobs: int) -> list[str]:
    return ["cmake", "--build", str(build_dir), "--target", spec.target, f"-j{jobs}"]


def run_logged(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    with log_path.open("w") as handle:
        handle.write(f"$ {display_command(command)}\n\n")
        handle.flush()
        proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, time.perf_counter() - begin


def command_record(command: list[str], log_path: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "argv": command,
        "display": display_command(command),
    }
    if log_path is not None:
        record["log"] = rel(log_path)
    return record


def task_manifest(
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    output_dir: Path,
    bin_dir: Path,
    build_dir: Path,
) -> dict[str, Any]:
    csv_path = output_dir / f"{spec.name}.csv"
    summary_path = output_dir / f"{spec.name}_summary.md"
    plot_dir = output_dir / "plots" / spec.name
    build_log = output_dir / "logs" / f"{spec.name}_build.log"
    benchmark_log = output_dir / "logs" / f"{spec.name}.log"
    summary_log = output_dir / "logs" / f"{spec.name}_summary.log"
    plot_log = output_dir / "logs" / f"{spec.name}_plots.log"

    commands: dict[str, Any] = {
        "benchmark": command_record(benchmark_command(spec, bin_dir, csv_path), benchmark_log),
    }
    if args.build:
        commands["build"] = command_record(build_command(spec, build_dir, args.jobs), build_log)
    if not args.skip_summary:
        commands["summary"] = command_record(summary_command(spec, csv_path, summary_path), summary_log)
    if args.plots:
        commands["plots"] = command_record(plot_command(csv_path, plot_dir), plot_log)

    return {
        "name": spec.name,
        "target": spec.target,
        "binary": spec.binary,
        "optional": spec.optional,
        "status": "planned",
        "commands": commands,
        "outputs": {
            "csv": rel(csv_path),
            "summary": rel(summary_path),
            "plots": rel(plot_dir),
        },
    }


def run_task(
    task: dict[str, Any],
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    bin_dir: Path,
) -> bool:
    if args.dry_run:
        task["status"] = "planned"
        return True

    begin = time.perf_counter()
    commands = task["commands"]

    if args.build:
        build = commands["build"]
        rc, duration = run_logged(build["argv"], ROOT / build["log"])
        build["returncode"] = rc
        build["duration_s"] = round(duration, 3)
        if rc != 0:
            task["status"] = "skipped_optional_build_failed" if spec.optional else "failed"
            task["duration_s"] = round(time.perf_counter() - begin, 3)
            return spec.optional and not args.strict_optional

    binary_path = bin_dir / spec.binary
    if not binary_path.exists():
        task["status"] = "skipped_optional_missing_binary" if spec.optional else "failed_missing_binary"
        task["duration_s"] = round(time.perf_counter() - begin, 3)
        return spec.optional and not args.strict_optional

    benchmark = commands["benchmark"]
    rc, duration = run_logged(benchmark["argv"], ROOT / benchmark["log"])
    benchmark["returncode"] = rc
    benchmark["duration_s"] = round(duration, 3)
    if rc != 0:
        task["status"] = "failed"
        task["duration_s"] = round(time.perf_counter() - begin, 3)
        return False

    if "summary" in commands:
        summary = commands["summary"]
        rc, duration = run_logged(summary["argv"], ROOT / summary["log"])
        summary["returncode"] = rc
        summary["duration_s"] = round(duration, 3)
        if rc != 0:
            task["status"] = "failed"
            task["duration_s"] = round(time.perf_counter() - begin, 3)
            return False

    if "plots" in commands:
        plots = commands["plots"]
        rc, duration = run_logged(plots["argv"], ROOT / plots["log"])
        plots["returncode"] = rc
        plots["duration_s"] = round(duration, 3)
        if rc != 0:
            task["status"] = "failed"
            task["duration_s"] = round(time.perf_counter() - begin, 3)
            return False

    task["status"] = "passed"
    task["duration_s"] = round(time.perf_counter() - begin, 3)
    return True


def print_catalog() -> None:
    print("Suites:")
    for suite, names in sorted(SUITES.items()):
        print(f"  {suite}: {', '.join(names)}")
    print("")
    print("Tasks:")
    for name, spec in sorted(BENCHMARKS.items()):
        suffix = " optional" if spec.optional else ""
        print(f"  {name}: {spec.binary}{suffix}")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.list:
        print_catalog()
        return 0

    output_dir = (ROOT / args.output_dir).resolve()
    build_dir = (ROOT / args.build_dir).resolve()
    bin_dir = (ROOT / args.bin_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = selected_specs(args)
    tasks = [task_manifest(spec, args, output_dir, bin_dir, build_dir) for spec in specs]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "suite": args.suite,
        "dry_run": bool(args.dry_run),
        "build_enabled": bool(args.build),
        "plots_enabled": bool(args.plots),
        "summary_enabled": not bool(args.skip_summary),
        "output_dir": rel(output_dir),
        "tasks": tasks,
    }

    manifest_path = output_dir / "manifest.json"
    print(f"Repro suite: {args.suite}")
    print(f"Output: {rel(output_dir)}")
    if args.dry_run:
        print("Dry run: no commands will be executed")

    all_ok = True
    for spec, task in zip(specs, tasks):
        print(f"\n[{spec.name}]")
        for command in task["commands"].values():
            print(f"  {command['display']}")

        ok = run_task(task, spec, args, bin_dir)
        print(f"  status: {task['status']}")
        all_ok = all_ok and ok
        write_manifest(manifest_path, manifest)
        if not ok and not args.continue_on_error:
            break

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "passed" if all_ok else "failed"
    write_manifest(manifest_path, manifest)
    print(f"\nManifest: {rel(manifest_path)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
