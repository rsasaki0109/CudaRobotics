#!/usr/bin/env python3
"""Run resumable closed-loop CUDA-controller transfer against a MuJoCo plant."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from contact_external_fidelity import (
    ExternalCondition,
    add_condition,
    profile_spec,
    summarize_external,
    validate_condition_rows,
    write_csv,
    write_report,
)
from contact_robustness import load_csv, sha256_file
from run_contact_robustness import (
    canonical_sha256,
    output_is_safe,
    resolve_binary,
    run_logged,
    write_json_atomic,
)
from run_cudanav_closed_loop import command_output, git_dirty, gpu_identity


ROOT = Path(__file__).resolve().parents[1]
TARGET = "benchmark_diff_mppi_pushing_box_mujoco"
DEFAULT_MODEL = Path("mujoco_models") / "contact_box_push.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("bin") / TARGET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime-library", type=Path)
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int)
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def resolve_runtime(binary: Path, requested: Path | None) -> Path:
    if requested is not None:
        return resolve_input(requested)
    candidates = [binary.parent / "mujoco.dll"]
    candidates += sorted(binary.parent.glob("libmujoco.so*"))
    candidates += sorted(binary.parent.glob("libmujoco*.dylib"))
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0])


def build_command(
    binary: Path,
    model: Path,
    condition: ExternalCondition,
    spec: dict[str, Any],
    csv_path: Path,
) -> list[str]:
    return [
        str(binary),
        "--model",
        str(model),
        "--scenarios",
        ",".join(spec["scenarios"]),
        "--planners",
        ",".join(spec["planners"]),
        "--k-values",
        ",".join(str(value) for value in spec["k_values"]),
        "--seed-count",
        str(spec["seed_count"]),
        "--seed-offset",
        str(spec["seed_offset"]),
        "--horizon",
        str(spec["horizon"]),
        "--frame-skip",
        str(spec["frame_skip"]),
        *condition.arguments(),
        "--csv",
        str(csv_path),
    ]


def experiment_identity(
    *,
    profile: str,
    spec: dict[str, Any],
    source_binary: Path,
    binary: Path,
    binary_sha256: str,
    source_model: Path,
    model: Path,
    model_sha256: str,
    runtime_source: Path,
    runtime: Path,
    runtime_sha256: str,
    commit: str,
    engine: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": profile,
        "git_commit": commit,
        "binary_source": str(source_binary),
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "model_source": str(source_model),
        "model": str(model),
        "model_sha256": model_sha256,
        "runtime_library_source": str(runtime_source),
        "runtime_library": str(runtime),
        "runtime_library_sha256": runtime_sha256,
        "engine": engine,
        "conditions": [asdict(condition) for condition in spec["conditions"]],
        "scenarios": list(spec["scenarios"]),
        "planners": list(spec["planners"]),
        "k_values": list(spec["k_values"]),
        "seed_count": spec["seed_count"],
        "seed_offset": spec["seed_offset"],
        "horizon": spec["horizon"],
        "frame_skip": spec["frame_skip"],
        "comparison_planners": list(spec["comparison_planners"]),
        "bootstrap_resamples": spec["bootstrap_resamples"],
    }


def completed_attempt(
    run: dict[str, Any], spec: dict[str, Any]
) -> tuple[Path, list[dict[str, str]]] | None:
    if not run.get("passed"):
        return None
    path = Path(run.get("csv", ""))
    if not path.is_file() or sha256_file(path) != run.get("csv_sha256"):
        return None
    rows = load_csv(path)
    return (path, rows) if not validate_condition_rows(rows, spec) else None


def runtime_environment(runtime_directory: Path) -> dict[str, str]:
    environment = os.environ.copy()
    variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    existing = environment.get(variable, "")
    environment[variable] = str(runtime_directory) + (
        os.pathsep + existing if existing else ""
    )
    return environment


def engine_identity(
    binary: Path, runtime_directory: Path | None = None
) -> dict[str, Any]:
    result = subprocess.run(
        [str(binary), "--engine-info"],
        cwd=ROOT,
        env=(
            runtime_environment(runtime_directory)
            if runtime_directory is not None
            else None
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"MuJoCo engine query failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    if (
        payload.get("engine") != "MuJoCo"
        or not payload.get("version")
        or not isinstance(payload.get("version_number"), int)
        or payload.get("header_version_number") != payload.get("version_number")
    ):
        raise RuntimeError("invalid or mismatched MuJoCo header/library identity")
    return payload


def run_benchmark_logged(
    command: list[str], log_path: Path, runtime_directory: Path
) -> tuple[int, float]:
    environment = runtime_environment(runtime_directory)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result.returncode, time.perf_counter() - start


def main() -> int:
    args = parse_args()
    spec = profile_spec(args.profile)
    if args.bootstrap_resamples is not None:
        minimum = 5000 if args.profile == "release" else 100
        if args.bootstrap_resamples < minimum:
            raise SystemExit(f"--bootstrap-resamples must be at least {minimum}")
        spec["bootstrap_resamples"] = args.bootstrap_resamples
    output = args.output_dir.resolve()
    source_binary = resolve_binary(args.binary)
    source_model = resolve_input(args.model)
    if args.build and not args.dry_run:
        build_dir = args.build_dir.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        returncode, _ = run_logged(
            ["cmake", "--build", str(build_dir), "--target", TARGET, "--parallel"],
            build_dir / "contact_external_fidelity_build.log",
        )
        if returncode:
            raise SystemExit("MuJoCo contact benchmark build failed")
        source_binary = resolve_binary(args.binary)
    source_runtime = resolve_runtime(source_binary, args.runtime_library)
    binary_sha = sha256_file(source_binary) if source_binary.is_file() else ""
    model_sha = sha256_file(source_model) if source_model.is_file() else ""
    runtime_sha = (
        sha256_file(source_runtime) if source_runtime.is_file() else ""
    )
    engine = (
        engine_identity(source_binary, source_runtime.parent)
        if source_binary.is_file() and not args.dry_run
        else {}
    )
    staged_binary = output / "artifacts" / source_binary.name
    staged_model = output / "artifacts" / source_model.name
    staged_runtime = output / "artifacts" / source_runtime.name
    experiment = experiment_identity(
        profile=args.profile,
        spec=spec,
        source_binary=source_binary,
        binary=staged_binary,
        binary_sha256=binary_sha,
        source_model=source_model,
        model=staged_model,
        model_sha256=model_sha,
        runtime_source=source_runtime,
        runtime=staged_runtime,
        runtime_sha256=runtime_sha,
        commit=command_output(["git", "rev-parse", "HEAD"]),
        engine=engine,
    )
    experiment["identity_sha256"] = canonical_sha256(experiment)
    if args.dry_run:
        commands = [
            build_command(
                staged_binary,
                staged_model,
                condition,
                spec,
                output / "raw" / f"{condition.name}.attempt-001.csv",
            )
            for condition in spec["conditions"]
        ]
        print(json.dumps({"experiment": experiment, "commands": commands}, indent=2))
        return 0
    if not source_binary.is_file():
        raise SystemExit(f"benchmark binary does not exist: {source_binary}")
    if not source_model.is_file():
        raise SystemExit(f"MuJoCo model does not exist: {source_model}")
    if not source_runtime.is_file():
        raise SystemExit(f"MuJoCo runtime library does not exist: {source_runtime}")
    if not output_is_safe(output):
        raise SystemExit("output inside the repository must be git-ignored")
    dirty = git_dirty()
    gpus = gpu_identity()
    if args.profile == "release" and dirty is not False:
        raise SystemExit("release evidence requires a clean worktree")
    if not gpus:
        raise SystemExit("external-fidelity evidence requires an identified NVIDIA GPU")

    plan_path = output / "plan.json"
    state_path = output / "state.json"
    if args.resume:
        if not plan_path.is_file() or not state_path.is_file():
            raise SystemExit("--resume requires existing plan.json and state.json")
        if json.loads(plan_path.read_text(encoding="utf-8")) != experiment:
            raise SystemExit("refusing resume: experiment identity changed")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not staged_binary.is_file()
            or sha256_file(staged_binary) != binary_sha
            or not staged_model.is_file()
            or sha256_file(staged_model) != model_sha
            or not staged_runtime.is_file()
            or sha256_file(staged_runtime) != runtime_sha
        ):
            raise SystemExit(
                "refusing resume: staged binary, model, or runtime changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"refusing non-empty output directory: {output}")
        for directory in ("raw", "logs", "artifacts"):
            (output / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_binary, staged_binary)
        shutil.copy2(source_model, staged_model)
        shutil.copy2(source_runtime, staged_runtime)
        write_json_atomic(plan_path, experiment)
        state = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "runs": {},
        }
        write_json_atomic(state_path, state)

    all_rows: list[dict[str, str]] = []
    all_passed = True
    for condition in spec["conditions"]:
        history = state["runs"].setdefault(condition.name, [])
        successful = next(
            (
                completed
                for attempt in reversed(history)
                if (completed := completed_attempt(attempt, spec)) is not None
            ),
            None,
        )
        if successful is None:
            attempt_number = len(history) + 1
            csv_path = output / "raw" / (
                f"{condition.name}.attempt-{attempt_number:03d}.csv"
            )
            log_path = output / "logs" / (
                f"{condition.name}.attempt-{attempt_number:03d}.log"
            )
            command = build_command(
                staged_binary, staged_model, condition, spec, csv_path
            )
            returncode, elapsed = run_benchmark_logged(
                command, log_path, staged_runtime.parent
            )
            rows: list[dict[str, str]] = []
            errors: list[str] = []
            if returncode == 0 and csv_path.is_file():
                try:
                    rows = load_csv(csv_path)
                    errors = validate_condition_rows(rows, spec)
                except (OSError, csv.Error, ValueError) as exception:
                    errors = [str(exception)]
            else:
                errors = [f"benchmark return code {returncode} or missing CSV"]
            attempt = {
                "attempt": attempt_number,
                "command": command,
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "csv": str(csv_path),
                "csv_sha256": sha256_file(csv_path) if csv_path.is_file() else "",
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
                "validation_errors": errors,
                "passed": returncode == 0 and not errors,
            }
            history.append(attempt)
            write_json_atomic(state_path, state)
            if attempt["passed"]:
                successful = (csv_path, rows)
        if successful is None:
            all_passed = False
        else:
            all_rows.extend(add_condition(successful[1], condition))
    if not all_passed:
        print(f"incomplete matrix; resume with: {sys.executable} {__file__} --resume ...")
        return 1

    paths = {
        "episodes": output / "episodes.csv",
        "summary": output / "summary.csv",
        "comparisons": output / "comparisons.csv",
        "report": output / "report.md",
    }
    write_csv(all_rows, paths["episodes"])
    summaries, comparisons = summarize_external(all_rows, spec)
    write_csv(summaries, paths["summary"])
    write_csv(comparisons, paths["comparisons"])
    write_report(summaries, comparisons, paths["report"])
    positive = sum(
        row["mcnemar_holm_p"] < 0.05 and row["success_delta"] > 0.0
        for row in comparisons
    )
    negative = sum(
        row["mcnemar_holm_p"] < 0.05 and row["success_delta"] < 0.0
        for row in comparisons
    )
    if engine_identity(staged_binary, staged_runtime.parent) != engine:
        raise SystemExit("MuJoCo engine identity changed during the experiment")
    artifact_paths = {
        **paths,
        "state": state_path,
        "plan": plan_path,
        "binary": staged_binary,
        "model": staged_model,
        "runtime_library": staged_runtime,
    }
    manifest = {
        "schema_version": 1,
        "evidence_mode": "contact_external_fidelity_mujoco_gpu",
        "profile": args.profile,
        "experiment": experiment,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_dirty": dirty,
        "gpu": gpus,
        "engine": engine,
        "matrix": {
            "conditions": len(spec["conditions"]),
            "scenarios": len(spec["scenarios"]),
            "planners": len(spec["planners"]),
            "k_values": len(spec["k_values"]),
            "seeds": spec["seed_count"],
            "episodes": len(all_rows),
        },
        "integrity_gate": {
            "complete_matrix": True,
            "all_raw_runs_valid": True,
            "clean_worktree": dirty is False,
            "gpu_identified": bool(gpus),
            "mujoco_identified": True,
        },
        "outcome": {
            "hypothesis_is_integrity_gate": False,
            "holm_significant_positive_success_cells": positive,
            "holm_significant_negative_success_cells": negative,
            "comparison_family_size": len(comparisons),
        },
        "artifacts": {
            name: {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
        "passed": True,
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(manifest["outcome"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
