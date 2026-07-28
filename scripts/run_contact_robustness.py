#!/usr/bin/env python3
"""Run the paper-grade contact-rich Diff-MPPI robustness matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from contact_robustness import (
    Condition,
    add_condition,
    load_csv,
    profile_spec,
    summarize,
    validate_rows,
    write_csv,
    write_report,
)
from run_cudanav_closed_loop import command_output, git_dirty, gpu_identity


ROOT = Path(__file__).resolve().parents[1]
TARGET = "benchmark_diff_mppi_pushing_box"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("bin") / TARGET)
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_binary(path: Path) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    candidate = candidate.resolve()
    if candidate.is_file():
        return candidate
    executable = candidate.with_suffix(".exe")
    return executable if executable.is_file() else candidate


def output_is_safe(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return True
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def build_command(binary: Path, condition: Condition, spec: dict[str, Any], csv_path: Path) -> list[str]:
    return [
        str(binary),
        "--scenarios",
        ",".join(spec["scenarios"]),
        "--planners",
        ",".join(spec["planners"]),
        "--k-values",
        ",".join(str(value) for value in spec["k_values"]),
        "--seed-count",
        str(spec["seed_count"]),
        "--horizon",
        str(spec["horizon"]),
        *condition.arguments(),
        "--csv",
        str(csv_path),
    ]


def experiment_identity(
    *,
    profile: str,
    spec: dict[str, Any],
    binary_source: Path,
    binary: Path,
    binary_sha256: str,
    commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": profile,
        "git_commit": commit,
        "binary_source": str(binary_source),
        "binary": str(binary),
        "binary_sha256": binary_sha256,
        "conditions": [asdict(condition) for condition in spec["conditions"]],
        "scenarios": list(spec["scenarios"]),
        "planners": list(spec["planners"]),
        "k_values": list(spec["k_values"]),
        "seed_count": spec["seed_count"],
        "horizon": spec["horizon"],
        "comparison_planners": list(spec["comparison_planners"]),
        "bootstrap_resamples": spec["bootstrap_resamples"],
    }


def run_logged(command: list[str], log_path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result.returncode, time.perf_counter() - start


def completed_attempt(
    run: dict[str, Any], spec: dict[str, Any]
) -> tuple[Path, list[dict[str, str]]] | None:
    if not run.get("passed"):
        return None
    path = Path(run.get("csv", ""))
    if not path.is_file() or sha256_file(path) != run.get("csv_sha256"):
        return None
    rows = load_csv(path)
    errors = validate_rows(
        rows,
        scenarios=spec["scenarios"],
        planners=spec["planners"],
        k_values=spec["k_values"],
        seed_count=spec["seed_count"],
    )
    return (path, rows) if not errors else None


def main() -> int:
    args = parse_args()
    spec = profile_spec(args.profile)
    if args.bootstrap_resamples is not None:
        minimum_resamples = 5000 if args.profile == "release" else 100
        if args.bootstrap_resamples < minimum_resamples:
            raise SystemExit(
                f"--bootstrap-resamples must be at least {minimum_resamples} "
                f"for {args.profile}"
            )
        spec["bootstrap_resamples"] = args.bootstrap_resamples
    output = args.output_dir.resolve()
    source_binary = resolve_binary(args.binary)
    if args.build and not args.dry_run:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise SystemExit(f"refusing non-empty output directory: {output}")
        build_dir = args.build_dir.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        build_log = build_dir / "contact_robustness_build.log"
        returncode, _ = run_logged(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                TARGET,
                "--parallel",
            ],
            build_log,
        )
        if returncode != 0:
            raise SystemExit(f"build failed; see {build_log}")
        source_binary = resolve_binary(args.binary)
    binary_sha = sha256_file(source_binary) if source_binary.is_file() else ""
    staged_binary = output / "artifacts" / source_binary.name
    commit = command_output(["git", "rev-parse", "HEAD"])
    experiment = experiment_identity(
        profile=args.profile,
        spec=spec,
        binary_source=source_binary,
        binary=staged_binary,
        binary_sha256=binary_sha,
        commit=commit,
    )
    experiment["identity_sha256"] = canonical_sha256(experiment)
    if args.dry_run:
        commands = [
            build_command(
                staged_binary,
                condition,
                spec,
                output / "raw" / f"{condition.name}.attempt-001.csv",
            )
            for condition in spec["conditions"]
        ]
        print(
            json.dumps(
                {"experiment": experiment, "commands": commands},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not source_binary.is_file():
        raise SystemExit(f"benchmark binary does not exist: {source_binary}")
    if not output_is_safe(output):
        raise SystemExit("output inside the repository must be git-ignored")
    dirty = git_dirty()
    gpus = gpu_identity()
    if args.profile == "release" and dirty is not False:
        raise SystemExit("release evidence requires a clean worktree")
    if not gpus:
        raise SystemExit("contact robustness evidence requires an identified NVIDIA GPU")

    plan_path = output / "plan.json"
    state_path = output / "state.json"
    manifest_path = output / "manifest.json"
    if args.resume:
        if not plan_path.is_file() or not state_path.is_file():
            raise SystemExit("--resume requires existing plan.json and state.json")
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != experiment:
            raise SystemExit("refusing resume: experiment identity changed")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not staged_binary.is_file()
            or sha256_file(staged_binary) != binary_sha
        ):
            raise SystemExit("refusing resume: staged benchmark binary changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"refusing non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "raw").mkdir()
        (output / "logs").mkdir()
        (output / "artifacts").mkdir()
        shutil.copy2(source_binary, staged_binary)
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
        successful: tuple[Path, list[dict[str, str]]] | None = None
        for attempt in reversed(history):
            successful = completed_attempt(attempt, spec)
            if successful is not None:
                break
        if successful is None:
            attempt_number = len(history) + 1
            csv_path = output / "raw" / (
                f"{condition.name}.attempt-{attempt_number:03d}.csv"
            )
            log_path = output / "logs" / (
                f"{condition.name}.attempt-{attempt_number:03d}.log"
            )
            command = build_command(staged_binary, condition, spec, csv_path)
            returncode, elapsed = run_logged(command, log_path)
            errors: list[str] = []
            rows: list[dict[str, str]] = []
            if returncode == 0 and csv_path.is_file():
                try:
                    rows = load_csv(csv_path)
                    errors = validate_rows(
                        rows,
                        scenarios=spec["scenarios"],
                        planners=spec["planners"],
                        k_values=spec["k_values"],
                        seed_count=spec["seed_count"],
                    )
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

    combined_path = output / "episodes.csv"
    summary_path = output / "summary.csv"
    comparisons_path = output / "comparisons.csv"
    report_path = output / "report.md"
    write_csv(all_rows, combined_path)
    summaries, comparisons = summarize(
        all_rows,
        baseline="mppi",
        comparison_planners=spec["comparison_planners"],
        bootstrap_resamples=spec["bootstrap_resamples"],
    )
    write_csv(summaries, summary_path)
    write_csv(comparisons, comparisons_path)
    write_report(summaries, comparisons, report_path)
    significant_positive = sum(
        row["mcnemar_holm_p"] < 0.05 and row["success_delta"] > 0.0
        for row in comparisons
    )
    significant_negative = sum(
        row["mcnemar_holm_p"] < 0.05 and row["success_delta"] < 0.0
        for row in comparisons
    )
    manifest = {
        "schema_version": 1,
        "evidence_mode": "contact_robustness_gpu",
        "profile": args.profile,
        "experiment": experiment,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_dirty": dirty,
        "gpu": gpus,
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
        },
        "outcome": {
            "hypothesis_is_integrity_gate": False,
            "holm_significant_positive_success_cells": significant_positive,
            "holm_significant_negative_success_cells": significant_negative,
            "comparison_family_size": len(comparisons),
        },
        "artifacts": {
            name: {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in (
                ("episodes", combined_path),
                ("summary", summary_path),
                ("comparisons", comparisons_path),
                ("report", report_path),
                ("state", state_path),
                ("plan", plan_path),
                ("binary", staged_binary),
            )
        },
        "passed": True,
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest["outcome"], indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
