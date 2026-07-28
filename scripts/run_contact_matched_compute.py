#!/usr/bin/env python3
"""Run calibrated, deadline-matched contact-control evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

from contact_matched_compute import (
    profile_spec,
    select_largest_feasible_k,
    summarize_evaluation,
    validate_rows,
    write_csv,
    write_report,
)
from contact_robustness import load_csv
from run_contact_robustness import (
    canonical_sha256,
    output_is_safe,
    resolve_binary,
    run_logged,
    sha256_file,
    write_json_atomic,
)
from run_cudanav_closed_loop import command_output, git_dirty, gpu_identity


ROOT = Path(__file__).resolve().parents[1]
TARGET = "benchmark_diff_mppi_pushing_box"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("bin") / TARGET)
    parser.add_argument("--profile", choices=("smoke", "release"), default="smoke")
    parser.add_argument("--deadline-ms", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def benchmark_command(
    binary: Path,
    spec: dict[str, Any],
    *,
    planner: str,
    phase: str,
    csv_path: Path,
    selected_k: int | None = None,
) -> list[str]:
    k_values = (
        spec["k_candidates"]
        if phase == "calibration"
        else (selected_k,)
    )
    return [
        str(binary),
        "--scenarios",
        ",".join(spec["scenarios"]),
        "--planners",
        planner,
        "--k-values",
        ",".join(str(value) for value in k_values),
        "--seed-count",
        str(spec[f"{phase}_seed_count"]),
        "--seed-offset",
        str(spec[f"{phase}_seed_offset"]),
        "--horizon",
        str(spec["horizon"]),
        "--control-deadline-ms",
        f"{spec['deadline_ms']:g}",
        "--csv",
        str(csv_path),
    ]


def experiment_identity(
    profile: str,
    spec: dict[str, Any],
    source_binary: Path,
    staged_binary: Path,
    binary_sha256: str,
    commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": profile,
        "git_commit": commit,
        "binary_source": str(source_binary),
        "binary": str(staged_binary),
        "binary_sha256": binary_sha256,
        "scenarios": list(spec["scenarios"]),
        "planners": list(spec["planners"]),
        "k_candidates": list(spec["k_candidates"]),
        "calibration_seed_count": spec["calibration_seed_count"],
        "calibration_seed_offset": spec["calibration_seed_offset"],
        "evaluation_seed_count": spec["evaluation_seed_count"],
        "evaluation_seed_offset": spec["evaluation_seed_offset"],
        "deadline_ms": spec["deadline_ms"],
        "horizon": spec["horizon"],
        "bootstrap_resamples": spec["bootstrap_resamples"],
    }


def _run_spec(spec: dict[str, Any], planner: str) -> dict[str, Any]:
    result = deepcopy(spec)
    result["planners"] = (planner,)
    return result


def _attempt_rows(
    attempt: dict[str, Any],
    *,
    spec: dict[str, Any],
    planner: str,
    phase: str,
    selected_k: int | None,
) -> list[dict[str, str]] | None:
    if not attempt.get("passed"):
        return None
    path = Path(attempt.get("csv", ""))
    if not path.is_file() or sha256_file(path) != attempt.get("csv_sha256"):
        return None
    rows = load_csv(path)
    run_spec = _run_spec(spec, planner)
    selection = {planner: selected_k} if selected_k is not None else None
    errors = validate_rows(
        rows,
        spec=run_spec,
        phase=phase,
        selected_k=selection,
    )
    return rows if not errors else None


def _execute_phase(
    *,
    output: Path,
    state: dict[str, Any],
    state_path: Path,
    binary: Path,
    spec: dict[str, Any],
    phase: str,
    selected_k: dict[str, int] | None = None,
) -> tuple[list[dict[str, str]], bool]:
    all_rows: list[dict[str, str]] = []
    all_passed = True
    for planner in spec["planners"]:
        key = f"{phase}:{planner}"
        history = state["runs"].setdefault(key, [])
        chosen_k = selected_k[planner] if selected_k is not None else None
        rows: list[dict[str, str]] | None = None
        for attempt in reversed(history):
            rows = _attempt_rows(
                attempt,
                spec=spec,
                planner=planner,
                phase=phase,
                selected_k=chosen_k,
            )
            if rows is not None:
                break
        if rows is None:
            attempt_number = len(history) + 1
            stem = f"{phase}_{planner}.attempt-{attempt_number:03d}"
            csv_path = output / "raw" / f"{stem}.csv"
            log_path = output / "logs" / f"{stem}.log"
            command = benchmark_command(
                binary,
                spec,
                planner=planner,
                phase=phase,
                csv_path=csv_path,
                selected_k=chosen_k,
            )
            returncode, elapsed = run_logged(command, log_path)
            errors: list[str] = []
            candidate_rows: list[dict[str, str]] = []
            if returncode == 0 and csv_path.is_file():
                try:
                    candidate_rows = load_csv(csv_path)
                    run_spec = _run_spec(spec, planner)
                    selection = (
                        {planner: chosen_k} if chosen_k is not None else None
                    )
                    errors = validate_rows(
                        candidate_rows,
                        spec=run_spec,
                        phase=phase,
                        selected_k=selection,
                    )
                except (OSError, csv.Error, ValueError) as exception:
                    errors = [str(exception)]
            else:
                errors = [f"benchmark return code {returncode} or missing CSV"]
            attempt = {
                "attempt": attempt_number,
                "phase": phase,
                "planner": planner,
                "selected_k": chosen_k,
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
                rows = candidate_rows
        if rows is None:
            all_passed = False
        else:
            all_rows.extend(rows)
    return all_rows, all_passed


def main() -> int:
    args = parse_args()
    spec = profile_spec(args.profile)
    if args.deadline_ms is not None:
        if not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0.0:
            raise SystemExit("--deadline-ms must be finite and positive")
        spec["deadline_ms"] = args.deadline_ms
    output = args.output_dir.resolve()
    source_binary = resolve_binary(args.binary)
    binary_sha = sha256_file(source_binary) if source_binary.is_file() else ""
    staged_binary = output / "artifacts" / source_binary.name
    commit = command_output(["git", "rev-parse", "HEAD"])
    experiment = experiment_identity(
        args.profile,
        spec,
        source_binary,
        staged_binary,
        binary_sha,
        commit,
    )
    experiment["identity_sha256"] = canonical_sha256(experiment)
    if args.dry_run:
        commands = [
            benchmark_command(
                staged_binary,
                spec,
                planner=planner,
                phase="calibration",
                csv_path=output / "raw" / f"calibration_{planner}.attempt-001.csv",
            )
            for planner in spec["planners"]
        ]
        print(
            json.dumps(
                {
                    "experiment": experiment,
                    "calibration_commands": commands,
                    "evaluation_commands": "selected K is derived from calibration",
                },
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
        raise SystemExit("matched-compute evidence requires an identified NVIDIA GPU")

    plan_path = output / "plan.json"
    state_path = output / "state.json"
    manifest_path = output / "manifest.json"
    if args.resume:
        if not plan_path.is_file() or not state_path.is_file():
            raise SystemExit("--resume requires plan.json and state.json")
        if json.loads(plan_path.read_text(encoding="utf-8")) != experiment:
            raise SystemExit("refusing resume: experiment identity changed")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not staged_binary.is_file() or sha256_file(staged_binary) != binary_sha:
            raise SystemExit("refusing resume: staged benchmark binary changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"refusing non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        for directory in ("raw", "logs", "artifacts"):
            (output / directory).mkdir()
        shutil.copy2(source_binary, staged_binary)
        write_json_atomic(plan_path, experiment)
        state = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "runs": {},
        }
        write_json_atomic(state_path, state)

    calibration_rows, calibration_passed = _execute_phase(
        output=output,
        state=state,
        state_path=state_path,
        binary=staged_binary,
        spec=spec,
        phase="calibration",
    )
    if not calibration_passed:
        print("calibration incomplete; rerun with --resume")
        return 1
    selected_k, calibration_table = select_largest_feasible_k(
        calibration_rows, spec
    )
    calibration_episodes_path = output / "calibration_episodes.csv"
    calibration_path = output / "calibration.csv"
    write_csv(calibration_rows, calibration_episodes_path)
    write_csv(calibration_table, calibration_path)

    evaluation_rows, evaluation_passed = _execute_phase(
        output=output,
        state=state,
        state_path=state_path,
        binary=staged_binary,
        spec=spec,
        phase="evaluation",
        selected_k=selected_k,
    )
    if not evaluation_passed:
        print("evaluation incomplete; rerun with --resume")
        return 1
    evaluation_path = output / "evaluation_episodes.csv"
    summary_path = output / "summary.csv"
    comparisons_path = output / "comparisons.csv"
    report_path = output / "report.md"
    write_csv(evaluation_rows, evaluation_path)
    summaries, comparisons = summarize_evaluation(
        evaluation_rows, spec, selected_k
    )
    write_csv(summaries, summary_path)
    write_csv(comparisons, comparisons_path)
    write_report(selected_k, summaries, comparisons, report_path)

    artifacts = {
        name: {
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in (
            ("calibration_episodes", calibration_episodes_path),
            ("calibration", calibration_path),
            ("evaluation_episodes", evaluation_path),
            ("summary", summary_path),
            ("comparisons", comparisons_path),
            ("report", report_path),
            ("state", state_path),
            ("plan", plan_path),
            ("binary", staged_binary),
        )
    }
    manifest = {
        "schema_version": 1,
        "evidence_mode": "contact_matched_compute_gpu",
        "profile": args.profile,
        "experiment": experiment,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_dirty": dirty,
        "gpu": gpus,
        "selected_k": selected_k,
        "matrix": {
            "calibration_episodes": len(calibration_rows),
            "evaluation_episodes": len(evaluation_rows),
            "scenarios": len(spec["scenarios"]),
            "planners": len(spec["planners"]),
            "evaluation_seeds": spec["evaluation_seed_count"],
        },
        "integrity_gate": {
            "calibration_complete": True,
            "held_out_evaluation_complete": True,
            "calibration_evaluation_seeds_disjoint": (
                spec["calibration_seed_offset"] + spec["calibration_seed_count"]
                <= spec["evaluation_seed_offset"]
            ),
            "all_selected_budgets_zero_miss_in_calibration": True,
            "clean_worktree": dirty is False,
            "gpu_identified": bool(gpus),
        },
        "artifacts": artifacts,
        "passed": True,
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"selected_k": selected_k, "matrix": manifest["matrix"]}, indent=2))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
