#!/usr/bin/env python3
"""Aggregate opt-in onboarding result files without user identifiers."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


INITIAL_SURFACES = {"colab_quickstart", "python_quickstart", "ros2_cudanav"}
CONTINUATION_SURFACES = {"python_planning_variant", "dlpack_costmap"}
INTEGRATION_SURFACES = {"dlpack_costmap", "ros2_cudanav"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--cohort", required=True, help="stable label such as 2026-W35")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def surface_name(payload: dict) -> str:
    if (
        payload.get("surface") == "python_quickstart"
        and payload.get("recipe") == "planning_variant"
    ):
        return "python_planning_variant"
    if payload.get("surface") in {"colab_quickstart", "python_quickstart"}:
        return payload["surface"]
    if payload.get("recipe") == "dlpack_costmap":
        return "dlpack_costmap"
    if payload.get("profile") in {"smoke", "release"} and "summary_gate" in payload:
        return "ros2_cudanav"
    if "mppi" in payload and "registration" in payload:
        return "colab_quickstart"
    raise ValueError("unrecognized onboarding result schema")


def duration_seconds(payload: dict) -> float | None:
    duration = payload.get("duration_seconds")
    if isinstance(duration, (int, float)) and duration >= 0:
        return float(duration)
    started = payload.get("started_at")
    finished = payload.get("finished_at")
    if isinstance(started, str) and isinstance(finished, str):
        try:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            finish_dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        except ValueError:
            return None
        elapsed = (finish_dt - start_dt).total_seconds()
        return elapsed if elapsed >= 0 else None
    return None


def percentile_nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def load_attempt(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError("passed must be boolean")
    return {
        "surface": surface_name(payload),
        "passed": payload["passed"],
        "failure_category": payload.get("failure_category"),
        "duration_seconds": duration_seconds(payload),
    }


def summarize(paths: list[Path], cohort: str) -> dict:
    attempts = [load_attempt(path) for path in paths]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt["surface"]].append(attempt)

    surfaces = {}
    for surface in sorted(grouped):
        rows = grouped[surface]
        passed = sum(row["passed"] for row in rows)
        failures = Counter(
            row["failure_category"] or "unclassified"
            for row in rows
            if not row["passed"]
        )
        durations = [
            row["duration_seconds"]
            for row in rows
            if row["passed"] and row["duration_seconds"] is not None
        ]
        surfaces[surface] = {
            "attempts": len(rows),
            "passed": passed,
            "activation_rate": round(passed / len(rows), 4),
            "duration_samples": len(durations),
            "duration_seconds_median": round(statistics.median(durations), 3)
            if durations
            else None,
            "duration_seconds_p90": percentile_nearest_rank(durations, 0.9),
            "failure_categories": dict(sorted(failures.items())),
        }

    failed_rows = [attempt for attempt in attempts if not attempt["passed"]]
    known_failures = sum(
        attempt["failure_category"] not in {None, "", "unknown"}
        for attempt in failed_rows
    )
    initial = [attempt for attempt in attempts if attempt["surface"] in INITIAL_SURFACES]
    initial_passed = sum(attempt["passed"] for attempt in initial)
    continuation_completions = sum(
        attempt["passed"] and attempt["surface"] in CONTINUATION_SURFACES
        for attempt in attempts
    )
    integration_completions = sum(
        attempt["passed"] and attempt["surface"] in INTEGRATION_SURFACES
        for attempt in attempts
    )
    return {
        "schema_version": 1,
        "cohort": cohort,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attempts": len(attempts),
        "initial_attempts": len(initial),
        "initial_activations": initial_passed,
        "initial_activation_rate": round(initial_passed / len(initial), 4)
        if initial
        else None,
        "failed_attempts": len(failed_rows),
        "known_failure_classification_rate": round(known_failures / len(failed_rows), 4)
        if failed_rows
        else None,
        "continuation_recipe_completion_proxy": continuation_completions,
        "integration_completion_proxy": integration_completions,
        "surfaces": surfaces,
        "limitations": [
            "inputs are opt-in result files, not all starts",
            "continuation recipe completions are not a user retention rate",
            "no user identifier or cross-run identity is collected",
        ],
    }


def render_markdown(summary: dict) -> str:
    lines = [
        f"# Adoption cohort {summary['cohort']}",
        "",
        f"- Attempts: {summary['attempts']}",
        f"- Initial activations: {summary['initial_activations']}/{summary['initial_attempts']}",
        f"- Continuation recipe completion proxy: {summary['continuation_recipe_completion_proxy']}",
        f"- Integration completion proxy: {summary['integration_completion_proxy']}",
        "",
        "| Surface | Attempts | Passed | Activation | TTFS median s | TTFS p90 s | Failures |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for surface, values in summary["surfaces"].items():
        failures = ", ".join(
            f"{name}={count}" for name, count in values["failure_categories"].items()
        ) or "-"
        activation = f"{values['activation_rate']:.1%}"
        median = values["duration_seconds_median"]
        p90 = values["duration_seconds_p90"]
        lines.append(
            f"| {surface} | {values['attempts']} | {values['passed']} | {activation} | "
            f"{median if median is not None else '-'} | {p90 if p90 is not None else '-'} | {failures} |"
        )
    lines.extend(
        [
            "",
            "> This report contains opt-in artifact counts. The continuation proxy is not a user retention rate.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize([path.resolve() for path in args.results], args.cohort)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"cannot summarize adoption results: {exc}", file=sys.stderr)
        return 1
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(
        f"Adoption cohort {summary['cohort']}: "
        f"{summary['initial_activations']}/{summary['initial_attempts']} initial activations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
