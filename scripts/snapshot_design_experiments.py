#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.support import problem_report_to_dict
from scripts.run_design_experiments import build_problem_reports, discover_csvs, load_rows


DEFAULT_HISTORY_DIR = ROOT / "experiments" / "history"
DEFAULT_HISTORY_DOC = ROOT / "docs" / "experiments_history.md"
HISTORY_POLICY_FILENAME = "policy.json"
LEADER_METRICS = [
    ("Avg Regret", "min"),
    ("Oracle Match", "max"),
    ("Budget Hit", "max"),
    ("Runtime ms/request", "min"),
    ("Readability", "max"),
    ("Extensibility", "max"),
]
TABLE_TEXT_COLUMNS = {"Variant", "Paradigm", "Source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot experiment-first reports and render design history docs.")
    parser.add_argument(
        "--csv",
        nargs="*",
        help="Benchmark CSV files to aggregate. Defaults to the checked-in fixture CSVs.",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=200,
        help="How many repeated recommendation passes to use for runtime benchmarking.",
    )
    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory containing design-history snapshots.",
    )
    parser.add_argument(
        "--docs-path",
        default=str(DEFAULT_HISTORY_DOC),
        help="Output path for the generated experiments history markdown.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional snapshot label to store alongside the timestamp.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Do not create a new snapshot; only regenerate the history markdown from existing snapshots.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()
    return slug


def build_snapshot_payload(csv_paths: list[Path], benchmark_iterations: int, label: str) -> dict[str, object]:
    timestamp = datetime.now().astimezone()
    snapshot_id = timestamp.strftime("%Y%m%dT%H%M%S%z")
    rows = load_rows(csv_paths)
    reports = build_problem_reports(rows, benchmark_iterations)
    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "created_at": timestamp.isoformat(timespec="seconds"),
        "label": label,
        "inputs": [str(path.relative_to(ROOT)) for path in csv_paths],
        "benchmark_iterations": benchmark_iterations,
        "problems": [problem_report_to_dict(report) for report in reports],
    }


def snapshot_filename(snapshot: dict[str, object]) -> str:
    label = str(snapshot.get("label", "")).strip()
    if label:
        return f"{snapshot['snapshot_id']}_{slugify(label)}.json"
    return f"{snapshot['snapshot_id']}.json"


def load_snapshots(history_dir: Path) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for path in sorted(history_dir.glob("*.json")):
        if path.name == HISTORY_POLICY_FILENAME:
            continue
        snapshots.append(json.loads(path.read_text()))
    return snapshots


def table_records(problem: dict[str, object]) -> list[dict[str, str]]:
    table = problem["aggregate_table"]
    headers = list(table["headers"])
    rows = list(table["rows"])
    return [
        {header: str(value) for header, value in zip(headers, row)}
        for row in rows
    ]


def numeric_headers(problem: dict[str, object]) -> list[str]:
    headers = list(problem["aggregate_table"]["headers"])
    return [header for header in headers if header not in TABLE_TEXT_COLUMNS]


def parse_metric_value(value: str) -> float:
    return float(value.replace("<runtime>", "0"))


def format_delta(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.1:
        return f"{value:+.4f}"
    return f"{value:+.3f}"


def problem_map(snapshot: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(problem["slug"]): problem for problem in snapshot["problems"]}


def variant_records(problem: dict[str, object]) -> dict[str, dict[str, str]]:
    return {record["Variant"]: record for record in table_records(problem)}


def leader_cell(problem: dict[str, object], metric: str, direction: str) -> str:
    records = table_records(problem)
    if not records or metric not in records[0]:
        return "-"

    def score(record: dict[str, str]) -> float:
        return parse_metric_value(record[metric])

    best = records[0]
    for record in records[1:]:
        if direction == "min":
            if score(record) < score(best):
                best = record
        else:
            if score(record) > score(best):
                best = record
    return f"{best['Variant']} ({best[metric]})"


def snapshot_summary_rows(snapshots: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    lines.append("| Snapshot | Created At | Label | Inputs | Problems | File |")
    lines.append("|---|---|---|---:|---:|---|")
    for snapshot in snapshots:
        label = str(snapshot.get("label", "")) or "-"
        filename = snapshot_filename(snapshot)
        lines.append(
            f"| {snapshot['snapshot_id']} | {snapshot['created_at']} | {label} | "
            f"{len(snapshot['inputs'])} | {len(snapshot['problems'])} | "
            f"`experiments/history/{filename}` |"
        )
    return lines


def problem_history_rows(problem_slug: str, snapshots: list[dict[str, object]]) -> tuple[str, list[str]]:
    title = problem_slug
    problem_snapshots: list[tuple[dict[str, object], dict[str, object]]] = []
    for snapshot in snapshots:
        for problem in snapshot["problems"]:
            if problem["slug"] == problem_slug:
                problem_snapshots.append((snapshot, problem))
                title = str(problem["title"])
                break

    lines: list[str] = []
    headers = ["Snapshot", "Requests"]
    present_metrics: list[tuple[str, str]] = []
    if problem_snapshots:
        first_problem = problem_snapshots[0][1]
        available_headers = set(first_problem["aggregate_table"]["headers"])
        present_metrics = [(metric, direction) for metric, direction in LEADER_METRICS if metric in available_headers]
        headers.extend(metric for metric, _ in present_metrics)

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for snapshot, problem in problem_snapshots:
        row = [str(snapshot["snapshot_id"]), str(problem["request_count"])]
        for metric, direction in present_metrics:
            row.append(leader_cell(problem, metric, direction))
        lines.append("| " + " | ".join(row) + " |")
    return title, lines


def latest_delta_rows(left_problem: dict[str, object], right_problem: dict[str, object]) -> list[str]:
    available_headers = set(left_problem["aggregate_table"]["headers"]) & set(right_problem["aggregate_table"]["headers"])
    present_metrics = [(metric, direction) for metric, direction in LEADER_METRICS if metric in available_headers]
    lines: list[str] = []
    lines.append("| Metric | Previous | Current | Changed |")
    lines.append("|---|---|---|---|")
    for metric, direction in present_metrics:
        previous = leader_cell(left_problem, metric, direction)
        current = leader_cell(right_problem, metric, direction)
        changed = "yes" if previous != current else "no"
        lines.append(f"| {metric} | {previous} | {current} | {changed} |")
    return lines


def latest_variant_delta_rows(left_problem: dict[str, object], right_problem: dict[str, object]) -> list[str]:
    left_records = variant_records(left_problem)
    right_records = variant_records(right_problem)
    variant_names = sorted(set(left_records) & set(right_records))
    metric_headers = [header for header in numeric_headers(right_problem) if header in left_problem["aggregate_table"]["headers"]]
    lines: list[str] = []
    lines.append("| Variant | " + " | ".join(f"Δ {header}" for header in metric_headers) + " |")
    lines.append("|" + "|".join("---" for _ in range(len(metric_headers) + 1)) + "|")
    for variant_name in variant_names:
        row = [variant_name]
        for header in metric_headers:
            left_value = parse_metric_value(left_records[variant_name][header])
            right_value = parse_metric_value(right_records[variant_name][header])
            row.append(format_delta(right_value - left_value))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def generate_snapshot_delta_lines(left_snapshot: dict[str, object], right_snapshot: dict[str, object]) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"Comparing `{left_snapshot['snapshot_id']}` -> `{right_snapshot['snapshot_id']}`."
    )
    lines.append("")
    lines.append("_Runtime deltas remain machine-dependent; quality and structure metrics are the more stable signals._")
    lines.append("")

    left_problems = problem_map(left_snapshot)
    right_problems = problem_map(right_snapshot)
    problem_slugs = list(dict.fromkeys([*left_problems.keys(), *right_problems.keys()]))

    for problem_slug in problem_slugs:
        left_problem = left_problems.get(problem_slug)
        right_problem = right_problems.get(problem_slug)
        if left_problem is None or right_problem is None:
            title = str((right_problem or left_problem)["title"])
            lines.append(f"### {title}")
            lines.append("")
            if left_problem is None:
                lines.append(f"Problem added in `{right_snapshot['snapshot_id']}`.")
            else:
                lines.append(f"Problem removed after `{left_snapshot['snapshot_id']}`.")
            lines.append("")
            continue

        title = str(right_problem["title"])
        lines.append(f"### {title}")
        lines.append("")
        lines.append(
            f"Requests: `{left_problem['request_count']} -> {right_problem['request_count']}`"
        )
        lines.append("")
        lines.append("Leader changes:")
        lines.append("")
        lines.extend(latest_delta_rows(left_problem, right_problem))
        lines.append("")
        lines.append("Variant metric deltas:")
        lines.append("")
        lines.extend(latest_variant_delta_rows(left_problem, right_problem))
        lines.append("")

    return lines


def generate_history_markdown(snapshots: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Experiments History")
    lines.append("")
    lines.append("_Generated by `python3 scripts/snapshot_design_experiments.py --render-only`._")
    lines.append("")
    lines.append(f"Snapshots tracked: `{len(snapshots)}`")
    lines.append("")
    lines.append("## Snapshots")
    lines.append("")
    if not snapshots:
        lines.append("No snapshots recorded yet.")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.extend(snapshot_summary_rows(snapshots))
    lines.append("")

    lines.append("## Latest Delta")
    lines.append("")
    if len(snapshots) < 2:
        lines.append("Add another snapshot to render inter-snapshot deltas.")
        lines.append("")
    else:
        lines.extend(generate_snapshot_delta_lines(snapshots[-2], snapshots[-1]))

    problem_slugs: list[str] = []
    seen: set[str] = set()
    for snapshot in snapshots:
        for problem in snapshot["problems"]:
            if problem["slug"] not in seen:
                seen.add(problem["slug"])
                problem_slugs.append(problem["slug"])

    lines.append("## Problem Timelines")
    lines.append("")
    for problem_slug in problem_slugs:
        title, table_lines = problem_history_rows(problem_slug, snapshots)
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(table_lines)
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    docs_path = Path(args.docs_path).resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.render_only:
        csv_paths = discover_csvs(args.csv)
        if not csv_paths:
            print("No benchmark CSVs found. Pass --csv or refresh the fixture CSVs in experiments/data/.", file=sys.stderr)
            return 1
        snapshot = build_snapshot_payload(csv_paths, args.benchmark_iterations, args.label)
        snapshot_path = history_dir / snapshot_filename(snapshot)
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"Wrote {snapshot_path.relative_to(ROOT)}")

    snapshots = load_snapshots(history_dir)
    docs_path.write_text(generate_history_markdown(snapshots))
    print(f"Generated {docs_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
