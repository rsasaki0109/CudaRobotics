#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.snapshot_design_experiments import (
    DEFAULT_HISTORY_DIR,
    HISTORY_POLICY_FILENAME,
    leader_cell,
    load_snapshots,
    parse_metric_value,
    problem_map,
    table_records,
)


DEFAULT_POLICY_PATH = ROOT / "experiments" / "history" / "policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the latest design snapshot against regression guardrails.")
    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory containing design-history snapshots.",
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="Regression policy JSON file.",
    )
    parser.add_argument(
        "--left",
        default="",
        help="Optional left snapshot id or filename. Defaults to the previous snapshot.",
    )
    parser.add_argument(
        "--right",
        default="",
        help="Optional right snapshot id or filename. Defaults to the latest snapshot.",
    )
    return parser.parse_args()


def load_policy(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise RuntimeError(f"{path.relative_to(ROOT)} must declare schema_version 1")
    problems = data.get("problems")
    if not isinstance(problems, dict) or not problems:
        raise RuntimeError(f"{path.relative_to(ROOT)} must contain a non-empty problems object")
    return data


def resolve_snapshot(selector: str, snapshots: list[dict[str, object]], history_dir: Path) -> dict[str, object]:
    for snapshot in snapshots:
        filename = snapshot["snapshot_id"]
        if selector == snapshot["snapshot_id"] or selector in {filename, f"{filename}.json"}:
            return snapshot
    for path in sorted(history_dir.glob("*.json")):
        if path.name == HISTORY_POLICY_FILENAME:
            continue
        if selector == path.name:
            data = json.loads(path.read_text())
            return data
    raise RuntimeError(f"Unknown snapshot selector: {selector}")


def best_metric(problem: dict[str, object], metric: str, direction: str) -> float:
    records = table_records(problem)
    values = [parse_metric_value(record[metric]) for record in records if metric in record]
    if not values:
        raise RuntimeError(f"{problem['slug']} is missing metric {metric}")
    if direction == "min":
        return min(values)
    if direction == "max":
        return max(values)
    raise RuntimeError(f"Unsupported direction {direction!r}")


def regression_amount(previous: float, current: float, direction: str) -> float:
    if direction == "min":
        return current - previous
    return previous - current


def main() -> int:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    policy_path = Path(args.policy).resolve()

    snapshots = load_snapshots(history_dir)
    if len(snapshots) < 2:
        print("Need at least two snapshots to check regressions.", file=sys.stderr)
        return 1

    left_snapshot = resolve_snapshot(args.left, snapshots, history_dir) if args.left else snapshots[-2]
    right_snapshot = resolve_snapshot(args.right, snapshots, history_dir) if args.right else snapshots[-1]
    policy = load_policy(policy_path)

    left_problems = problem_map(left_snapshot)
    right_problems = problem_map(right_snapshot)
    problems = policy["problems"]

    failures: list[str] = []
    notes: list[str] = []

    for problem_slug, config in problems.items():
        if problem_slug not in left_problems or problem_slug not in right_problems:
            failures.append(f"{problem_slug}: missing from one of the compared snapshots")
            continue

        left_problem = left_problems[problem_slug]
        right_problem = right_problems[problem_slug]
        title = str(right_problem["title"])
        require_same_request_count = bool(config.get("require_same_request_count", False))
        if require_same_request_count and left_problem["request_count"] != right_problem["request_count"]:
            failures.append(
                f"{problem_slug}: request count changed {left_problem['request_count']} -> {right_problem['request_count']}"
            )

        metric_rules = config.get("metrics", {})
        if not isinstance(metric_rules, dict) or not metric_rules:
            failures.append(f"{problem_slug}: metrics policy is missing or invalid")
            continue

        for metric, rule in metric_rules.items():
            if not isinstance(rule, dict):
                failures.append(f"{problem_slug}/{metric}: metric rule must be an object")
                continue
            direction = str(rule.get("direction", ""))
            max_regression = float(rule.get("max_regression", 0.0))
            previous = best_metric(left_problem, metric, direction)
            current = best_metric(right_problem, metric, direction)
            regression = regression_amount(previous, current, direction)
            previous_leader = leader_cell(left_problem, metric, direction)
            current_leader = leader_cell(right_problem, metric, direction)
            notes.append(
                f"{title} / {metric}: {previous_leader} -> {current_leader} "
                f"(regression={regression:+.4f}, limit={max_regression:.4f})"
            )
            if regression > max_regression + 1.0e-12:
                failures.append(
                    f"{title} / {metric}: regression {regression:+.4f} exceeded limit {max_regression:.4f}"
                )

    print(f"Comparing {left_snapshot['snapshot_id']} -> {right_snapshot['snapshot_id']}")
    for note in notes:
        print(f"OK {note}")

    if failures:
        print("Regression check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Design regression check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
