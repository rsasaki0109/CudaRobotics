#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.snapshot_design_experiments import DEFAULT_HISTORY_DIR, leader_cell, load_snapshots, problem_map


DEFAULT_CONVERGENCE_DOC = ROOT / "docs" / "convergence.md"
QUALITY_METRICS = ("Avg Regret", "Oracle Match", "Budget Hit")
STRUCTURAL_METRICS = ("Readability", "Extensibility")
VOLATILE_METRICS = {"Runtime ms/request"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render convergence signals from design-history snapshots.")
    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory containing design-history snapshots.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CONVERGENCE_DOC),
        help="Output markdown path.",
    )
    return parser.parse_args()


def metric_headers(problem: dict[str, object]) -> list[str]:
    headers = list(problem["aggregate_table"]["headers"])
    return [header for header in headers if header not in {"Variant", "Paradigm", "Source"}]


def metric_direction(metric: str) -> str:
    if metric in {"Avg Regret", "Runtime ms/request"}:
        return "min"
    return "max"


def leader_variant(problem: dict[str, object], metric: str) -> str:
    direction = metric_direction(metric)
    cell = leader_cell(problem, metric, direction)
    if cell == "-":
        return "-"
    return cell.split(" (", 1)[0]


def leader_history(problem_slug: str, snapshots: list[dict[str, object]], metric: str) -> list[str]:
    history: list[str] = []
    for snapshot in snapshots:
        problem = problem_map(snapshot).get(problem_slug)
        if problem is None:
            continue
        if metric not in metric_headers(problem):
            continue
        history.append(leader_variant(problem, metric))
    return history


def current_streak(items: list[str]) -> int:
    if not items:
        return 0
    tail = items[-1]
    streak = 0
    for item in reversed(items):
        if item != tail:
            break
        streak += 1
    return streak


def leader_change_count(items: list[str]) -> int:
    if len(items) < 2:
        return 0
    changes = 0
    previous = items[0]
    for item in items[1:]:
        if item != previous:
            changes += 1
            previous = item
    return changes


def metric_summary_rows(problem_slug: str, snapshots: list[dict[str, object]]) -> list[str]:
    latest_problem = problem_map(snapshots[-1])[problem_slug]
    metrics = metric_headers(latest_problem)
    lines: list[str] = []
    lines.append("| Metric | Current Leader | Current Streak | Leader Changes | Leaders Seen |")
    lines.append("|---|---|---:|---:|---|")
    for metric in metrics:
        history = leader_history(problem_slug, snapshots, metric)
        leaders_seen = ", ".join(dict.fromkeys(history))
        current = history[-1] if history else "-"
        lines.append(
            f"| {metric} | {current} | {current_streak(history)} | {leader_change_count(history)} | {leaders_seen or '-'} |"
        )
    return lines


def quality_signal(problem_slug: str, snapshots: list[dict[str, object]]) -> str:
    latest_problem = problem_map(snapshots[-1])[problem_slug]
    available_metrics = [metric for metric in QUALITY_METRICS if metric in metric_headers(latest_problem)]
    if not available_metrics:
        return "No quality metrics available."
    leaders = {metric: leader_history(problem_slug, snapshots, metric) for metric in available_metrics}
    current_leaders = {metric: history[-1] for metric, history in leaders.items() if history}
    unique_current = sorted(set(current_leaders.values()))
    min_streak = min(current_streak(history) for history in leaders.values())
    if len(unique_current) == 1:
        variant = unique_current[0]
        return f"Soft quality signal: `{variant}` leads all quality metrics with a current streak of `{min_streak}` snapshot(s)."
    joined = ", ".join(f"`{metric}` -> `{variant}`" for metric, variant in current_leaders.items())
    return f"Quality is still split across variants: {joined}."


def structural_signal(problem_slug: str, snapshots: list[dict[str, object]]) -> str:
    latest_problem = problem_map(snapshots[-1])[problem_slug]
    available_metrics = [metric for metric in STRUCTURAL_METRICS if metric in metric_headers(latest_problem)]
    if not available_metrics:
        return "No structural metrics available."
    current_leaders = {}
    for metric in available_metrics:
        history = leader_history(problem_slug, snapshots, metric)
        if history:
            current_leaders[metric] = history[-1]
    unique_current = sorted(set(current_leaders.values()))
    if len(unique_current) == 1:
        variant = unique_current[0]
        return f"Structural signal: `{variant}` leads the structural metrics currently tracked."
    joined = ", ".join(f"`{metric}` -> `{variant}`" for metric, variant in current_leaders.items())
    return f"Structural signals remain split: {joined}."


def promotion_note(problem_slug: str, snapshots: list[dict[str, object]]) -> str:
    latest_problem = problem_map(snapshots[-1])[problem_slug]
    snapshot_count = len(snapshots)
    quality_metrics = [metric for metric in QUALITY_METRICS if metric in metric_headers(latest_problem)]
    if not quality_metrics:
        return "Promotion status: no call; the problem does not expose the expected quality metrics yet."
    leaders = [leader_history(problem_slug, snapshots, metric) for metric in quality_metrics]
    same_variant = len({history[-1] for history in leaders if history}) == 1
    stable_quality = min(current_streak(history) for history in leaders if history)
    if snapshot_count >= 3 and same_variant and stable_quality >= 3:
        return "Promotion status: quality has stayed aligned for at least 3 snapshots; consider extracting only the parts that all active variants still share."
    return "Promotion status: keep convergence at the interface level; implementation-level promotion is still premature."


def generate_convergence_markdown(snapshots: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Convergence")
    lines.append("")
    lines.append("_Generated by `python3 scripts/render_design_convergence.py`._")
    lines.append("")
    lines.append(f"Snapshots analyzed: `{len(snapshots)}`")
    lines.append("")
    lines.append("Convergence is inferred from repeated survival, not from a single benchmark win.")
    lines.append("")
    lines.append(f"Volatile metrics kept out of promotion calls: `{', '.join(sorted(VOLATILE_METRICS))}`")
    lines.append("")

    if not snapshots:
        lines.append("No snapshots recorded yet.")
        lines.append("")
        return "\n".join(lines) + "\n"

    latest_problem_slugs = [problem["slug"] for problem in snapshots[-1]["problems"]]
    for problem_slug in latest_problem_slugs:
        latest_problem = problem_map(snapshots[-1])[problem_slug]
        lines.append(f"## {latest_problem['title']}")
        lines.append("")
        lines.append(quality_signal(problem_slug, snapshots))
        lines.append("")
        lines.append(structural_signal(problem_slug, snapshots))
        lines.append("")
        lines.append(promotion_note(problem_slug, snapshots))
        lines.append("")
        lines.append("Leader stability:")
        lines.append("")
        lines.extend(metric_summary_rows(problem_slug, snapshots))
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshots(history_dir)
    output_path.write_text(generate_convergence_markdown(snapshots))
    print(f"Generated {output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
