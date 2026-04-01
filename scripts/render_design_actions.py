#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_design_convergence import (
    DEFAULT_HISTORY_DIR,
    QUALITY_METRICS,
    STRUCTURAL_METRICS,
    current_streak,
    leader_history,
    metric_headers,
)
from scripts.snapshot_design_experiments import load_snapshots, problem_map


DEFAULT_ACTIONS_DOC = ROOT / "docs" / "next_actions.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render next-action suggestions from design-history snapshots.")
    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory containing design-history snapshots.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ACTIONS_DOC),
        help="Output markdown path.",
    )
    return parser.parse_args()


def quality_state(problem_slug: str, snapshots: list[dict[str, object]]) -> tuple[list[str], int]:
    latest_problem = problem_map(snapshots[-1])[problem_slug]
    quality_metrics = [metric for metric in QUALITY_METRICS if metric in metric_headers(latest_problem)]
    histories = [leader_history(problem_slug, snapshots, metric) for metric in quality_metrics]
    leaders = [history[-1] for history in histories if history]
    streak = min((current_streak(history) for history in histories if history), default=0)
    return leaders, streak


def structural_state(problem_slug: str, snapshots: list[dict[str, object]]) -> tuple[list[str], int]:
    latest_problem = problem_map(snapshots[-1])[problem_slug]
    structural_metrics = [metric for metric in STRUCTURAL_METRICS if metric in metric_headers(latest_problem)]
    histories = [leader_history(problem_slug, snapshots, metric) for metric in structural_metrics]
    leaders = [history[-1] for history in histories if history]
    streak = min((current_streak(history) for history in histories if history), default=0)
    return leaders, streak


def classify_action(problem_slug: str, snapshots: list[dict[str, object]]) -> tuple[str, str, list[str]]:
    latest_problem = problem_map(snapshots[-1])[problem_slug]
    snapshot_count = len(snapshots)
    quality_leaders, quality_streak = quality_state(problem_slug, snapshots)
    structural_leaders, structural_streak = structural_state(problem_slug, snapshots)

    unique_quality = sorted(set(quality_leaders))
    unique_structural = sorted(set(structural_leaders))

    if not quality_leaders:
        return (
            "needs_metrics",
            "Add missing quality metrics before making design calls.",
            [
                "Expose at least one quality metric in the generated aggregate table.",
                "Keep all current variants alive until the missing metrics exist.",
            ],
        )

    if len(unique_quality) > 1:
        return (
            "diversify",
            "Quality leadership is still split across variants.",
            [
                "Keep all current variants alive.",
                "Add one more discriminating request set or fixture so the split is not hidden behind the current inputs.",
            ],
        )

    quality_winner = unique_quality[0]

    if snapshot_count < 3 or quality_streak < 3:
        return (
            "hold",
            f"`{quality_winner}` is leading on quality, but the streak is still short.",
            [
                "Keep convergence at the interface level.",
                "Take another snapshot after the next meaningful fixture or implementation change.",
            ],
        )

    if len(unique_structural) > 1:
        return (
            "promotion_watch",
            f"`{quality_winner}` has a stable quality lead, but structural metrics still favor different variants.",
            [
                "Do not promote one implementation into `core/`.",
                f"Inspect `{quality_winner}` for helpers that are now shared implicitly and extract only those if at least two variants can use them.",
            ],
        )

    structural_winner = unique_structural[0]
    if structural_winner != quality_winner:
        return (
            "hold",
            f"Quality favors `{quality_winner}`, while structure favors `{structural_winner}`.",
            [
                "Keep both variants alive.",
                "Prefer extracting narrow helpers over converging on a single implementation.",
            ],
        )

    if structural_streak >= 3:
        return (
            "promote_shared_helpers",
            f"`{quality_winner}` is stable on both quality and structure.",
            [
                "Consider promoting only the helper layer that other live variants can adopt without increasing the interface surface.",
                "Keep at least one alternative implementation alive after the extraction.",
            ],
        )

    return (
        "hold",
        f"`{quality_winner}` leads quality and structure now, but the structural streak is still short.",
        [
            "Delay implementation promotion.",
            "Wait for one more stable snapshot before extracting anything into `core/`.",
        ],
    )


def action_rows(snapshots: list[dict[str, object]]) -> list[str]:
    latest_problems = problem_map(snapshots[-1])
    lines: list[str] = []
    lines.append("| Problem | Action | Reason |")
    lines.append("|---|---|---|")
    for problem_slug, problem in latest_problems.items():
        action, reason, _ = classify_action(problem_slug, snapshots)
        lines.append(f"| {problem['title']} | `{action}` | {reason} |")
    return lines


def generate_actions_markdown(snapshots: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Next Actions")
    lines.append("")
    lines.append("_Generated by `python3 scripts/render_design_actions.py`._")
    lines.append("")
    lines.append("This document turns snapshot history into the next process move. It does not pick a final implementation by default.")
    lines.append("")

    if not snapshots:
        lines.append("No snapshots recorded yet.")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append("## Summary")
    lines.append("")
    lines.extend(action_rows(snapshots))
    lines.append("")

    latest_problems = problem_map(snapshots[-1])
    for problem_slug, problem in latest_problems.items():
        action, reason, suggestions = classify_action(problem_slug, snapshots)
        lines.append(f"## {problem['title']}")
        lines.append("")
        lines.append(f"Recommended action: `{action}`")
        lines.append("")
        lines.append(reason)
        lines.append("")
        lines.append("Suggested next moves:")
        lines.append("")
        for suggestion in suggestions:
            lines.append(f"- {suggestion}")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshots(history_dir)
    output_path.write_text(generate_actions_markdown(snapshots))
    print(f"Generated {output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
