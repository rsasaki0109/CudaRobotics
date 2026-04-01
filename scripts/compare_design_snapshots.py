#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.snapshot_design_experiments import DEFAULT_HISTORY_DIR, generate_snapshot_delta_lines, load_snapshots, snapshot_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a markdown comparison between two design snapshots.")
    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory containing design-history snapshots.",
    )
    parser.add_argument(
        "--left",
        default="",
        help="Left snapshot id or filename. Defaults to the previous snapshot.",
    )
    parser.add_argument(
        "--right",
        default="",
        help="Right snapshot id or filename. Defaults to the latest snapshot.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output markdown path. Defaults to stdout.",
    )
    return parser.parse_args()


def resolve_snapshot(selector: str, snapshots: list[dict[str, object]]) -> dict[str, object]:
    if not selector:
        raise ValueError("empty snapshot selector")
    for snapshot in snapshots:
        if selector == snapshot["snapshot_id"] or selector == snapshot_filename(snapshot):
            return snapshot
    raise RuntimeError(f"Unknown snapshot selector: {selector}")


def comparison_markdown(left_snapshot: dict[str, object], right_snapshot: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Snapshot Comparison")
    lines.append("")
    lines.extend(generate_snapshot_delta_lines(left_snapshot, right_snapshot))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    snapshots = load_snapshots(history_dir)
    if len(snapshots) < 2:
        print("Need at least two snapshots to compare.", file=sys.stderr)
        return 1

    left_snapshot = resolve_snapshot(args.left, snapshots) if args.left else snapshots[-2]
    right_snapshot = resolve_snapshot(args.right, snapshots) if args.right else snapshots[-1]
    markdown = comparison_markdown(left_snapshot, right_snapshot)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
        print(f"Generated {output_path.relative_to(ROOT)}")
        return 0

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
