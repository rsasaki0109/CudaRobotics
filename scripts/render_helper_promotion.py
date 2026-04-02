#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.snapshot_design_experiments import DEFAULT_HISTORY_DIR, load_snapshots, problem_map


DEFAULT_OUTPUT = ROOT / "docs" / "helper_promotion.md"
DEFAULT_POLICY = ROOT / "experiments" / "history" / "helper_policy.json"


@dataclass(frozen=True)
class HelperUsage:
    helper: str
    files: tuple[str, ...]
    problems: tuple[str, ...]
    min_problem_snapshots: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render helper-promotion watch signals from current experiment usage.")
    parser.add_argument(
        "--history-dir",
        default=str(DEFAULT_HISTORY_DIR),
        help="Directory containing design-history snapshots.",
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY),
        help="Helper promotion policy JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output markdown path.",
    )
    return parser.parse_args()


def load_policy(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise RuntimeError(f"{path.relative_to(ROOT)} must declare schema_version 1")
    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise RuntimeError(f"{path.relative_to(ROOT)} must contain thresholds")
    return data


def experiment_variant_files() -> list[Path]:
    paths: list[Path] = []
    for path in sorted((ROOT / "experiments").glob("*/*.py")):
        if path.name in {"__init__.py", "common.py", "support.py"}:
            continue
        paths.append(path)
    return paths


def support_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "experiments.support":
            imports.extend(alias.name for alias in node.names)
    return imports


def problem_snapshot_counts(history_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for snapshot in load_snapshots(history_dir):
        for slug in problem_map(snapshot):
            counts[slug] = counts.get(slug, 0) + 1
    return counts


def collect_helper_usage(history_dir: Path) -> list[HelperUsage]:
    snapshot_counts = problem_snapshot_counts(history_dir)
    helper_files: dict[str, set[str]] = {}
    helper_problems: dict[str, set[str]] = {}

    for path in experiment_variant_files():
        imports = support_imports(path)
        if not imports:
            continue
        problem_slug = path.parent.name
        relative = str(path.relative_to(ROOT))
        for helper in imports:
            helper_files.setdefault(helper, set()).add(relative)
            helper_problems.setdefault(helper, set()).add(problem_slug)

    usages: list[HelperUsage] = []
    for helper in sorted(helper_files):
        problems = tuple(sorted(helper_problems[helper]))
        min_snapshots = min(snapshot_counts.get(problem, 0) for problem in problems) if problems else 0
        usages.append(
            HelperUsage(
                helper=helper,
                files=tuple(sorted(helper_files[helper])),
                problems=problems,
                min_problem_snapshots=min_snapshots,
            )
        )
    return usages


def classify_usage(usage: HelperUsage, policy: dict[str, object]) -> tuple[str, str]:
    thresholds = policy["thresholds"]
    problem_count = len(usage.problems)
    variant_count = len(usage.files)
    snapshot_count = usage.min_problem_snapshots

    core = thresholds["core_candidate"]
    if (
        problem_count >= int(core["min_problem_count"])
        and variant_count >= int(core["min_variant_count"])
        and snapshot_count >= int(core["min_problem_snapshots"])
    ):
        return (
            "core_candidate",
            "shared across enough problems and variants to justify a core-promotion review",
        )

    watch = thresholds["promotion_watch"]
    if (
        problem_count >= int(watch["min_problem_count"])
        and variant_count >= int(watch["min_variant_count"])
        and snapshot_count >= int(watch["min_problem_snapshots"])
    ):
        return (
            "promotion_watch",
            "shared across multiple problems and has survived enough snapshots to watch closely",
        )

    return (
        "keep_in_experiments",
        "still too local or too young to promote beyond the experiment layer",
    )


def generate_markdown(usages: list[HelperUsage], policy: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Helper Promotion")
    lines.append("")
    lines.append("_Generated by `python3 scripts/render_helper_promotion.py`._")
    lines.append("")
    lines.append("This report watches shared helper extraction without auto-promoting helpers into `core/`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Helper | Problems | Variant Files | Min Problem Snapshots | Classification |")
    lines.append("|---|---:|---:|---:|---|")
    for usage in usages:
        classification, _ = classify_usage(usage, policy)
        lines.append(
            f"| `{usage.helper}` | {len(usage.problems)} | {len(usage.files)} | {usage.min_problem_snapshots} | `{classification}` |"
        )
    lines.append("")

    lines.append("## Details")
    lines.append("")
    for usage in usages:
        classification, reason = classify_usage(usage, policy)
        lines.append(f"### `{usage.helper}`")
        lines.append("")
        lines.append(f"Classification: `{classification}`")
        lines.append("")
        lines.append(reason)
        lines.append("")
        lines.append(f"Problems: `{', '.join(usage.problems) or '-'}`")
        lines.append("")
        lines.append(f"Min snapshots across those problems: `{usage.min_problem_snapshots}`")
        lines.append("")
        lines.append("Imported by:")
        lines.append("")
        for file_path in usage.files:
            lines.append(f"- `{file_path}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    output_path = Path(args.output).resolve()
    policy_path = Path(args.policy).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy = load_policy(policy_path)
    usages = collect_helper_usage(history_dir)
    output_path.write_text(generate_markdown(usages, policy))
    print(f"Generated {output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
