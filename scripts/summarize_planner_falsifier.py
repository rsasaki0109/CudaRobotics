#!/usr/bin/env python3
"""Render GPU planner falsifier JSON into a compact Markdown report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_JSON = "gif/gpu_planner_falsifier_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize gpu_planner_falsifier_benchmark JSON output.")
    parser.add_argument("--json", default=DEFAULT_JSON,
                        help=f"Input falsifier JSON path (default: {DEFAULT_JSON})")
    parser.add_argument("--markdown-out",
                        help="Optional Markdown report output path.")
    parser.add_argument("--strict", action="store_true",
                        help="Return non-zero if the falsifier gate misses.")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if payload.get("benchmark") != "gpu_planner_falsifier_benchmark":
        raise ValueError(f"unexpected benchmark field in {path}")
    if "falsifier_gate" not in payload or "worst_cases" not in payload:
        raise ValueError(f"missing falsifier_gate or worst_cases in {path}")
    return payload


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def pass_text(value: bool) -> str:
    return "PASS" if value else "MISS"


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(payload: dict[str, Any], source: Path) -> str:
    gate = payload["falsifier_gate"]
    runtime = payload["runtime"]
    worst = payload["worst_learned"]
    cases = payload["worst_cases"]
    top_k = int(payload["top_k"])

    lines: list[str] = []
    lines.append("# GPU Planner Falsifier")
    lines.append("")
    lines.append(f"- Source: `{source}`")
    lines.append(f"- Candidates scanned: {int(payload['candidates_scanned'])}")
    lines.append(f"- Worst-K target: **{pass_text(bool(gate['target_pass']))}**")
    lines.append(
        "- Falsifier gate: "
        f"no-pressure fails {int(gate['no_pressure_failures'])}/{top_k}, "
        f"no-regret fails {int(gate['no_regret_failures'])}/{top_k}, "
        f"learned passes {int(gate['learned_passes'])}/{top_k}, "
        f"extra evaluated {int(gate['extra_evaluated'])}, "
        f"accepted {int(gate['accepted_extra'])}")
    lines.append(
        "- Worst learned target row: "
        f"CVaR {fmt_float(float(worst['collision_cvar']))}, "
        f"residual {fmt_float(float(worst['residual_pct']))}%, "
        f"runtime {fmt_float(float(worst['runtime_ms']), 3)} ms")
    lines.append(
        "- Runtime: "
        f"{fmt_float(float(runtime['gpu_ms']), 3)} ms GPU, "
        f"{fmt_float(float(runtime['speedup']), 1)}x vs CPU surrogate")
    lines.append("")

    lines.append("## Worst Cases")
    lines.append("")
    lines.append("| Rank | Score | Lane | Jitter | Shift | Phase | Goal | Flip | Pressure | No-pressure | Learned target | Budget |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|")
    for rank, row in enumerate(cases, start=1):
        no_pressure = row["no_pressure"]
        learned = row["learned_target"]
        budget = row["adaptive_budget"]
        budget_cell = "accepted" if budget["accepted_extra"] else (
            "eval-only" if budget["extra_evaluated"] else "fixed")
        lines.append(
            f"| {rank} | {fmt_float(float(row['rank_score']))} | "
            f"{fmt_float(float(row['lane_scale']), 2)} | "
            f"{fmt_float(float(row['jitter_scale']), 2)} | "
            f"{fmt_float(float(row['cross_shift']), 2)} | "
            f"{fmt_float(float(row['spawn_phase']), 2)} | "
            f"{fmt_float(float(row['goal_offset']), 2)} | "
            f"{bool_text(bool(row['priority_flip']))} | "
            f"{fmt_float(float(row['scenario_pressure']), 2)} | "
            f"{int(no_pressure['collisions'])} coll, CVaR {fmt_float(float(no_pressure['collision_cvar']))} | "
            f"{int(learned['collisions'])} coll, CVaR {fmt_float(float(learned['collision_cvar']))}, "
            f"{fmt_float(float(learned['residual_pct']))}% | "
            f"{budget_cell} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    source = Path(args.json)
    try:
        payload = load_payload(source)
        report = render_markdown(payload, source)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.markdown_out:
        out_path = Path(args.markdown_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n")
        print(f"Wrote {out_path}")
    else:
        print(report)

    if args.strict and not payload["falsifier_gate"].get("target_pass", False):
        print("falsifier gate failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
