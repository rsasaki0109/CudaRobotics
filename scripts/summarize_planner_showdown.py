#!/usr/bin/env python3
"""Render GPU planner showdown JSON into a compact Markdown report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_JSON = "gif/gpu_planner_showdown_benchmark.json"
DEFAULT_TARGET = "Trainable safety-dual MPPI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize gpu_planner_showdown_benchmark JSON output.")
    parser.add_argument("--json", action="append", dest="json_paths",
                        help=("Input showdown JSON path. Repeat to render a scenario "
                              f"matrix (default: {DEFAULT_JSON})"))
    parser.add_argument("--markdown-out",
                        help="Optional Markdown report output path.")
    parser.add_argument("--target-planner", default=DEFAULT_TARGET,
                        help=f"Planner row to treat as the target (default: {DEFAULT_TARGET})")
    parser.add_argument("--strict", action="store_true",
                        help="Return non-zero if the target planner misses the hard gates.")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if payload.get("benchmark") != "gpu_planner_showdown_benchmark":
        raise ValueError(f"unexpected benchmark field in {path}")
    if "hard_target" not in payload or "planners" not in payload:
        raise ValueError(f"missing hard_target or planners in {path}")
    return payload


def find_row(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for row in payload["planners"]:
        if row.get("name") == name:
            return row
    raise ValueError(f"planner row not found: {name}")


def maybe_row(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in payload["planners"]:
        if row.get("name") == name:
            return row
    return None


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_residual(value: float) -> str:
    if value < 0:
        return "n/a"
    return f"{value:.2f}%"


def pass_text(value: bool) -> str:
    return "PASS" if value else "MISS"


def pressure_mode(payload: dict[str, Any]) -> str:
    pressure = payload.get("pressure_controller") or {}
    return str(payload.get("pressure_mode") or pressure.get("mode") or "learned")


def pressure_controller_line(payload: dict[str, Any]) -> str | None:
    mode = pressure_mode(payload)
    if mode == "teacher":
        return "teacher safety-pressure formula"
    if mode == "none":
        return "none, pressure multiplier bypassed"
    pressure = payload.get("pressure_controller")
    if not pressure:
        return None
    return (
        "learned safety-pressure, "
        f"{int(pressure['samples'])} labels, "
        f"loss {fmt_float(pressure['initial_loss'], 5)} -> "
        f"{fmt_float(pressure['final_loss'], 5)}")


def pressure_context(payload: dict[str, Any]) -> dict[str, Any] | None:
    context = payload.get("pressure_context")
    return context if isinstance(context, dict) else None


def pressure_context_line(payload: dict[str, Any]) -> str | None:
    context = pressure_context(payload)
    if not context:
        return None
    return (
        f"lane {fmt_float(float(context['lane_tightness']), 2)}, "
        f"conflict {fmt_float(float(context['conflict_density']), 2)}, "
        f"shift {fmt_float(float(context['cross_shift_load']), 2)}, "
        f"priority_flip {fmt_float(float(context['priority_flip']), 1)}")


def budget_decision(payload: dict[str, Any]) -> dict[str, Any] | None:
    budget = payload.get("budget_decision")
    return budget if isinstance(budget, dict) else None


def budget_line(payload: dict[str, Any]) -> str | None:
    budget = budget_decision(payload)
    if not budget:
        return None
    extra_pass = bool(budget.get("extra_pass"))
    accepted = bool(budget.get("accepted_extra"))
    extra = "fixed"
    refinement = ""
    if extra_pass:
        extra = "extra accepted" if accepted else "extra eval-only"
        refinement = (
            f", refine {fmt_float(float(budget.get('refinement_score_before', 0.0)), 2)}"
            f" -> {fmt_float(float(budget.get('refinement_score_after', 0.0)), 2)}"
            f" (delta {fmt_float(float(budget.get('refinement_score_delta', 0.0)), 2)})")
    return (
        f"{budget.get('mode', 'learned')} {extra}, "
        f"score {fmt_float(float(budget['score']), 3)}, "
        f"pass {int(budget['decision_pass'])}, "
        f"fixed {fmt_float(float(budget['fixed_gpu_ms']), 3)} ms -> "
        f"final {fmt_float(float(budget['final_gpu_ms']), 3)} ms"
        f"{refinement}")


def budget_cell(payload: dict[str, Any]) -> str:
    budget = budget_decision(payload)
    if not budget:
        return "n/a"
    extra_pass = bool(budget.get("extra_pass"))
    accepted = bool(budget.get("accepted_extra"))
    if extra_pass:
        extra = "accepted" if accepted else "eval-only"
    else:
        extra = "fixed"
    return f"{budget.get('mode', 'learned')} {extra}"


def scenario_label(payload: dict[str, Any], source: Path) -> str:
    label = str(payload.get("scenario") or source.stem)
    mode = pressure_mode(payload)
    if mode != "learned":
        label = f"{label}/{mode}"
    return label


def gate_rows(row: dict[str, Any], target: dict[str, Any]) -> list[tuple[str, str, str, bool]]:
    checks = [
        ("reach", f"{int(row['reached'])}", f">= {int(target['reach'])}",
         row["reached"] >= target["reach"]),
        ("deadlocks", f"{int(row['deadlocks'])}", f"<= {int(target['deadlocks_max'])}",
         row["deadlocks"] <= target["deadlocks_max"]),
        ("collisions", f"{int(row['collisions'])}", f"<= {int(target['collisions_max'])}",
         row["collisions"] <= target["collisions_max"]),
        ("collision CVaR", fmt_float(row["collision_cvar"]), f"<= {fmt_float(target['collision_cvar_max'])}",
         row["collision_cvar"] <= target["collision_cvar_max"]),
        ("residual", fmt_residual(row["residual_pct"]), f"<= {fmt_float(target['residual_pct_max'])}%",
         0 <= row["residual_pct"] <= target["residual_pct_max"]),
        ("runtime", f"{fmt_float(row['runtime_ms'], 3)} ms", f"<= {fmt_float(target['runtime_ms_max'], 3)} ms",
         row["runtime_ms"] <= target["runtime_ms_max"]),
    ]
    return checks


def render_markdown(payload: dict[str, Any], source: Path, target_name: str) -> str:
    hard_target = payload["hard_target"]
    target_row = find_row(payload, target_name)
    noregret = maybe_row(payload, "No-regret graph MPPI")
    lines: list[str] = []

    lines.append("# GPU Planner Showdown")
    lines.append("")
    lines.append(f"- Source: `{source}`")
    if payload.get("scenario"):
        lines.append(f"- Scenario: `{payload['scenario']}`")
    lines.append(f"- Overall target: **{pass_text(bool(payload.get('target_pass')))}**")
    lines.append(
        "- Runtime: "
        f"{fmt_float(payload['runtime']['gpu_ms'], 3)} ms GPU, "
        f"{fmt_float(payload['runtime']['speedup'], 1)}x vs CPU equivalent")
    lines.append(
        "- Training: "
        f"{int(payload['training']['samples'])} labels, "
        f"loss {fmt_float(payload['training']['initial_loss'], 5)} -> "
        f"{fmt_float(payload['training']['final_loss'], 5)}")
    pressure_line = pressure_controller_line(payload)
    if pressure_line:
        lines.append(f"- Pressure controller: {pressure_line}")
    context_line = pressure_context_line(payload)
    if context_line:
        lines.append(f"- Pressure context: {context_line}")
    budget = budget_line(payload)
    if budget:
        lines.append(f"- Adaptive budget: {budget}")
    if noregret:
        lines.append(
            "- Main enemy: no-regret MPPI misses the hard gate "
            f"({int(noregret['collisions'])} collisions, "
            f"CVaR {fmt_float(noregret['collision_cvar'])}); "
            f"target planner cuts collisions by "
            f"{int(noregret['collisions'] - target_row['collisions'])} and CVaR by "
            f"{fmt_float(noregret['collision_cvar'] - target_row['collision_cvar'])}.")
    lines.append("")

    lines.append("## Hard Gates")
    lines.append("")
    lines.append("| Gate | Measured | Target | Status |")
    lines.append("|---|---:|---:|---|")
    for name, measured, target, ok in gate_rows(target_row, hard_target):
        lines.append(f"| {name} | {measured} | {target} | {pass_text(ok)} |")
    lines.append("")

    lines.append("## Planner Table")
    lines.append("")
    lines.append("| Planner | Status | Reach | Deadlocks | Collisions | CVaR | Residual | Runtime ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["planners"]:
        lines.append(
            f"| {row['name']} | {pass_text(bool(row['target_pass']))} | "
            f"{int(row['reached'])}/48 | {int(row['deadlocks'])} | "
            f"{int(row['collisions'])} | {fmt_float(row['collision_cvar'])} | "
            f"{fmt_residual(row['residual_pct'])} | {fmt_float(row['runtime_ms'], 3)} |")
    lines.append("")
    return "\n".join(lines)


def render_matrix_markdown(runs: list[tuple[Path, dict[str, Any]]],
                           target_name: str) -> str:
    target_rows = [(source, payload, find_row(payload, target_name))
                   for source, payload in runs]
    worst_collisions = max(row["collisions"] for _, _, row in target_rows)
    worst_cvar = max(row["collision_cvar"] for _, _, row in target_rows)
    worst_residual = max(row["residual_pct"] for _, _, row in target_rows)
    worst_runtime = max(row["runtime_ms"] for _, _, row in target_rows)
    all_pass = all(bool(row.get("target_pass")) for _, _, row in target_rows)

    lines: list[str] = []
    lines.append("# GPU Planner Showdown Matrix")
    lines.append("")
    lines.append("- Sources: " + ", ".join(f"`{source}`" for source, _ in runs))
    lines.append("- Scenarios: " + ", ".join(
        f"`{scenario_label(payload, source)}`" for source, payload in runs))
    lines.append(f"- Overall target: **{pass_text(all_pass)}**")
    lines.append(
        "- Worst target row: "
        f"{int(worst_collisions)} collisions, "
        f"CVaR {fmt_float(worst_cvar)}, "
        f"residual {fmt_residual(worst_residual)}, "
        f"runtime {fmt_float(worst_runtime, 3)} ms")
    pressure_lines = sorted({pressure_controller_line(payload)
                             for _, payload in runs
                             if pressure_controller_line(payload)})
    if len(pressure_lines) == 1:
        lines.append(f"- Pressure controller: {pressure_lines[0]}")
    elif pressure_lines:
        lines.append("- Pressure controllers: " + "; ".join(pressure_lines))
    contexts = [pressure_context(payload) for _, payload in runs
                if pressure_context(payload)]
    if contexts:
        lane_values = [float(context["lane_tightness"]) for context in contexts]
        conflict_values = [float(context["conflict_density"]) for context in contexts]
        shift_values = [float(context["cross_shift_load"]) for context in contexts]
        flips = sum(1 for context in contexts
                    if float(context["priority_flip"]) > 0.5)
        lines.append(
            "- Pressure context range: "
            f"lane {fmt_float(min(lane_values), 2)}-{fmt_float(max(lane_values), 2)}, "
            f"conflict {fmt_float(min(conflict_values), 2)}-{fmt_float(max(conflict_values), 2)}, "
            f"shift {fmt_float(min(shift_values), 2)}-{fmt_float(max(shift_values), 2)}, "
            f"priority flips {flips}/{len(contexts)}")
    budgets = [budget_decision(payload) for _, payload in runs
               if budget_decision(payload)]
    if budgets:
        extra_count = sum(1 for budget in budgets if bool(budget.get("extra_pass")))
        accepted_count = sum(1 for budget in budgets
                             if bool(budget.get("accepted_extra")))
        accepted_tail = (f", accepted {accepted_count}/{extra_count} evaluated"
                         if extra_count else ", accepted 0/0 evaluated")
        lines.append(
            f"- Adaptive budget: extra evaluated {extra_count}/{len(budgets)} runs"
            f"{accepted_tail}")
    lines.append("")

    lines.append("## Scenario Matrix")
    lines.append("")
    lines.append("| Scenario | Status | Reach | Deadlocks | Collisions | CVaR | Residual | Runtime ms | Speedup | Budget | Enemy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for source, payload, target_row in target_rows:
        noregret = maybe_row(payload, "No-regret graph MPPI")
        enemy = "n/a"
        if noregret:
            enemy = (
                f"no-regret: {int(noregret['collisions'])} collisions, "
                f"CVaR {fmt_float(noregret['collision_cvar'])}")
        hard_target = payload["hard_target"]
        lines.append(
            f"| {scenario_label(payload, source)} | "
            f"{pass_text(bool(target_row['target_pass']))} | "
            f"{int(target_row['reached'])}/{int(hard_target['reach'])} | "
            f"{int(target_row['deadlocks'])} | "
            f"{int(target_row['collisions'])} | "
            f"{fmt_float(target_row['collision_cvar'])} | "
            f"{fmt_residual(target_row['residual_pct'])} | "
            f"{fmt_float(target_row['runtime_ms'], 3)} | "
            f"{fmt_float(payload['runtime']['speedup'], 1)}x | "
            f"{budget_cell(payload)} | "
            f"{enemy} |")
    lines.append("")

    lines.append("## Hard Gate Margins")
    lines.append("")
    lines.append("| Scenario | Collision margin | CVaR margin | Residual margin | Runtime margin ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for source, payload, target_row in target_rows:
        hard_target = payload["hard_target"]
        lines.append(
            f"| {scenario_label(payload, source)} | "
            f"{int(hard_target['collisions_max'] - target_row['collisions'])} | "
            f"{fmt_float(hard_target['collision_cvar_max'] - target_row['collision_cvar'])} | "
            f"{fmt_float(hard_target['residual_pct_max'] - target_row['residual_pct'])}% | "
            f"{fmt_float(hard_target['runtime_ms_max'] - target_row['runtime_ms'], 3)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    json_paths = [Path(path) for path in (args.json_paths or [DEFAULT_JSON])]
    try:
        runs = [(json_path, load_payload(json_path)) for json_path in json_paths]
        if len(runs) == 1:
            report = render_markdown(runs[0][1], runs[0][0], args.target_planner)
        else:
            report = render_matrix_markdown(runs, args.target_planner)
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

    failed = []
    for source, payload in runs:
        target_row = find_row(payload, args.target_planner)
        if not target_row.get("target_pass", False):
            failed.append(f"{scenario_label(payload, source)} ({source})")
    if args.strict and failed:
        print(f"target planner failed gates: {args.target_planner}: "
              + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
