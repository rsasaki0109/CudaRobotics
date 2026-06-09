#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = "docs/results/mppi_zoo_suite_2026-06-09.csv"
DEFAULT_SVG = "docs/results/mppi_zoo_suite_2026-06-09.svg"
PLANNER_ORDER = [
    "mppi",
    "step_mppi_smooth",
    "tsallis_mppi_smooth",
    "ducct_mppi_smooth",
    "dra_mppi_soft",
    "lp_mppi_smooth",
    "c2u_mppi_smooth",
    "sc_mppi_smooth",
]
PANELS = [
    {
        "scenario": "dynamic_crossing",
        "title": "Dynamic crossing",
        "subtitle": "Baseline MPPI fails here; bars show final-distance reduction versus vanilla MPPI.",
        "metric": "final_distance_gain",
        "metric_label": "final-distance reduction vs MPPI",
        "higher_is_better": True,
        "baseline_metric": "final_distance",
        "detail_fmt": "+{gain:.2f}  s={success:.2f}",
    },
    {
        "scenario": "model_mismatch_crossing",
        "title": "Model-mismatch crossing",
        "subtitle": "Planner/model timing mismatch; bars show mean success rate.",
        "metric": "success",
        "metric_label": "mean success rate",
        "higher_is_better": True,
        "baseline_metric": None,
        "detail_fmt": "s={success:.2f}  d={final_distance:.2f}",
    },
    {
        "scenario": "dynamic_pincer",
        "title": "Dynamic pincer",
        "subtitle": "Risk-aware stress scene; bars show mean success rate.",
        "metric": "success",
        "metric_label": "mean success rate",
        "higher_is_better": True,
        "baseline_metric": None,
        "detail_fmt": "s={success:.2f}  d={final_distance:.2f}",
    },
    {
        "scenario": "narrow_passage",
        "title": "Narrow passage",
        "subtitle": "Efficiency check; bars show average step reduction versus vanilla MPPI.",
        "metric": "step_savings",
        "metric_label": "step reduction vs MPPI",
        "higher_is_better": True,
        "baseline_metric": "steps",
        "detail_fmt": "+{gain:.1f}  s={success:.2f}",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a compact SVG chart from an MPPI zoo suite CSV."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input MPPI zoo suite CSV.")
    parser.add_argument("--svg-out", default=DEFAULT_SVG, help="Output SVG path.")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "scenario": row["scenario"],
                    "planner": row["planner"],
                    "k_samples": int(float(row["k_samples"])),
                    "success": float(row["success"]),
                    "final_distance": float(row["final_distance"]),
                    "steps": float(row["steps"]),
                }
            )
    return rows


def aggregate(rows: list[dict[str, object]], scenario: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["scenario"] == scenario:
            groups[str(row["planner"])].append(row)

    summary: list[dict[str, object]] = []
    for planner, entries in groups.items():
        summary.append(
            {
                "planner": planner,
                "success": mean(float(e["success"]) for e in entries),
                "final_distance": mean(float(e["final_distance"]) for e in entries),
                "steps": mean(float(e["steps"]) for e in entries),
            }
        )
    return summary


def planner_index(planner: str) -> int:
    try:
        return PLANNER_ORDER.index(planner)
    except ValueError:
        return len(PLANNER_ORDER)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def color_for(planner: str, is_best: bool) -> str:
    if planner == "mppi":
        return "#D84A45"
    if is_best:
        return "#188F5A"
    return "#3478C5"


def text(x: float, y: float, value: str, size: int = 15, color: str = "#1F2933", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#CDD6E0") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1"/>'


def rect(x: float, y: float, width: float, height: float, fill: str, radius: float = 0.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.0):.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}"/>'
    )


def panel(
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    metric: str,
    metric_label: str,
    x: float,
    y: float,
    width: float,
    height: float,
    higher_is_better: bool,
) -> list[str]:
    if not rows:
        return []

    values = [float(r[metric]) for r in rows]
    best_value = max(values) if higher_is_better else min(values)
    min_value = min(0.0, min(values))
    max_value = max(values)
    span = max(max_value - min_value, 1.0e-6)

    label_w = 190.0
    plot_x = x + label_w
    plot_w = width - label_w - 92.0
    bar_h = 14.0
    gap = 5.0
    chart_y = y + 48.0
    max_rows = max(1, len(rows))
    available_h = height - 72.0
    if max_rows * (bar_h + gap) > available_h:
        bar_h = max(10.0, (available_h - gap * max_rows) / max_rows)

    lines: list[str] = []
    lines.append(text(x, y, title, 17, "#101820"))
    lines.append(text(x, y + 20, subtitle, 11, "#586A7A"))

    for tick in (min_value, min_value + span / 2.0, max_value):
        tx = plot_x + (tick - min_value) / span * plot_w
        lines.append(line(tx, chart_y - 10, tx, chart_y + len(rows) * (bar_h + gap) + 2, "#E5EAF0"))
        lines.append(text(tx, chart_y - 16, f"{tick:.2f}", 10, "#6B7C8D", "middle"))

    for idx, row in enumerate(rows):
        planner = str(row["planner"])
        value = float(row[metric])
        y0 = chart_y + idx * (bar_h + gap)
        value_w = (value - min_value) / span * plot_w
        is_best = abs(value - best_value) < 1.0e-9
        detail = str(row.get("detail", f"{value:.2f}"))
        lines.append(text(x, y0 + 11, planner, 11, "#24313D"))
        lines.append(rect(plot_x, y0, value_w, bar_h, color_for(planner, is_best), 3.0))
        lines.append(text(plot_x + value_w + 6, y0 + 11, detail, 11, "#24313D"))

    lines.append(text(plot_x + plot_w, chart_y + len(rows) * (bar_h + gap) + 16, metric_label, 11, "#586A7A", "end"))
    return lines


def prepare_panel_rows(panel_cfg: dict[str, object], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scenario_rows = aggregate(rows, str(panel_cfg["scenario"]))
    if not scenario_rows:
        return []

    baseline_metric = panel_cfg.get("baseline_metric")
    if baseline_metric:
        baseline = next((r for r in scenario_rows if r["planner"] == "mppi"), None)
        if baseline is None:
            return []
        for row in scenario_rows:
            gain = float(baseline[str(baseline_metric)]) - float(row[str(baseline_metric)])
            if panel_cfg["metric"] == "final_distance_gain":
                row["final_distance_gain"] = gain
            elif panel_cfg["metric"] == "step_savings":
                row["step_savings"] = gain
            row["detail"] = str(panel_cfg["detail_fmt"]).format(
                gain=gain,
                success=float(row["success"]),
                final_distance=float(row["final_distance"]),
            )
        sort_metric = str(panel_cfg["metric"])
        scenario_rows.sort(key=lambda r: (-float(r[sort_metric]), planner_index(str(r["planner"]))))
        return scenario_rows

    for row in scenario_rows:
        row["detail"] = str(panel_cfg["detail_fmt"]).format(
            gain=float(row["success"]),
            success=float(row["success"]),
            final_distance=float(row["final_distance"]),
        )
    scenario_rows.sort(key=lambda r: (-float(r["success"]), planner_index(str(r["planner"]))))
    return scenario_rows


def render_svg(rows: list[dict[str, object]]) -> str:
    width = 1060
    height = 980
    panel_w = 490.0
    panel_h = 360.0
    positions = [(42, 92), (528, 92), (42, 500), (528, 500)]

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">MPPI Zoo fixed-seed suite chart</title>",
        "<desc id=\"desc\">A four-panel chart comparing curated MPPI zoo planners across navigation stress scenarios.</desc>",
        rect(0, 0, width, height, "#FFFFFF"),
        text(42, 45, "MPPI Zoo Fixed-Seed Suite Result", 28, "#101820"),
        text(
            42,
            72,
            "Mean over K=64,128 and 3 seeds per scenario/planner/K cell across five navigation scenarios.",
            14,
            "#586A7A",
        ),
    ]

    for panel_cfg, (x, y) in zip(PANELS, positions):
        panel_rows = prepare_panel_rows(panel_cfg, rows)
        elements.extend(
            panel(
                str(panel_cfg["title"]),
                str(panel_cfg["subtitle"]),
                panel_rows,
                str(panel_cfg["metric"]),
                str(panel_cfg["metric_label"]),
                x,
                y,
                panel_w,
                panel_h,
                bool(panel_cfg["higher_is_better"]),
            )
        )

    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def main() -> None:
    args = parse_args()
    csv_path = repo_path(args.csv)
    svg_path = repo_path(args.svg_out)
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"CSV has no rows: {csv_path}")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(rows), encoding="utf-8")
    print(f"SVG chart saved to {svg_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
