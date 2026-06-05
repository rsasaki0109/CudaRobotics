#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = "docs/results/mppi_zoo_smoke_2026-06-05.csv"
DEFAULT_SVG = "docs/results/mppi_zoo_smoke_2026-06-05.svg"
PLANNER_ORDER = [
    "mppi",
    "lp_mppi_smooth",
    "step_mppi_smooth",
    "tsallis_mppi_smooth",
    "dra_mppi_soft",
    "c2u_mppi_smooth",
    "ducct_mppi_smooth",
    "dbas_log_mppi_agile",
    "pa_mppi_smooth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a compact SVG chart from an MPPI zoo smoke CSV."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input MPPI zoo smoke CSV.")
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
    higher_is_better: bool,
) -> list[str]:
    if not rows:
        return []

    values = [float(r[metric]) for r in rows]
    best_value = max(values) if higher_is_better else min(values)
    min_value = 0.0
    max_value = max(values)
    span = max(max_value - min_value, 1.0e-6)

    label_w = 205.0
    plot_x = x + label_w
    plot_w = width - label_w - 105.0
    bar_h = 17.0
    gap = 7.0
    chart_y = y + 58.0

    lines: list[str] = []
    lines.append(text(x, y, title, 20, "#101820"))
    lines.append(text(x, y + 24, subtitle, 13, "#586A7A"))

    for i, tick in enumerate((min_value, min_value + span / 2.0, max_value)):
        tx = plot_x + (tick - min_value) / span * plot_w
        lines.append(line(tx, chart_y - 15, tx, chart_y + len(rows) * (bar_h + gap) + 4, "#E5EAF0"))
        lines.append(text(tx, chart_y - 21, f"{tick:.2f}", 11, "#6B7C8D", "middle"))

    for idx, row in enumerate(rows):
        planner = str(row["planner"])
        value = float(row[metric])
        success = float(row["success"])
        y0 = chart_y + idx * (bar_h + gap)
        value_w = (value - min_value) / span * plot_w
        is_best = abs(value - best_value) < 1.0e-9
        detail = str(row.get("detail", f"{value:.2f}  s={success:.2f}"))
        lines.append(text(x, y0 + 13, planner, 12, "#24313D"))
        lines.append(rect(plot_x, y0, value_w, bar_h, color_for(planner, is_best), 3.0))
        lines.append(text(plot_x + value_w + 8, y0 + 13, detail, 12, "#24313D"))

    lines.append(text(plot_x + plot_w, chart_y + len(rows) * (bar_h + gap) + 20, metric_label, 12, "#586A7A", "end"))
    return lines


def render_svg(rows: list[dict[str, object]]) -> str:
    dynamic = aggregate(rows, "dynamic_crossing")
    narrow = aggregate(rows, "narrow_passage")
    dynamic_baseline = next(r for r in dynamic if r["planner"] == "mppi")
    narrow_baseline = next(r for r in narrow if r["planner"] == "mppi")

    for row in dynamic:
        gain = float(dynamic_baseline["final_distance"]) - float(row["final_distance"])
        row["final_distance_gain"] = gain
        row["detail"] = f"+{gain:.2f}  s={float(row['success']):.2f}"
    for row in narrow:
        gain = float(narrow_baseline["steps"]) - float(row["steps"])
        row["step_savings"] = gain
        row["detail"] = f"+{gain:.1f}  s={float(row['success']):.2f}"

    dynamic.sort(key=lambda r: (-float(r["final_distance_gain"]), planner_index(str(r["planner"]))))
    narrow.sort(key=lambda r: (-float(r["step_savings"]), planner_index(str(r["planner"]))))

    width = 1060
    height = 760
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">MPPI Zoo fixed-seed smoke chart</title>",
        "<desc id=\"desc\">A two-panel chart comparing MPPI zoo planners on dynamic crossing final distance and narrow passage steps.</desc>",
        rect(0, 0, width, height, "#FFFFFF"),
        text(42, 45, "MPPI Zoo Fixed-Seed Smoke Result", 28, "#101820"),
        text(42, 72, "Mean over K=64,128 and 3 seeds per scenario/planner/K cell. Longer bars improve more over vanilla MPPI.", 14, "#586A7A"),
    ]

    elements.extend(
        panel(
            "Dynamic crossing exposes the baseline failure",
            "Final-distance reduction versus vanilla MPPI; labels also show mean success rate.",
            dynamic,
            "final_distance_gain",
            "final-distance reduction vs MPPI",
            42,
            112,
            970,
            higher_is_better=True,
        )
    )
    elements.extend(
        panel(
            "Narrow passage is solved by MPPI, but smooth variants finish sooner",
            "Average step reduction versus vanilla MPPI; labels also show mean success rate.",
            narrow,
            "step_savings",
            "step reduction vs MPPI",
            42,
            445,
            970,
            higher_is_better=True,
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
