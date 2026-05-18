#!/usr/bin/env python3
"""Per-cell horizon recommendations from a sweep summary CSV.

Loads ``build/sweep_grad_horizon_*_summary.csv`` (the output of
``sweep_grad_horizon_difficulty.py``), groups by (scenario, speed_scale,
radius_scale) -- treating each cell as a synthetic "dataset" -- and asks
the ``MinimalSufficientHorizonSelector`` for a recommendation per cell.
Prints a Markdown table of (scenario, speed, radius) -> planner, horizon,
success_rate, final_distance, rationale.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.horizon_selector_interface import HorizonSelectionRequest
from core.planner_selector_interface import AggregateBenchmarkRow
from experiments.horizon_selection.minimal_sufficient_selector import (
    MinimalSufficientHorizonSelector,
)


def dataset_label(speed: float, radius: float) -> str:
    return f"speed={speed:+.2f}_radius={radius:.2f}"


def load_rows(summary_csv: Path) -> list[AggregateBenchmarkRow]:
    rows: list[AggregateBenchmarkRow] = []
    with open(summary_csv) as f:
        for r in csv.DictReader(f):
            ds = dataset_label(
                float(r["dyn_speed_scale"]), float(r["dyn_radius_scale"]))
            rows.append(AggregateBenchmarkRow(
                dataset=ds,
                scenario=r["scenario"],
                planner=r["planner"],
                k_samples=4096,
                success=float(r["success_rate"]),
                steps=0.0,
                final_distance=float(r["final_distance"]),
                cumulative_cost=float(r["cumulative_cost"]),
                avg_control_ms=float(r["avg_control_ms"]),
            ))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", required=True,
                   help="Sweep summary CSV produced by "
                        "sweep_grad_horizon_difficulty.py")
    p.add_argument("--success-threshold", type=float, default=0.999)
    p.add_argument("--full-horizon-steps", type=int, default=30,
                   help="DEFAULT_T_HORIZON in benchmark_diff_mppi.cu")
    p.add_argument("--md-out", default=None,
                   help="Optional path to write the Markdown table")
    args = p.parse_args()

    rows = load_rows(Path(args.summary_csv))
    selector = MinimalSufficientHorizonSelector(
        full_horizon_steps=args.full_horizon_steps)

    keys = sorted({(r.scenario, r.dataset) for r in rows})

    md = ["| scenario | speed | radius | planner | horizon | succ | "
          "final_d | rationale |",
          "|---|---|---|---|---|---|---|---|"]
    for (scenario, dataset) in keys:
        request = HorizonSelectionRequest(
            dataset=dataset,
            scenario=scenario,
            success_threshold=args.success_threshold,
        )
        try:
            rec = selector.recommend(rows, request)
        except ValueError as e:
            md.append(f"| {scenario} | {dataset} | -- | -- | -- | -- | "
                      f"-- | error: {e} |")
            continue
        # extract speed/radius from dataset label
        sp = dataset.split("speed=")[1].split("_radius=")[0]
        rad = dataset.split("_radius=")[1]
        md.append(
            f"| {scenario} | {sp} | {rad} | {rec.planner} | "
            f"{rec.grad_update_horizon} | {rec.success_rate:.2f} | "
            f"{rec.final_distance:.2f} | {rec.rationale} |"
        )

    table = "\n".join(md)
    print(table)
    if args.md_out:
        Path(args.md_out).write_text(table + "\n")
        print(f"\nWritten to {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
