#!/usr/bin/env python3
"""Drive benchmark_diff_mppi from the HorizonSelector.

Default (dry-run) mode reads a sweep summary CSV, asks
``MinimalSufficientHorizonSelector`` for a recommendation per (scenario,
speed, radius) cell, and compares the recommended planner's recorded
performance against the always-full (``diff_mppi_3``) baseline pulled
from the same CSV. No CUDA work is required.

``--verify`` mode also re-runs ``benchmark_diff_mppi`` for the
recommended planner + the always-full baseline on the listed cells, so
the recommendation can be cross-checked end-to-end. Use ``--cells`` to
limit verification to a handful of representative cells.

Output: a Markdown comparison table (recommended planner, horizon,
final_distance, avg_control_ms, baseline final_distance/ms, gaps).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.horizon_selector_interface import HorizonSelectionRequest
from core.planner_selector_interface import AggregateBenchmarkRow
from experiments.horizon_selection.minimal_sufficient_selector import (
    MinimalSufficientHorizonSelector,
)

BASELINE_PLANNER = "diff_mppi_3"


def dataset_label(speed: float, radius: float) -> str:
    return f"speed={speed:+.2f}_radius={radius:.2f}"


def parse_cell_label(label: str) -> tuple[float, float]:
    sp = float(label.split("speed=")[1].split("_radius=")[0])
    rad = float(label.split("_radius=")[1])
    return sp, rad


def load_rows(path: Path) -> list[AggregateBenchmarkRow]:
    rows: list[AggregateBenchmarkRow] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(AggregateBenchmarkRow(
                dataset=dataset_label(
                    float(r["dyn_speed_scale"]),
                    float(r["dyn_radius_scale"])),
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


def parse_cells(arg: str | None) -> set[tuple[str, float, float]] | None:
    if not arg:
        return None
    out: set[tuple[str, float, float]] = set()
    for item in arg.split(";"):
        item = item.strip()
        if not item:
            continue
        scenario, sp, rad = item.split(",")
        out.add((scenario.strip(), float(sp), float(rad)))
    return out


def run_verify(bin_path: Path, scenario: str, planners: list[str],
               speed: float, radius: float, seeds: int,
               k_samples: int) -> dict[str, dict[str, float]]:
    """Run benchmark_diff_mppi once for the given (scenario, speed, radius)
    cell with the listed planners; return per-planner aggregate."""
    out_csv = Path("build/_orchestrator")
    out_csv.mkdir(parents=True, exist_ok=True)
    csv_path = out_csv / (
        f"{scenario}__s{speed:+.2f}__r{radius:.2f}.csv")
    cmd = [
        str(bin_path),
        "--scenarios", scenario,
        "--planners", ",".join(planners),
        "--k-values", str(k_samples),
        "--seed-count", str(seeds),
        "--override-dyn-speed-scale", str(speed),
        "--override-dyn-radius-scale", str(radius),
        "--csv", str(csv_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"benchmark failed for {scenario} {speed} {radius}:\n"
            f"{proc.stderr}")
    grouped: dict[str, list[dict[str, float]]] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            grouped.setdefault(r["planner"], []).append({
                "success": float(r["success"]),
                "final_distance": float(r["final_distance"]),
                "avg_control_ms": float(r["avg_control_ms"]),
                "cumulative_cost": float(r["cumulative_cost"]),
            })
    return {
        planner: {
            "success": mean(e["success"] for e in entries),
            "final_distance": mean(e["final_distance"] for e in entries),
            "avg_control_ms": mean(e["avg_control_ms"] for e in entries),
            "cumulative_cost": mean(e["cumulative_cost"] for e in entries),
            "n": float(len(entries)),
        }
        for planner, entries in grouped.items()
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--success-threshold", type=float, default=0.999)
    p.add_argument("--full-horizon-steps", type=int, default=30)
    p.add_argument("--md-out", default=None)
    p.add_argument("--verify", action="store_true",
                   help="Also re-run benchmark_diff_mppi for each cell")
    p.add_argument("--cells",
                   help="Restrict verification to specific cells. "
                        "Semicolon-separated list of "
                        "'scenario,speed,radius'. "
                        "Example: 'dynamic_crossing,1.5,1.0;"
                        "dynamic_slalom,0.0,1.0'")
    p.add_argument("--bin", default="./bin/benchmark_diff_mppi")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--k-samples", type=int, default=4096)
    args = p.parse_args()

    rows = load_rows(Path(args.summary_csv))
    selector = MinimalSufficientHorizonSelector(
        full_horizon_steps=args.full_horizon_steps)

    cells = sorted({(r.scenario, r.dataset) for r in rows})
    cell_filter = parse_cells(args.cells) if args.verify else None

    md = [
        "| scenario | speed | radius | recommended | h | succ | final_d "
        "| ms | baseline succ | baseline final_d | baseline ms "
        "| ms saved | final_d gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for (scenario, dataset) in cells:
        request = HorizonSelectionRequest(
            dataset=dataset, scenario=scenario,
            success_threshold=args.success_threshold)
        try:
            rec = selector.recommend(rows, request)
        except ValueError:
            continue
        speed, radius = parse_cell_label(dataset)

        # baseline = always-full row from sweep
        baseline = next(
            (r for r in rows
             if r.scenario == scenario and r.dataset == dataset
             and r.planner == BASELINE_PLANNER),
            None)
        if baseline is None:
            continue

        # recommended row from sweep
        rec_row = next(
            (r for r in rows
             if r.scenario == scenario and r.dataset == dataset
             and r.planner == rec.planner),
            None)
        if rec_row is None:
            continue

        verify_note = ""
        rec_succ = rec_row.success
        rec_final = rec_row.final_distance
        rec_ms = rec_row.avg_control_ms
        base_succ = baseline.success
        base_final = baseline.final_distance
        base_ms = baseline.avg_control_ms

        if args.verify and (cell_filter is None
                            or (scenario, speed, radius) in cell_filter):
            planners = sorted({rec.planner, BASELINE_PLANNER})
            verified = run_verify(
                Path(args.bin), scenario, planners,
                speed, radius, args.seeds, args.k_samples)
            v_rec = verified.get(rec.planner)
            v_base = verified.get(BASELINE_PLANNER)
            if v_rec and v_base:
                rec_succ = v_rec["success"]
                rec_final = v_rec["final_distance"]
                rec_ms = v_rec["avg_control_ms"]
                base_succ = v_base["success"]
                base_final = v_base["final_distance"]
                base_ms = v_base["avg_control_ms"]
                verify_note = " *(verified)*"

        ms_saved = base_ms - rec_ms
        final_gap = rec_final - base_final
        md.append(
            f"| {scenario}{verify_note} | {speed:+.2f} | {radius:.2f} "
            f"| {rec.planner} | {rec.grad_update_horizon} | "
            f"{rec_succ:.2f} | {rec_final:.2f} | {rec_ms:.2f} | "
            f"{base_succ:.2f} | {base_final:.2f} | {base_ms:.2f} | "
            f"{ms_saved:+.2f} | {final_gap:+.2f} |"
        )

    table = "\n".join(md)
    print(table)
    if args.md_out:
        Path(args.md_out).write_text(table + "\n")
        print(f"\nWritten to {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
