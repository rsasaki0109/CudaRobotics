#!/usr/bin/env python3
"""Test whether the offline-trained HorizonSelector generalises to
difficulty cells that were not in the sweep grid.

For each (scenario, off-grid probe cell), the difficulty-index
finds the nearest known cell from the sweep summary, runs the
HorizonSelector on its rows, and reports the recommended planner.
We then re-run benchmark_diff_mppi at the probe cell with that
recommended planner and with the always-full baseline, and report
the gap on success rate and final distance.

The off-grid probe cells are intentionally placed between known
sweep points (e.g. speed=0.25 when the sweep has 0.0 and 0.5), so
the test measures interpolation, not extrapolation.
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

from experiments.horizon_selection.difficulty_index import (
    load_indexed_rows,
    recommend_for_probe,
    recommend_for_probe_robust,
)
from experiments.horizon_selection.minimal_sufficient_selector import (
    MinimalSufficientHorizonSelector,
)


BASELINE_PLANNER = "diff_mppi_3"


SUMMARY_BY_SCENARIO = {
    "dynamic_crossing": "build/sweep_grad_horizon_difficulty_summary.csv",
    "dynamic_slalom":   "build/sweep_slalom_summary.csv",
    "dynamic_pincer":   "build/sweep_pincer_summary.csv",
}


# Off-grid probe cells: chosen between known sweep points so each
# probe has nearest-neighbour distance > 0 but small. Sweep grid
# was speed in {-1.0, 0.0, 0.5, 1.0, 1.5, 2.0}, radius in {1.0, 1.3, 1.6}.
DEFAULT_PROBES = {
    "dynamic_crossing": [
        (0.25, 1.0),   # between speed 0.0 and 0.5
        (1.25, 1.0),   # between 1.0 and 1.5 -- crosses easy->hard transition
        (1.75, 1.0),   # between 1.5 and 2.0 -- back to easy regime
        (0.5, 1.15),   # between radius 1.0 and 1.3
    ],
    "dynamic_slalom": [
        (0.25, 1.0),
        (1.25, 1.0),   # easy -> hard transition
        (1.75, 1.0),
        (0.5, 1.45),
    ],
    "dynamic_pincer": [
        (0.25, 1.0),
        (0.75, 1.0),   # close to sweet spot at 0.5
        (1.25, 1.3),   # mid easy/hard transition
        (1.75, 1.3),   # hard regime
    ],
}


def run_bench(bin_path: Path, scenario: str, planners: list[str],
              speed: float, radius: float, seeds: int,
              k_samples: int) -> dict[str, dict[str, float]]:
    out_dir = Path("build/_online")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (
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
            })
    return {
        planner: {
            "success": mean(e["success"] for e in entries),
            "final_distance": mean(e["final_distance"] for e in entries),
            "avg_control_ms": mean(e["avg_control_ms"] for e in entries),
        }
        for planner, entries in grouped.items()
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+",
                   default=list(SUMMARY_BY_SCENARIO.keys()))
    p.add_argument("--bin", default="./bin/benchmark_diff_mppi")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--k-samples", type=int, default=4096)
    p.add_argument("--md-out",
                   default="build/online_horizon_generalization.md")
    p.add_argument("--robust", action="store_true",
                   help="Use k-NN max-horizon (conservative) recommender")
    p.add_argument("--k", type=int, default=3,
                   help="Number of nearest sweep cells for --robust mode")
    args = p.parse_args()

    selector = MinimalSufficientHorizonSelector()
    mode_label = f"robust (k={args.k})" if args.robust else "minimal"

    md = [
        "# Online HorizonSelector generalization test",
        "",
        f"Mode: **{mode_label}**. ",
        "Off-grid (speed, radius) probes are matched to the nearest cell"
        " in the sweep summary; the selector recommends a planner from"
        " that cell; we then run benchmark_diff_mppi at the probe and"
        " compare the recommended planner against always-full.",
        "",
        "| scenario | probe (speed, radius) | matched cell (dist) | "
        "recommended | h | succ | final_d | full succ | full final_d "
        "| final_d gap |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for scenario in args.scenarios:
        summary_csv = Path(SUMMARY_BY_SCENARIO[scenario])
        if not summary_csv.exists():
            print(f"[skip] no sweep summary for {scenario} "
                  f"({summary_csv})")
            continue
        rows = load_indexed_rows(summary_csv)
        for (sp, rad) in DEFAULT_PROBES[scenario]:
            if args.robust:
                indexed = recommend_for_probe_robust(
                    selector, rows, scenario, sp, rad, k=args.k)
            else:
                indexed = recommend_for_probe(
                    selector, rows, scenario, sp, rad)
            rec = indexed.recommendation
            planners = sorted({rec.planner, BASELINE_PLANNER})
            verified = run_bench(
                Path(args.bin), scenario, planners,
                sp, rad, args.seeds, args.k_samples)
            rec_data = verified.get(rec.planner, {})
            base_data = verified.get(BASELINE_PLANNER, {})
            final_gap = (rec_data.get("final_distance", 0)
                         - base_data.get("final_distance", 0))
            md.append(
                f"| {scenario} | ({sp:+.2f}, {rad:.2f}) | "
                f"({indexed.matched_speed:+.2f}, "
                f"{indexed.matched_radius:.2f}) "
                f"[d={indexed.distance:.2f}] "
                f"| {rec.planner} | {rec.grad_update_horizon} | "
                f"{rec_data.get('success', 0):.2f} | "
                f"{rec_data.get('final_distance', 0):.2f} | "
                f"{base_data.get('success', 0):.2f} | "
                f"{base_data.get('final_distance', 0):.2f} | "
                f"{final_gap:+.2f} |"
            )

    table = "\n".join(md)
    print(table)
    Path(args.md_out).write_text(table + "\n")
    print(f"\nWritten to {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
