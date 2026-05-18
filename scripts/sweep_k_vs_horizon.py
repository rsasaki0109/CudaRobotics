#!/usr/bin/env python3
"""Sweep K_samples x gradient-update-horizon on a fixed difficulty cell.

For a single (scenario, speed_scale, radius_scale) cell, run
benchmark_diff_mppi across all (K, planner) combinations in one shot
(the benchmark supports both axes natively), then aggregate per
(planner, K) -> success rate, final distance, cumulative cost,
avg_control_ms.

The point of fixing the cell is to ask the substitution question:
*can higher K compensate for a shorter horizon?* The classical
expectation is yes (more samples = better gradient estimates), but
in our findings short horizons (early8) sometimes beat the full
horizon outright in hard regimes, so the trade-off may be more
nuanced.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from statistics import mean


DEFAULT_PLANNERS = [
    "diff_mppi_3_early1",
    "diff_mppi_3_early2",
    "diff_mppi_3_early4",
    "diff_mppi_3_early8",
    "diff_mppi_3_early16",
    "diff_mppi_3",
]
DEFAULT_KS = [512, 1024, 2048, 4096, 8192]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", default="./bin/benchmark_diff_mppi")
    p.add_argument("--scenario", required=True)
    p.add_argument("--speed-scale", type=float, required=True)
    p.add_argument("--radius-scale", type=float, default=1.0)
    p.add_argument("--planners", nargs="+", default=DEFAULT_PLANNERS)
    p.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_KS)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--csv-out", default=None)
    p.add_argument("--summary-out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    Path("build").mkdir(exist_ok=True)

    tag = f"{args.scenario}__s{args.speed_scale:+.2f}_r{args.radius_scale:.2f}"
    csv_out = args.csv_out or f"build/k_vs_horizon_{tag}.csv"
    summary_out = (args.summary_out
                   or f"build/k_vs_horizon_{tag}_summary.csv")

    cmd = [
        args.bin,
        "--scenarios", args.scenario,
        "--planners", ",".join(args.planners),
        "--k-values", ",".join(str(k) for k in args.k_values),
        "--seed-count", str(args.seeds),
        "--override-dyn-speed-scale", str(args.speed_scale),
        "--override-dyn-radius-scale", str(args.radius_scale),
        "--csv", csv_out,
    ]
    print("[sweep] running benchmark...")
    print("  " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit("benchmark failed")
    print(f"long-form CSV -> {csv_out}")

    grouped: dict[tuple[str, int], list[dict]] = {}
    with open(csv_out) as f:
        for r in csv.DictReader(f):
            key = (r["planner"], int(r["k_samples"]))
            grouped.setdefault(key, []).append({
                "success": float(r["success"]),
                "final_distance": float(r["final_distance"]),
                "min_goal_distance": float(r["min_goal_distance"]),
                "cumulative_cost": float(r["cumulative_cost"]),
                "avg_control_ms": float(r["avg_control_ms"]),
                "collisions": int(r["collisions"]),
            })

    summary_rows = []
    for (planner, k), entries in sorted(grouped.items()):
        n = len(entries)
        summary_rows.append({
            "scenario": args.scenario,
            "dyn_speed_scale": args.speed_scale,
            "dyn_radius_scale": args.radius_scale,
            "planner": planner,
            "k_samples": k,
            "n": n,
            "success_rate": sum(e["success"] for e in entries) / n,
            "final_distance":
                mean(e["final_distance"] for e in entries),
            "min_goal_distance":
                mean(e["min_goal_distance"] for e in entries),
            "cumulative_cost":
                mean(e["cumulative_cost"] for e in entries),
            "collisions": mean(e["collisions"] for e in entries),
            "avg_control_ms":
                mean(e["avg_control_ms"] for e in entries),
        })

    fields = list(summary_rows[0].keys())
    with open(summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"summary -> {summary_out} ({len(summary_rows)} rows)")

    # Quick stdout: success grid by (planner, K)
    planners = sorted({r["planner"] for r in summary_rows},
                      key=lambda p: (len(p), p))
    ks = sorted({r["k_samples"] for r in summary_rows})
    succ = {(r["planner"], r["k_samples"]): r["success_rate"]
            for r in summary_rows}
    print(f"\n=== success_rate grid ({tag}) ===")
    header = ["planner"] + [f"K={k}" for k in ks]
    print("  ".join(f"{c:>20}" for c in header))
    for p in planners:
        row = [p] + [f"{succ[(p, k)]:.2f}" for k in ks]
        print("  ".join(f"{c:>20}" for c in row))


if __name__ == "__main__":
    main()
