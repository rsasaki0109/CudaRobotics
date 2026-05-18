#!/usr/bin/env python3
"""Sweep Diff-MPPI gradient horizon against dynamic-obstacle difficulty.

For each (scenario, speed_scale, radius_scale) cell we run benchmark_diff_mppi
with the planner set ``mppi, diff_mppi_3_early{1,2,4,8,16}, diff_mppi_3`` and
several seeds, then aggregate success rate, final distance, cumulative cost,
collisions and average control latency per planner.

The point of the sweep is to find the regime where a short gradient-update
horizon (early1/early2) is enough, versus where the full horizon is required.
"""

import argparse
import csv
import subprocess
from pathlib import Path
from statistics import mean

DEFAULT_PLANNERS = [
    "mppi",
    "diff_mppi_3_early1",
    "diff_mppi_3_early2",
    "diff_mppi_3_early4",
    "diff_mppi_3_early8",
    "diff_mppi_3_early16",
    "diff_mppi_3",
]

DEFAULT_SPEED_SCALES = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_RADIUS_SCALES = [1.0, 1.3, 1.6]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", default="./bin/benchmark_diff_mppi")
    p.add_argument("--scenarios", nargs="+", default=["dynamic_crossing"])
    p.add_argument("--planners", nargs="+", default=DEFAULT_PLANNERS)
    p.add_argument("--speed-scales", nargs="+", type=float,
                   default=DEFAULT_SPEED_SCALES)
    p.add_argument("--radius-scales", nargs="+", type=float,
                   default=DEFAULT_RADIUS_SCALES)
    p.add_argument("--k-samples", type=int, default=4096)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--csv-out",
                   default="build/sweep_grad_horizon_difficulty.csv")
    p.add_argument("--summary-out",
                   default="build/sweep_grad_horizon_difficulty_summary.csv")
    p.add_argument("--cells-dir", default="build/_sweep_cells")
    return p.parse_args()


def run_cell(args, scenario, speed, radius):
    cells_dir = Path(args.cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cells_dir / f"{scenario}__s{speed:+.2f}__r{radius:.2f}.csv"
    cmd = [
        args.bin,
        "--scenarios", scenario,
        "--planners", ",".join(args.planners),
        "--k-values", str(args.k_samples),
        "--seed-count", str(args.seeds),
        "--override-dyn-speed-scale", str(speed),
        "--override-dyn-radius-scale", str(radius),
        "--csv", str(csv_path),
    ]
    print(f"[sweep] {scenario} speed={speed:+.2f} radius={radius:.2f}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(
            f"benchmark failed for {scenario} speed={speed} radius={radius}")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["dyn_speed_scale"] = f"{speed:.4f}"
            r["dyn_radius_scale"] = f"{radius:.4f}"
            rows.append(r)
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["scenario"], r["planner"],
               float(r["dyn_speed_scale"]), float(r["dyn_radius_scale"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (scenario, planner, sp, rad), entries in groups.items():
        n = len(entries)
        out.append({
            "scenario": scenario,
            "planner": planner,
            "dyn_speed_scale": sp,
            "dyn_radius_scale": rad,
            "n": n,
            "success_rate": sum(1 for e in entries if e["success"] == "1") / n,
            "final_distance": mean(float(e["final_distance"]) for e in entries),
            "min_goal_distance": mean(
                float(e["min_goal_distance"]) for e in entries),
            "cumulative_cost": mean(
                float(e["cumulative_cost"]) for e in entries),
            "collisions": mean(int(e["collisions"]) for e in entries),
            "avg_control_ms": mean(
                float(e["avg_control_ms"]) for e in entries),
        })
    return out


def planner_sort_key(planner):
    order = {
        "mppi": -1,
        "diff_mppi_3_early1": 1,
        "diff_mppi_3_early2": 2,
        "diff_mppi_3_early4": 4,
        "diff_mppi_3_early8": 8,
        "diff_mppi_3_early16": 16,
        "diff_mppi_3": 99,
    }
    return order.get(planner, 50)


def main():
    args = parse_args()
    Path("build").mkdir(exist_ok=True)

    all_rows = []
    for scenario in args.scenarios:
        for sp in args.speed_scales:
            for rad in args.radius_scales:
                all_rows.extend(run_cell(args, scenario, sp, rad))

    if all_rows:
        fields = list(all_rows[0].keys())
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"long-form CSV -> {args.csv_out} ({len(all_rows)} rows)")

    summary = summarize(all_rows)
    summary.sort(key=lambda r: (
        r["scenario"], r["dyn_speed_scale"], r["dyn_radius_scale"],
        planner_sort_key(r["planner"])))

    if summary:
        with open(args.summary_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"summary -> {args.summary_out} ({len(summary)} rows)")

    print("\n=== Sweep summary ===")
    header = ["scenario", "planner", "speed", "radius",
              "succ", "final_d", "min_d", "cost", "coll", "ms"]
    print("  ".join(f"{c:>18}" for c in header))
    for r in summary:
        print(
            f"  {r['scenario']:>18}  {r['planner']:>18}  "
            f"{r['dyn_speed_scale']:>18.2f}  {r['dyn_radius_scale']:>18.2f}  "
            f"{r['success_rate']:>18.2f}  {r['final_distance']:>18.2f}  "
            f"{r['min_goal_distance']:>18.2f}  {r['cumulative_cost']:>18.1f}  "
            f"{r['collisions']:>18.2f}  {r['avg_control_ms']:>18.2f}")


if __name__ == "__main__":
    main()
