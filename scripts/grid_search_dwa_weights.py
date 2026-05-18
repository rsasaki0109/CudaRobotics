#!/usr/bin/env python3
"""Grid-search DWA cost weights on the hard-cell regime.

The hand-tuned DWA cost weights in benchmark_diff_mppi were picked by eye to
match MPPI's cost scale. This script sweeps (w_goal, w_obs, w_terminal) on
the hard cells (dyn_speed_scale >= 1.5) where dwa_med already does well but
final_distance still has slack, and reports the Pareto-best setting.

Search is intentionally coarse (3^3 = 27 combos) to land in ~10 min wall
time. The objective ranks combos lexicographically by:

  1. mean success rate over hard cells (higher is better)
  2. mean final_distance over hard cells (lower is better)
  3. mean avg_control_ms (lower is better; near-tie tiebreak)

The script prints a sorted leaderboard, writes a CSV, and emits the top-1
recommendation for promotion into the C++ default. It does NOT modify the
.cu file -- the recommended weights need a human eyeball before being
written into source.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
from pathlib import Path
from statistics import mean


HARD_CELLS = [
    (1.5, 1.0),
    (1.5, 1.3),
    (1.5, 1.6),
    (2.0, 1.0),
    (2.0, 1.3),
    (2.0, 1.6),
]

DEFAULT_W_GOALS = [3.0, 5.0, 8.0]
DEFAULT_W_OBS = [8.0, 11.5, 16.0]
DEFAULT_W_TERMINAL = [6.0, 12.0, 20.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", default="./bin/benchmark_diff_mppi")
    p.add_argument("--scenario", default="dynamic_crossing")
    p.add_argument("--variant", default="dwa_med",
                   help="Which DWA variant to tune (dwa_fast/dwa_med/dwa_fine)")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--k-samples", type=int, default=4096)
    p.add_argument("--w-goals", nargs="+", type=float, default=DEFAULT_W_GOALS)
    p.add_argument("--w-obs", nargs="+", type=float, default=DEFAULT_W_OBS)
    p.add_argument("--w-terminals", nargs="+", type=float,
                   default=DEFAULT_W_TERMINAL)
    p.add_argument("--csv-out",
                   default="build/grid_search_dwa_weights.csv")
    p.add_argument("--cells-dir",
                   default="build/_dwa_grid_cells")
    return p.parse_args()


def run_combo(args: argparse.Namespace, w_goal: float, w_obs: float,
              w_terminal: float) -> list[dict]:
    cells_dir = Path(args.cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for sp, rad in HARD_CELLS:
        tag = (f"g{w_goal:.1f}_o{w_obs:.1f}_t{w_terminal:.1f}"
               f"__s{sp:+.2f}__r{rad:.2f}")
        cell_csv = cells_dir / f"{args.variant}__{tag}.csv"
        cmd = [
            args.bin,
            "--scenarios", args.scenario,
            "--planners", args.variant,
            "--k-values", str(args.k_samples),
            "--seed-count", str(args.seeds),
            "--override-dyn-speed-scale", str(sp),
            "--override-dyn-radius-scale", str(rad),
            "--override-dwa-w-goal", str(w_goal),
            "--override-dwa-w-obs", str(w_obs),
            "--override-dwa-w-terminal", str(w_terminal),
            "--csv", str(cell_csv),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(
                f"benchmark failed for {tag}: {proc.returncode}")
        with open(cell_csv) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "speed": sp,
                    "radius": rad,
                    "success": float(r["success"]),
                    "final_distance": float(r["final_distance"]),
                    "avg_control_ms": float(r["avg_control_ms"]),
                    "collisions": int(r["collisions"]),
                })
    return rows


def summarise(rows: list[dict]) -> dict:
    return {
        "n_episodes": len(rows),
        "success_rate": mean(r["success"] for r in rows),
        "mean_final_distance": mean(r["final_distance"] for r in rows),
        "mean_avg_ms": mean(r["avg_control_ms"] for r in rows),
        "mean_collisions": mean(r["collisions"] for r in rows),
    }


def main() -> int:
    args = parse_args()
    Path("build").mkdir(exist_ok=True)
    combos = list(itertools.product(
        args.w_goals, args.w_obs, args.w_terminals))
    print(f"[grid] {len(combos)} combos × {len(HARD_CELLS)} cells "
          f"× {args.seeds} seeds = "
          f"{len(combos) * len(HARD_CELLS) * args.seeds} episodes")

    results: list[dict] = []
    for i, (w_goal, w_obs, w_terminal) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] "
              f"w_goal={w_goal:.1f} w_obs={w_obs:.1f} "
              f"w_terminal={w_terminal:.1f}")
        rows = run_combo(args, w_goal, w_obs, w_terminal)
        summary = summarise(rows)
        summary.update({
            "w_goal": w_goal,
            "w_obs": w_obs,
            "w_terminal": w_terminal,
        })
        results.append(summary)
        print(f"    succ={summary['success_rate']:.2f} "
              f"final_d={summary['mean_final_distance']:.2f} "
              f"ms={summary['mean_avg_ms']:.2f} "
              f"coll={summary['mean_collisions']:.2f}")

    # Sort: higher success first, then lower final_d, then lower ms.
    results.sort(key=lambda r: (
        -r["success_rate"],
        r["mean_final_distance"],
        r["mean_avg_ms"]))

    with open(args.csv_out, "w", newline="") as f:
        fields = ["w_goal", "w_obs", "w_terminal",
                  "success_rate", "mean_final_distance",
                  "mean_avg_ms", "mean_collisions", "n_episodes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    print(f"\n[grid] leaderboard -> {args.csv_out}")
    print("\n=== Top 5 ===")
    print(f"{'w_goal':>8} {'w_obs':>8} {'w_term':>8} "
          f"{'succ':>6} {'final_d':>8} {'ms':>6}")
    for r in results[:5]:
        print(f"{r['w_goal']:>8.1f} {r['w_obs']:>8.1f} "
              f"{r['w_terminal']:>8.1f} "
              f"{r['success_rate']:>6.2f} "
              f"{r['mean_final_distance']:>8.2f} "
              f"{r['mean_avg_ms']:>6.2f}")
    best = results[0]
    print(f"\n[grid] recommended: w_goal={best['w_goal']:.1f}, "
          f"w_obs={best['w_obs']:.1f}, "
          f"w_terminal={best['w_terminal']:.1f}")
    print("       update src/benchmark_diff_mppi.cu PlannerVariant defaults "
          "or the dwa_med variant block if these beat the current values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
