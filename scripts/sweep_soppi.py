#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SoppiConfig:
    step_size: float
    bandwidth: float
    iters: int
    neighbors: int

    @property
    def label(self) -> str:
        return f"soppi_s{self.step_size:g}_b{self.bandwidth:g}_i{self.iters}_n{self.neighbors}"


def parse_string_list(text: str) -> list[str]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(token)
    return sorted(set(values))


def parse_float_list(text: str) -> list[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return sorted(set(values))


def parse_int_list(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(max(1, int(token)))
    return sorted(set(values))


def parse_nonnegative_int_list(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(max(0, int(token)))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep SOPPI hyperparameters against MPPI in benchmark_diff_mppi."
    )
    parser.add_argument("--bin", default="bin/benchmark_diff_mppi", help="benchmark_diff_mppi binary path")
    parser.add_argument("--output-dir", default="build/soppi_sweep", help="output directory")
    parser.add_argument("--scenarios", default="dynamic_crossing,cluttered", help="comma-separated scenarios")
    parser.add_argument("--k-values", default="128,256", help="comma-separated sample counts")
    parser.add_argument("--seed-count", type=int, default=1, help="seed count per run")
    parser.add_argument("--step-sizes", default="0.015,0.025,0.045,0.075", help="comma-separated SOPPI step sizes")
    parser.add_argument("--bandwidths", default="1.0,2.0,4.0", help="comma-separated SOPPI RBF bandwidths")
    parser.add_argument("--iters", default="1", help="comma-separated SOPPI SVGD iteration counts")
    parser.add_argument("--neighbors", default="0", help="comma-separated SOPPI particle subset counts; 0 uses all particles")
    parser.add_argument("--t-horizon", type=int, help="optional benchmark horizon override")
    parser.add_argument("--quick", action="store_true", default=True, help="pass --quick to the benchmark")
    parser.add_argument("--no-quick", action="store_false", dest="quick", help="do not pass --quick")
    parser.add_argument("--continue-on-error", action="store_true", help="continue after a failed run")
    parser.add_argument(
        "--baseline-planners",
        default="mppi",
        help="comma-separated planners to run once before the SOPPI grid (default: mppi)",
    )
    return parser.parse_args()


def command_display(command: list[str]) -> str:
    return shlex.join(command)


def run_logged(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {command_display(command)}\n\n")
        handle.flush()
        proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, time.perf_counter() - begin


def load_csv(path: Path, run_label: str, config: SoppiConfig | None) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out = dict(row)
            out["run_label"] = run_label
            out["soppi_step_size"] = "" if config is None else f"{config.step_size:g}"
            out["soppi_bandwidth"] = "" if config is None else f"{config.bandwidth:g}"
            out["soppi_iters"] = "" if config is None else str(config.iters)
            out["soppi_neighbors"] = "" if config is None else str(config.neighbors)
            rows.append(out)
    return rows


def write_combined_csv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_label", "soppi_step_size", "soppi_bandwidth", "soppi_iters", "soppi_neighbors"]
    fieldnames += [name for name in rows[0].keys() if name not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"],
            row["planner"],
            row["k_samples"],
            row["run_label"],
            row["soppi_step_size"],
            row["soppi_bandwidth"],
            row["soppi_neighbors"],
        )
        groups[key].append(row)

    summary = []
    for key, group in sorted(groups.items()):
        item: dict[str, object] = {
            "scenario": key[0],
            "planner": key[1],
            "k_samples": int(key[2]),
            "run_label": key[3],
            "soppi_step_size": key[4],
            "soppi_bandwidth": key[5],
            "soppi_neighbors": key[6],
            "soppi_iters": group[0]["soppi_iters"],
            "episodes": len(group),
            "success": mean([float(r["success"]) for r in group]),
            "final_distance": mean([float(r["final_distance"]) for r in group]),
            "min_goal_distance": mean([float(r["min_goal_distance"]) for r in group]),
            "cumulative_cost": mean([float(r["cumulative_cost"]) for r in group]),
            "avg_control_ms": mean([float(r["avg_control_ms"]) for r in group]),
            "collisions": mean([float(r["collisions"]) for r in group]),
        }
        summary.append(item)
    return summary


def write_summary(summary: list[dict[str, object]], path: Path) -> None:
    by_cell: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        by_cell[(str(row["scenario"]), int(row["k_samples"]))].append(row)

    lines = [
        "# SOPPI Sweep",
        "",
        "Generated by `python3 scripts/sweep_soppi.py`.",
        "",
        "## Best SOPPI vs MPPI",
        "",
        "| Scenario | K | Best SOPPI | Success | Final Distance | Cost | Avg ms | MPPI Final Distance | MPPI Cost | MPPI Avg ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (scenario, k_samples), group in sorted(by_cell.items()):
        mppi = next((r for r in group if r["planner"] == "mppi"), None)
        if mppi is None:
            baselines = [r for r in group if r["run_label"].endswith("_baseline")]
            mppi = baselines[0] if baselines else None
        soppi_rows = [r for r in group if r["planner"] == "soppi"]
        if not mppi or not soppi_rows:
            continue
        best = min(
            soppi_rows,
            key=lambda r: (
                -float(r["success"]),
                float(r["final_distance"]),
                float(r["cumulative_cost"]),
                float(r["avg_control_ms"]),
            ),
        )
        label = (
            f"s={best['soppi_step_size']}, b={best['soppi_bandwidth']}, "
            f"i={best['soppi_iters']}, n={best['soppi_neighbors']}"
        )
        lines.append(
            f"| {scenario} | {k_samples} | {label} | {float(best['success']):.2f} | "
            f"{float(best['final_distance']):.2f} | {float(best['cumulative_cost']):.1f} | "
            f"{float(best['avg_control_ms']):.2f} | {float(mppi['final_distance']):.2f} | "
            f"{float(mppi['cumulative_cost']):.1f} | {float(mppi['avg_control_ms']):.2f} |"
        )

    lines += [
        "",
        "## All Configs",
        "",
        "| Scenario | K | Planner | Config | Success | Final Distance | Cost | Avg ms | Collisions |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["planner"] == "mppi":
            label = "baseline"
        else:
            label = (
                f"s={row['soppi_step_size']}, b={row['soppi_bandwidth']}, "
                f"i={row['soppi_iters']}, n={row['soppi_neighbors']}"
            )
        lines.append(
            f"| {row['scenario']} | {row['k_samples']} | {row['planner']} | {label} | "
            f"{float(row['success']):.2f} | {float(row['final_distance']):.2f} | "
            f"{float(row['cumulative_cost']):.1f} | {float(row['avg_control_ms']):.2f} | "
            f"{float(row['collisions']):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    csv_dir = out_dir / "csv"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    bin_path = str(Path(args.bin))
    scenarios = args.scenarios
    k_values = args.k_values
    configs = [
        SoppiConfig(step, bandwidth, iters, neighbors)
        for neighbors in parse_nonnegative_int_list(args.neighbors)
        for iters in parse_int_list(args.iters)
        for bandwidth in parse_float_list(args.bandwidths)
        for step in parse_float_list(args.step_sizes)
    ]

    base_args = []
    if args.quick:
        base_args.append("--quick")
    base_args += ["--scenarios", scenarios, "--k-values", k_values, "--seed-count", str(args.seed_count)]
    if args.t_horizon:
        base_args += ["--t-horizon", str(args.t_horizon)]

    runs: list[tuple[str, list[str], Path, SoppiConfig | None]] = []
    baseline_planners = parse_string_list(args.baseline_planners)
    if not baseline_planners:
        baseline_planners = ["mppi"]
    baseline_label = "_".join(baseline_planners)
    baseline_csv = csv_dir / f"{baseline_label}_baseline.csv"
    runs.append((
        f"{baseline_label}_baseline",
        [bin_path, *base_args, "--planners", ",".join(baseline_planners), "--csv", str(baseline_csv)],
        baseline_csv,
        None,
    ))
    for config in configs:
        csv_path = csv_dir / f"{config.label}.csv"
        command = [
            bin_path,
            *base_args,
            "--planners", "soppi",
            "--override-soppi-step-size", f"{config.step_size:g}",
            "--override-soppi-bandwidth", f"{config.bandwidth:g}",
            "--override-soppi-iters", str(config.iters),
            "--override-soppi-neighbors", str(config.neighbors),
            "--csv", str(csv_path),
        ]
        runs.append((config.label, command, csv_path, config))

    all_rows: list[dict[str, str]] = []
    manifest = []
    for label, command, csv_path, config in runs:
        log_path = log_dir / f"{label}.log"
        print(command_display(command), flush=True)
        returncode, duration = run_logged(command, log_path)
        manifest.append({
            "label": label,
            "returncode": returncode,
            "duration_s": round(duration, 3),
            "csv": str(csv_path),
            "log": str(log_path),
        })
        if returncode != 0:
            print(f"{label}: failed with return code {returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return returncode
            continue
        all_rows.extend(load_csv(csv_path, label, config))

    combined_csv = out_dir / "soppi_sweep_combined.csv"
    summary_md = out_dir / "soppi_sweep_summary.md"
    write_combined_csv(all_rows, combined_csv)
    write_summary(summarize(all_rows), summary_md)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "returncode", "duration_s", "csv", "log"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Combined CSV: {combined_csv}")
    print(f"Summary: {summary_md}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
