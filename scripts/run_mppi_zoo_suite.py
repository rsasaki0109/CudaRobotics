#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mppi_zoo_benchmark import rel, run_benchmark


DEFAULT_SCENARIOS = (
    "dynamic_crossing,"
    "narrow_passage,"
    "model_mismatch_crossing,"
    "dynamic_pincer,"
    "uncertain_crossing"
)
DEFAULT_PLANNERS = (
    "mppi,"
    "step_mppi_smooth,"
    "tsallis_mppi_smooth,"
    "ducct_mppi_smooth,"
    "dra_mppi_soft,"
    "lp_mppi_smooth,"
    "c2u_mppi_smooth,"
    "sc_mppi_smooth,"
    "soppi,"
    "soppi_fast"
)
DEFAULT_K_VALUES = "64,128"
DEFAULT_RESULTS_STEM = "mppi_zoo_suite_2026-06-10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the expanded MPPI reproduction zoo fixed-seed suite and render a Markdown report."
    )
    parser.add_argument("--bin", default="bin/benchmark_diff_mppi", help="Path to benchmark_diff_mppi binary.")
    parser.add_argument("--out-dir", default="docs/results", help="Directory for CSV and Markdown output.")
    parser.add_argument("--stem", default=DEFAULT_RESULTS_STEM, help="Output file stem under out-dir.")
    parser.add_argument("--csv", help="CSV path. Defaults to <out-dir>/<stem>.csv.")
    parser.add_argument("--markdown-out", help="Markdown report path. Defaults to <out-dir>/<stem>.md.")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Comma-separated scenario names.")
    parser.add_argument("--planners", default=DEFAULT_PLANNERS, help="Comma-separated planner names.")
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES, help="Comma-separated K sample counts.")
    parser.add_argument("--seed-count", default="3", help="Number of seeds per scenario/planner/K cell.")
    parser.add_argument("--dry-run", action="store_true", help="Write the command report without running the binary.")
    parser.add_argument("--skip-run", action="store_true", help="Summarize an existing CSV without running the binary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_default = f"{args.stem}.csv"
    markdown_default = f"{args.stem}.md"
    csv_path, markdown_path, command = run_benchmark(
        bin_path=args.bin,
        out_dir=args.out_dir,
        csv_path=args.csv or f"{args.out_dir}/{csv_default}",
        markdown_path=args.markdown_out or f"{args.out_dir}/{markdown_default}",
        scenarios=args.scenarios,
        planners=args.planners,
        k_values=args.k_values,
        seed_count=args.seed_count,
        title="MPPI Zoo Suite Report",
        generator="python3 scripts/run_mppi_zoo_suite.py",
        csv_basename=csv_default,
        markdown_basename=markdown_default,
        dry_run=args.dry_run,
        skip_run=args.skip_run,
    )
    if args.dry_run:
        print(shlex.join(command))
        print(f"Dry-run report saved to {rel(markdown_path)}")
        return
    print(f"CSV saved to {rel(csv_path)}")
    print(f"Markdown report saved to {rel(markdown_path)}")


if __name__ == "__main__":
    main()
