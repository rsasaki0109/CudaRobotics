#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh generated experiment-first docs in-place.")
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=200,
        help="How many repeated recommendation passes to use for runtime benchmarking.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subprocess.run(
        [
            sys.executable,
            "scripts/run_design_experiments.py",
            "--docs-dir",
            "docs",
            "--benchmark-iterations",
            str(args.benchmark_iterations),
        ],
        cwd=ROOT,
        check=True,
    )
    print("Refreshed docs/experiments.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
