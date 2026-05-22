#!/usr/bin/env python3
"""
Run a curated set of `comparison_*` GPU demos, parse their headline GPU
timing from stdout, and compare against scripts/perf_baseline.json.

Exits non-zero if any measurement is slower than `baseline * (1 + tolerance)`.

Usage:
    python3 scripts/perf_check.py              # use default baseline / tolerance
    python3 scripts/perf_check.py --update      # rewrite baseline with new times
    python3 scripts/perf_check.py --tolerance 0.25  # 25% slack

Designed for self-hosted CI with a CUDA-capable GPU; meant to be invoked
from .github/workflows/perf.yml (workflow_dispatch trigger).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
BASELINE_PATH = ROOT / "scripts" / "perf_baseline.json"

# Each entry: (binary_name, regex_for_gpu_ms, label)
# The regex MUST capture the GPU time in group(1), in milliseconds (float).
BENCHMARKS = [
    (
        "comparison_esdf",
        re.compile(r"Avg GPU\s+([\d.]+)\s*ms\s*/\s*ESDF"),
        "esdf_2d_gpu_ms",
    ),
    (
        "comparison_esdf_3d",
        re.compile(r"Avg GPU\s+([\d.]+)\s*ms\s*/\s*ESDF"),
        "esdf_3d_gpu_ms",
    ),
    (
        "comparison_voxel_map",
        re.compile(r"Avg GPU\s+([\d.]+)\s*ms\s*/\s*scan"),
        "voxel_map_gpu_ms",
    ),
    (
        "comparison_collision_check",
        re.compile(r"Avg GPU\s+([\d.]+)\s*ms\s*/\s*scan"),
        "collision_check_gpu_ms",
    ),
    (
        "comparison_rrtstar_rewire",
        re.compile(r"Avg GPU\s+([\d.]+)\s*ms\s*/\s*rewire"),
        "rrtstar_rewire_gpu_ms",
    ),
    (
        "esdf_mppi",
        re.compile(r"Avg rollout time .*?:\s*([\d.]+)\s*ms"),
        "esdf_mppi_rollout_ms",
    ),
]


def run_binary(name: str) -> str:
    binary = BIN / name
    if not binary.exists():
        raise FileNotFoundError(f"binary not found: {binary}")
    out = subprocess.run(
        [str(binary)],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return out.stdout


def parse(text: str, pattern: re.Pattern[str]) -> float:
    m = pattern.search(text)
    if not m:
        raise ValueError(f"pattern not found in stdout: {pattern.pattern}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="Overwrite baseline JSON with current measurements.")
    ap.add_argument("--tolerance", type=float, default=0.30,
                    help="Allowed slack (default 0.30 = 30%% slower than baseline).")
    args = ap.parse_args()

    if BASELINE_PATH.exists():
        baseline: dict[str, float] = json.loads(BASELINE_PATH.read_text())
    else:
        baseline = {}

    measured: dict[str, float] = {}
    failures: list[str] = []
    for binary, pattern, label in BENCHMARKS:
        try:
            stdout = run_binary(binary)
            value = parse(stdout, pattern)
            measured[label] = value
            print(f"[OK]  {label:32s} = {value:8.3f} ms", flush=True)
        except FileNotFoundError:
            # Missing binary (e.g., feature not built on this branch). Skip,
            # don't fail — the perf gate is for regressions, not coverage.
            print(f"[SKIP] {label}: binary not built", flush=True)
        except (subprocess.TimeoutExpired,
                subprocess.CalledProcessError, ValueError) as e:
            failures.append(f"{label}: {type(e).__name__}: {e}")
            print(f"[ERR] {label}: {e}", flush=True)

    if args.update:
        BASELINE_PATH.write_text(json.dumps(measured, indent=2) + "\n")
        print(f"\nUpdated {BASELINE_PATH} with {len(measured)} measurements.")
        return 0

    regressions: list[str] = []
    for label, value in measured.items():
        base = baseline.get(label)
        if base is None:
            print(f"[NEW] {label}: no baseline, measured {value:.3f} ms")
            continue
        limit = base * (1.0 + args.tolerance)
        if value > limit:
            regressions.append(
                f"{label}: measured {value:.3f} ms > limit {limit:.3f} ms "
                f"(baseline {base:.3f} ms, +{args.tolerance*100:.0f}%)")

    if failures:
        print("\n=== FAILED BENCHMARKS ===")
        for f in failures:
            print(" -", f)

    if regressions:
        print("\n=== PERFORMANCE REGRESSIONS ===")
        for r in regressions:
            print(" -", r)
        return 1

    if not failures:
        print("\nAll benchmarks within tolerance.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
