#!/usr/bin/env python3
"""Run the GPU-labelled CTest suite on a machine with an NVIDIA GPU.

CI has no GPU, so tests labelled "gpu" (smoke tests, headless demos, benchmark
gates) only run where someone runs them. This wraps ctest for that:

    python3 scripts/run_gpu_checks.py --build           # build what the tests need, then run
    python3 scripts/run_gpu_checks.py --labels gpu,cpu  # more labels
    python3 scripts/run_gpu_checks.py --json build/gpu_checks.json

It handles single- and multi-config generators, prints one line per test and
exits non-zero if any test fails. The JSON report records the commit, whether
the tree was dirty, and each test's result and time; it does not record the GPU
model.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def multi_config(build_dir: Path) -> bool:
    cache = build_dir / "CMakeCache.txt"
    return cache.exists() and "CMAKE_CONFIGURATION_TYPES:" in cache.read_text(errors="replace")


def ctest_base(build_dir: Path, config: str, labels: list[str], exclude: str | None) -> list[str]:
    cmd = ["ctest", "--test-dir", str(build_dir), "-L", "|".join(labels)]
    if multi_config(build_dir):
        cmd += ["-C", config]
    if exclude:
        cmd += ["-E", exclude]
    return cmd


def required_targets(tests: list[dict]) -> list[str]:
    """Executable targets the selected tests run (by the command's file name)."""
    targets = []
    for test in tests:
        command = test.get("command") or []
        if not command:
            continue
        exe = Path(command[0]).name
        if exe.lower().endswith(".exe"):
            exe = exe[:-4]
        if exe.lower().startswith("python") or exe in targets:
            continue
        targets.append(exe)
    return targets


def parse_junit(xml_text: str) -> list[dict]:
    """Per-test status, time and output from ctest --output-junit."""
    results = []
    for case in ET.fromstring(xml_text).iter("testcase"):
        status = case.get("status", "")
        failed = case.find("failure") is not None or status == "fail"
        skipped = case.find("skipped") is not None or status == "notrun"
        out = case.find("system-out")
        results.append({
            "name": case.get("name", ""),
            "status": "failed" if failed else ("not run" if skipped else "passed"),
            "seconds": round(float(case.get("time", "0") or 0), 2),
            "output": out.text if out is not None and out.text else "",
        })
    return results


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--config", default="Release", help="configuration for multi-config generators")
    parser.add_argument("--labels", default="gpu", help="comma-separated CTest labels (default: gpu)")
    parser.add_argument("--exclude", help="regex of test names to skip (ctest -E)")
    parser.add_argument("--build", action="store_true", help="build the tests' executables first")
    parser.add_argument("--jobs", type=int, default=0, help="parallel build jobs (default: CMake's)")
    parser.add_argument("--timeout", type=int, default=600, help="per-test timeout in seconds")
    parser.add_argument("--json", type=Path, help="write a machine-readable report here")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if not (build_dir / "CMakeCache.txt").exists():
        print(f"error: {build_dir} is not a configured CMake build directory", file=sys.stderr)
        return 2
    labels = [l for l in args.labels.split(",") if l]
    base = ctest_base(build_dir, args.config, labels, args.exclude)

    listing = subprocess.run(base + ["--show-only=json-v1"], capture_output=True, text=True)
    if listing.returncode != 0:
        print(listing.stderr, file=sys.stderr)
        return 2
    tests = json.loads(listing.stdout).get("tests", [])
    if not tests:
        print(f"no tests with labels {labels}", file=sys.stderr)
        return 2

    if args.build:
        targets = required_targets(tests)
        print(f"building {len(targets)} targets: {' '.join(targets)}")
        cmd = ["cmake", "--build", str(build_dir), "--target", *targets]
        if multi_config(build_dir):
            cmd += ["--config", args.config]
        if args.jobs:
            cmd += ["--parallel", str(args.jobs)]
        if subprocess.run(cmd).returncode != 0:
            print("error: build failed", file=sys.stderr)
            return 1

    print(f"running {len(tests)} tests (labels: {', '.join(labels)})")
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "ctest.xml"
        t0 = time.time()
        run = subprocess.run(base + ["--timeout", str(args.timeout), "--output-junit", str(junit)],
                             capture_output=True, text=True)
        elapsed = time.time() - t0
        results = parse_junit(junit.read_text(encoding="utf-8", errors="replace")) if junit.exists() else []
    failed = [r for r in results if r["status"] != "passed"]
    for r in results:
        mark = "ok  " if r["status"] == "passed" else "FAIL"
        print(f"  {mark} {r['name']:<42} {r['seconds']:8.2f} s  {'' if r['status'] == 'passed' else r['status']}")
    for r in failed:
        tail = "\n".join(r["output"].rstrip().splitlines()[-25:])
        print(f"\n--- {r['name']} ({r['status']}), last lines of output ---\n{tail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed in {elapsed:.0f} s")

    if args.json:
        report = {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "platform": platform.platform(),
            "config": args.config if multi_config(build_dir) else None,
            "labels": labels,
            "elapsed_s": round(elapsed, 1),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "tests": [{k: v for k, v in r.items() if k != "output"} for r in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report: {args.json}")
    if len(results) != len(tests):
        print(f"warning: parsed {len(results)} results for {len(tests)} tests", file=sys.stderr)
    return 0 if not failed and run.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
