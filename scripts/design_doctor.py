#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh and validate the experiment-first design workflow.")
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=200,
        help="How many repeated recommendation passes to use for runtime benchmarking.",
    )
    parser.add_argument(
        "--snapshot-label",
        default="",
        help="When set, append a new history snapshot instead of only regenerating the history doc.",
    )
    parser.add_argument(
        "--skip-fixtures",
        action="store_true",
        help="Skip fixture refresh or fixture-sync checks.",
    )
    parser.add_argument(
        "--check-fixture-sync",
        action="store_true",
        help="Check that fixture CSVs still match their configured build outputs instead of copying them.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip refreshing docs/experiments.md.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Skip regenerating docs/experiments_history.md or writing a new snapshot.",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Skip regenerating docs/convergence.md.",
    )
    parser.add_argument(
        "--skip-actions",
        action="store_true",
        help="Skip regenerating docs/next_actions.md.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip scripts/validate_design_workflow.py.",
    )
    parser.add_argument(
        "--skip-scaffold-check",
        action="store_true",
        help="Skip scripts/check_scaffold_design_problem.py.",
    )
    parser.add_argument(
        "--skip-regression-check",
        action="store_true",
        help="Skip scripts/check_design_regressions.py.",
    )
    args = parser.parse_args()
    if args.skip_fixtures and args.check_fixture_sync:
        parser.error("--skip-fixtures and --check-fixture-sync cannot be used together")
    return args


def run_step(label: str, command: list[str]) -> None:
    print(f"[doctor] {label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()

    if not args.skip_fixtures:
        fixture_command = [sys.executable, "scripts/refresh_design_fixtures.py"]
        if args.check_fixture_sync:
            fixture_command.append("--check-sync")
            run_step("check fixture sync", fixture_command)
        else:
            run_step("refresh fixtures", fixture_command)

    if not args.skip_docs:
        run_step(
            "refresh experiment docs",
            [
                sys.executable,
                "scripts/refresh_design_docs.py",
                "--benchmark-iterations",
                str(args.benchmark_iterations),
            ],
        )

    if not args.skip_history:
        history_command = [
            sys.executable,
            "scripts/snapshot_design_experiments.py",
            "--benchmark-iterations",
            str(args.benchmark_iterations),
        ]
        if args.snapshot_label:
            history_command.extend(["--label", args.snapshot_label])
            run_step("record design snapshot", history_command)
        else:
            history_command.append("--render-only")
            run_step("refresh experiments history", history_command)

    if not args.skip_convergence:
        run_step("refresh convergence doc", [sys.executable, "scripts/render_design_convergence.py"])

    if not args.skip_actions:
        run_step("refresh next-actions doc", [sys.executable, "scripts/render_design_actions.py"])

    if not args.skip_validate:
        run_step("validate design workflow", [sys.executable, "scripts/validate_design_workflow.py"])

    if not args.skip_regression_check:
        run_step("check design regressions", [sys.executable, "scripts/check_design_regressions.py"])

    if not args.skip_scaffold_check:
        run_step("check scaffold workflow", [sys.executable, "scripts/check_scaffold_design_problem.py"])

    print("Design doctor completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
