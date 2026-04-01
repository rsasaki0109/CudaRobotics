#!/usr/bin/env python3

from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that scaffold_design_problem.py still emits the current workflow contract.")
    parser.add_argument("--problem", default="cache_policy", help="Problem slug to scaffold during the check.")
    return parser.parse_args()


def require_contains(path: Path, fragments: list[str]) -> None:
    text = path.read_text()
    for fragment in fragments:
        if fragment not in text:
            raise RuntimeError(f"{path.relative_to(path.parents[2])} is missing expected fragment: {fragment}")


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="design-scaffold-") as tmp:
        root = Path(tmp)
        subprocess.run(
            [
                sys.executable,
                "scripts/scaffold_design_problem.py",
                args.problem,
                "--root",
                str(root),
            ],
            cwd=ROOT,
            check=True,
        )

        expected_files = [
            "core/__init__.py",
            f"core/{args.problem}_interface.py",
            "experiments/__init__.py",
            f"experiments/{args.problem}/__init__.py",
            f"experiments/{args.problem}/functional_variant.py",
            f"experiments/{args.problem}/oop_variant.py",
            f"experiments/{args.problem}/pipeline_variant.py",
        ]

        for relative_path in expected_files:
            path = root / relative_path
            if not path.exists():
                raise RuntimeError(f"Missing scaffolded file: {relative_path}")
            py_compile.compile(str(path), doraise=True)

        init_path = root / "experiments" / args.problem / "__init__.py"
        require_contains(
            init_path,
            [
                f'PROBLEM_KIND = "{args.problem}"',
                "INTERFACE_FILE",
                "build_requests",
                "build_report",
                "build_variants",
            ],
        )
        print("Scaffold design problem check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
