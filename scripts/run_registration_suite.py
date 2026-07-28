#!/usr/bin/env python3
"""Run the unified CudaRobotics rigid-registration benchmark suite.

This is the stable entry point for the v0.2 suite. It delegates cell isolation
and optional external baselines to ``benchmark_registration_external.py`` while
using a version-neutral output location.
"""

from __future__ import annotations

import sys
from pathlib import Path

import benchmark_registration_external as benchmark


DEFAULT_CSV = Path("build/registration_suite/registration_suite.csv")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--csv" not in arguments:
        arguments += ["--csv", str(DEFAULT_CSV)]
    return benchmark.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
