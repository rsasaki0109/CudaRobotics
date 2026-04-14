#!/usr/bin/env python3
"""Export selected paper PDF figures to web-friendly PNG assets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_EXPORTS = {
    "paper/figures/fig_pareto.pdf": "gif/diff_mppi_pareto.png",
    "paper/figures/fig_mechanism.pdf": "gif/diff_mppi_mechanism.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=220, help="PNG export DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool = shutil.which("pdftoppm")
    if not tool:
        raise SystemExit("pdftoppm not found")

    for src_str, dst_str in DEFAULT_EXPORTS.items():
        src = Path(src_str)
        dst = Path(dst_str)
        if not src.exists():
            raise SystemExit(f"Missing source PDF: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        prefix = dst.with_suffix("")
        subprocess.run(
            [tool, "-png", "-singlefile", "-r", str(args.dpi), str(src), str(prefix)],
            check=True,
        )
        print(f"Saved {dst}")


if __name__ == "__main__":
    main()
