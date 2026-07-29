#!/usr/bin/env python3
"""Validate a portable CudaRobotics systems-paper artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from systems_paper_bundle import load_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    gate = load_bundle(args.manifest, args.commit)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["valid"] and (gate["ready"] or not args.require_ready) else 1


if __name__ == "__main__":
    raise SystemExit(main())
