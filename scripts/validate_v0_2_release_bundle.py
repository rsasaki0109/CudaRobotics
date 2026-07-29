#!/usr/bin/env python3
"""Validate a portable v0.2.0 release-candidate evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v0_2_release_bundle import load_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    gate = load_bundle(args.bundle, args.commit)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["valid"] and gate["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
