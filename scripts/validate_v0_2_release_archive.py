#!/usr/bin/env python3
"""Safely validate a canonical v0.2.0 release evidence ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v0_2_release_archive import load_archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    gate = load_archive(
        args.archive,
        args.commit,
        checksum_path=args.checksum,
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["valid"] and gate["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
