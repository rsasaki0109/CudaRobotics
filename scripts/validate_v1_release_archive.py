#!/usr/bin/env python3
"""Safely validate a canonical post-tag v1 release evidence ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v1_release_archive import load_archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    gate = load_archive(
        args.archive,
        target_version=args.version,
        target_tag=args.tag,
        expected_commit=args.commit,
        checksum_path=args.checksum,
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["valid"] and gate["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
