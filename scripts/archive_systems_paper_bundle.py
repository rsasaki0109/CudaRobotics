#!/usr/bin/env python3
"""Create a canonical systems-paper ZIP and SHA-256 sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from systems_paper_archive import create_archive, write_checksum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    checksum_path = args.checksum or Path(f"{args.output}.sha256")
    try:
        if checksum_path.exists():
            raise ValueError(
                f"refusing to overwrite checksum: {checksum_path.resolve()}"
            )
        result = create_archive(args.manifest, args.output, args.commit)
        result.update(write_checksum(args.output, checksum_path))
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(
            f"cannot archive systems paper bundle: {error}"
        ) from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
