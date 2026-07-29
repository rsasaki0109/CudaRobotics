#!/usr/bin/env python3
"""Write the source-tag manifest deployed beside the v1 documentation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from v1_documentation_evidence import SITE


def build_manifest(tag: str, commit: str) -> dict:
    if tag != "v1.0.0":
        raise ValueError("documentation release tag must be v1.0.0")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full lowercase commit")
    return {
        "schema_version": 1,
        "version": "1.0.0",
        "target_tag": tag,
        "source_commit": commit,
        "site": SITE,
        "source_url": (
            f"https://github.com/rsasaki0109/CudaRobotics/tree/{tag}"
        ),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        payload = build_manifest(args.tag, args.commit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
