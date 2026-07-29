#!/usr/bin/env python3
"""Validate staged v1 documentation content before a Pages deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v1_documentation_evidence import evaluate_site_content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_directory", type=Path)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    checks = evaluate_site_content(
        args.site_directory.resolve(),
        version=args.version,
        target_tag=args.tag,
        git_commit=args.commit,
    )
    result = {"passed": all(checks.values()), "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
