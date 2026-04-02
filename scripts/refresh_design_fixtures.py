#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "experiments" / "data"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh version-controlled design fixtures from the configured benchmark outputs.")
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Fixture manifest to use. Defaults to experiments/data/manifest.json.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify that all configured sources exist without copying files.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text())
    fixtures = data.get("fixtures", [])
    if not isinstance(fixtures, list) or not fixtures:
        raise RuntimeError(f"{path.relative_to(ROOT)} must contain a non-empty fixtures list")
    return fixtures


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    fixtures = load_manifest(manifest_path)

    for fixture in fixtures:
        source = ROOT / fixture["source"]
        target = manifest_path.parent / fixture["filename"]
        if not source.exists():
            raise RuntimeError(f"Missing fixture source: {source.relative_to(ROOT)}")
        if args.check_only:
            print(f"OK {source.relative_to(ROOT)}")
            continue
        shutil.copy2(source, target)
        print(f"{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")

    if args.check_only:
        print("Fixture sources are available")
    else:
        print(f"Refreshed {len(fixtures)} fixture files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
