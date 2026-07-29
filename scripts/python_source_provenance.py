#!/usr/bin/env python3
"""Build a content manifest for every source consumed by Python artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROVENANCE_RELATIVE = Path("src/cudarobotics/_source_provenance.json")


def source_paths() -> list[tuple[str, Path]]:
    python_root = REPO / "python"
    paths = [
        ("CMakeLists.txt", python_root / "CMakeLists.txt"),
        ("pyproject.toml", python_root / "pyproject.toml"),
        ("README.md", python_root / "README.md"),
    ]
    package_root = python_root / "src" / "cudarobotics"
    paths.extend(
        (path.relative_to(python_root).as_posix(), path)
        for path in sorted(
            package_root.rglob("*"), key=lambda item: item.as_posix()
        )
        if path.is_file()
        and path.suffix in {".cpp", ".py"}
        and path.name != PROVENANCE_RELATIVE.name
    )
    paths.extend(
        (path.relative_to(python_root).as_posix(), path)
        for path in sorted(
            (python_root / "core").rglob("*"),
            key=lambda item: item.as_posix(),
        )
        if path.is_file() and path.suffix in {".cu", ".cuh", ".hpp"}
    )
    return paths


def normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def expected_payload() -> dict[str, Any]:
    sources = {
        relative: hashlib.sha256(normalized_bytes(path)).hexdigest()
        for relative, path in source_paths()
    }
    canonical = json.dumps(
        sources, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "algorithm": "sha256-text-lf",
        "source_digest": hashlib.sha256(canonical).hexdigest(),
        "sources": sources,
    }


def serialized_payload() -> bytes:
    return (
        json.dumps(expected_payload(), indent=2, sort_keys=True) + "\n"
    ).encode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the checked-in manifest is stale",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = REPO / "python" / PROVENANCE_RELATIVE
    contents = serialized_payload()
    if args.check:
        assert destination.is_file(), (
            f"Python source provenance is missing: {destination}; "
            "run python scripts/python_source_provenance.py"
        )
        assert destination.read_bytes() == contents, (
            "Python source provenance is stale; run "
            "python scripts/python_source_provenance.py"
        )
        action = "verified"
    else:
        destination.write_bytes(contents)
        action = "wrote"
    print(
        f"{action} {destination.relative_to(REPO)} with "
        f"{len(expected_payload()['sources'])} source hashes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
