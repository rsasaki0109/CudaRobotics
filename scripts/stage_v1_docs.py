#!/usr/bin/env python3
"""Stage immutable v1 documentation without changing the live v0.2 source."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


VERSION = "1.0.0"
TAG = "v1.0.0"
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".txt"}


def transform_text(text: str, *, tag: str) -> str:
    replacements = (
        (
            "docs/releases/v0.2.0_smoke_checklist.md",
            "docs/releases/v1.0.0_release_checklist.md",
        ),
        (
            "docs/releases/v0.2.0_notes.md",
            "docs/releases/v1.0.0_notes.md",
        ),
        ("CudaRobotics/tree/master/", f"CudaRobotics/tree/{tag}/"),
        ("CudaRobotics/blob/master/", f"CudaRobotics/blob/{tag}/"),
        ("CudaRobotics/master/", f"CudaRobotics/{tag}/"),
        ("v0.2.0", f"v{VERSION}"),
    )
    transformed = text
    for old, new in replacements:
        transformed = transformed.replace(old, new)
    return transformed


def stage(source_dir: Path, output_dir: Path, *, tag: str = TAG) -> None:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if tag != TAG:
        raise ValueError(f"unsupported documentation tag: {tag}")
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        destination = output / relative
        if source_path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source_path.suffix.lower() in TEXT_SUFFIXES:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                transform_text(
                    source_path.read_text(encoding="utf-8"),
                    tag=tag,
                ).encode("utf-8")
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()
    try:
        stage(args.source_dir, args.output_dir, tag=args.tag)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"cannot stage v1 documentation: {error}") from error
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
