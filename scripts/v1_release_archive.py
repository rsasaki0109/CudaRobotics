#!/usr/bin/env python3
"""Create and validate canonical ZIPs of post-tag v1 release evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from canonical_evidence_archive import (
    MAX_TOTAL_BYTES,
    create_archive as create_canonical_archive,
    evaluate_archive as evaluate_canonical_archive,
    load_archive as load_canonical_archive,
    read_checksum,
    write_checksum,
)
from v1_release_bundle import load_bundle


ARCHIVE_ROOT = "cudarobotics-v1.0.0-evidence"


def _bundle_loader(
    *,
    target_version: str,
    target_tag: str,
    expected_commit: str,
):
    def load(path: Path) -> dict[str, Any]:
        return load_bundle(
            path,
            target_version=target_version,
            target_tag=target_tag,
            expected_commit=expected_commit,
        )

    return load


def create_archive(
    bundle_path: Path,
    archive_path: Path,
    *,
    target_version: str,
    target_tag: str,
    expected_commit: str,
) -> dict[str, Any]:
    bundle = bundle_path.resolve()
    if bundle.name != "bundle.json":
        raise ValueError("bundle manifest must be named bundle.json")
    loader = _bundle_loader(
        target_version=target_version,
        target_tag=target_tag,
        expected_commit=expected_commit,
    )
    gate = loader(bundle)
    if not gate["passed"]:
        failed = [
            name for name, passed in gate.get("checks", {}).items() if not passed
        ]
        raise ValueError("bundle is not ready: " + ", ".join(failed))

    result = create_canonical_archive(
        bundle.parent,
        archive_path,
        ARCHIVE_ROOT,
    )
    validation = evaluate_canonical_archive(
        archive_path,
        archive_root=ARCHIVE_ROOT,
        manifest_relative="bundle.json",
        validate_manifest=loader,
        manifest_passes=lambda payload: payload.get("passed") is True,
    )
    if not validation["valid"]:
        failed = [
            name
            for name, passed in validation["checks"].items()
            if not passed
        ]
        raise ValueError("created archive is invalid: " + ", ".join(failed))
    return result


def evaluate_archive(
    archive_path: Path,
    *,
    target_version: str,
    target_tag: str,
    expected_commit: str,
    expected_sha256: str | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    loader = _bundle_loader(
        target_version=target_version,
        target_tag=target_tag,
        expected_commit=expected_commit,
    )
    return evaluate_canonical_archive(
        archive_path,
        archive_root=ARCHIVE_ROOT,
        manifest_relative="bundle.json",
        validate_manifest=loader,
        manifest_passes=lambda payload: payload.get("passed") is True,
        expected_sha256=expected_sha256,
        max_total_bytes=max_total_bytes,
    )


def load_archive(
    archive_path: Path,
    *,
    target_version: str,
    target_tag: str,
    expected_commit: str,
    checksum_path: Path | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    loader = _bundle_loader(
        target_version=target_version,
        target_tag=target_tag,
        expected_commit=expected_commit,
    )
    return load_canonical_archive(
        archive_path,
        archive_root=ARCHIVE_ROOT,
        manifest_relative="bundle.json",
        validate_manifest=loader,
        manifest_passes=lambda payload: payload.get("passed") is True,
        checksum_path=checksum_path,
        max_total_bytes=max_total_bytes,
    )


__all__ = [
    "ARCHIVE_ROOT",
    "create_archive",
    "evaluate_archive",
    "load_archive",
    "read_checksum",
    "write_checksum",
]
