#!/usr/bin/env python3
"""Create and validate canonical CudaRobotics systems-paper archives."""

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
from systems_paper_bundle import load_bundle


ARCHIVE_ROOT = "cudarobotics-systems-paper-artifact"


def _bundle_loader(expected_commit: str):
    def load(path: Path) -> dict[str, Any]:
        return load_bundle(path, expected_commit)

    return load


def create_archive(
    manifest_path: Path,
    archive_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    manifest = manifest_path.resolve()
    if manifest.name != "submission_manifest.json":
        raise ValueError(
            "submission manifest must be named submission_manifest.json"
        )
    loader = _bundle_loader(expected_commit)
    gate = loader(manifest)
    if not gate["valid"] or not gate["ready"]:
        failed = [
            name for name, passed in gate.get("checks", {}).items() if not passed
        ]
        failed.extend(
            name
            for name, passed in gate.get("readiness_checks", {}).items()
            if not passed
        )
        raise ValueError("systems paper bundle is not ready: " + ", ".join(failed))

    result = create_canonical_archive(
        manifest.parent,
        archive_path,
        ARCHIVE_ROOT,
    )
    validation = evaluate_canonical_archive(
        archive_path,
        archive_root=ARCHIVE_ROOT,
        manifest_relative="submission_manifest.json",
        validate_manifest=loader,
        manifest_passes=lambda payload: bool(
            payload.get("valid") and payload.get("ready")
        ),
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
    expected_commit: str,
    *,
    expected_sha256: str | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    loader = _bundle_loader(expected_commit)
    return evaluate_canonical_archive(
        archive_path,
        archive_root=ARCHIVE_ROOT,
        manifest_relative="submission_manifest.json",
        validate_manifest=loader,
        manifest_passes=lambda payload: bool(
            payload.get("valid") and payload.get("ready")
        ),
        expected_sha256=expected_sha256,
        max_total_bytes=max_total_bytes,
    )


def load_archive(
    archive_path: Path,
    expected_commit: str,
    *,
    checksum_path: Path | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    loader = _bundle_loader(expected_commit)
    return load_canonical_archive(
        archive_path,
        archive_root=ARCHIVE_ROOT,
        manifest_relative="submission_manifest.json",
        validate_manifest=loader,
        manifest_passes=lambda payload: bool(
            payload.get("valid") and payload.get("ready")
        ),
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
