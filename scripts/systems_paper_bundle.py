#!/usr/bin/env python3
"""Validate a portable CudaRobotics systems-paper artifact bundle."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from canonical_evidence_archive import sha256_file
from paper_artifact_contract import validate_manifest


MODE = "cudarobotics_systems_paper_bundle"
PAPER_ID = "cudarobotics-systems"
MANIFEST_KEYS = {
    "schema_version",
    "evidence_mode",
    "paper_id",
    "title",
    "source_commit",
    "git_dirty",
    "ledger",
    "manuscript",
    "files",
}
FINAL_DRAFT_FORBIDDEN = (
    "not a submission-ready manuscript",
    "ready: false",
)


def _safe_file(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (root / relative).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def manuscript_links(manuscript: str) -> list[str]:
    return [
        target.split("#", 1)[0]
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", manuscript)
        if "://" not in target
        and not target.startswith("#")
        and target.split("#", 1)[0]
    ]


def evaluate_manuscript(
    manuscript_path: Path,
    ledger: dict[str, Any],
    root: Path,
) -> dict[str, bool]:
    try:
        text = manuscript_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        text = ""
    first_line = text.splitlines()[0] if text else ""
    claim_rows = all(
        f"`{claim['id']}`" in text
        and bool(
            re.search(
                rf"\| `{re.escape(claim['id'])}` \| "
                rf"{re.escape(claim['status'].title())} \|",
                text,
            )
        )
        for claim in ledger.get("claims", [])
        if isinstance(claim, dict)
    )
    links = manuscript_links(text)
    links_valid = bool(links)
    for target in links:
        path = (manuscript_path.parent / target).resolve()
        links_valid = links_valid and path.is_relative_to(root) and path.is_file()
    lowered = text.lower()
    return {
        "manuscript_readable": bool(text),
        "title": isinstance(ledger.get("title"), str)
        and ledger["title"] in first_line,
        "claim_status_rows": claim_rows,
        "local_links": links_valid,
        "final_status": not any(
            phrase in lowered for phrase in FINAL_DRAFT_FORBIDDEN
        ),
    }


def evaluate_bundle(
    manifest: dict[str, Any],
    bundle_root: Path,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = bundle_root.resolve()
    files = manifest.get("files")
    file_checks: dict[str, bool] = {}
    declared: set[str] = set()
    categories: set[str] = set()
    if isinstance(files, list):
        for index, entry in enumerate(files):
            key = (
                str(entry.get("path", f"entry-{index}"))
                if isinstance(entry, dict)
                else f"entry-{index}"
            )
            path = (
                _safe_file(root, entry.get("path"))
                if isinstance(entry, dict)
                else None
            )
            valid = (
                path is not None
                and isinstance(entry.get("bytes"), int)
                and entry["bytes"] == path.stat().st_size
                and bool(
                    re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(entry.get("sha256", "")),
                    )
                )
                and sha256_file(path) == entry["sha256"]
                and isinstance(entry.get("category"), str)
                and bool(entry["category"])
            )
            file_checks[key] = valid
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                declared.add(entry["path"])
            if isinstance(entry, dict) and isinstance(
                entry.get("category"), str
            ):
                categories.add(entry["category"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() != "submission_manifest.json"
    }

    ledger_reference = manifest.get("ledger")
    ledger_path = _safe_file(
        root,
        ledger_reference.get("path")
        if isinstance(ledger_reference, dict)
        else None,
    )
    ledger: dict[str, Any] = {}
    ledger_gate: dict[str, Any] = {"valid": False, "ready": False}
    if ledger_path is not None:
        try:
            candidate = json.loads(ledger_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                ledger = candidate
                ledger_gate = validate_manifest(ledger, root)
        except (json.JSONDecodeError, OSError, UnicodeError):
            pass

    manuscript_reference = manifest.get("manuscript")
    manuscript_path = _safe_file(
        root,
        manuscript_reference.get("path")
        if isinstance(manuscript_reference, dict)
        else None,
    )
    manuscript_gate = (
        evaluate_manuscript(manuscript_path, ledger, root)
        if manuscript_path is not None
        else {
            "manuscript_readable": False,
            "title": False,
            "claim_status_rows": False,
            "local_links": False,
            "final_status": False,
        }
    )

    source_commit = manifest.get("source_commit")
    manifest_keys = set(manifest)
    structural_checks = {
        "schema": manifest.get("schema_version") == 1,
        "manifest_schema": manifest_keys
        in (MANIFEST_KEYS, MANIFEST_KEYS | {"validation"}),
        "mode": manifest.get("evidence_mode") == MODE,
        "paper_id": manifest.get("paper_id") == PAPER_ID,
        "title": manifest.get("title") == ledger.get("title"),
        "source_commit": bool(
            re.fullmatch(r"[0-9a-f]{40}", str(source_commit))
        )
        and (expected_commit is None or source_commit == expected_commit),
        "files": isinstance(files, list)
        and bool(files)
        and len(declared) == len(files)
        and all(file_checks.values()),
        "complete_inventory": declared == actual,
        "categories": {
            "manuscript",
            "ledger",
            "evidence",
            "linked_document",
        }
        <= categories,
        "ledger_reference": (
            ledger_path is not None
            and isinstance(ledger_reference, dict)
            and set(ledger_reference) == {"path", "sha256"}
            and ledger_reference["sha256"] == sha256_file(ledger_path)
        ),
        "ledger_valid": ledger_gate.get("valid") is True,
        "manuscript_reference": (
            manuscript_path is not None
            and isinstance(manuscript_reference, dict)
            and set(manuscript_reference) == {"path", "sha256"}
            and manuscript_reference["sha256"] == sha256_file(manuscript_path)
        ),
        **manuscript_gate,
    }
    readiness_checks = {
        "clean_commit": manifest.get("git_dirty") is False,
        "ledger_ready": ledger_gate.get("ready") is True,
    }
    base_valid = all(structural_checks.values())
    base_ready = base_valid and all(readiness_checks.values())
    expected_validation = {
        "valid": base_valid,
        "ready": base_ready,
        "checks": structural_checks.copy(),
        "readiness_checks": readiness_checks,
    }
    if "validation" in manifest:
        structural_checks["validation_record"] = (
            manifest["validation"] == expected_validation
        )
    valid = all(structural_checks.values())
    return {
        "valid": valid,
        "ready": valid and all(readiness_checks.values()),
        "checks": structural_checks,
        "readiness_checks": readiness_checks,
        "file_checks": file_checks,
        "ledger": ledger_gate,
    }


def load_bundle(
    manifest_path: Path,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    try:
        path = manifest_path.resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest root is not an object")
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError):
        return {
            "valid": False,
            "ready": False,
            "checks": {"manifest_readable": False},
            "readiness_checks": {},
            "file_checks": {},
            "ledger": {"valid": False, "ready": False},
        }
    return evaluate_bundle(manifest, path.parent, expected_commit)
