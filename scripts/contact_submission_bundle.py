#!/usr/bin/env python3
"""Validate a portable, anonymized contact-rich Diff-MPPI submission bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from paper_artifact_contract import validate_manifest


MODE = "contact_rich_diff_mppi_submission_bundle"
PAPER_ID = "contact_rich_diff_mppi"
REQUIRED_FIGURES = {
    "contact_robustness",
    "contact_matched_compute",
    "contact_external_fidelity",
}
FORBIDDEN_IDENTITY_TOKENS = (
    b"ryohei",
    b"sasaki",
    b"rsasa",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = (root / value).resolve()
    return (
        candidate
        if candidate.is_relative_to(root) and candidate.is_file()
        else None
    )


def _artifact_url_ready(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    lowered = value.lower()
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.netloc.lower() not in {"example.com", "example.org"}
        and ".invalid" not in parsed.netloc.lower()
        and not any(token.decode() in lowered for token in FORBIDDEN_IDENTITY_TOKENS)
    )


def evaluate_bundle(
    manifest: dict[str, Any],
    bundle_root: Path,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = bundle_root.resolve()
    files = manifest.get("files")
    file_checks: dict[str, bool] = {}
    paths: list[Path] = []
    categories: set[str] = set()
    category_by_path: dict[str, str] = {}
    if isinstance(files, list):
        for index, item in enumerate(files):
            key = str(item.get("path", f"entry-{index}")) if isinstance(
                item, dict
            ) else f"entry-{index}"
            path = _safe_file(root, item.get("path")) if isinstance(item, dict) else None
            valid = (
                path is not None
                and isinstance(item.get("bytes"), int)
                and item["bytes"] == path.stat().st_size
                and bool(re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))))
                and sha256_file(path) == item["sha256"]
                and isinstance(item.get("category"), str)
                and bool(item["category"])
            )
            file_checks[key] = valid
            if path is not None:
                paths.append(path)
            if isinstance(item, dict) and isinstance(item.get("category"), str):
                categories.add(item["category"])
                if isinstance(item.get("path"), str):
                    category_by_path[item["path"]] = item["category"]
    unique_paths = len(paths) == len(set(paths))
    declared_paths = set(category_by_path)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "submission_manifest.json"
    }

    ledger_path = _safe_file(
        root, manifest.get("anonymous_ledger", {}).get("path")
        if isinstance(manifest.get("anonymous_ledger"), dict)
        else None
    )
    ledger_gate: dict[str, Any] = {"valid": False, "ready": False}
    if ledger_path is not None:
        try:
            ledger_gate = validate_manifest(
                json.loads(ledger_path.read_text(encoding="utf-8")), root
            )
        except (json.JSONDecodeError, OSError, UnicodeError):
            pass

    figure_manifest_path = _safe_file(
        root, manifest.get("figure_manifest", {}).get("path")
        if isinstance(manifest.get("figure_manifest"), dict)
        else None
    )
    figure_sources = False
    figure_stems: set[str] = set()
    figure_formats: set[tuple[str, str]] = set()
    if figure_manifest_path is not None:
        try:
            figures = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
            sources = figures.get("sources", {})
            figure_sources = (
                figures.get("schema_version") == 1
                and figures.get("evidence_mode") == "contact_submission_figures"
                and isinstance(sources, dict)
                and bool(sources)
                and all(
                    (source_path := _safe_file(root, source.get("path"))) is not None
                    and sha256_file(source_path) == source.get("sha256")
                    for source in sources.values()
                    if isinstance(source, dict)
                )
                and len(sources)
                == sum(isinstance(source, dict) for source in sources.values())
            )
        except (json.JSONDecodeError, OSError, UnicodeError):
            figure_sources = False
    for path in paths:
        if path.suffix.lower() in {".pdf", ".svg", ".png"}:
            figure_stems.add(path.stem)
            figure_formats.add((path.stem, path.suffix.lower()))

    redactions = manifest.get("redactions")
    redactions_valid = isinstance(redactions, list) and bool(redactions)
    if redactions_valid:
        for item in redactions:
            path = (
                _safe_file(root, item.get("path"))
                if isinstance(item, dict)
                else None
            )
            redactions_valid = redactions_valid and (
                path is not None
                and bool(
                    re.fullmatch(
                        r"[0-9a-f]{64}", str(item.get("source_sha256", ""))
                    )
                )
                and sha256_file(path) == item.get("bundle_sha256")
                and isinstance(item.get("replacements"), int)
                and item["replacements"] > 0
            )

    identity_clean = True
    for path in paths:
        try:
            lowered = path.read_bytes().lower()
        except OSError:
            identity_clean = False
            break
        if any(token in lowered for token in FORBIDDEN_IDENTITY_TOKENS):
            identity_clean = False
            break
        if re.search(
            rb"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", lowered
        ):
            identity_clean = False
            break

    source_commit = manifest.get("source_commit")
    structural_checks = {
        "schema": manifest.get("schema_version") == 1,
        "mode": manifest.get("evidence_mode") == MODE,
        "paper_id": manifest.get("paper_id") == PAPER_ID,
        "anonymous": manifest.get("anonymous") is True,
        "files": isinstance(files, list)
        and bool(files)
        and all(file_checks.values()),
        "unique_paths": unique_paths,
        "complete_inventory": declared_paths == actual_paths,
        "categories": {
            "manuscript",
            "ledger",
            "generated_results",
            "protocol",
            "evidence",
            "figure",
            "figure_manifest",
        }
        <= categories,
        "category_bindings": (
            ledger_path is not None
            and category_by_path.get(ledger_path.relative_to(root).as_posix())
            == "ledger"
            and figure_manifest_path is not None
            and category_by_path.get(
                figure_manifest_path.relative_to(root).as_posix()
            )
            == "figure_manifest"
        ),
        "source_commit": bool(
            re.fullmatch(r"[0-9a-f]{40}", str(source_commit))
        )
        and (expected_commit is None or source_commit == expected_commit),
        "source_ledger_hash": bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(manifest.get("source_ledger_sha256", "")),
            )
        ),
        "anonymous_ledger": ledger_gate.get("valid") is True
        and ledger_gate.get("ready") is True,
        "figure_manifest": figure_sources,
        "required_figures": REQUIRED_FIGURES <= figure_stems
        and all(
            (stem, suffix) in figure_formats
            for stem in REQUIRED_FIGURES
            for suffix in (".pdf", ".svg", ".png")
        ),
        "redactions": redactions_valid,
        "identity_clean": identity_clean,
    }
    readiness_checks = {
        "clean_commit": manifest.get("git_dirty") is False,
        "venue_selected": isinstance(manifest.get("venue"), str)
        and manifest["venue"].strip().lower()
        not in {"", "unselected", "tbd", "todo"},
        "artifact_url": _artifact_url_ready(manifest.get("artifact_url")),
    }
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
    manifest_path: Path, expected_commit: str | None = None
) -> dict[str, Any]:
    try:
        path = manifest_path.resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return {
            "valid": False,
            "ready": False,
            "checks": {"manifest_readable": False},
            "readiness_checks": {},
            "file_checks": {},
            "ledger": {"valid": False, "ready": False},
        }
    return evaluate_bundle(manifest, path.parent, expected_commit)
