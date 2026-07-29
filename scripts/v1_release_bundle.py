#!/usr/bin/env python3
"""Validate a portable post-tag bundle of all v1 release attestations."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from v1_release_attestation import MODES, load_reference


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def evaluate_bundle(
    bundle: dict[str, Any],
    directory: Path,
    *,
    target_version: str,
    target_tag: str,
    expected_commit: str,
) -> dict[str, Any]:
    root = directory.resolve()
    references = bundle.get("attestations")
    if not isinstance(references, dict):
        references = {}
    gates = {
        key: load_reference(
            references.get(key),
            repo_root=root,
            key=key,
            target_version=target_version,
            target_tag=target_tag,
        )
        for key in MODES
    }
    commits = {
        gate.get("git_commit")
        for gate in gates.values()
        if gate.get("passed") is True
    }
    checks = {
        "schema": bundle.get("schema_version") == 1,
        "evidence_mode": bundle.get("evidence_mode")
        == "v1_release_evidence_bundle",
        "status": bundle.get("status") == "passed",
        "version": bundle.get("version") == target_version,
        "target_tag": bundle.get("target_tag") == target_tag,
        "git_commit": bool(
            re.fullmatch(
                r"[0-9a-f]{40}", str(bundle.get("git_commit", ""))
            )
        )
        and bundle.get("git_commit") == expected_commit,
        "attestation_table": set(references) == set(MODES),
        "all_attestations": all(
            gate.get("passed") is True for gate in gates.values()
        ),
        "same_subject_commit": commits == {expected_commit}
        and bundle.get("git_commit") == expected_commit,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "git_commit": bundle.get("git_commit"),
        "attestations": gates,
        "references": references,
    }


def load_bundle(
    path: Path,
    *,
    target_version: str,
    target_tag: str,
    expected_commit: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        bundle = read_object(resolved)
        return evaluate_bundle(
            bundle,
            resolved.parent,
            target_version=target_version,
            target_tag=target_tag,
            expected_commit=expected_commit,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "passed": False,
            "checks": {"bundle_readable": False},
            "error": str(error),
            "attestations": {},
            "references": {},
        }
