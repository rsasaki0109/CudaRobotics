#!/usr/bin/env python3
"""Validate claim-to-evidence manifests for CudaRobotics papers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


CLAIM_STATUSES = {"planned", "partial", "supported", "refuted"}
EVIDENCE_STATUSES = {"pending", "complete"}
EVIDENCE_KINDS = {"file_set", "csv_assertions", "json_assertions"}
OPERATORS = {"eq", "approx", "lt", "le", "gt", "ge"}


def sha256_file(path: Path, normalization: str | None = None) -> str:
    digest = hashlib.sha256()
    if normalization == "text_lf":
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
    elif normalization is None:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    else:
        raise ValueError(f"unsupported artifact normalization: {normalization}")
    return digest.hexdigest()


def _safe_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = (root / relative).resolve()
    return candidate if candidate.is_relative_to(root) else None


def _compare(actual: Any, assertion: dict[str, Any]) -> bool:
    operator = assertion.get("op")
    expected = assertion.get("value")
    if operator not in OPERATORS:
        return False
    if operator == "eq":
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                return math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=float(assertion.get("tolerance", 0.0)),
                )
            except (TypeError, ValueError):
                return False
        return actual == expected
    try:
        number = float(actual)
        target = float(expected)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number) or not math.isfinite(target):
        return False
    if operator == "approx":
        return math.isclose(
            number,
            target,
            rel_tol=0.0,
            abs_tol=float(assertion.get("tolerance", 0.0)),
        )
    return {
        "lt": number < target,
        "le": number <= target,
        "gt": number > target,
        "ge": number >= target,
    }[operator]


def _aggregate_csv(rows: list[dict[str, str]], assertion: dict[str, Any]) -> Any:
    filters = assertion.get("filters", {})
    selected = [
        row
        for row in rows
        if all(row.get(key) == str(value) for key, value in filters.items())
    ]
    if len(selected) < assertion.get("min_rows", 1):
        raise ValueError("insufficient filtered rows")
    aggregate = assertion.get("aggregate", "mean")
    field = assertion.get("field")
    if aggregate == "count":
        return len(selected)
    values = [float(row[field]) for row in selected]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite CSV value")
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "min":
        return min(values)
    if aggregate == "max":
        return max(values)
    if aggregate == "sum":
        return sum(values)
    raise ValueError("unsupported CSV aggregate")


def _json_value(payload: Any, dotted_path: str) -> Any:
    current = payload
    for token in dotted_path.split("."):
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(token)
    return current


def _validate_artifact(
    root: Path, artifact: dict[str, Any]
) -> tuple[bool, Path | None]:
    path = _safe_path(root, artifact.get("path"))
    try:
        valid = (
            path is not None
            and path.is_file()
            and path.stat().st_size > 0
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", ""))))
            and sha256_file(path, artifact.get("normalization"))
            == artifact["sha256"]
        )
    except (OSError, ValueError):
        valid = False
    return valid, path


def validate_manifest(
    manifest: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    root = repo_root.resolve()
    errors: list[str] = []
    checks: dict[str, bool] = {
        "schema_version": manifest.get("schema_version") == 1,
        "paper_id": bool(
            re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(manifest.get("paper_id", "")))
        ),
        "title": isinstance(manifest.get("title"), str) and bool(manifest["title"]),
    }
    claims = manifest.get("claims")
    evidence = manifest.get("evidence")
    checks["claims_table"] = isinstance(claims, list) and bool(claims)
    checks["evidence_table"] = isinstance(evidence, list) and bool(evidence)
    if not checks["claims_table"] or not checks["evidence_table"]:
        return {
            "valid": False,
            "ready": False,
            "checks": checks,
            "errors": ["claims and evidence must be non-empty lists"],
            "claims": {},
            "evidence": {},
        }

    claim_by_id: dict[str, dict[str, Any]] = {}
    for claim in claims:
        claim_id = claim.get("id") if isinstance(claim, dict) else None
        if (
            not isinstance(claim_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", claim_id)
            or claim_id in claim_by_id
        ):
            errors.append(f"invalid or duplicate claim id: {claim_id!r}")
            continue
        if claim.get("status") not in CLAIM_STATUSES:
            errors.append(f"{claim_id}: invalid claim status")
        if not isinstance(claim.get("statement"), str) or not claim["statement"]:
            errors.append(f"{claim_id}: missing statement")
        if not isinstance(claim.get("submission_required"), bool):
            errors.append(f"{claim_id}: submission_required must be boolean")
        if not isinstance(claim.get("evidence"), list) or not claim["evidence"]:
            errors.append(f"{claim_id}: evidence must be a non-empty list")
        if not isinstance(claim.get("limitations"), list):
            errors.append(f"{claim_id}: limitations must be a list")
        elif (
            claim.get("status") in {"planned", "partial", "refuted"}
            and not claim["limitations"]
        ):
            errors.append(f"{claim_id}: non-supported claim needs limitations")
        claim_by_id[claim_id] = claim

    evidence_by_id: dict[str, dict[str, Any]] = {}
    for item in evidence:
        evidence_id = item.get("id") if isinstance(item, dict) else None
        if (
            not isinstance(evidence_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", evidence_id)
            or evidence_id in evidence_by_id
        ):
            errors.append(f"invalid or duplicate evidence id: {evidence_id!r}")
            continue
        if item.get("status") not in EVIDENCE_STATUSES:
            errors.append(f"{evidence_id}: invalid evidence status")
        if item.get("kind") not in EVIDENCE_KINDS:
            errors.append(f"{evidence_id}: invalid evidence kind")
        command = item.get("generator_command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(token, str) and token for token in command)
        ):
            errors.append(f"{evidence_id}: generator_command must be argv")
        evidence_by_id[evidence_id] = item

    evidence_results: dict[str, dict[str, Any]] = {}
    for evidence_id, item in evidence_by_id.items():
        complete = item.get("status") == "complete"
        result = {"declared_complete": complete, "valid": not complete, "checks": {}}
        if complete:
            artifacts = item.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                result["valid"] = False
                result["checks"]["artifacts"] = False
            else:
                artifact_paths: list[Path] = []
                artifact_ok = True
                for artifact in artifacts:
                    valid, path = _validate_artifact(root, artifact)
                    artifact_ok = artifact_ok and valid
                    if path is not None:
                        artifact_paths.append(path)
                result["checks"]["artifacts"] = artifact_ok
                result["valid"] = artifact_ok
                assertions = item.get("assertions", [])
                if item["kind"] in {"csv_assertions", "json_assertions"}:
                    assertion_ok = bool(assertions) and len(artifact_paths) == 1
                    if assertion_ok:
                        try:
                            if item["kind"] == "csv_assertions":
                                with artifact_paths[0].open(
                                    newline="", encoding="utf-8"
                                ) as stream:
                                    rows = list(csv.DictReader(stream))
                                assertion_ok = bool(rows) and all(
                                    _compare(_aggregate_csv(rows, assertion), assertion)
                                    for assertion in assertions
                                )
                            else:
                                payload = json.loads(
                                    artifact_paths[0].read_text(encoding="utf-8")
                                )
                                assertion_ok = all(
                                    _compare(
                                        _json_value(payload, assertion["json_path"]),
                                        assertion,
                                    )
                                    for assertion in assertions
                                )
                        except (
                            csv.Error,
                            json.JSONDecodeError,
                            KeyError,
                            OSError,
                            TypeError,
                            ValueError,
                        ):
                            assertion_ok = False
                    result["checks"]["assertions"] = assertion_ok
                    result["valid"] = result["valid"] and assertion_ok
        evidence_results[evidence_id] = result

    claim_results: dict[str, dict[str, Any]] = {}
    for claim_id, claim in claim_by_id.items():
        identifiers = claim.get("evidence", [])
        references_exist = all(
            isinstance(identifier, str) and identifier in evidence_by_id
            for identifier in identifiers
        )
        evidence_valid = references_exist and all(
            evidence_results[identifier]["declared_complete"]
            and evidence_results[identifier]["valid"]
            for identifier in identifiers
        )
        status_consistent = (
            claim.get("status") not in {"supported", "refuted"}
            or evidence_valid
        )
        claim_results[claim_id] = {
            "references_exist": references_exist,
            "evidence_valid": evidence_valid,
            "status_consistent": status_consistent,
            "submission_required": claim.get("submission_required") is True,
            "status": claim.get("status"),
        }
        if not references_exist:
            errors.append(f"{claim_id}: references unknown evidence")
        if not status_consistent:
            errors.append(
                f"{claim_id}: supported/refuted claim lacks complete valid evidence"
            )

    valid = all(checks.values()) and not errors and all(
        result["valid"] for result in evidence_results.values()
    )
    required = [
        result for result in claim_results.values() if result["submission_required"]
    ]
    ready = (
        valid
        and bool(required)
        and all(
            result["status"] == "supported" and result["evidence_valid"]
            for result in required
        )
    )
    return {
        "valid": valid,
        "ready": ready,
        "checks": checks,
        "errors": errors,
        "claims": claim_results,
        "evidence": evidence_results,
    }
