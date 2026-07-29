#!/usr/bin/env python3
"""Assemble a ready systems-paper ledger into a portable artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from canonical_evidence_archive import sha256_file
from paper_artifact_contract import validate_manifest
from systems_paper_bundle import (
    evaluate_bundle,
    evaluate_manuscript,
    manuscript_links,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = Path("paper/artifacts/cudarobotics_systems.json")
MANUSCRIPT = Path("paper/cudarobotics_systems_paper.md")


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def copy(source_root: Path, output: Path, relative: str) -> Path:
    source = (source_root / relative).resolve()
    if not source.is_relative_to(source_root) or not source.is_file():
        raise ValueError(f"unsafe or missing source artifact: {relative}")
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def file_entry(
    output: Path,
    relative: str,
    category: str,
) -> dict[str, Any]:
    path = output / relative
    return {
        "path": relative,
        "category": category,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assemble(
    source_root: Path,
    output_dir: Path,
    source_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    source = source_root.resolve()
    output = output_dir.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full lowercase commit")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    ledger_path = source / LEDGER
    manuscript_path = source / MANUSCRIPT
    ledger = read_object(ledger_path)
    ledger_gate = validate_manifest(ledger, source)
    if not ledger_gate["valid"] or not ledger_gate["ready"]:
        raise ValueError("source systems-paper ledger is not ready")
    manuscript_gate = evaluate_manuscript(
        manuscript_path,
        ledger,
        source,
    )
    if not all(manuscript_gate.values()):
        failed = [
            name for name, passed in manuscript_gate.items() if not passed
        ]
        raise ValueError(
            "source systems manuscript is not final: " + ", ".join(failed)
        )

    output.mkdir(parents=True, exist_ok=True)
    categories = {
        LEDGER.as_posix(): "ledger",
        MANUSCRIPT.as_posix(): "manuscript",
    }
    manuscript = manuscript_path.read_text(encoding="utf-8")
    for target in manuscript_links(manuscript):
        path = (manuscript_path.parent / target).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"unsafe or missing manuscript link: {target}")
        relative = path.relative_to(source).as_posix()
        categories.setdefault(relative, "linked_document")
    for evidence in ledger["evidence"]:
        if evidence.get("status") != "complete":
            continue
        for artifact in evidence.get("artifacts", []):
            relative = artifact.get("path")
            if not isinstance(relative, str):
                raise ValueError("ledger artifact path is invalid")
            categories[relative] = "evidence"

    for relative in sorted(categories):
        copy(source, output, relative)
    files = [
        file_entry(output, relative, category)
        for relative, category in sorted(categories.items())
    ]
    manifest = {
        "schema_version": 1,
        "evidence_mode": "cudarobotics_systems_paper_bundle",
        "paper_id": "cudarobotics-systems",
        "title": ledger["title"],
        "source_commit": source_commit,
        "git_dirty": git_dirty,
        "ledger": {
            "path": LEDGER.as_posix(),
            "sha256": sha256_file(output / LEDGER),
        },
        "manuscript": {
            "path": MANUSCRIPT.as_posix(),
            "sha256": sha256_file(output / MANUSCRIPT),
        },
        "files": files,
    }
    gate = evaluate_bundle(manifest, output, source_commit)
    if not gate["valid"] or not gate["ready"]:
        failed = [
            name for name, passed in gate["checks"].items() if not passed
        ]
        failed.extend(
            name
            for name, passed in gate["readiness_checks"].items()
            if not passed
        )
        raise ValueError("assembled systems bundle failed: " + ", ".join(failed))
    manifest["validation"] = {
        "valid": gate["valid"],
        "ready": gate["ready"],
        "checks": gate["checks"],
        "readiness_checks": gate["readiness_checks"],
    }
    write_object(output / "submission_manifest.json", manifest)
    return manifest


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit")
    args = parser.parse_args()
    commit = args.commit or git_text("rev-parse", "HEAD")
    dirty = bool(git_text("status", "--porcelain"))
    if dirty:
        raise SystemExit("tracked worktree is dirty; commit paper sources first")
    try:
        assemble(ROOT, args.output_dir, commit, dirty)
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"cannot assemble systems paper bundle: {error}") from error
    print(args.output_dir.resolve() / "submission_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
