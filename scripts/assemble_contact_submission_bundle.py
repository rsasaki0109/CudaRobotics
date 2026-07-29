#!/usr/bin/env python3
"""Assemble an anonymized, content-bound contact-paper submission bundle."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from contact_submission_bundle import evaluate_bundle
from paper_artifact_contract import sha256_file, validate_manifest
from render_contact_submission_figures import render


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "paper" / "artifacts" / "contact_rich_diff_mppi.json"
PRIMARY_FILES = {
    "paper/diff_mppi_submission_draft.md": "manuscript",
    "paper/latex/contact_rich_diff_mppi.tex": "manuscript_source",
    "paper/latex/references.bib": "bibliography",
    "paper/contact_rich_diff_mppi_results.md": "generated_results",
    "paper/contact_rich_diff_mppi_plan.md": "plan",
    "docs/contact_diff_mppi_robustness.md": "protocol",
    "docs/contact_matched_compute.md": "protocol",
    "docs/contact_external_fidelity.md": "protocol",
}
RESULT_PREFIXES = (
    "contact_robustness_2026-07-28_",
    "contact_matched_compute_2026-07-28_",
    "contact_external_fidelity_2026-07-28_",
)


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _redact_json(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        result = {}
        count = 0
        for key, item in value.items():
            redacted, item_count = _redact_json(item)
            result[key] = redacted
            count += item_count
        return result, count
    if isinstance(value, list):
        result = []
        count = 0
        for item in value:
            redacted, item_count = _redact_json(item)
            result.append(redacted)
            count += item_count
        return result, count
    if isinstance(value, str) and (
        re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/")
    ):
        basename = value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return f"<REDACTED_PATH>/{basename}", 1
    return value, 0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _file_entry(root: Path, relative: str, category: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "category": category,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assemble(
    output_dir: Path,
    source_commit: str,
    venue: str,
    artifact_url: str,
    git_dirty: bool,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    source_gate = validate_manifest(source_ledger, ROOT)
    if not source_gate["valid"] or not source_gate["ready"]:
        raise ValueError("source contact-paper ledger is not ready")
    anonymous_ledger = copy.deepcopy(source_ledger)
    redactions = []

    for evidence in anonymous_ledger["evidence"]:
        for artifact in evidence.get("artifacts", []):
            relative = artifact["path"]
            source = ROOT / relative
            destination = output / relative
            if source.suffix.lower() == ".json":
                payload = json.loads(source.read_text(encoding="utf-8"))
                sanitized, count = _redact_json(payload)
                _write_json(destination, sanitized)
                bundled_hash = sha256_file(
                    destination, artifact.get("normalization")
                )
                redactions.append(
                    {
                        "path": relative,
                        "source_sha256": artifact["sha256"],
                        "bundle_sha256": sha256_file(destination),
                        "normalization": artifact.get("normalization"),
                        "replacements": count,
                    }
                )
                artifact["sha256"] = bundled_hash
            else:
                _copy(source, destination)

    anonymous_ledger_path = output / "paper/artifacts/contact_rich_diff_mppi.json"
    _write_json(anonymous_ledger_path, anonymous_ledger)
    anonymous_gate = validate_manifest(anonymous_ledger, output)
    if not anonymous_gate["valid"] or not anonymous_gate["ready"]:
        raise ValueError("anonymized ledger did not retain ready status")

    categories = dict(PRIMARY_FILES)
    for relative in PRIMARY_FILES:
        _copy(ROOT / relative, output / relative)
    result_dir = ROOT / "docs" / "results"
    for source in sorted(result_dir.iterdir()):
        if source.is_file() and (
            source.name == "soppi_box_pushing_2026-06-14.csv"
            or any(source.name.startswith(prefix) for prefix in RESULT_PREFIXES)
        ):
            relative = source.relative_to(ROOT).as_posix()
            if not (output / relative).exists():
                if source.name.endswith("_provenance.json"):
                    payload = json.loads(source.read_text(encoding="utf-8"))
                    sanitized, count = _redact_json(payload)
                    _write_json(output / relative, sanitized)
                    redactions.append(
                        {
                            "path": relative,
                            "source_sha256": sha256_file(source),
                            "bundle_sha256": sha256_file(output / relative),
                            "replacements": count,
                        }
                    )
                else:
                    _copy(source, output / relative)
            categories[relative] = "evidence"

    figure_dir = output / "paper" / "figures" / "submission"
    render(figure_dir)
    for figure in sorted(figure_dir.iterdir()):
        relative = figure.relative_to(output).as_posix()
        categories[relative] = (
            "figure_manifest"
            if figure.name == "figure_manifest.json"
            else "figure"
        )
    categories["paper/artifacts/contact_rich_diff_mppi.json"] = "ledger"
    for evidence in anonymous_ledger["evidence"]:
        for artifact in evidence.get("artifacts", []):
            categories[artifact["path"]] = "evidence"

    files = [
        _file_entry(output, relative, category)
        for relative, category in sorted(categories.items())
    ]
    figure_manifest_relative = (
        "paper/figures/submission/figure_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "evidence_mode": "contact_rich_diff_mppi_submission_bundle",
        "paper_id": "contact_rich_diff_mppi",
        "title": source_ledger["title"],
        "source_commit": source_commit,
        "git_dirty": git_dirty,
        "anonymous": True,
        "venue": venue,
        "artifact_url": artifact_url,
        "source_ledger_sha256": sha256_file(LEDGER),
        "anonymous_ledger": {
            "path": "paper/artifacts/contact_rich_diff_mppi.json",
            "sha256": sha256_file(anonymous_ledger_path),
        },
        "figure_manifest": {
            "path": figure_manifest_relative,
            "sha256": sha256_file(output / figure_manifest_relative),
        },
        "redactions": redactions,
        "files": files,
    }
    gate = evaluate_bundle(manifest, output, source_commit)
    manifest["validation"] = {
        "valid": gate["valid"],
        "ready": gate["ready"],
        "checks": gate["checks"],
        "readiness_checks": gate["readiness_checks"],
    }
    _write_json(output / "submission_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--venue", default="unselected")
    parser.add_argument("--artifact-url", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    commit = args.commit or git_text("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("--commit must be a full lowercase Git commit")
    dirty = bool(git_text("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree is dirty; commit the submission sources first "
            "or use --allow-dirty for a non-ready diagnostic bundle"
        )
    try:
        manifest = assemble(
            args.output_dir,
            commit,
            args.venue,
            args.artifact_url,
            dirty,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot assemble contact submission bundle: {error}") from error
    print((args.output_dir.resolve() / "submission_manifest.json"))
    return 0 if manifest["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
