#!/usr/bin/env python3
"""Publish compact, content-addressed contact robustness evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from contact_robustness import evaluate_manifest, sha256_file


PUBLISHED_ARTIFACTS = ("summary", "comparisons", "report")


def _result_date(manifest: dict[str, Any]) -> str:
    finished_at = manifest.get("finished_at")
    if not isinstance(finished_at, str):
        raise ValueError("manifest is missing finished_at")
    try:
        return datetime.fromisoformat(finished_at.replace("Z", "+00:00")).date().isoformat()
    except ValueError as exception:
        raise ValueError("manifest finished_at is not ISO-8601") from exception


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def publish(
    evidence_directory: Path,
    output_directory: Path,
    *,
    profile: str = "release",
    result_id: str | None = None,
) -> dict[str, Any]:
    evidence_directory = evidence_directory.resolve()
    manifest_path = evidence_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = evaluate_manifest(manifest, evidence_directory, profile)
    if not validation["passed"]:
        failed = sorted(name for name, passed in validation["checks"].items() if not passed)
        raise ValueError(f"source evidence failed validation: {', '.join(failed)}")

    identifier = result_id or f"contact_robustness_{_result_date(manifest)}"
    if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in identifier):
        raise ValueError("result id must contain only lowercase letters, numbers, '_' or '-'")

    output_directory.mkdir(parents=True, exist_ok=True)
    published: dict[str, dict[str, str]] = {}
    for name in PUBLISHED_ARTIFACTS:
        artifact = manifest["artifacts"][name]
        source = (evidence_directory / artifact["path"]).resolve()
        suffix = source.suffix
        destination = output_directory / f"{identifier}_{name}{suffix}"
        _atomic_copy(source, destination)
        published[name] = {
            "path": destination.name,
            "sha256": sha256_file(destination),
        }

    provenance_path = output_directory / f"{identifier}_provenance.json"
    provenance = {
        "schema_version": 1,
        "evidence_mode": "published_contact_robustness",
        "result_id": identifier,
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "profile": manifest["profile"],
            "finished_at": manifest["finished_at"],
            "experiment": manifest["experiment"],
            "git_dirty": manifest["git_dirty"],
            "gpu": manifest["gpu"],
            "matrix": manifest["matrix"],
            "integrity_gate": manifest["integrity_gate"],
            "outcome": manifest["outcome"],
            "artifacts": {
                name: {"sha256": artifact["sha256"]}
                for name, artifact in sorted(manifest["artifacts"].items())
            },
        },
        "validation": {
            "passed": True,
            "checks": validation["checks"],
        },
        "published_artifacts": published,
    }
    _atomic_json(provenance, provenance_path)
    provenance["provenance"] = {
        "path": provenance_path.name,
        "sha256": sha256_file(provenance_path),
    }
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--profile", choices=("smoke", "release"), default="release")
    parser.add_argument("--result-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = publish(
        args.evidence_directory,
        args.output_dir,
        profile=args.profile,
        result_id=args.result_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
