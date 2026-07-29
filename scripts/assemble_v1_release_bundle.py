#!/usr/bin/env python3
"""Assemble four post-tag v1 attestations into one portable release bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any

from v1_release_attestation import MODES, sha256_file
from v1_release_bundle import evaluate_bundle


FILENAMES = {
    "quickstart_15_minute_evidence": "quickstart.json",
    "cudanav_release_evidence": "cudanav.json",
    "docker_gpu_evidence": "docker_gpu.json",
    "documentation_deployment": "documentation.json",
}


def encoded(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def assemble(
    sources: dict[str, Path],
    output_directory: Path,
    *,
    version: str,
    target_tag: str,
    git_commit: str,
) -> dict[str, Any]:
    if set(sources) != set(MODES):
        raise ValueError("exactly four named attestation sources are required")
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("git_commit must be a full lowercase commit")
    output = output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    references = {}
    for key, filename in FILENAMES.items():
        source = sources[key].resolve()
        if not source.is_file():
            raise ValueError(f"attestation is missing: {source}")
        target = output / filename
        shutil.copyfile(source, target)
        references[key] = {
            "path": filename,
            "sha256": sha256_file(target),
        }
    bundle = {
        "schema_version": 1,
        "evidence_mode": "v1_release_evidence_bundle",
        "status": "passed",
        "version": version,
        "target_tag": target_tag,
        "git_commit": git_commit,
        "attestations": references,
    }
    gate = evaluate_bundle(
        bundle,
        output,
        target_version=version,
        target_tag=target_tag,
        expected_commit=git_commit,
    )
    if not gate["passed"]:
        failed = sorted(
            name for name, passed in gate["checks"].items() if not passed
        )
        raise ValueError(
            "release evidence bundle failed: " + ", ".join(failed)
        )
    (output / "bundle.json").write_text(
        encoded(bundle), encoding="utf-8"
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quickstart", type=Path, required=True)
    parser.add_argument("--cudanav", type=Path, required=True)
    parser.add_argument("--docker-gpu", type=Path, required=True)
    parser.add_argument("--documentation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        assemble(
            {
                "quickstart_15_minute_evidence": args.quickstart,
                "cudanav_release_evidence": args.cudanav,
                "docker_gpu_evidence": args.docker_gpu,
                "documentation_deployment": args.documentation,
            },
            args.output_dir,
            version=args.version,
            target_tag=args.tag,
            git_commit=args.commit,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot assemble v1 release bundle: {error}") from error
    print(args.output_dir.resolve() / "bundle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
