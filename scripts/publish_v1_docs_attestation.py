#!/usr/bin/env python3
"""Publish validated v1 documentation deployment evidence as an attestation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v1_documentation_evidence import SITE, evaluate_manifest
from v1_quickstart_evidence import sha256_file
from v1_release_attestation import MODES, validate_payload


ROOT = Path(__file__).resolve().parents[1]
KEY = "documentation_deployment"


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def build_attestation(directory: Path) -> dict[str, Any]:
    root = directory.resolve()
    manifest_path = root / "manifest.json"
    manifest = read_object(manifest_path)
    commit = manifest.get("git_commit")
    gate = evaluate_manifest(
        manifest,
        root,
        expected_commit=commit if isinstance(commit, str) else None,
    )
    if not gate["passed"]:
        failed = sorted(
            name for name, passed in gate["checks"].items() if not passed
        )
        raise ValueError(
            "documentation deployment evidence failed: "
            + ", ".join(failed)
        )
    attestation = {
        "schema_version": 1,
        "evidence_mode": MODES[KEY],
        "status": "passed",
        "version": manifest["version"],
        "target_tag": manifest["target_tag"],
        "git_commit": commit,
        "git_dirty": manifest["git_dirty"],
        "payload_sha256": sha256_file(manifest_path),
        "checks": gate["checks"],
        "details": {
            "site": SITE,
            "deployed_tag": manifest["target_tag"],
            "install_page_pass": True,
            "nav2_page_pass": True,
            "release_links_pass": True,
            "source_manifest_sha256": sha256_file(manifest_path),
        },
    }
    attestation_gate = validate_payload(
        attestation,
        key=KEY,
        target_version=manifest["version"],
        target_tag=manifest["target_tag"],
    )
    if not attestation_gate["passed"]:
        failed = sorted(
            name
            for name, passed in attestation_gate["checks"].items()
            if not passed
        )
        raise ValueError(
            "generated documentation v1 attestation failed: "
            + ", ".join(failed)
        )
    return attestation


def encoded(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit("--output must remain inside the repository")
    try:
        content = encoded(build_attestation(args.evidence_dir))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot publish documentation v1 attestation: {error}"
        ) from error
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            print(f"stale documentation v1 attestation: {output}")
            return 1
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    reference = {
        "path": output.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(output),
    }
    print(json.dumps(reference, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
