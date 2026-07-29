#!/usr/bin/env python3
"""Create portable, commit-bound GitHub CI evidence for v0.2 gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

GATE_CONTRACTS = {
    "github_build": {
        "workflow": "Build",
        "checks": {
            "cmake_configure",
            "cuda_build",
            "python_ctest",
            "cpu_ctest",
        },
        "artifacts": set(),
        "artifact_manifest": False,
    },
    "python_manylinux_wheels": {
        "workflow": "Python package",
        "checks": {
            "python_3_10_3_12_build_test",
            "sdist_and_native_wheel",
            "manylinux_cp310_cp312",
            "artifacts_uploaded",
        },
        "artifacts": {
            "cudarobotics-wheels",
            "cudarobotics-manylinux-wheels",
        },
        "artifact_manifest": True,
    },
}


def evaluate(
    payload: dict[str, Any],
    *,
    expected_gate: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    gate = payload.get("gate")
    contract = GATE_CONTRACTS.get(str(gate))
    github = payload.get("github", {})
    platform = payload.get("platform", {})
    checks_table = payload.get("checks")
    artifacts = payload.get("artifacts")
    commit = payload.get("git_commit")
    required_checks = contract["checks"] if contract else set()
    required_artifacts = contract["artifacts"] if contract else set()
    artifact_manifest = payload.get("artifact_manifest")
    checks = {
        "schema": payload.get("schema_version") == 1,
        "evidence_mode": payload.get("evidence_mode") == "release_ci",
        "status": payload.get("status") == "passed",
        "gate": contract is not None
        and (expected_gate is None or gate == expected_gate),
        "workflow": contract is not None
        and github.get("workflow") == contract["workflow"],
        "git_commit": isinstance(commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", commit))
        and (expected_commit is None or commit == expected_commit),
        "clean_checkout": payload.get("git_dirty") is False,
        "repository": github.get("repository")
        == "rsasaki0109/CudaRobotics",
        "run_id": isinstance(github.get("run_id"), int)
        and github["run_id"] > 0,
        "run_attempt": isinstance(github.get("run_attempt"), int)
        and github["run_attempt"] > 0,
        "run_url": github.get("run_url")
        == (
            f"https://github.com/{github.get('repository')}/actions/runs/"
            f"{github.get('run_id')}"
        ),
        "event": github.get("event") in {"push", "workflow_dispatch"},
        "ref": isinstance(github.get("ref"), str)
        and github["ref"].startswith("refs/"),
        "runner": platform.get("os") == "Linux"
        and platform.get("arch") in {"X64", "ARM64"},
        "checks": isinstance(checks_table, dict)
        and required_checks <= set(checks_table)
        and all(
            checks_table.get(name) == "passed"
            for name in required_checks
        ),
        "artifacts": isinstance(artifacts, list)
        and len(artifacts) == len(set(artifacts))
        and required_artifacts <= set(artifacts),
        "artifact_manifest": (
            isinstance(artifact_manifest, dict)
            and artifact_manifest.get("name") == "python_artifacts.json"
            and isinstance(artifact_manifest.get("bytes"), int)
            and artifact_manifest["bytes"] > 0
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(artifact_manifest.get("sha256", "")),
                )
            )
        )
        if contract and contract["artifact_manifest"]
        else artifact_manifest is None,
    }
    return {"passed": all(checks.values()), "checks": checks}


def git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", choices=sorted(GATE_CONTRACTS), required=True)
    parser.add_argument("--git-commit", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", "")
    )
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument(
        "--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    )
    parser.add_argument(
        "--event", default=os.environ.get("GITHUB_EVENT_NAME", "")
    )
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--runner-os", default=os.environ.get("RUNNER_OS", ""))
    parser.add_argument(
        "--runner-arch", default=os.environ.get("RUNNER_ARCH", "")
    )
    parser.add_argument("--check", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--artifact-manifest", type=Path)
    return parser.parse_args()


def describe_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    args = parse_args()
    try:
        run_id = int(args.run_id)
        run_attempt = int(args.run_attempt)
    except ValueError as error:
        raise SystemExit("GitHub run identifiers must be integers") from error
    artifact_manifest = None
    if args.artifact_manifest is not None:
        path = args.artifact_manifest.resolve()
        if not path.is_file():
            raise SystemExit(f"artifact manifest is missing: {path}")
        artifact_manifest = describe_file(path)
    payload = {
        "schema_version": 1,
        "evidence_mode": "release_ci",
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": args.gate,
        "git_commit": args.git_commit,
        "git_dirty": git_dirty(),
        "github": {
            "repository": args.repository,
            "workflow": args.workflow,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "run_url": (
                f"https://github.com/{args.repository}/actions/runs/{run_id}"
            ),
            "event": args.event,
            "ref": args.ref,
        },
        "platform": {
            "os": args.runner_os,
            "arch": args.runner_arch,
        },
        "checks": {
            name: "passed" for name in sorted(set(args.check))
        },
        "artifacts": sorted(set(args.artifact)),
        "artifact_manifest": artifact_manifest,
    }
    result = evaluate(payload, expected_gate=args.gate)
    if not result["passed"]:
        failed = sorted(
            name for name, passed in result["checks"].items() if not passed
        )
        raise SystemExit("invalid release CI evidence: " + ", ".join(failed))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
