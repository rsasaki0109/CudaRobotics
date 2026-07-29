#!/usr/bin/env python3
"""Run and retain a GPU smoke of the published v1 Docker image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

from run_v1_quickstart import docker_output, gpu_identity, run_logged
from v1_docker_gpu_evidence import (
    IMAGE_REPOSITORY,
    REQUIRED_ARTIFACTS,
    describe_artifacts,
    evaluate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.tag != "v1.0.0":
        raise SystemExit("the v1 release gate requires --tag v1.0.0")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_dir = output / "result"
    result_dir.mkdir()
    image_reference = f"{IMAGE_REPOSITORY}:{args.tag}"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
    )
    devices = gpu_identity()
    engine_version = docker_output(
        "version", "--format", "{{.Server.Version}}"
    )
    pull_command = ["docker", "pull", image_reference]
    run_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{result_dir}:/out",
        image_reference,
        "cudanav",
    ]
    started = datetime.now(timezone.utc).isoformat()
    began = time.perf_counter()
    pull_code, pull_seconds = run_logged(
        pull_command,
        output / "docker_pull.log",
        cwd=ROOT,
        timeout=args.timeout_seconds,
    )
    inspect: dict[str, Any] = {}
    if pull_code == 0:
        try:
            inspect_payload = json.loads(
                docker_output("image", "inspect", image_reference)
            )
            if isinstance(inspect_payload, list) and inspect_payload:
                inspect = inspect_payload[0]
        except (OSError, ValueError, subprocess.CalledProcessError):
            inspect = {}
    if pull_code == 0 and inspect:
        run_code, run_seconds = run_logged(
            run_command,
            output / "docker_run.log",
            cwd=ROOT,
            timeout=args.timeout_seconds,
        )
    else:
        run_code = 1
        run_seconds = 0.0
        (output / "docker_run.log").write_text(
            "run skipped: image pull or inspection failed\n",
            encoding="utf-8",
        )
    repo_digests = inspect.get("RepoDigests", [])
    digest = ""
    prefix = f"{IMAGE_REPOSITORY}@"
    if isinstance(repo_digests, list):
        for value in repo_digests:
            if isinstance(value, str) and value.startswith(prefix):
                digest = value[len(prefix) :]
                break
    labels = inspect.get("Config", {}).get("Labels", {})
    raw_passed = (
        pull_code == 0
        and run_code == 0
        and isinstance(labels, dict)
        and labels.get("org.opencontainers.image.revision") == commit
        and not dirty
    )
    manifest = {
        "schema_version": 1,
        "evidence_mode": "v1_published_docker_gpu_smoke",
        "status": "passed" if raw_passed else "failed",
        "version": "1.0.0",
        "target_tag": args.tag,
        "git_commit": commit,
        "git_dirty": dirty,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - began, 3),
        "phase_seconds": {
            "pull": round(pull_seconds, 3),
            "run": round(run_seconds, 3),
        },
        "platform": platform.platform(),
        "docker_engine_version": engine_version,
        "gpu": devices,
        "image": {
            "reference": image_reference,
            "digest": digest,
            "repo_digests": repo_digests,
            "image_id": inspect.get("Id"),
            "labels": labels,
        },
        "commands": {"pull": pull_command, "run": run_command},
        "returncodes": {"pull": pull_code, "run": run_code},
        "artifacts": describe_artifacts(output, set(REQUIRED_ARTIFACTS)),
    }
    gate = evaluate_manifest(
        manifest, output, expected_commit=commit
    )
    manifest["gate"] = gate
    manifest["status"] = "passed" if gate["passed"] else "failed"
    write_json(output / "manifest.json", manifest)
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(output / "manifest.json")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
