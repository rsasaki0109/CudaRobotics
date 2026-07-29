#!/usr/bin/env python3
"""Assemble every passing v0.2.0 RC gate into one portable evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any

from v0_2_release_bundle import PATHS, evaluate_bundle, sha256_file
from v0_2_release_evidence import evaluate_release


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


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ValueError(f"required file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_preflight(source_dir: Path, destination: Path) -> list[Path]:
    manifest = read_object(source_dir / "manifest.json")
    copied = []
    copy_file(source_dir / "manifest.json", destination / "manifest.json")
    copied.append(destination / "manifest.json")
    evidence = manifest.get("evidence_files")
    if not isinstance(evidence, list):
        raise ValueError(f"preflight has no evidence table: {source_dir}")
    for entry in evidence:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"invalid preflight evidence entry: {entry!r}")
        source = (source_dir / relative).resolve()
        if not source.is_relative_to(source_dir.resolve()):
            raise ValueError(f"unsafe preflight evidence path: {relative}")
        target = destination / relative
        copy_file(source, target)
        copied.append(target)
    return copied


def file_entry(root: Path, path: Path, category: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "category": category,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assemble(
    *,
    output_dir: Path,
    expected_commit: str,
    cpu_preflight_dir: Path,
    gpu_preflight_dir: Path,
    build_ci_path: Path,
    python_ci_path: Path,
    ros_ci_path: Path,
    python_artifacts_path: Path,
    dist_dir: Path,
    rosbag_report_path: Path,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    categories: dict[Path, str] = {}

    for path in copy_preflight(
        cpu_preflight_dir.resolve(), output / "evidence/cpu_preflight"
    ):
        categories[path] = "cpu_preflight"
    for path in copy_preflight(
        gpu_preflight_dir.resolve(), output / "evidence/gpu_preflight"
    ):
        categories[path] = "gpu_preflight"

    ci_sources = {
        build_ci_path.resolve(): output / PATHS["github_build"],
        python_ci_path.resolve(): output / PATHS["python_manylinux_wheels"],
        ros_ci_path.resolve(): output / PATHS["ros2_cuda_mppi"],
    }
    for source, target in ci_sources.items():
        copy_file(source, target)
        categories[target] = "remote_ci"

    artifact_manifest = read_object(python_artifacts_path.resolve())
    copied_manifest = output / PATHS["python_artifacts"]
    copy_file(python_artifacts_path.resolve(), copied_manifest)
    categories[copied_manifest] = "python_artifact_manifest"
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Python artifact manifest has no artifacts")
    for entry in artifacts:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"unsafe Python artifact name: {name!r}")
        target = output / "dist" / name
        copy_file(dist_dir.resolve() / name, target)
        categories[target] = "distribution"

    copied_rosbag = output / PATHS["real_rosbag_negative"]
    copy_file(rosbag_report_path.resolve(), copied_rosbag)
    categories[copied_rosbag] = "negative_result"

    gate = evaluate_release(
        expected_commit=expected_commit,
        cpu_preflight_dir=output / "evidence/cpu_preflight",
        gpu_preflight_dir=output / "evidence/gpu_preflight",
        build_ci_path=output / PATHS["github_build"],
        python_ci_path=output / PATHS["python_manylinux_wheels"],
        ros_ci_path=output / PATHS["ros2_cuda_mppi"],
        python_artifacts_path=output / PATHS["python_artifacts"],
        dist_dir=output / "dist",
        rosbag_report_path=output / PATHS["real_rosbag_negative"],
    )
    if not gate["passed"]:
        failed = [name for name, passed in gate["checks"].items() if not passed]
        raise ValueError("release gate is not ready: " + ", ".join(failed))
    for name, relative in PATHS.items():
        gate["sources"][name] = {
            "path": relative,
            "sha256": sha256_file(output / relative),
        }
    release_gate_path = output / "release_gate.json"
    write_object(release_gate_path, gate)
    categories[release_gate_path] = "release_gate"

    files = [
        file_entry(output, path, category)
        for path, category in sorted(
            categories.items(), key=lambda item: item[0].as_posix()
        )
    ]
    bundle = {
        "schema_version": 1,
        "evidence_mode": "v0_2_release_evidence_bundle",
        "version": "0.2.0",
        "status": "ready",
        "git_commit": expected_commit,
        "release_gate": {
            "path": "release_gate.json",
            "sha256": sha256_file(release_gate_path),
        },
        "files": files,
    }
    validation = evaluate_bundle(bundle, output, expected_commit)
    if not validation["valid"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise ValueError("assembled bundle is invalid: " + ", ".join(failed))
    write_object(output / "bundle.json", bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--cpu-preflight", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--build-ci", type=Path, required=True)
    parser.add_argument("--python-ci", type=Path, required=True)
    parser.add_argument("--ros-ci", type=Path, required=True)
    parser.add_argument("--python-artifacts", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--rosbag-report", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise SystemExit("--commit must be a full lowercase Git commit")
    try:
        assemble(
            output_dir=args.output_dir,
            expected_commit=args.commit,
            cpu_preflight_dir=args.cpu_preflight,
            gpu_preflight_dir=args.gpu_preflight,
            build_ci_path=args.build_ci,
            python_ci_path=args.python_ci,
            ros_ci_path=args.ros_ci,
            python_artifacts_path=args.python_artifacts,
            dist_dir=args.dist_dir,
            rosbag_report_path=args.rosbag_report,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot assemble v0.2 release bundle: {error}") from error
    print(args.output_dir.resolve() / "bundle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
