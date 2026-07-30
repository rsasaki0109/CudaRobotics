#!/usr/bin/env python3
"""Assemble and validate portable pre-tag v1 release-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from cudanav_ros_ci_evidence import evaluate as evaluate_ros_ci
from python_source_provenance import expected_payload
from release_ci_evidence import evaluate as evaluate_release_ci

MODE = "v1_release_candidate_bundle"
DECISION_MODE = "v1_release_candidate"
VERSION = "1.0.0"
REPOSITORY = "rsasaki0109/CudaRobotics"
PATHS = {
    "github_build": "evidence/github_build.json",
    "python_package": "evidence/python_package.json",
    "python_artifacts": "evidence/python_artifacts.json",
    "ros_jazzy": "evidence/ros_jazzy.json",
    "decision": "decision.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def describe(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _artifact_manifest_gate(
    payload: dict[str, Any], expected_commit: str
) -> dict[str, Any]:
    entries = payload.get("artifacts")
    names: list[str] = []
    entries_valid = isinstance(entries, list) and bool(entries)
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                entries_valid = False
                continue
            name = entry.get("name")
            if not (
                isinstance(name, str)
                and name
                and isinstance(entry.get("bytes"), int)
                and entry["bytes"] > 0
                and entry.get("kind") in {"sdist", "wheel"}
                and re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
            ):
                entries_valid = False
                continue
            names.append(name)
    name_set = set(names)
    checks = {
        "schema": payload.get("schema_version") == 1,
        "package": payload.get("package") == "cudarobotics",
        "version": payload.get("package_version") == VERSION,
        "git_commit": payload.get("git_commit") == expected_commit,
        "clean_checkout": payload.get("git_dirty") is False,
        "source_provenance": payload.get("source_provenance") == expected_payload(),
        "entries": entries_valid and len(names) == len(name_set) and len(names) >= 3,
        "sdist": f"cudarobotics-{VERSION}.tar.gz" in name_set,
        "manylinux_cp310": any(
            re.fullmatch(
                rf"cudarobotics-{re.escape(VERSION)}-cp310-cp310-"
                r".*manylinux.*x86_64\.whl",
                name,
            )
            for name in name_set
        ),
        "manylinux_cp312": any(
            re.fullmatch(
                rf"cudarobotics-{re.escape(VERSION)}-cp312-cp312-"
                r".*manylinux.*x86_64\.whl",
                name,
            )
            for name in name_set
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_inputs(
    *,
    expected_commit: str,
    build_path: Path,
    python_path: Path,
    python_artifacts_path: Path,
    ros_path: Path,
) -> dict[str, Any]:
    build = read_object(build_path)
    python = read_object(python_path)
    python_artifacts = read_object(python_artifacts_path)
    ros = read_object(ros_path)
    build_gate = evaluate_release_ci(
        build,
        expected_gate="github_build",
        expected_commit=expected_commit,
    )
    python_gate = evaluate_release_ci(
        python,
        expected_gate="python_manylinux_wheels",
        expected_commit=expected_commit,
    )
    ros_gate = evaluate_ros_ci(ros, expected_commit=expected_commit)
    artifacts_gate = _artifact_manifest_gate(python_artifacts, expected_commit)
    manifest_binding = python.get("artifact_manifest")
    github_tables = [
        payload.get("github", {})
        for payload in (build, python, ros)
        if isinstance(payload.get("github"), dict)
    ]
    refs = {str(table.get("ref", "")) for table in github_tables}
    run_ids = {table.get("run_id") for table in github_tables}
    repositories = {str(table.get("repository", "")) for table in github_tables}
    events = {str(table.get("event", "")) for table in github_tables}
    checks = {
        "expected_commit": bool(re.fullmatch(r"[0-9a-f]{40}", expected_commit)),
        "github_build": build_gate["passed"],
        "python_manylinux_wheels": python_gate["passed"],
        "ros2_jazzy": ros_gate["passed"],
        "python_artifacts": artifacts_gate["passed"],
        "python_manifest_binding": (
            isinstance(manifest_binding, dict)
            and manifest_binding.get("name") == python_artifacts_path.name
            and manifest_binding.get("bytes") == python_artifacts_path.stat().st_size
            and manifest_binding.get("sha256") == sha256_file(python_artifacts_path)
        ),
        "same_remote_ref": len(refs) == 1
        and next(iter(refs), "") == "refs/heads/master",
        "distinct_remote_runs": len(run_ids) == 3
        and all(isinstance(run_id, int) for run_id in run_ids),
        "repository": repositories == {REPOSITORY},
        "non_pr_evidence": bool(events) and events <= {"push", "workflow_dispatch"},
    }
    source_paths = {
        "github_build": build_path,
        "python_package": python_path,
        "python_artifacts": python_artifacts_path,
        "ros_jazzy": ros_path,
    }
    return {
        "schema_version": 1,
        "evidence_mode": DECISION_MODE,
        "version": VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "git_commit": expected_commit,
        "passed": all(checks.values()),
        "checks": checks,
        "gates": {
            "github_build": build_gate,
            "python_manylinux_wheels": python_gate,
            "python_artifacts": artifacts_gate,
            "ros2_jazzy": ros_gate,
        },
        "remote": {
            "ref": next(iter(refs)) if len(refs) == 1 else None,
            "run_ids": sorted(run_id for run_id in run_ids if isinstance(run_id, int)),
        },
        "sources": {
            name: describe(path, PATHS[name]) for name, path in source_paths.items()
        },
    }


def assemble(
    *,
    output_dir: Path,
    expected_commit: str,
    build_path: Path,
    python_path: Path,
    python_artifacts_path: Path,
    ros_path: Path,
) -> Path:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "github_build": build_path.resolve(),
        "python_package": python_path.resolve(),
        "python_artifacts": python_artifacts_path.resolve(),
        "ros_jazzy": ros_path.resolve(),
    }
    for name, source in inputs.items():
        if not source.is_file():
            raise ValueError(f"release-candidate input is missing: {source}")
        destination = output / PATHS[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    decision = evaluate_inputs(
        expected_commit=expected_commit,
        build_path=output / PATHS["github_build"],
        python_path=output / PATHS["python_package"],
        python_artifacts_path=output / PATHS["python_artifacts"],
        ros_path=output / PATHS["ros_jazzy"],
    )
    decision_path = output / PATHS["decision"]
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = []
    for relative in PATHS.values():
        path = output / relative
        files.append(describe(path, relative))
    bundle = {
        "schema_version": 1,
        "evidence_mode": MODE,
        "version": VERSION,
        "status": "passed" if decision["passed"] else "failed",
        "git_commit": expected_commit,
        "files": files,
        "decision": describe(decision_path, PATHS["decision"]),
    }
    bundle_path = output / "bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_path


def _safe_file(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (root / relative).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def evaluate_bundle(
    bundle_path: Path, expected_commit: str | None = None
) -> dict[str, Any]:
    try:
        path = bundle_path.resolve()
        root = path.parent
        bundle = read_object(path)
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError):
        return {
            "valid": False,
            "passed": False,
            "checks": {"bundle_readable": False},
        }
    entries = bundle.get("files")
    declared: list[str] = []
    file_checks: dict[str, bool] = {}
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                file_checks[f"entry-{index}"] = False
                continue
            relative = entry.get("path")
            target = _safe_file(root, relative)
            key = str(relative)
            file_checks[key] = (
                target is not None
                and isinstance(entry.get("bytes"), int)
                and entry["bytes"] == target.stat().st_size
                and entry.get("sha256") == sha256_file(target)
            )
            if isinstance(relative, str):
                declared.append(relative)
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item != path
    }
    decision_path = _safe_file(root, PATHS["decision"])
    stored: dict[str, Any] = {}
    recomputed: dict[str, Any] = {"passed": False}
    if decision_path is not None:
        try:
            stored = read_object(decision_path)
            recomputed = evaluate_inputs(
                expected_commit=str(bundle.get("git_commit", "")),
                build_path=root / PATHS["github_build"],
                python_path=root / PATHS["python_package"],
                python_artifacts_path=root / PATHS["python_artifacts"],
                ros_path=root / PATHS["ros_jazzy"],
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    decision_ref = bundle.get("decision")
    commit = bundle.get("git_commit")
    checks = {
        "schema": bundle.get("schema_version") == 1,
        "mode": bundle.get("evidence_mode") == MODE,
        "version": bundle.get("version") == VERSION,
        "status": bundle.get("status") == "passed",
        "git_commit": bool(re.fullmatch(r"[0-9a-f]{40}", str(commit)))
        and (expected_commit is None or commit == expected_commit),
        "file_table": isinstance(entries, list)
        and set(declared) == set(PATHS.values())
        and len(declared) == len(set(declared))
        and all(file_checks.values()),
        "complete_inventory": actual == set(PATHS.values()),
        "decision_binding": isinstance(decision_ref, dict)
        and decision_ref.get("path") == PATHS["decision"]
        and decision_path is not None
        and decision_ref.get("bytes") == decision_path.stat().st_size
        and decision_ref.get("sha256") == sha256_file(decision_path),
        "decision_identity": stored.get("evidence_mode") == DECISION_MODE
        and stored.get("git_commit") == commit
        and stored.get("passed") is True,
        "decision_recomputed": recomputed.get("passed") is True
        and stored == recomputed,
    }
    return {
        "valid": all(checks.values()),
        "passed": all(checks.values()),
        "checks": checks,
        "file_checks": file_checks,
        "decision": recomputed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--build", type=Path, required=True)
    assemble_parser.add_argument("--python", type=Path, required=True)
    assemble_parser.add_argument("--python-artifacts", type=Path, required=True)
    assemble_parser.add_argument("--ros", type=Path, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    assemble_parser.add_argument("--commit", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("bundle", type=Path)
    validate_parser.add_argument("--commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "assemble":
        bundle = assemble(
            output_dir=args.output_dir,
            expected_commit=args.commit,
            build_path=args.build,
            python_path=args.python,
            python_artifacts_path=args.python_artifacts,
            ros_path=args.ros,
        )
        result = evaluate_bundle(bundle, args.commit)
        print(bundle)
    else:
        result = evaluate_bundle(args.bundle, args.commit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
