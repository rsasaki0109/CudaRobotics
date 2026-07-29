#!/usr/bin/env python3
"""Publish validated real-dataset materialization as portable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from cudanav_real_dataset import read_json
from cudanav_rosbag_evidence import sha256_file
from validate_cudanav_real_dataset import evaluate


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
CONTRACT_SOURCES = (
    "scripts/prepare_cudanav_istanbul_dataset.py",
    "scripts/derive_cudanav_path_sidecar.py",
    "scripts/validate_cudanav_real_dataset.py",
    "scripts/run_cudanav_real_dataset_pipeline.py",
    "scripts/publish_cudanav_dataset_evidence.py",
)
REQUIRED_VALIDATION_CHECKS = {
    "acquisition_inspection_bound",
    "acquisition_inspection_content",
    "derived_content_unchanged",
    "derived_identity",
    "derived_path_present",
    "derived_sqlite_path_semantics",
    "generator_report_bound",
    "generator_report_content",
    "provenance_bound",
    "source_content_unchanged",
    "source_identity",
    "spec_content_bound",
}


def git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    return commit, dirty


def _portable_files(identity: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": entry["path"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in identity["files"]
    ]


def make_portable_evidence(
    spec_path: Path,
    materialization_path: Path,
    *,
    result_id: str,
    git_commit: str,
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    materialization_path = materialization_path.resolve()
    spec = read_json(spec_path)
    materialization = read_json(materialization_path)
    validation = evaluate(spec_path, materialization_path)
    if not validation["ready"]:
        raise ValueError(
            "dataset materialization is not ready: "
            + json.dumps(validation["checks"], sort_keys=True)
        )
    inspection = materialization["acquisition_inspection"]
    remote_probe = inspection["remote_probe"]
    generator = materialization["generator_report"]
    source_metadata = materialization["source_metadata"]
    derived = materialization["derived_path_bag"]
    return {
        "schema_version": 1,
        "result_id": result_id,
        "status": "materialized",
        "evidence_mode": materialization["evidence_mode"],
        "git_commit": git_commit,
        "dataset": {
            "dataset_id": spec["dataset_id"],
            "spec_sha256": sha256_file(spec_path),
            "canonical_documentation": spec["canonical_documentation"],
        },
        "acquisition": {
            "method": spec["acquisition"]["method"],
            "file_id": spec["acquisition"]["file_id"],
            "database": {
                "filename": spec["acquisition"]["expected_database"],
                "bytes": inspection["database"]["bytes"],
                "sha256": inspection["database"]["sha256"],
            },
            "remote_probe": {
                "filename": remote_probe["database"]["filename"],
                "bytes": remote_probe["database"]["bytes"],
                "file_id": remote_probe["database"]["file_id"],
                "passed": remote_probe["passed"],
                "reused": "reused_from_inspection" in remote_probe,
            },
        },
        "recorded_topics": source_metadata["topics"],
        "derived_path": {
            "tree_sha256": derived["tree_sha256"],
            "total_bytes": derived["total_bytes"],
            "files": _portable_files(derived),
            "metadata_sha256": materialization[
                "derived_path_metadata"
            ]["sha256"],
            "algorithm": generator["algorithm"],
            "source_topic": generator["source_topic"],
            "source_type": generator["source_type"],
            "output_topic": generator["output_topic"],
            "output_type": spec["path_derivation"]["output_type"],
            "storage_id": generator["storage_id"],
            "parameters": generator["parameters"],
            "input_samples": generator["input_samples"],
            "output_poses": generator["output_poses"],
            "first_stamp_ns": generator["first_stamp_ns"],
            "last_stamp_ns": generator["last_stamp_ns"],
            "frame_id": generator["frame_id"],
        },
        "validation": validation,
        "claims": {
            "real_file_acquisition": True,
            "derived_path": True,
            "gpu_controller_run": False,
            "closed_loop": False,
        },
        "contract_sources": [
            {
                "path": relative,
                "sha256": sha256_file(ROOT / relative),
            }
            for relative in CONTRACT_SOURCES
        ],
    }


def evaluate_portable_evidence(
    payload: dict[str, Any],
    *,
    expected_commit: str | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    dataset = payload.get("dataset", {})
    acquisition = payload.get("acquisition", {})
    database = acquisition.get("database", {})
    remote = acquisition.get("remote_probe", {})
    derived = payload.get("derived_path", {})
    validation_checks = payload.get("validation", {}).get("checks")
    checks = {
        "schema": payload.get("schema_version") == 1,
        "result_id": isinstance(payload.get("result_id"), str)
        and bool(payload["result_id"]),
        "status": payload.get("status") == "materialized",
        "evidence_mode": payload.get("evidence_mode")
        == "real_sensor_shadow_with_derived_path",
        "commit": bool(COMMIT.fullmatch(str(payload.get("git_commit", ""))))
        and (
            expected_commit is None
            or payload.get("git_commit") == expected_commit
        ),
        "dataset": (
            isinstance(payload.get("dataset"), dict)
            and bool(payload["dataset"].get("dataset_id"))
            and bool(
                SHA256.fullmatch(
                    str(payload["dataset"].get("spec_sha256", ""))
                )
            )
        ),
        "database": (
            isinstance(database, dict)
            and database.get("bytes", 0) > 0
            and bool(
                SHA256.fullmatch(
                    str(database.get("sha256", ""))
                )
            )
            and remote.get("passed") is True
            and remote.get("file_id") == acquisition.get("file_id")
            and remote.get("filename") == database.get("filename")
            and remote.get("bytes", 0) > 0
        ),
        "recorded_topics": (
            isinstance(payload.get("recorded_topics"), dict)
            and len(payload["recorded_topics"]) >= 3
            and all(
                isinstance(entry, dict)
                and entry.get("count", 0) > 0
                and bool(entry.get("type"))
                for entry in payload["recorded_topics"].values()
            )
        ),
        "derived_path": (
            isinstance(derived, dict)
            and derived.get("output_poses", 0) >= 2
            and derived.get("input_samples", 0)
            >= derived.get("output_poses", 0)
            and derived.get("storage_id") in {"sqlite3", "mcap"}
            and bool(
                SHA256.fullmatch(
                    str(derived.get("tree_sha256", ""))
                )
            )
            and isinstance(derived.get("files"), list)
            and bool(derived["files"])
            and all(
                isinstance(entry, dict)
                and isinstance(entry.get("path"), str)
                and bool(entry["path"])
                and entry.get("bytes", 0) > 0
                and bool(
                    SHA256.fullmatch(str(entry.get("sha256", "")))
                )
                for entry in derived["files"]
            )
        ),
        "validation": (
            payload.get("validation", {}).get("valid") is True
            and payload.get("validation", {}).get("ready") is True
            and isinstance(validation_checks, dict)
            and REQUIRED_VALIDATION_CHECKS <= set(validation_checks)
            and all(validation_checks.values())
        ),
        "claims": payload.get("claims")
        == {
            "real_file_acquisition": True,
            "derived_path": True,
            "gpu_controller_run": False,
            "closed_loop": False,
        },
        "contract_sources": False,
        "dataset_contract": False,
    }
    sources = payload.get("contract_sources")
    if isinstance(sources, list):
        checks["contract_sources"] = (
            len(sources) == len(CONTRACT_SOURCES)
            and all(isinstance(entry, dict) for entry in sources)
            and {
                entry.get("path")
                for entry in sources
                if isinstance(entry, dict)
            }
            == set(CONTRACT_SOURCES)
            and all(
                bool(SHA256.fullmatch(str(entry.get("sha256", ""))))
                for entry in sources
                if isinstance(entry, dict)
            )
        )
        if checks["contract_sources"] and verify_sources:
            checks["contract_sources"] = all(
                (ROOT / entry["path"]).is_file()
                and sha256_file(ROOT / entry["path"]) == entry["sha256"]
                for entry in sources
            )
    if checks["dataset"] and verify_sources:
        for candidate in sorted(
            (ROOT / "docs").glob("cudanav_real_dataset*.json")
        ):
            try:
                spec = read_json(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                spec.get("dataset_id") == dataset.get("dataset_id")
                and sha256_file(candidate) == dataset.get("spec_sha256")
                and dataset.get("canonical_documentation")
                == spec.get("canonical_documentation")
                and all(
                    payload.get("recorded_topics", {})
                    .get(contract["topic"], {})
                    .get("type")
                    == contract["type"]
                    for contract in spec["recorded_inputs"].values()
                )
            ):
                checks["dataset_contract"] = True
                break
    elif checks["dataset"]:
        checks["dataset_contract"] = True
    return {"valid": all(checks.values()), "checks": checks}


def render_markdown(payload: dict[str, Any]) -> str:
    database = payload["acquisition"]["database"]
    path = payload["derived_path"]
    topics = payload["recorded_topics"]
    lines = [
        f"# {payload['result_id']}",
        "",
        "Portable real-file acquisition and derived-Path evidence. This is "
        "not a GPU controller or closed-loop result.",
        "",
        f"- Commit: `{payload['git_commit']}`",
        f"- Dataset: `{payload['dataset']['dataset_id']}`",
        f"- Database: `{database['filename']}` ({database['bytes']} bytes)",
        f"- Database SHA-256: `{database['sha256']}`",
        f"- Derived poses: {path['output_poses']} from "
        f"{path['input_samples']} recorded samples",
        f"- Derived storage: `{path['storage_id']}`",
        f"- Derived tree SHA-256: `{path['tree_sha256']}`",
        f"- Validation checks: {len(payload['validation']['checks'])} / "
        f"{len(payload['validation']['checks'])} passed",
        "",
        "## Recorded topics",
        "",
        "| Topic | Type | Messages |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | `{entry['type']}` | {entry['count']} |"
        for name, entry in sorted(topics.items())
    )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Real-file acquisition: yes",
            "- Deterministic derived Path: yes",
            "- GPU controller run: no",
            "- Closed-loop evidence: no",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--materialization", type=Path)
    parser.add_argument("--result-id")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--no-verify-sources", action="store_true")
    args = parser.parse_args()
    if args.validate is not None:
        result = evaluate_portable_evidence(
            read_json(args.validate),
            expected_commit=args.commit,
            verify_sources=not args.no_verify_sources,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    required = (
        args.spec,
        args.materialization,
        args.result_id,
        args.output_json,
        args.output_markdown,
    )
    if any(value is None for value in required):
        parser.error(
            "publishing requires --spec, --materialization, --result-id, "
            "--output-json, and --output-markdown"
        )
    commit, dirty = git_identity()
    if dirty:
        raise SystemExit("refusing to publish from a dirty worktree")
    for output in (args.output_json, args.output_markdown):
        if output.exists():
            raise SystemExit(f"refusing existing output: {output}")
    payload = make_portable_evidence(
        args.spec,
        args.materialization,
        result_id=args.result_id,
        git_commit=commit,
    )
    validation = evaluate_portable_evidence(payload, expected_commit=commit)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
