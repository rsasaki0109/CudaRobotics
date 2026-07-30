#!/usr/bin/env python3
"""Tests for the portable pre-tag v1 release-candidate bundle."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cudanav_ros_ci_evidence import REQUIRED_CHECKS, REQUIRED_PACKAGES
from python_source_provenance import expected_payload
from release_ci_evidence import GATE_CONTRACTS
from v1_release_candidate import (
    assemble,
    evaluate_bundle,
    sha256_file,
)

COMMIT = "a" * 40
REF = "refs/heads/master"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ci_payload(gate: str, run_id: int, manifest_path: Path | None = None) -> dict:
    contract = GATE_CONTRACTS[gate]
    return {
        "schema_version": 1,
        "evidence_mode": "release_ci",
        "status": "passed",
        "gate": gate,
        "git_commit": COMMIT,
        "git_dirty": False,
        "github": {
            "repository": "rsasaki0109/CudaRobotics",
            "workflow": contract["workflow"],
            "run_id": run_id,
            "run_attempt": 1,
            "run_url": (
                "https://github.com/rsasaki0109/CudaRobotics/actions/runs/" f"{run_id}"
            ),
            "event": "workflow_dispatch",
            "ref": REF,
        },
        "platform": {"os": "Linux", "arch": "X64"},
        "checks": {name: "passed" for name in contract["checks"]},
        "artifacts": sorted(contract["artifacts"]),
        "artifact_manifest": (
            {
                "name": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            }
            if manifest_path is not None
            else None
        ),
    }


def ros_payload(run_id: int) -> dict:
    return {
        "schema_version": 1,
        "evidence_mode": "ros_jazzy_ci",
        "status": "passed",
        "git_commit": COMMIT,
        "git_dirty": False,
        "github": {
            "repository": "rsasaki0109/CudaRobotics",
            "workflow": "ROS2 CUDA MPPI",
            "run_id": run_id,
            "run_attempt": 1,
            "run_url": (
                "https://github.com/rsasaki0109/CudaRobotics/actions/runs/" f"{run_id}"
            ),
            "event": "workflow_dispatch",
            "ref": REF,
        },
        "platform": {
            "os": "Linux",
            "arch": "X64",
            "image": "ubuntu-24.04",
        },
        "ros": {"distro": "jazzy"},
        "cuda": {
            "toolkit": "12.6",
            "compiler": "Cuda compilation tools, release 12.6, V12.6.85",
        },
        "packages": sorted(REQUIRED_PACKAGES),
        "checks": {name: "passed" for name in REQUIRED_CHECKS},
    }


def artifact_manifest() -> dict:
    names = [
        "cudarobotics-1.0.0.tar.gz",
        ("cudarobotics-1.0.0-cp310-cp310-" "manylinux_2_17_x86_64.whl"),
        ("cudarobotics-1.0.0-cp312-cp312-" "manylinux_2_17_x86_64.whl"),
    ]
    return {
        "schema_version": 1,
        "package": "cudarobotics",
        "package_version": "1.0.0",
        "git_commit": COMMIT,
        "git_dirty": False,
        "source_provenance": expected_payload(),
        "artifacts": [
            {
                "name": name,
                "kind": "sdist" if name.endswith(".tar.gz") else "wheel",
                "bytes": 100 + index,
                "sha256": f"{index + 1:064x}",
            }
            for index, name in enumerate(names)
        ],
    }


def fixture(root: Path) -> dict:
    inputs = root / "inputs"
    manifest = inputs / "python_artifacts.json"
    write_json(manifest, artifact_manifest())
    build = inputs / "build.json"
    python = inputs / "python.json"
    ros = inputs / "ros.json"
    write_json(build, ci_payload("github_build", 101))
    write_json(python, ci_payload("python_manylinux_wheels", 102, manifest))
    write_json(ros, ros_payload(103))
    return {
        "output_dir": root / "bundle",
        "expected_commit": COMMIT,
        "build_path": build,
        "python_path": python,
        "python_artifacts_path": manifest,
        "ros_path": ros,
    }


class V1ReleaseCandidateTest(unittest.TestCase):
    def test_complete_candidate_is_portable_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = assemble(**fixture(Path(directory)))
            result = evaluate_bundle(bundle, COMMIT)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["decision"]["remote"]["run_ids"], [101, 102, 103])

    def test_post_assembly_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = assemble(**fixture(Path(directory)))
            target = bundle.parent / "evidence/python_artifacts.json"
            target.write_bytes(target.read_bytes() + b" ")
            result = evaluate_bundle(bundle, COMMIT)
            self.assertFalse(result["checks"]["file_table"])
            self.assertFalse(result["passed"])

    def test_remote_runs_must_share_master_ref(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = fixture(Path(directory))
            payload = json.loads(spec["python_path"].read_text(encoding="utf-8"))
            payload["github"]["ref"] = "refs/heads/other"
            write_json(spec["python_path"], payload)
            bundle = assemble(**spec)
            result = evaluate_bundle(bundle, COMMIT)
            self.assertFalse(result["decision"]["checks"]["same_remote_ref"])
            self.assertFalse(result["passed"])

    def test_manifest_must_bind_python_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = fixture(Path(directory))
            payload = json.loads(
                spec["python_artifacts_path"].read_text(encoding="utf-8")
            )
            payload["platform"] = "tampered"
            write_json(spec["python_artifacts_path"], payload)
            bundle = assemble(**spec)
            result = evaluate_bundle(bundle, COMMIT)
            self.assertFalse(result["decision"]["checks"]["python_manifest_binding"])
            self.assertFalse(result["passed"])

    def test_candidate_commit_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = assemble(**fixture(Path(directory)))
            result = evaluate_bundle(bundle, "b" * 40)
            self.assertFalse(result["checks"]["git_commit"])
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
