#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cudanav_ros_ci_evidence import REQUIRED_CHECKS, REQUIRED_PACKAGES
from release_ci_evidence import GATE_CONTRACTS
from test_release_preflight import evidence_fixture
from test_verify_python_release_artifacts import (
    VERSION,
    required_sdist_names,
    write_sdist,
    write_wheel,
)
from v0_2_release_evidence import evaluate_release, file_sha256


COMMIT = "a" * 40
REF = "refs/heads/release-candidate"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_preflight(root: Path, profile: str) -> None:
    root.mkdir()
    write_json(root / "manifest.json", evidence_fixture(root, profile))


def ci_payload(
    gate: str, run_id: int, artifact_manifest_path: Path | None = None
) -> dict:
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
                "https://github.com/rsasaki0109/CudaRobotics/actions/runs/"
                f"{run_id}"
            ),
            "event": "workflow_dispatch",
            "ref": REF,
        },
        "platform": {"os": "Linux", "arch": "X64"},
        "checks": {name: "passed" for name in contract["checks"]},
        "artifacts": sorted(contract["artifacts"]),
        "artifact_manifest": (
            {
                "name": artifact_manifest_path.name,
                "bytes": artifact_manifest_path.stat().st_size,
                "sha256": file_sha256(artifact_manifest_path),
            }
            if artifact_manifest_path is not None
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
                "https://github.com/rsasaki0109/CudaRobotics/actions/runs/"
                f"{run_id}"
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


def write_python_artifacts(root: Path) -> Path:
    root.mkdir()
    sdist = root / f"cudarobotics-{VERSION}.tar.gz"
    cp310 = (
        root
        / f"cudarobotics-{VERSION}-cp310-cp310-manylinux_2_17_x86_64.whl"
    )
    cp312 = (
        root
        / f"cudarobotics-{VERSION}-cp312-cp312-manylinux_2_17_x86_64.whl"
    )
    write_sdist(sdist, required_sdist_names())
    write_wheel(cp310)
    write_wheel(cp312)
    artifacts = [sdist, cp310, cp312]
    manifest = {
        "schema_version": 1,
        "package": "cudarobotics",
        "package_version": VERSION,
        "git_commit": COMMIT,
        "git_dirty": False,
        "artifacts": [
            {
                "name": path.name,
                "kind": "sdist" if path == sdist else "wheel",
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifacts
        ],
    }
    path = root / "python_artifacts.json"
    write_json(path, manifest)
    return path


def write_rosbag_report(path: Path) -> None:
    path.write_text(
        "Source: https://doi.org/10.5281/zenodo.10518775\n"
        "This is recorded-motion evidence, not a closed-loop success claim.\n"
        "Overall result: **FAIL**\n"
        "- FAIL: at least 90% scan/command pairing coverage\n"
        "- FAIL: minimum front clearance at least 0.10 m\n",
        encoding="utf-8",
    )


def complete_fixture(root: Path) -> dict:
    cpu = root / "cpu"
    gpu = root / "gpu"
    write_preflight(cpu, "cpu")
    write_preflight(gpu, "gpu")
    build_ci = root / "build_ci.json"
    python_ci = root / "python_ci.json"
    ros_ci = root / "ros_ci.json"
    write_json(build_ci, ci_payload("github_build", 101))
    write_json(ros_ci, ros_payload(103))
    dist = root / "dist"
    python_artifacts = write_python_artifacts(dist)
    write_json(
        python_ci,
        ci_payload(
            "python_manylinux_wheels", 102, python_artifacts
        ),
    )
    rosbag = root / "rosbag.md"
    write_rosbag_report(rosbag)
    return {
        "expected_commit": COMMIT,
        "cpu_preflight_dir": cpu,
        "gpu_preflight_dir": gpu,
        "build_ci_path": build_ci,
        "python_ci_path": python_ci,
        "ros_ci_path": ros_ci,
        "python_artifacts_path": python_artifacts,
        "dist_dir": dist,
        "rosbag_report_path": rosbag,
    }


class V02ReleaseEvidenceTest(unittest.TestCase):
    def test_complete_release_candidate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_release(**complete_fixture(Path(directory)))
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["status"], "ready")

    def test_remote_runs_must_share_one_ref(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = complete_fixture(Path(directory))
            payload = json.loads(
                fixture["python_ci_path"].read_text(encoding="utf-8")
            )
            payload["github"]["ref"] = "refs/heads/other"
            write_json(fixture["python_ci_path"], payload)
            result = evaluate_release(**fixture)
            self.assertFalse(result["checks"]["same_remote_ref"])
            self.assertFalse(result["passed"])

    def test_post_verification_artifact_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = complete_fixture(Path(directory))
            wheel = next(fixture["dist_dir"].glob("*cp310*.whl"))
            wheel.write_bytes(wheel.read_bytes() + b"replacement")
            result = evaluate_release(**fixture)
            self.assertFalse(
                result["gates"]["python_artifacts"]["checks"][
                    "content_unchanged"
                ]
            )
            self.assertFalse(result["passed"])

    def test_python_ci_must_bind_the_downloaded_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = complete_fixture(Path(directory))
            manifest = json.loads(
                fixture["python_artifacts_path"].read_text(encoding="utf-8")
            )
            manifest["platform"] = "replacement"
            write_json(fixture["python_artifacts_path"], manifest)
            result = evaluate_release(**fixture)
            self.assertFalse(
                result["checks"]["python_ci_artifact_binding"]
            )
            self.assertFalse(result["passed"])

    def test_both_portable_python_versions_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = complete_fixture(Path(directory))
            manifest = json.loads(
                fixture["python_artifacts_path"].read_text(encoding="utf-8")
            )
            manifest["artifacts"] = [
                entry
                for entry in manifest["artifacts"]
                if "cp310" not in entry["name"]
            ]
            for wheel in fixture["dist_dir"].glob("*cp310*.whl"):
                wheel.unlink()
            write_json(fixture["python_artifacts_path"], manifest)
            result = evaluate_release(**fixture)
            self.assertFalse(
                result["gates"]["python_artifacts"]["checks"][
                    "manylinux_cp310"
                ]
            )
            self.assertFalse(result["passed"])

    def test_rosbag_failure_cannot_be_relabelled_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = complete_fixture(Path(directory))
            fixture["rosbag_report_path"].write_text(
                "Overall result: **PASS**\n", encoding="utf-8"
            )
            result = evaluate_release(**fixture)
            self.assertFalse(
                result["checks"]["real_rosbag_explicit_negative"]
            )
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
