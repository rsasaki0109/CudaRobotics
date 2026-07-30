#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from v1_quickstart_evidence import (
    REQUIRED_ARTIFACTS,
    describe_artifacts,
    evaluate_manifest,
    sha256_file,
)
from v1_support_matrix import MATRIX_PATH, evaluate as evaluate_matrix, load


COMMIT = "a" * 40


def fixture(root: Path) -> dict:
    (root / "result").mkdir(parents=True)
    shutil.copyfile(MATRIX_PATH, root / "support_matrix.json")
    for relative in ("clone.log", "docker_build.log", "docker_run.log"):
        (root / relative).write_text(f"{relative} passed\n", encoding="utf-8")
    (root / "result" / "cudanav_closed_loop.log").write_text(
        "closed loop passed\n", encoding="utf-8"
    )
    (root / "result" / "cudanav_closed_loop.json").write_text(
        json.dumps(
            {"schema_version": 1, "success": True, "smoke_pass": True}
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = load()
    actual = evaluate_matrix(matrix)["actual"]
    return {
        "schema_version": 1,
        "evidence_mode": "v1_quickstart",
        "profile": "development",
        "status": "passed",
        "duration_seconds": 600.0,
        "phase_seconds": {"clone": 5.0, "build": 550.0, "run": 45.0},
        "time_budget_seconds": 900.0,
        "target_version": "1.0.0",
        "source_ref": "release-candidate",
        "repository": "https://github.com/rsasaki0109/CudaRobotics.git",
        "git_commit": COMMIT,
        "git_dirty": False,
        "component_versions": actual,
        "preexisting_image": False,
        "preexisting_container": False,
        "commands": {
            "clone": [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "release-candidate",
                "https://github.com/rsasaki0109/CudaRobotics.git",
                "checkout",
            ],
            "build": [
                "docker",
                "build",
                "--pull",
                "--no-cache",
                "-f",
                "docker/Dockerfile",
                "-t",
                "cudarobotics",
                ".",
            ],
            "run": [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--name",
                "cudarobotics-quickstart",
                "-v",
                "/tmp/out:/out",
                "cudarobotics",
                "cudanav",
            ],
        },
        "build_command_contract": matrix["main_demo"]["build_command"],
        "run_command_contract": matrix["main_demo"]["run_command"],
        "returncodes": {"clone": 0, "build": 0, "run": 0},
        "docker": {
            "engine_version": "27.0.0",
            "image_id": "sha256:" + "b" * 64,
        },
        "gpu": [
            {
                "name": "GPU A",
                "uuid": "GPU-a",
                "driver_version": "999",
            }
        ],
        "result": "out/cudanav_closed_loop.json",
        "support_matrix_sha256": sha256_file(
            root / "support_matrix.json"
        ),
        "artifacts": describe_artifacts(root, set(REQUIRED_ARTIFACTS)),
    }


class V1QuickstartEvidenceTest(unittest.TestCase):
    def test_development_quickstart_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = evaluate_manifest(
                fixture(root),
                root,
                expected_profile="development",
                expected_commit=COMMIT,
            )
            self.assertTrue(result["passed"], result)

    def test_release_profile_cannot_use_development_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            manifest["profile"] = "release"
            manifest["source_ref"] = "v1.0.0"
            result = evaluate_manifest(manifest, root)
            self.assertFalse(result["checks"]["matrix_release_status"])
            self.assertTrue(result["checks"]["python_at_target"])
            self.assertTrue(result["checks"]["ros_at_target"])
            self.assertFalse(result["passed"])

    def test_quickstart_over_900_seconds_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            manifest["duration_seconds"] = 900.001
            result = evaluate_manifest(manifest, root)
            self.assertFalse(result["checks"]["duration"])
            self.assertFalse(result["passed"])

    def test_post_run_result_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            (root / "result" / "cudanav_closed_loop.json").write_text(
                json.dumps(
                    {"schema_version": 1, "success": False, "smoke_pass": False}
                ),
                encoding="utf-8",
            )
            result = evaluate_manifest(manifest, root)
            self.assertFalse(result["checks"]["artifact_content"])
            self.assertFalse(result["checks"]["result"])
            self.assertFalse(result["passed"])

    def test_cached_project_image_is_not_fresh_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            manifest["preexisting_image"] = True
            result = evaluate_manifest(manifest, root)
            self.assertFalse(result["checks"]["fresh_image"])
            self.assertFalse(result["passed"])

    def test_retained_evidence_uses_content_bound_version_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            matrix_path = root / "support_matrix.json"
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix["surfaces"]["python_source"]["version"] = "9.9.9"
            matrix["surfaces"]["python_wheels"]["version"] = "9.9.9"
            for package in matrix["surfaces"]["ros2"]["package_versions"]:
                matrix["surfaces"]["ros2"]["package_versions"][
                    package
                ] = "9.9.9"
            matrix_path.write_text(
                json.dumps(matrix) + "\n", encoding="utf-8"
            )
            manifest["component_versions"] = {
                "python_version": "9.9.9",
                "ros_package_versions": matrix["surfaces"]["ros2"][
                    "package_versions"
                ],
            }
            manifest["support_matrix_sha256"] = sha256_file(matrix_path)
            manifest["artifacts"] = describe_artifacts(
                root, set(REQUIRED_ARTIFACTS)
            )
            result = evaluate_manifest(
                manifest,
                root,
                expected_profile="development",
                expected_commit=COMMIT,
            )
            self.assertTrue(result["checks"]["matrix_valid"], result)
            self.assertTrue(result["checks"]["component_versions"], result)
            self.assertTrue(result["passed"], result)

    def test_malformed_snapshot_is_rejected_without_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            matrix_path = root / "support_matrix.json"
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix["main_demo"] = ["not", "an", "object"]
            matrix_path.write_text(
                json.dumps(matrix) + "\n", encoding="utf-8"
            )
            manifest["commands"] = ["not", "an", "object"]
            manifest["support_matrix_sha256"] = sha256_file(matrix_path)
            manifest["artifacts"] = describe_artifacts(
                root, set(REQUIRED_ARTIFACTS)
            )
            result = evaluate_manifest(manifest, root)
            self.assertFalse(result["checks"]["matrix_valid"])
            self.assertFalse(result["checks"]["build_command"])
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
