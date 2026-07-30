#!/usr/bin/env python3
"""Tests for content-bound v1 release attestations."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from v1_release_attestation import (
    MODES,
    load_reference,
    sha256_file,
)

VERSION = "1.0.0"
TAG = "v1.0.0"
COMMIT = "a" * 40


def details(key: str) -> dict:
    if key == "quickstart_15_minute_evidence":
        return {
            "profile": "release",
            "surface": "docker_source",
            "duration_seconds": 600.0,
            "result": "out/cudanav_closed_loop.json",
            "fresh_clone": True,
            "no_cache_build": True,
        }
    if key == "cudanav_release_evidence":
        return {
            "ros2_closed_loop": True,
            "closed_loop_duration_seconds": 650.0,
            "real_rosbag_shadow": True,
            "physical_gpu_models": ["GPU A"],
            "ros_distribution": "jazzy",
        }
    if key == "docker_gpu_evidence":
        return {
            "image": ("ghcr.io/rsasaki0109/" "cuda-mppi-controller-demo:v1.0.0"),
            "image_digest": "sha256:" + "b" * 64,
            "gpu_uuid": "GPU-aaaa-bbbb",
            "smoke_pass": True,
        }
    return {
        "site": "https://rsasaki0109.github.io/CudaRobotics/docs/",
        "deployed_tag": TAG,
        "install_page_pass": True,
        "nav2_page_pass": True,
        "release_links_pass": True,
    }


def write_attestation(root: Path, key: str) -> tuple[dict, Path]:
    path = root / f"{key}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_mode": MODES[key],
                "status": "passed",
                "version": VERSION,
                "target_tag": TAG,
                "git_commit": COMMIT,
                "git_dirty": False,
                "payload_sha256": "c" * 64,
                "checks": {"source_gate": True},
                "details": details(key),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
    }, path


class V1ReleaseAttestationTest(unittest.TestCase):
    def test_all_attestation_modes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key in MODES:
                reference, _ = write_attestation(root, key)
                gate = load_reference(
                    reference,
                    repo_root=root,
                    key=key,
                    target_version=VERSION,
                    target_tag=TAG,
                )
                self.assertTrue(gate["passed"], (key, gate))
                self.assertEqual(gate["git_commit"], COMMIT)

    def test_post_attestation_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, path = write_attestation(root, "docker_gpu_evidence")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["details"]["smoke_pass"] = False
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            gate = load_reference(
                reference,
                repo_root=root,
                key="docker_gpu_evidence",
                target_version=VERSION,
                target_tag=TAG,
            )
            self.assertFalse(gate["checks"]["content_bound"])
            self.assertFalse(gate["passed"])

    def test_legacy_inline_self_report_is_rejected(self) -> None:
        gate = load_reference(
            {
                "status": "passed",
                "version": VERSION,
                "git_commit": COMMIT,
            },
            repo_root=Path.cwd(),
            key="quickstart_15_minute_evidence",
            target_version=VERSION,
            target_tag=TAG,
        )
        self.assertFalse(gate["checks"]["reference_schema"])
        self.assertFalse(gate["passed"])

    def test_cudanav_requires_a_physical_gpu_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, path = write_attestation(root, "cudanav_release_evidence")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["details"]["physical_gpu_models"] = []
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            reference["sha256"] = sha256_file(path)
            gate = load_reference(
                reference,
                repo_root=root,
                key="cudanav_release_evidence",
                target_version=VERSION,
                target_tag=TAG,
            )
            self.assertFalse(gate["checks"]["physical_gpu"])
            self.assertFalse(gate["passed"])

    def test_malformed_details_are_rejected_without_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, path = write_attestation(root, "cudanav_release_evidence")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["details"] = ["not", "an", "object"]
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            reference["sha256"] = sha256_file(path)
            gate = load_reference(
                reference,
                repo_root=root,
                key="cudanav_release_evidence",
                target_version=VERSION,
                target_tag=TAG,
            )
            self.assertFalse(gate["checks"]["details"])
            self.assertFalse(gate["passed"])

    def test_malformed_duration_and_gpu_models_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, path = write_attestation(root, "cudanav_release_evidence")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["details"]["closed_loop_duration_seconds"] = "600"
            payload["details"]["physical_gpu_models"] = [["GPU A"], "GPU B"]
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            reference["sha256"] = sha256_file(path)
            gate = load_reference(
                reference,
                repo_root=root,
                key="cudanav_release_evidence",
                target_version=VERSION,
                target_tag=TAG,
            )
            self.assertFalse(gate["checks"]["closed_loop_duration"])
            self.assertFalse(gate["checks"]["physical_gpu"])
            self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
