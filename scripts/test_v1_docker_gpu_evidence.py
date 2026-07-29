#!/usr/bin/env python3
"""Tests for published v1 Docker GPU evidence and attestation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from publish_v1_docker_attestation import build_attestation
from v1_docker_gpu_evidence import (
    IMAGE_REPOSITORY,
    REQUIRED_ARTIFACTS,
    describe_artifacts,
    evaluate_manifest,
)
from v1_release_attestation import validate_payload


COMMIT = "a" * 40
DIGEST = "sha256:" + "b" * 64
IMAGE = f"{IMAGE_REPOSITORY}:v1.0.0"
ROOT = Path(__file__).resolve().parents[1]


def fixture(root: Path) -> dict:
    (root / "result").mkdir(parents=True)
    (root / "docker_pull.log").write_text(
        "pulled immutable image\n", encoding="utf-8"
    )
    (root / "docker_run.log").write_text(
        "CudaNav smoke passed\n", encoding="utf-8"
    )
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
    run = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{root / 'result'}:/out",
        IMAGE,
        "cudanav",
    ]
    manifest = {
        "schema_version": 1,
        "evidence_mode": "v1_published_docker_gpu_smoke",
        "status": "passed",
        "version": "1.0.0",
        "target_tag": "v1.0.0",
        "git_commit": COMMIT,
        "git_dirty": False,
        "docker_engine_version": "27.0.0",
        "gpu": [
            {
                "name": "GPU A",
                "uuid": "GPU-aaaa-bbbb",
                "driver_version": "999",
            }
        ],
        "image": {
            "reference": IMAGE,
            "digest": DIGEST,
            "repo_digests": [f"{IMAGE_REPOSITORY}@{DIGEST}"],
            "image_id": "sha256:" + "c" * 64,
            "labels": {
                "org.opencontainers.image.revision": COMMIT,
                "org.opencontainers.image.source": (
                    "https://github.com/rsasaki0109/CudaRobotics"
                ),
                "org.opencontainers.image.version": "v1.0.0",
            },
        },
        "commands": {
            "pull": ["docker", "pull", IMAGE],
            "run": run,
        },
        "returncodes": {"pull": 0, "run": 0},
        "artifacts": describe_artifacts(root, set(REQUIRED_ARTIFACTS)),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return manifest


class V1DockerGpuEvidenceTest(unittest.TestCase):
    def test_workflow_uses_tag_checkout_and_self_hosted_gpu(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "v1-docker-gpu-evidence.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "runs-on: [self-hosted, Linux, X64, gpu]", workflow
        )
        self.assertIn("ref: ${{ inputs.tag }}", workflow)
        self.assertIn("run_v1_docker_gpu_smoke.py", workflow)
        self.assertIn("publish_v1_docker_attestation.py", workflow)
        self.assertIn("actions/upload-artifact@v4", workflow)

    def test_valid_published_image_smoke_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            gate = evaluate_manifest(
                manifest, root, expected_commit=COMMIT
            )
            self.assertTrue(gate["passed"], gate)

    def test_valid_smoke_builds_matrix_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            attestation = build_attestation(root)
            gate = validate_payload(
                attestation,
                key="docker_gpu_evidence",
                target_version="1.0.0",
                target_tag="v1.0.0",
            )
            self.assertTrue(gate["passed"], gate)
            self.assertEqual(attestation["details"]["image_digest"], DIGEST)
            self.assertEqual(
                attestation["details"]["gpu_uuid"], "GPU-aaaa-bbbb"
            )

    def test_image_revision_must_match_release_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            manifest["image"]["labels"][
                "org.opencontainers.image.revision"
            ] = "d" * 40
            gate = evaluate_manifest(manifest, root)
            self.assertFalse(gate["checks"]["image_revision"])
            self.assertFalse(gate["passed"])

    def test_post_smoke_artifact_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            (root / "docker_run.log").write_text(
                "edited after capture\n", encoding="utf-8"
            )
            gate = evaluate_manifest(manifest, root)
            self.assertFalse(gate["checks"]["artifact_content"])
            with self.assertRaisesRegex(ValueError, "artifact_content"):
                build_attestation(root)

    def test_malformed_repo_digests_are_rejected_without_exception(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            manifest["image"]["repo_digests"] = None
            gate = evaluate_manifest(manifest, root)
            self.assertFalse(gate["checks"]["image_digest"])
            self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
