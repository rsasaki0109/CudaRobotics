#!/usr/bin/env python3
"""Tests for deployed v1 documentation evidence and attestation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from publish_v1_docs_attestation import build_attestation
from v1_documentation_evidence import (
    REQUIRED_ARTIFACTS,
    SITE,
    describe_artifacts,
    evaluate_manifest,
)
from v1_release_attestation import validate_payload
from write_v1_docs_release_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
TAG = "v1.0.0"


def fixture(root: Path) -> dict:
    site = root / "site"
    site.mkdir(parents=True)
    (site / "index.html").write_text(
        '<h1>v1.0.0</h1><a href="install.html">Install</a>'
        '<a href="nav2.html">Nav2</a>\n',
        encoding="utf-8",
    )
    (site / "install.html").write_text(
        "v1.0.0 releases/tag/v1.0.0 "
        "cuda-mppi-controller-demo:v1.0.0 "
        "/blob/v1.0.0/docs/v1_support_matrix.json\n",
        encoding="utf-8",
    )
    (site / "nav2.html").write_text(
        "v1.0.0 cuda_nav_bringup cudanav_closed_loop.launch.py "
        "/blob/v1.0.0/ros2_ws/src/cuda_nav_bringup/README.md\n",
        encoding="utf-8",
    )
    release = build_manifest(TAG, COMMIT)
    (site / "release.json").write_text(
        json.dumps(release) + "\n", encoding="utf-8"
    )
    urls = {
        "index": SITE,
        "install": SITE + "install.html",
        "nav2": SITE + "nav2.html",
        "release": SITE + "release.json",
    }
    manifest = {
        "schema_version": 1,
        "evidence_mode": "v1_documentation_http_deployment",
        "status": "passed",
        "version": "1.0.0",
        "target_tag": TAG,
        "git_commit": COMMIT,
        "git_dirty": False,
        "site": SITE,
        "urls": urls,
        "http_status": {key: 200 for key in urls},
        "artifacts": describe_artifacts(root, set(REQUIRED_ARTIFACTS)),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return manifest


class V1DocumentationEvidenceTest(unittest.TestCase):
    def test_workflow_preserves_gallery_and_refetches_deployment(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "v1-docs-deploy.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("ref: gh-pages", workflow)
        self.assertIn(
            "rsync -a --delete source/docs/site/ pages/docs/", workflow
        )
        self.assertIn("path: pages", workflow)
        self.assertIn("actions/deploy-pages@v4", workflow)
        self.assertIn("validate_v1_docs_source.py", workflow)
        self.assertIn("run_v1_docs_deployment_check.py", workflow)
        self.assertIn("publish_v1_docs_attestation.py", workflow)

    def test_deployed_pages_build_valid_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            evidence_gate = evaluate_manifest(
                manifest, root, expected_commit=COMMIT
            )
            self.assertTrue(evidence_gate["passed"], evidence_gate)
            attestation = build_attestation(root)
            gate = validate_payload(
                attestation,
                key="documentation_deployment",
                target_version="1.0.0",
                target_tag=TAG,
            )
            self.assertTrue(gate["passed"], gate)

    def test_wrong_deployed_source_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            release_path = root / "site" / "release.json"
            release = json.loads(release_path.read_text(encoding="utf-8"))
            release["source_commit"] = "b" * 40
            release_path.write_text(
                json.dumps(release) + "\n", encoding="utf-8"
            )
            manifest["artifacts"] = describe_artifacts(
                root, set(REQUIRED_ARTIFACTS)
            )
            gate = evaluate_manifest(manifest, root)
            self.assertFalse(gate["checks"]["release_schema"])
            self.assertFalse(gate["passed"])

    def test_post_fetch_page_edit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            (root / "site" / "nav2.html").write_text(
                "edited after fetch\n", encoding="utf-8"
            )
            gate = evaluate_manifest(manifest, root)
            self.assertFalse(gate["checks"]["artifact_content"])
            self.assertFalse(gate["checks"]["nav2_page"])


if __name__ == "__main__":
    unittest.main()
