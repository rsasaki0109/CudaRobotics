#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from stage_v1_docs import TAG, stage
from v1_documentation_evidence import evaluate_site_content
from write_v1_docs_release_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


class StageV1DocsTest(unittest.TestCase):
    def test_staged_site_is_tag_pinned_and_release_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            stage(ROOT / "docs" / "site", output)
            (output / "release.json").write_text(
                json.dumps(build_manifest(TAG, COMMIT)) + "\n",
                encoding="utf-8",
            )
            checks = evaluate_site_content(
                output,
                version="1.0.0",
                target_tag=TAG,
                git_commit=COMMIT,
            )
            self.assertTrue(all(checks.values()), checks)
            for name in ("index.html", "install.html", "nav2.html"):
                text = (output / name).read_text(encoding="utf-8")
                self.assertNotIn("0.2.0", text)
                self.assertNotIn("CudaRobotics/blob/master/", text)
                self.assertNotIn("CudaRobotics/tree/master/", text)

    def test_live_site_remains_the_published_v0_2_surface(self) -> None:
        index = (ROOT / "docs" / "site" / "index.html").read_text(
            encoding="utf-8"
        )
        install = (ROOT / "docs" / "site" / "install.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("published v0.2.0", index)
        self.assertIn("releases/tag/v0.2.0", install)

    def test_wrong_tag_and_existing_output_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                stage(ROOT / "docs" / "site", root / "wrong", tag="main")
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                stage(ROOT / "docs" / "site", output)


if __name__ == "__main__":
    unittest.main()
