#!/usr/bin/env python3
"""Contract tests for strict cross-runner CudaNav multi-GPU collection."""

from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/cudanav-multi-gpu.yml"
DOCS = ROOT / "docs/cudanav_multi_gpu.md"


class CudaNavMultiGpuWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.docs = DOCS.read_text(encoding="utf-8")

    def test_two_distinct_self_hosted_runner_labels_are_required(self) -> None:
        self.assertIn("runner_a_label:", self.workflow)
        self.assertIn("runner_b_label:", self.workflow)
        self.assertIn(
            'test "$RUNNER_A_LABEL" != "$RUNNER_B_LABEL"',
            self.workflow,
        )
        self.assertEqual(self.workflow.count("- self-hosted"), 2)
        self.assertIn("- ${{ inputs.runner_a_label }}", self.workflow)
        self.assertIn("- ${{ inputs.runner_b_label }}", self.workflow)

    def test_each_node_runs_the_unweakened_native_release(self) -> None:
        self.assertEqual(self.workflow.count("--evidence-kind native-release"), 3)
        self.assertEqual(self.workflow.count("--repetitions 1"), 2)
        self.assertEqual(self.workflow.count("--minimum-gpu-devices 1"), 2)
        self.assertEqual(self.workflow.count("--minimum-gpu-models 1"), 2)
        self.assertEqual(
            self.workflow.count("--target cudanav_gpu_closed_loop_s_course"),
            2,
        )

    def test_aggregator_restores_strict_publication_thresholds(self) -> None:
        self.assertIn("needs:\n      - gpu-a\n      - gpu-b", self.workflow)
        self.assertIn('test "${#node_a_runs[@]}" -eq 1', self.workflow)
        self.assertIn('test "${#node_b_runs[@]}" -eq 1', self.workflow)
        self.assertIn("--minimum-gpu-devices 2", self.workflow)
        self.assertIn("--minimum-gpu-models 2", self.workflow)
        self.assertIn(
            "python3 scripts/validate_cudanav_multi_gpu.py",
            self.workflow,
        )

    def test_artifacts_are_bound_to_the_exact_workflow_commit(self) -> None:
        self.assertGreaterEqual(self.workflow.count("${{ github.sha }}"), 5)
        self.assertEqual(
            self.workflow.count('test "$(git rev-parse HEAD)" = "$GITHUB_SHA"'),
            2,
        )
        self.assertEqual(
            self.workflow.count('test -z "$(git status --porcelain)"'),
            5,
        )

    def test_operator_documentation_names_the_workflow_and_hard_gate(
        self,
    ) -> None:
        docs = " ".join(self.docs.split())
        self.assertIn("cudanav-multi-gpu.yml", docs)
        self.assertIn("two distinct self-hosted runner labels", docs)
        self.assertIn("two distinct physical GPU UUIDs", docs)
        self.assertIn("two distinct GPU model names", docs)


if __name__ == "__main__":
    unittest.main()
