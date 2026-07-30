#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
import unittest

from v1_support_matrix import (
    attestations_share_release_commit,
    evaluate,
    load,
)


class V1SupportMatrixTest(unittest.TestCase):
    def test_current_development_contract_is_valid_but_not_ready(self) -> None:
        result = evaluate(load())
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["ready"])
        self.assertFalse(result["readiness"]["release_status"])
        self.assertFalse(result["readiness"]["python_at_target"])
        self.assertFalse(result["readiness"]["ros_at_target"])

    def test_declared_component_version_must_match_source(self) -> None:
        matrix = deepcopy(load())
        matrix["surfaces"]["ros2"]["package_versions"][
            "cuda_kiss_icp"
        ] = "9.9.9"
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["ros_versions"])
        self.assertFalse(result["valid"])

    def test_fifteen_minute_budget_cannot_be_weakened(self) -> None:
        matrix = deepcopy(load())
        matrix["main_demo"]["time_budget_seconds"] = 1800
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["time_budget"])
        self.assertFalse(result["valid"])

    def test_main_demo_command_must_exist_on_every_user_surface(self) -> None:
        matrix = deepcopy(load())
        matrix["main_demo"]["run_command"] = "docker run imaginary"
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["main_command"])
        self.assertFalse(result["valid"])

    def test_colab_is_pinned_to_the_immutable_source_tag(self) -> None:
        result = evaluate(load())
        self.assertTrue(result["checks"]["colab"], result)
        self.assertFalse(result["readiness"]["colab_target_ref"], result)

    def test_colab_clone_cannot_float_to_master(self) -> None:
        matrix = deepcopy(load())
        matrix["source_tag"] = "v0.3.1"
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["source_tag"])
        self.assertFalse(result["checks"]["colab"])
        self.assertFalse(result["readiness"]["colab_target_ref"])

    def test_release_documents_are_part_of_the_surface_contract(self) -> None:
        matrix = deepcopy(load())
        matrix["surfaces"]["documentation"]["release_notes"] = (
            "docs/releases/v0.2.0_notes.md"
        )
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["documentation"])
        self.assertFalse(result["valid"])

    def test_development_published_version_must_match_docs(self) -> None:
        matrix = deepcopy(load())
        matrix["release_readiness"]["published_version"] = "0.1.0"
        result = evaluate(matrix)
        self.assertFalse(result["checks"]["published_version"])
        self.assertFalse(result["valid"])

    def test_legacy_inline_readiness_self_report_is_rejected(self) -> None:
        matrix = deepcopy(load())
        matrix["release_readiness"]["quickstart_15_minute_evidence"] = {
            "status": "passed",
            "version": "1.0.0",
            "git_commit": "a" * 40,
        }
        result = evaluate(matrix)
        self.assertFalse(result["readiness"]["quickstart_evidence"])
        self.assertFalse(
            result["attestations"]["quickstart_15_minute_evidence"][
                "checks"
            ]["reference_schema"]
        )

    def test_release_attestations_must_share_one_commit(self) -> None:
        gates = {
            key: {"passed": True, "git_commit": "a" * 40}
            for key in (
                "quickstart_15_minute_evidence",
                "cudanav_release_evidence",
                "docker_gpu_evidence",
                "documentation_deployment",
            )
        }
        self.assertTrue(attestations_share_release_commit(gates))
        gates["docker_gpu_evidence"]["git_commit"] = "b" * 40
        self.assertFalse(attestations_share_release_commit(gates))
        gates["docker_gpu_evidence"]["git_commit"] = "a" * 40
        gates["documentation_deployment"]["passed"] = False
        self.assertFalse(attestations_share_release_commit(gates))


if __name__ == "__main__":
    unittest.main()
