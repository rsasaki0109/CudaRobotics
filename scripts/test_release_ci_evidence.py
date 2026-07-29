#!/usr/bin/env python3

from __future__ import annotations

import unittest

from release_ci_evidence import GATE_CONTRACTS, evaluate


def valid_payload(gate: str) -> dict:
    contract = GATE_CONTRACTS[gate]
    return {
        "schema_version": 1,
        "evidence_mode": "release_ci",
        "status": "passed",
        "generated_at": "2026-07-29T00:00:00+00:00",
        "gate": gate,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "github": {
            "repository": "rsasaki0109/CudaRobotics",
            "workflow": contract["workflow"],
            "run_id": 123,
            "run_attempt": 1,
            "run_url": (
                "https://github.com/rsasaki0109/CudaRobotics/actions/runs/123"
            ),
            "event": "workflow_dispatch",
            "ref": "refs/heads/release-candidate",
        },
        "platform": {"os": "Linux", "arch": "X64"},
        "checks": {name: "passed" for name in contract["checks"]},
        "artifacts": sorted(contract["artifacts"]),
        "artifact_manifest": (
            {
                "name": "python_artifacts.json",
                "bytes": 100,
                "sha256": "b" * 64,
            }
            if contract["artifact_manifest"]
            else None
        ),
    }


class ReleaseCiEvidenceTest(unittest.TestCase):
    def test_each_complete_gate_passes(self) -> None:
        for gate in GATE_CONTRACTS:
            with self.subTest(gate=gate):
                result = evaluate(
                    valid_payload(gate),
                    expected_gate=gate,
                    expected_commit="a" * 40,
                )
                self.assertTrue(result["passed"], result)

    def test_commit_mismatch_is_rejected(self) -> None:
        result = evaluate(
            valid_payload("github_build"),
            expected_commit="b" * 40,
        )
        self.assertFalse(result["checks"]["git_commit"])
        self.assertFalse(result["passed"])

    def test_pull_request_run_is_not_release_evidence(self) -> None:
        payload = valid_payload("github_build")
        payload["github"]["event"] = "pull_request"
        result = evaluate(payload)
        self.assertFalse(result["checks"]["event"])
        self.assertFalse(result["passed"])

    def test_missing_check_or_artifact_is_rejected(self) -> None:
        for mutation in ("check", "artifact"):
            with self.subTest(mutation=mutation):
                payload = valid_payload("python_manylinux_wheels")
                if mutation == "check":
                    payload["checks"].pop("manylinux_cp310_cp312")
                else:
                    payload["artifacts"].remove(
                        "cudarobotics-manylinux-wheels"
                    )
                self.assertFalse(evaluate(payload)["passed"])

    def test_workflow_identity_cannot_be_relabelled(self) -> None:
        payload = valid_payload("github_build")
        payload["github"]["workflow"] = "Python package"
        result = evaluate(payload)
        self.assertFalse(result["checks"]["workflow"])
        self.assertFalse(result["passed"])

    def test_python_artifact_manifest_is_required(self) -> None:
        payload = valid_payload("python_manylinux_wheels")
        payload["artifact_manifest"] = None
        result = evaluate(payload)
        self.assertFalse(result["checks"]["artifact_manifest"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
