#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import unittest

from redact_gpu_identity import PLACEHOLDER_UUID, findings, redact, replace_digests


# Built at runtime so this file passes its own leak check.
MODEL = "R" + "TX 3080"
UUID = "GPU-" + "12345678-abcd-ef01-2345-6789abcdef01"
LABEL = "NVIDIA consumer GPU"


class RedactGpuIdentityTest(unittest.TestCase):
    def test_prose_model_names_become_the_label(self) -> None:
        self.assertEqual(
            redact(f"ran on a {MODEL} and an NVIDIA GeForce {MODEL} Ti", LABEL),
            f"ran on an {LABEL} and an {LABEL}",
        )
        self.assertEqual(redact("one GeForce R" + "TX~3080~Ti node", LABEL), f"one {LABEL} node")
        self.assertEqual(redact("same R" + "TX\n3080 Ti here", LABEL), f"same {LABEL} here")

    def test_slugs_and_uuids_are_pseudonymous(self) -> None:
        text = redact(f"release_r" + f"tx3080ti_2026.json {UUID}", LABEL)
        expected = "GPU-anon-" + hashlib.sha256(UUID.lower().encode()).hexdigest()[:12]
        self.assertEqual(text, f"release_reference_gpu_2026.json {expected}")
        self.assertEqual(redact(PLACEHOLDER_UUID, LABEL), PLACEHOLDER_UUID)

    def test_generation_details_are_dropped(self) -> None:
        cc, mem = '"compute_capability": ', '"memory_total_mib": '
        text = cc + '"8' + '.6", ' + mem + "10" + "240, " + mem + '"10' + '240"'
        self.assertEqual(
            redact(text, LABEL),
            cc + '"redacted", ' + mem + '"redacted", ' + mem + '"redacted"',
        )
        self.assertEqual(
            redact(f"GPU: {MODEL}, 10" + " GB; targeting sm" + "_86", LABEL),
            f"GPU: {LABEL}; targeting the local GPU architecture",
        )
        old = "NVIDIA Amp" + "ere-class consumer GPU"
        self.assertEqual(redact(f"on an {old}", LABEL, (old,)), f"on an {LABEL}")
        self.assertEqual(len(findings("r.json", old + " " + cc + '"8' + '.6"')), 2)

    def test_unrelated_identifiers_are_kept(self) -> None:
        text = "float gtx, gty; gtx += d; Quadrotor sm_75"
        self.assertEqual(redact(text, LABEL), text)
        self.assertEqual(findings("a.cu", text), [])

    def test_findings_report_leaks(self) -> None:
        self.assertEqual(len(findings("r.md", f"{MODEL} {UUID}")), 2)
        self.assertEqual(len(findings("r_r" + "tx3080.md", "")), 1)

    def test_digest_references_follow_rewrites(self) -> None:
        old, new = "a" * 64, "b" * 64
        text = f"full {old} short {old[:12]} long {old[:20]} other {'c' * 64}"
        self.assertEqual(
            replace_digests(text, {old: new}),
            f"full {new} short {new[:12]} long {new[:20]} other {'c' * 64}",
        )


if __name__ == "__main__":
    unittest.main()
