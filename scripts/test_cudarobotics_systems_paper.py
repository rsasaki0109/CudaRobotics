#!/usr/bin/env python3
"""Keep the CudaNav systems draft aligned with its claim ledger."""

from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paper = (ROOT / "paper" / "cudarobotics_systems_paper.md").read_text(
        encoding="utf-8"
    )
    ledger = json.loads(
        (ROOT / "paper" / "artifacts" / "cudarobotics_systems.json").read_text(
            encoding="utf-8"
        )
    )
    latex = (ROOT / "paper" / "latex" / "cudanav_systems.tex").read_text(
        encoding="utf-8"
    )
    workflow = (ROOT / ".github" / "workflows" / "systems-paper.yml").read_text(
        encoding="utf-8"
    )
    assert paper.startswith("# CudaNav: A Reproducible End-to-End GPU Autonomy Stack")
    assert ledger["title"] in paper.splitlines()[0]
    for claim in ledger["claims"]:
        claim_id = claim["id"]
        assert (
            f"`{claim_id}`" in paper
        ), f"systems draft does not map ledger claim {claim_id}"
        status_row = (
            rf"\| `{re.escape(claim_id)}` \| "
            rf"{re.escape(claim['status'].title())} \|"
        )
        assert re.search(
            status_row, paper
        ), f"systems draft status is stale for {claim_id}"
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", paper):
        if "://" in target or target.startswith("#"):
            continue
        local = (ROOT / "paper" / target.split("#", 1)[0]).resolve()
        assert (
            local.is_relative_to(ROOT) and local.exists()
        ), f"systems draft has a broken local link: {target}"
    required_boundaries = [
        "recorded-data shadow",
        "command-driven closed-loop simulation",
        "optional cross-device",
        "release or submission gate",
        "ros 2",
        "mcap",
    ]
    paper_lower = paper.lower()
    for phrase in required_boundaries:
        assert phrase in paper_lower, f"systems draft omits boundary: {phrase}"
    stale_claims = [
        "97 GPU-parallel implementations",
        "53,404x",
        "86,613x",
        "License: [TBD]",
    ]
    for phrase in stale_claims:
        assert phrase not in paper, f"legacy unsupported claim remains: {phrase}"
    assert "0.812 m ATE RMSE" in paper
    assert "0.819 m ATE RMSE" not in paper
    assert r"\documentclass[conference]{IEEEtran}" in latex
    assert r"\author{\IEEEauthorblockN{Anonymous Authors}}" in latex
    assert r"\bibliography{references}" in latex
    for claim in ledger["claims"]:
        escaped = claim["id"].replace("_", r"\_")
        assert (
            escaped in latex
        ), f"systems LaTeX source omits ledger claim {claim['id']}"
    latex_required = [
        "second distinct physical GPU model",
        "optional extension",
        "recorded-data shadow",
        "real-robot closed-loop navigation",
        "1,059.4",
        "352.748",
        "0.003493",
        "0.815",
        "0.812",
        "1,325.5",
        "4.801",
    ]
    latex_lower = latex.lower()
    for phrase in latex_required:
        assert (
            phrase.lower() in latex_lower
        ), f"systems LaTeX source omits result/boundary: {phrase}"
    for token in ("Ryohei", "Sasaki", "rsasa", "@"):
        assert (
            token.lower() not in latex_lower
        ), f"systems LaTeX source contains identity token: {token}"
    workflow_required = [
        "validate_paper_artifacts.py",
        "test_cudarobotics_systems_paper.py",
        "cudanav_systems.tex",
        "latex-action@6549dc21effb2730855a1281407ecfcececc6c1b",
        "cudanav-systems-paper-candidate-${{ github.sha }}",
        "steps.readiness.outputs.ready == 'true'",
    ]
    for phrase in workflow_required:
        assert phrase in workflow, f"systems paper workflow omits contract: {phrase}"
    print(
        "CudaNav systems Markdown, anonymous IEEE source, and workflow "
        "are aligned with "
        f"{len(ledger['claims'])} ledger claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
