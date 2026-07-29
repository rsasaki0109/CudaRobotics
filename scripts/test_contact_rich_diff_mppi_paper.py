#!/usr/bin/env python3
"""Keep the contact-rich submission draft aligned with its ready ledger."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    draft_path = ROOT / "paper" / "diff_mppi_submission_draft.md"
    paper = draft_path.read_text(encoding="utf-8")
    latex_path = ROOT / "paper" / "latex" / "contact_rich_diff_mppi.tex"
    latex = latex_path.read_text(encoding="utf-8")
    ledger = json.loads(
        (
            ROOT / "paper" / "artifacts" / "contact_rich_diff_mppi.json"
        ).read_text(encoding="utf-8")
    )
    assert paper.startswith(f"# {ledger['title']}")
    for claim in ledger["claims"]:
        claim_id = claim["id"]
        assert re.search(
            rf"\| `{re.escape(claim_id)}` \| "
            rf"{re.escape(claim['status'].title())} \|",
            paper,
        ), f"contact draft status is stale or missing for {claim_id}"
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", paper):
        if "://" in target or target.startswith("#"):
            continue
        local = (draft_path.parent / target.split("#", 1)[0]).resolve()
        assert local.is_relative_to(ROOT) and local.exists(), (
            f"contact draft has a broken local link: {target}"
        )
    required = [
        "32,400-episode",
        "33 Holm-significant positive",
        "6 Holm-significant negative",
        "enforced 10 ms",
        "0.000305",
        "3,150 closed-loop",
        "three Holm-significant positive",
        "zero negative",
        "ready: true",
        "not universal planner dominance",
        "not real-robot evidence",
    ]
    paper_lower = paper.lower()
    for phrase in required:
        assert phrase.lower() in paper_lower, (
            f"contact draft omits frozen result/boundary: {phrase}"
        )
    stale = [
        "# Diff-MPPI Main-Paper Draft",
        "dynamic_slalom",
        "7-DOF serial-arm",
        "the only method family",
    ]
    for phrase in stale:
        assert phrase not in paper, f"legacy broad claim remains: {phrase}"
    assert r"\documentclass[conference]{IEEEtran}" in latex
    assert r"\author{\IEEEauthorblockN{Anonymous Authors}}" in latex
    assert r"\graphicspath{{../figures/submission/}}" in latex
    assert r"\bibliography{references}" in latex
    for stem in (
        "contact_robustness.pdf",
        "contact_matched_compute.pdf",
        "contact_external_fidelity.pdf",
    ):
        assert stem in latex, f"contact LaTeX source omits figure: {stem}"
    latex_required = [
        "32,400-episode",
        "33 Holm-significant",
        "6 negative",
        "enforced 10 ms",
        "0.000305",
        "3,150",
        "three Holm-significant positive",
        "zero negative",
        "not universal planner dominance",
        "not real-robot evidence",
    ]
    latex_lower = latex.lower()
    for phrase in latex_required:
        assert phrase.lower() in latex_lower, (
            f"contact LaTeX source omits frozen result/boundary: {phrase}"
        )
    for token in ("Ryohei", "Sasaki", "rsasa", "@"):
        assert token.lower() not in latex_lower, (
            f"contact LaTeX source contains identity token: {token}"
        )
    print(
        "Contact-rich Markdown and anonymous IEEE source are aligned with "
        f"{len(ledger['claims'])} ready claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
