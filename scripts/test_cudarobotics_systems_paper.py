#!/usr/bin/env python3
"""Keep the CudaNav systems draft aligned with its claim ledger."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paper = (
        ROOT / "paper" / "cudarobotics_systems_paper.md"
    ).read_text(encoding="utf-8")
    ledger = json.loads(
        (
            ROOT / "paper" / "artifacts" / "cudarobotics_systems.json"
        ).read_text(encoding="utf-8")
    )
    assert paper.startswith(
        "# CudaNav: A Reproducible End-to-End GPU Autonomy Stack"
    )
    assert ledger["title"] in paper.splitlines()[0]
    for claim in ledger["claims"]:
        claim_id = claim["id"]
        assert f"`{claim_id}`" in paper, (
            f"systems draft does not map ledger claim {claim_id}"
        )
        status_row = (
            rf"\| `{re.escape(claim_id)}` \| "
            rf"{re.escape(claim['status'].title())} \|"
        )
        assert re.search(status_row, paper), (
            f"systems draft status is stale for {claim_id}"
        )
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", paper):
        if "://" in target or target.startswith("#"):
            continue
        local = (
            ROOT / "paper" / target.split("#", 1)[0]
        ).resolve()
        assert local.is_relative_to(ROOT) and local.exists(), (
            f"systems draft has a broken local link: {target}"
        )
    required_boundaries = [
        "recorded-data shadow",
        "command-driven closed-loop simulation",
        "second physical gpu",
        "ready: false",
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
    print(
        "CudaNav systems draft is aligned with "
        f"{len(ledger['claims'])} ledger claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
