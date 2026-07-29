#!/usr/bin/env python3
"""Render the contact-rich Diff-MPPI paper results from published evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from paper_artifact_contract import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
PAPER_MANIFEST = ROOT / "paper" / "artifacts" / "contact_rich_diff_mppi.json"
DEFAULT_OUTPUT = ROOT / "paper" / "contact_rich_diff_mppi_results.md"

ROBUSTNESS_PREFIX = "contact_robustness_2026-07-28"
MATCHED_PREFIX = "contact_matched_compute_2026-07-28"
EXTERNAL_PREFIX = "contact_external_fidelity_2026-07-28"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(
    rows: list[dict[str, str]],
    *,
    rate_field: str,
    count_field: str = "episodes",
    latency_field: str | None = None,
    misses_field: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["planner"], []).append(row)
    output = []
    for planner, planner_rows in grouped.items():
        episodes = sum(int(row[count_field]) for row in planner_rows)
        successes = sum(
            float(row[rate_field]) * int(row[count_field])
            for row in planner_rows
        )
        item: dict[str, Any] = {
            "planner": planner,
            "episodes": episodes,
            "rate": successes / episodes,
        }
        if latency_field:
            item["latency"] = sum(
                float(row[latency_field]) * int(row[count_field])
                for row in planner_rows
            ) / episodes
        if misses_field:
            item["misses"] = sum(
                int(row[misses_field]) for row in planner_rows
            )
        output.append(item)
    return output


def significant_counts(
    rows: list[dict[str, str]],
) -> tuple[int, int]:
    significant = [
        row for row in rows if float(row["mcnemar_holm_p"]) < 0.05
    ]
    positive = sum(float(row["success_delta"]) > 0.0 for row in significant)
    negative = sum(float(row["success_delta"]) < 0.0 for row in significant)
    return positive, negative


def matched_cell(
    rows: list[dict[str, str]], scenario: str, planner: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["scenario"] == scenario and row["planner"] == planner
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one matched-compute cell for {scenario}/{planner}"
        )
    return matches[0]


def evidence_paths(prefix: str) -> dict[str, Path]:
    return {
        "summary": RESULTS / f"{prefix}_summary.csv",
        "comparisons": RESULTS / f"{prefix}_comparisons.csv",
        "provenance": RESULTS / f"{prefix}_provenance.json",
    }


def validate_sources() -> None:
    manifest = load_json(PAPER_MANIFEST)
    validation = validate_manifest(manifest, ROOT)
    if not validation["valid"] or not validation["ready"]:
        raise ValueError("contact paper evidence ledger is not ready")
    required = []
    for prefix in (
        ROBUSTNESS_PREFIX,
        MATCHED_PREFIX,
        EXTERNAL_PREFIX,
    ):
        required.extend(evidence_paths(prefix).values())
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise ValueError("missing published evidence: " + ", ".join(missing))


def render() -> str:
    validate_sources()
    robustness = evidence_paths(ROBUSTNESS_PREFIX)
    matched = evidence_paths(MATCHED_PREFIX)
    external = evidence_paths(EXTERNAL_PREFIX)

    robust_summary = load_csv(robustness["summary"])
    robust_comparisons = load_csv(robustness["comparisons"])
    robust_provenance = load_json(robustness["provenance"])
    matched_summary = load_csv(matched["summary"])
    matched_comparisons = load_csv(matched["comparisons"])
    matched_provenance = load_json(matched["provenance"])
    external_summary = load_csv(external["summary"])
    external_comparisons = load_csv(external["comparisons"])
    external_provenance = load_json(external["provenance"])

    robust_aggregate = aggregate(
        robust_summary,
        rate_field="success_rate",
        latency_field="control_ms_mean",
    )
    matched_aggregate = aggregate(
        matched_summary,
        rate_field="real_time_success_rate",
        misses_field="deadline_misses",
    )
    external_aggregate = aggregate(
        external_summary,
        rate_field="success_rate",
        latency_field="control_ms_mean",
    )
    robust_positive, robust_negative = significant_counts(
        robust_comparisons
    )
    external_positive, external_negative = significant_counts(
        external_comparisons
    )
    matched_diff = matched_cell(
        matched_comparisons, "box_align_contact_loss", "diff_mppi_3"
    )
    matched_soppi = matched_cell(
        matched_comparisons, "box_align_contact_loss", "soppi_fast"
    )

    lines = [
        "# Contact-Rich Diff-MPPI Release Results",
        "",
        "This chapter is generated from the validated, content-addressed release "
        "artifacts. Statistical outcomes are reported independently of evidence "
        "integrity; negative and zero-success cells remain in the source tables.",
        "",
        "## Evidence freeze",
        "",
        "| Block | Episodes | Commit | GPU |",
        "|---|---:|---|---|",
        (
            f"| Robustness | "
            f"{robust_provenance['source']['matrix']['episodes']:,} | "
            f"`{robust_provenance['source']['experiment']['git_commit'][:12]}` | "
            f"{robust_provenance['source']['gpu'][0]['name']} |"
        ),
        (
            f"| Matched compute | "
            f"{matched_provenance['source']['matrix']['calibration_episodes']} "
            f"calibration + "
            f"{matched_provenance['source']['matrix']['evaluation_episodes']} "
            f"held-out | "
            f"`{matched_provenance['source']['experiment']['git_commit'][:12]}` | "
            f"{matched_provenance['source']['gpu'][0]['name']} |"
        ),
        (
            f"| MuJoCo transfer | "
            f"{external_provenance['source']['matrix']['episodes']:,} | "
            f"`{external_provenance['source']['experiment']['git_commit'][:12]}` | "
            f"{external_provenance['source']['gpu'][0]['name']} |"
        ),
        "",
        "## Broad robustness",
        "",
        "The fixed release matrix spans 12 plant conditions, five contact tasks, "
        "six planners, K={128,256,512}, and 30 paired seeds. It contains "
        f"{robust_positive} Holm-significant positive and {robust_negative} "
        "Holm-significant negative success cells versus MPPI.",
        "",
        "| Planner | Episodes | Success | Mean control ms |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(
        robust_aggregate, key=lambda item: (-item["rate"], item["planner"])
    ):
        lines.append(
            f"| {row['planner']} | {row['episodes']:,} | "
            f"{row['rate']:.3f} | {row['latency']:.3f} |"
        )

    lines += [
        "",
        "The aggregate ordering favors Diff-MPPI-3, but the effect is not "
        "universal. All six Holm-significant negative cells are Diff-MPPI-3 on "
        "the tall-box condition. The detour task remains a visible negative "
        "control rather than being removed from the family.",
        "",
        "## Exact 10 ms matched compute",
        "",
        "Calibration and evaluation seeds are disjoint. Each planner selected "
        "K=1024 and received the same enforced 10 ms control slot. "
        "`real_time_success` requires task success and zero deadline misses.",
        "",
        "| Planner | Held-out episodes | Real-time success | Deadline misses |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(
        matched_aggregate, key=lambda item: (-item["rate"], item["planner"])
    ):
        lines.append(
            f"| {row['planner']} | {row['episodes']} | "
            f"{row['rate']:.3f} | {row['misses']} |"
        )
    lines += [
        "",
        "On `box_align_contact_loss`, Diff-MPPI-3 improves real-time success "
        f"by {float(matched_diff['real_time_success_delta']):+.3f} "
        f"(95% bootstrap CI "
        f"[{float(matched_diff['success_delta_ci_low']):+.3f}, "
        f"{float(matched_diff['success_delta_ci_high']):+.3f}], "
        f"Holm p={float(matched_diff['mcnemar_holm_p']):.6f}). "
        "SOPPI-fast improves it by "
        f"{float(matched_soppi['real_time_success_delta']):+.3f} "
        f"(Holm p={float(matched_soppi['mcnemar_holm_p']):.6f}). "
        "The other four scenario families are not Holm-significant, and every "
        "planner remains at 0/30 on `box_align_detour`.",
        "",
        "## Closed-loop MuJoCo transfer",
        "",
        "The CUDA planners retain the nominal smooth rollout model while "
        f"MuJoCo {external_provenance['source']['engine']['version']} executes "
        "every selected command and returns the next state. The matrix declares "
        "friction, mass, and observation-noise variations.",
        "",
        "| Planner | Episodes | Success | Mean control ms |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(
        external_aggregate, key=lambda item: (-item["rate"], item["planner"])
    ):
        lines.append(
            f"| {row['planner']} | {row['episodes']} | "
            f"{row['rate']:.3f} | {row['latency']:.3f} |"
        )
    lines += [
        "",
        f"The full 70-cell family contains {external_positive} "
        f"Holm-significant positive and {external_negative} negative cells. "
        "No individual observation-noise cell survives full-family Holm "
        "correction, so sensing-noise effects remain descriptive.",
        "",
        "## Claim boundary",
        "",
        "- These experiments support a contact-rich, compute-quality result; "
        "they do not establish universal planner dominance.",
        "- SOPPI-fast contains one nominal gradient step and is not a pure "
        "sampling-only or pure-SVGD baseline.",
        "- The MuJoCo task is a custom planar closed-loop sim-to-sim transfer, "
        "not a standard manipulator benchmark or real-robot result.",
        "- All results are from one GTX 1660 Ti. Independent hardware "
        "replication is desirable but is not silently implied.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 scripts/validate_paper_artifacts.py "
        "paper/artifacts/contact_rich_diff_mppi.json --require-ready",
        "python3 scripts/render_contact_paper_results.py --check",
        "```",
        "",
        "Source artifacts:",
        "",
        f"- `docs/results/{ROBUSTNESS_PREFIX}_provenance.json`",
        f"- `docs/results/{MATCHED_PREFIX}_provenance.json`",
        f"- `docs/results/{EXTERNAL_PREFIX}_provenance.json`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            print(f"generated contact paper results are stale: {output}")
            return 1
        print(f"contact paper results are current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
