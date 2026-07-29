#!/usr/bin/env python3
"""Render the frozen contact-paper figures only from published CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
SOURCES = {
    "robustness": RESULTS / "contact_robustness_2026-07-28_comparisons.csv",
    "matched_compute": RESULTS / "contact_matched_compute_2026-07-28_summary.csv",
    "external_fidelity": RESULTS
    / "contact_external_fidelity_2026-07-28_summary.csv",
}
PLANNERS = ("mppi", "diff_mppi_3", "soppi_fast")
COLORS = {
    "mppi": "#4477AA",
    "diff_mppi_3": "#228833",
    "soppi_fast": "#CCBB44",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"published CSV is empty: {path}")
    return rows


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    outputs = []
    for suffix in (".pdf", ".svg", ".png"):
        path = stem.with_suffix(suffix)
        metadata = {"Creator": "CudaRobotics"}
        if suffix == ".pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif suffix == ".svg":
            metadata["Date"] = None
        else:
            metadata = {"Software": "CudaRobotics"}
        figure.savefig(path, metadata=metadata)
        outputs.append(path)
    return outputs


def _short_scenario(value: str) -> str:
    return value.removeprefix("box_").replace("_", "\n")


def render_robustness(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0, "neutral": 0}
    )
    selected = [
        row
        for row in rows
        if row["planner"] == "diff_mppi_3" and row["baseline"] == "mppi"
    ]
    for row in selected:
        significant = float(row["mcnemar_holm_p"]) < 0.05
        delta = float(row["success_delta"])
        key = (
            "positive"
            if significant and delta > 0.0
            else "negative"
            if significant and delta < 0.0
            else "neutral"
        )
        counts[row["condition"]][key] += 1
    conditions = sorted(counts)
    positives = [counts[name]["positive"] for name in conditions]
    negatives = [counts[name]["negative"] for name in conditions]

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    positions = list(range(len(conditions)))
    axis.barh(positions, positives, color="#228833", label="positive")
    axis.barh(
        positions,
        [-value for value in negatives],
        color="#CC6677",
        label="negative",
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(positions, [name.replace("_", " ") for name in conditions])
    axis.set_xlabel("Holm-significant cells vs MPPI (Diff-MPPI-3)")
    axis.set_title("Contact robustness across model and geometry shifts")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    files = _save_figure(figure, output / "contact_robustness")
    plt.close(figure)
    return {
        "source_rows": len(rows),
        "selected_comparisons": len(selected),
        "conditions": [
            {"condition": name, **counts[name]} for name in conditions
        ],
        "files": [path.name for path in files],
    }


def render_matched_compute(
    rows: list[dict[str, str]], output: Path
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    indexed = {(row["scenario"], row["planner"]): row for row in rows}
    missing = [
        (scenario, planner)
        for scenario in scenarios
        for planner in PLANNERS
        if (scenario, planner) not in indexed
    ]
    if missing:
        raise ValueError(f"matched-compute matrix is incomplete: {missing}")

    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    width = 0.24
    positions = list(range(len(scenarios)))
    semantic_rows = []
    for planner_index, planner in enumerate(PLANNERS):
        values = []
        lower = []
        upper = []
        for scenario in scenarios:
            row = indexed[(scenario, planner)]
            rate = float(row["real_time_success_rate"])
            values.append(rate)
            lower.append(max(0.0, rate - float(row["success_ci_low"])))
            upper.append(max(0.0, float(row["success_ci_high"]) - rate))
            semantic_rows.append(
                {
                    "scenario": scenario,
                    "planner": planner,
                    "success_rate": rate,
                    "ci_low": float(row["success_ci_low"]),
                    "ci_high": float(row["success_ci_high"]),
                    "mean_control_ms": float(row["mean_control_ms"]),
                    "deadline_misses": int(row["deadline_misses"]),
                }
            )
        offsets = [
            position + (planner_index - 1) * width for position in positions
        ]
        axis.bar(
            offsets,
            values,
            width,
            yerr=[lower, upper],
            capsize=2,
            color=COLORS[planner],
            label=planner.replace("_", " "),
        )
    axis.set_xticks(positions, [_short_scenario(name) for name in scenarios])
    axis.set_ylim(0.0, 1.12)
    axis.set_ylabel("real-time success rate")
    axis.set_title("Held-out performance in the enforced 10 ms control slot")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    files = _save_figure(figure, output / "contact_matched_compute")
    plt.close(figure)
    return {
        "source_rows": len(rows),
        "scenarios": scenarios,
        "rows": semantic_rows,
        "files": [path.name for path in files],
    }


def render_external_fidelity(
    rows: list[dict[str, str]], output: Path
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "successes": 0}
    )
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    for row in rows:
        key = (row["condition"], row["planner"])
        totals[key]["episodes"] += int(row["episodes"])
        totals[key]["successes"] += int(row["successes"])
    missing = [
        (condition, planner)
        for condition in conditions
        for planner in PLANNERS
        if (condition, planner) not in totals
    ]
    if missing:
        raise ValueError(f"external-fidelity matrix is incomplete: {missing}")

    figure, axis = plt.subplots(figsize=(7.0, 3.9))
    positions = list(range(len(conditions)))
    semantic_rows = []
    for planner in PLANNERS:
        rates = []
        for condition in conditions:
            aggregate = totals[(condition, planner)]
            rate = aggregate["successes"] / aggregate["episodes"]
            rates.append(rate)
            semantic_rows.append(
                {
                    "condition": condition,
                    "planner": planner,
                    **aggregate,
                    "success_rate": rate,
                }
            )
        axis.plot(
            positions,
            rates,
            marker="o",
            linewidth=1.8,
            color=COLORS[planner],
            label=planner.replace("_", " "),
        )
    axis.set_xticks(
        positions,
        [name.replace("_", "\n") for name in conditions],
    )
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("aggregate success rate")
    axis.set_title("Closed-loop MuJoCo transfer across physical variations")
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    files = _save_figure(figure, output / "contact_external_fidelity")
    plt.close(figure)
    return {
        "source_rows": len(rows),
        "conditions": conditions,
        "rows": semantic_rows,
        "files": [path.name for path in files],
    }


def render(output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "svg.hashsalt": "cudarobotics-contact-submission-v1",
        }
    )
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = {name: read_rows(path) for name, path in SOURCES.items()}
    figures = {
        "robustness": render_robustness(rows["robustness"], output),
        "matched_compute": render_matched_compute(
            rows["matched_compute"], output
        ),
        "external_fidelity": render_external_fidelity(
            rows["external_fidelity"], output
        ),
    }
    manifest = {
        "schema_version": 1,
        "evidence_mode": "contact_submission_figures",
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in SOURCES.items()
        },
        "figures": figures,
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "contact_submission_figures",
    )
    args = parser.parse_args()
    render(args.output_dir)
    print((args.output_dir.resolve() / "figure_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
