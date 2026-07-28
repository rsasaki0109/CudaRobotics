#!/usr/bin/env python3
"""Produce a quality-gated CUDA MPPI evaluation from a real rosbag2 DB3."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import analyze_rosbag_clearance
import export_rosbag_motion
import render_cuda_mppi_diagnostics


def diagnostics_metrics(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    rows = render_cuda_mppi_diagnostics.load_rows(path)
    solve = [
        render_cuda_mppi_diagnostics.as_float(row, "solve_ms") for row in rows
    ]
    valid = [
        render_cuda_mppi_diagnostics.as_float(row, "valid_rollout_ratio")
        for row in rows
    ]
    return {
        "source": str(path),
        "samples": len(rows),
        "solve_mean_ms": render_cuda_mppi_diagnostics.mean(solve),
        "solve_p95_ms": render_cuda_mppi_diagnostics.percentile(solve, 0.95),
        "solve_max_ms": max(render_cuda_mppi_diagnostics.finite(solve)),
        "valid_rollout_ratio_mean": render_cuda_mppi_diagnostics.mean(valid),
        "valid_rollout_ratio_min": min(
            render_cuda_mppi_diagnostics.finite(valid)
        ),
        "all_colliding_cycles": sum(
            row.get("all_colliding") == "1" for row in rows
        ),
        "retreat_cycles": sum(row.get("retreating") == "1" for row in rows),
    }


def build_report(
    motion: dict[str, object],
    clearance: dict[str, object],
    diagnostics: dict[str, object] | None,
    *,
    minimum_clearance_m: float,
    maximum_solve_p95_ms: float,
    minimum_valid_ratio: float,
) -> dict[str, object]:
    checks = {
        "motion_has_duration": float(motion["duration_s"]) > 0.0,
        "motion_has_commands": int(motion["command_samples"]) > 0,
        "motion_has_odometry": int(motion["odometry_samples"]) > 1,
        "clearance_pair_coverage": float(clearance["command_pair_ratio"]) >= 0.90,
        "minimum_clearance": (
            float(clearance["minimum_front_range_m"]) >= minimum_clearance_m
        ),
    }
    if diagnostics is not None:
        checks.update(
            {
                "solve_p95_budget": (
                    float(diagnostics["solve_p95_ms"]) <= maximum_solve_p95_ms
                ),
                "valid_rollout_ratio": (
                    float(diagnostics["valid_rollout_ratio_mean"])
                    >= minimum_valid_ratio
                ),
                "no_all_colliding_cycles": (
                    int(diagnostics["all_colliding_cycles"]) == 0
                ),
            }
        )
    return {
        "schema_version": 1,
        "evidence_mode": (
            "shadow_controller_with_recorded_motion"
            if diagnostics is not None
            else "recorded_motion_only"
        ),
        "quality_pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_clearance_m": minimum_clearance_m,
            "maximum_solve_p95_ms": maximum_solve_p95_ms,
            "minimum_valid_rollout_ratio": minimum_valid_ratio,
            "minimum_command_pair_coverage": 0.90,
        },
        "motion": motion,
        "clearance": clearance,
        "diagnostics": diagnostics,
        "limitations": (
            "Recorded motion is open-loop evidence. Closed-loop success requires "
            "a live robot or simulator whose state reacts to CUDA MPPI commands."
        ),
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def write_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    checks = report["checks"]
    with (output_dir / "checks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "passed"])
        writer.writeheader()
        writer.writerows(
            {"check": name, "passed": passed} for name, passed in checks.items()
        )
    motion = report["motion"]
    clearance = report["clearance"]
    diagnostics = report["diagnostics"]
    lines = [
        "# CUDA MPPI Real-Rosbag Evaluation",
        "",
        f"Overall quality gate: **{'PASS' if report['quality_pass'] else 'FAIL'}**",
        "",
        f"Evidence mode: `{report['evidence_mode']}`",
        "",
        "## Recorded Motion",
        "",
        "| Duration (s) | Path (m) | Mean speed (m/s) | Commands | Odometry |",
        "|---:|---:|---:|---:|---:|",
        f"| {fmt(motion['duration_s'])} | {fmt(motion['path_length_m'])} | "
        f"{fmt(motion['mean_speed_mps'])} | {motion['command_samples']} | "
        f"{motion['odometry_samples']} |",
        "",
        "## Clearance",
        "",
        "| Mean front (m) | Minimum (m) | Below 0.5 m | Pair coverage |",
        "|---:|---:|---:|---:|",
        f"| {fmt(clearance['mean_front_clearance_m'])} | "
        f"{fmt(clearance['minimum_front_range_m'])} | "
        f"{float(clearance['front_below_0_5m_ratio']):.1%} | "
        f"{float(clearance['command_pair_ratio']):.1%} |",
    ]
    if diagnostics is not None:
        lines += [
            "",
            "## CUDA MPPI Diagnostics",
            "",
            "| Solve mean (ms) | Solve p95 (ms) | Valid ratio mean | "
            "All-colliding | Retreat |",
            "|---:|---:|---:|---:|---:|",
            f"| {fmt(diagnostics['solve_mean_ms'])} | "
            f"{fmt(diagnostics['solve_p95_ms'])} | "
            f"{fmt(diagnostics['valid_rollout_ratio_mean'])} | "
            f"{diagnostics['all_colliding_cycles']} | "
            f"{diagnostics['retreat_cycles']} |",
        ]
    lines += ["", "## Quality Gates", ""]
    lines += [
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in checks.items()
    ]
    lines += ["", f"Limitation: {report['limitations']}", ""]
    (output_dir / "evaluation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    parser.add_argument("--diagnostics-csv", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/mppi_rosbag_evaluation")
    )
    parser.add_argument(
        "--command-topic", default="/mobile_base_controller/cmd_vel"
    )
    parser.add_argument(
        "--odometry-topic", default="/mobile_base_controller/odom"
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--minimum-clearance-m", type=float, default=0.10)
    parser.add_argument("--maximum-solve-p95-ms", type=float, default=50.0)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db = args.db.resolve()
    motion = export_rosbag_motion.export_motion(
        db,
        args.output_dir / "motion",
        command_topic=args.command_topic,
        odometry_topic=args.odometry_topic,
    )
    clearance = analyze_rosbag_clearance.analyze(
        db,
        args.output_dir / "clearance_samples.csv",
        scan_topic=args.scan_topic,
        command_topic=args.command_topic,
    )
    diagnostics = diagnostics_metrics(args.diagnostics_csv)
    report = build_report(
        motion,
        clearance,
        diagnostics,
        minimum_clearance_m=args.minimum_clearance_m,
        maximum_solve_p95_ms=args.maximum_solve_p95_ms,
        minimum_valid_ratio=args.minimum_valid_ratio,
    )
    write_report(report, args.output_dir)
    print(f"wrote {args.output_dir / 'evaluation.md'}")
    return 0 if report["quality_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
