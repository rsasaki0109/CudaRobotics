#!/usr/bin/env python3
"""Paired KISS-ICP reduction/dense host-map comparison on real LiDAR scans.

All methods use exact spatial queries and the same mapping/ESDF/MPPI workload.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from benchmark_kiss_icp_spatial import digest, summarize

ROOT = Path(__file__).resolve().parents[1]
MODES = {"baseline": ("atomic", "unordered", "unordered", "input"),
         "reduction": ("block", "unordered", "unordered", "input"),
         "map": ("block", "dense", "unordered", "input"),
         "optimized": ("block", "dense", "cached", "cell")}
SOURCES = ["src/gpu_kiss_icp.cu", "include/cudarobotics/kiss_icp_gpu.hpp",
           "include/kiss_icp_spatial.cuh", "include/kiss_icp_reduction.cuh",
           "include/kiss_icp_host_map.hpp", "include/kiss_icp_downsample.hpp",
           "include/kiss_icp_order.cuh",
           "tools/cudanav_real_gpu_stack_sequence.cu",
           "scripts/benchmark_kiss_icp_pipeline.py", "scripts/benchmark_kiss_icp_spatial.py"]


def source_digest(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--executable", type=Path, default=ROOT / (
        "bin/Release/cudanav_real_gpu_stack_sequence.exe" if os.name == "nt"
        else "bin/cudanav_real_gpu_stack_sequence"))
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--maximum-frames", type=int, default=0)
    p.add_argument("--dll-dir", type=Path)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if a.repeats < 1 or a.maximum_frames < 0:
        p.error("repeats must be positive; maximum-frames must be nonnegative")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    exe = a.out_dir / ("runner.exe" if os.name == "nt" else "runner")
    shutil.copy2(a.executable, exe)
    env = os.environ.copy()
    if a.dll_dir:
        env["PATH"] = str(a.dll_dir.resolve()) + os.pathsep + env.get("PATH", "")
    result = {"schema": "cudarobotics.kiss_icp_pipeline.v1", "gpu": "NVIDIA consumer GPU",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0,
              "source_hash_normalization": "sha256-text-lf",
              "sources_sha256": {s: source_digest(ROOT / s) for s in SOURCES},
              "executable_sha256": digest(exe), "sequence": a.sequence.name,
              "sequence_sha256": digest(a.sequence), "runs": []}
    for repeat in range(a.repeats):
        for mode in (list(MODES) if repeat % 2 == 0 else list(reversed(MODES))):
            reduction, map_backend, downsample, normal_order = MODES[mode]
            stem = a.out_dir / f"{mode}_{repeat}"
            command = [str(exe.resolve()), "--sequence", str(a.sequence.resolve()),
                       "--json", str(stem.with_suffix(".json").resolve()),
                       "--csv", str(stem.with_suffix(".csv").resolve()),
                       "--kiss-reduction-backend", reduction, "--kiss-map-backend", map_backend,
                       "--kiss-downsample-backend", downsample,
                       "--kiss-normal-query-order", normal_order,
                       "--maximum-ate-rmse-m", "3", "--maximum-final-drift-percent", "5",
                       "--minimum-inliers", "100", "--maximum-all-colliding-evaluations", "6", "--check"]
            if a.maximum_frames:
                command.extend(["--maximum-frames", str(a.maximum_frames), "--minimum-control-evaluations", "1"])
            print(f"Running {mode}, repeat {repeat}", flush=True)
            with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
                status = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT).returncode
            metrics = json.loads(stem.with_suffix(".json").read_text())
            with stem.with_suffix(".csv").open(newline="") as stream:
                timing = summarize(list(csv.DictReader(stream)))
            result["runs"].append({"mode": mode, "repeat": repeat, "returncode": status,
                                   "argv": command, "timing": timing,
                                   "quality_pass": metrics["quality_pass"],
                                   "ate_rmse_m": metrics["ate_rmse_m"],
                                   "final_drift_percent": metrics["final_drift_percent"],
                                   "inliers_min": metrics["inliers_min"], "mppi": metrics["mppi"]})
            (a.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"  p95={timing['frame_ms']['p95']:.3f} ms; ATE={metrics['ate_rmse_m']:.4f} m; quality={metrics['quality_pass']}", flush=True)
    result["paired_checks"] = []
    for repeat in range(a.repeats):
        old = next(r for r in result["runs"] if r["mode"] == "baseline" and r["repeat"] == repeat)
        new = next(r for r in result["runs"] if r["mode"] == "optimized" and r["repeat"] == repeat)
        result["paired_checks"].append({"repeat": repeat,
            "frame_p95_under_75ms": new["timing"]["frame_ms"]["p95"] < 75,
            "faster_than_baseline": new["timing"]["frame_ms"]["p95"] < old["timing"]["frame_ms"]["p95"],
            "ate_within_2cm_of_baseline": new["ate_rmse_m"] <= old["ate_rmse_m"] + .02,
            "drift_within_0_02_percentage_points": new["final_drift_percent"] <= old["final_drift_percent"] + .02})
    (a.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = ["# KISS-ICP pipeline comparison", "", "Same full route and point density; native shadow execution.", "",
              "| Method / repeat | Frame mean ms | p95 ms | p99 ms | Normal equation mean ms | Map update mean ms | ATE m | Drift % | Quality |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for run in result["runs"]:
        t = run["timing"]
        report.append(f"| {run['mode']} / {run['repeat']} | {t['frame_ms']['mean']:.3f} | {t['frame_ms']['p95']:.3f} | {t['frame_ms']['p99']:.3f} | {t['normal_equation_ms']['mean']:.3f} | {t['map_update_ms']['mean']:.3f} | {run['ate_rmse_m']:.4f} | {run['final_drift_percent']:.4f} | {run['quality_pass']} |")
    report += ["", "FIFO metrics are calculated serial replay, not observed ROS scheduling. MPPI runs every tenth scan. Commands are not applied. All failures are retained.", ""]
    (a.out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = list(MODES)
        colors = ["#9e5665", "#d2a34c", "#5196ae", "#388b82"]
        for i, mode in enumerate(labels):
            runs = [r for r in result["runs"] if r["mode"] == mode]
            axes[0].bar(i, sum(r["timing"]["frame_ms"]["p95"] for r in runs)/len(runs), color=colors[i])
            for r in runs:
                axes[0].plot(i, r["timing"]["frame_ms"]["p95"], "o", color="black", markersize=3)
        axes[0].set(title="Frame p95 (mean across replays)", ylabel="milliseconds", xticks=range(len(labels)), xticklabels=labels)
        axes[0].axhline(75, ls="--", color="gray", label="75 ms target")
        axes[0].legend()
        for field, color, offset, label in [
            ("normal_equation_ms", "#80609d", -.24, "Normal equations"),
            ("map_update_ms", "#409ba4", 0., "Map update"),
            ("downsample_ms", "#d2a34c", .24, "Scan centroids")]:
            values = [sum(r["timing"][field]["mean"] for r in result["runs"] if r["mode"] == m)/a.repeats for m in labels]
            axes[1].bar([i+offset for i in range(len(labels))], values, width=.23, color=color, label=label)
        axes[1].set(title="Measured stage means", ylabel="milliseconds", xticks=range(len(labels)), xticklabels=labels)
        axes[1].legend()
        fig.suptitle("Same real scans, voxel resolutions and ICP limits")
        fig.tight_layout()
        fig.savefig(a.out_dir / "comparison.png", dpi=160)
        plt.close(fig)
    if any(r["returncode"] or not r["quality_pass"] for r in result["runs"]):
        raise SystemExit("One or more quality gates failed; failures are retained.")
    if any(not all(v for k, v in c.items() if k != "repeat") for c in result["paired_checks"]):
        raise SystemExit("One or more paired performance/accuracy checks failed.")


if __name__ == "__main__":
    main()
