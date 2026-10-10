#!/usr/bin/env python3
"""Paired real-scan comparison of exact KISS-ICP spatial query backends.

Runs the complete odometry/mapping/ESDF/MPPI shadow workload sequentially.
The FIFO queue metric is a replay calculation, not measured ROS scheduling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODES = {"legacy": ("brute", "linked"), "normals": ("voxel", "linked"), "spatial": ("voxel", "voxel")}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def summarize(rows: list[dict[str, str]]) -> dict:
    if not rows:
        raise ValueError("no measured frames")
    result = {"frames": len(rows)}
    fields = ["frame_ms", "odometry_ms", "normal_ms", "index_ms", "nn_ms"]
    fields.extend(field for field in (
        "validation_ms", "deskew_wall_ms", "downsample_ms", "map_upload_ms", "icp_ms",
        "normal_equation_ms", "map_update_ms", "map_prune_ms", "map_insert_ms", "map_pack_ms", "map_reorder_ms"
    ) if field in rows[0])
    for field in fields:
        values = [float(row[field]) for row in rows]
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError(f"invalid timing: {field}")
        result[field] = {"mean": statistics.mean(values), "p95": percentile(values, .95),
                         "p99": percentile(values, .99), "max": max(values)}
    first_stamp = int(rows[0]["stamp_ns"])
    finished = 0.0
    responses = []
    misses = 0
    for i, row in enumerate(rows):
        arrival = (int(row["stamp_ns"]) - first_stamp) / 1e6
        finished = max(finished, arrival) + float(row["frame_ms"])
        responses.append(finished - arrival)
        # Use observed next-scan interval; use the preceding interval for the last frame.
        other = rows[i + 1] if i + 1 < len(rows) else rows[max(0, i - 1)]
        interval = abs(int(other["stamp_ns"]) - int(row["stamp_ns"])) / 1e6
        misses += finished > arrival + interval
    result["fifo_response_p95_ms"] = percentile(responses, .95)
    result["fifo_response_p99_ms"] = percentile(responses, .99)
    result["fifo_final_response_ms"] = responses[-1]
    result["fifo_deadline_miss_fraction"] = misses / len(rows)
    result["over_100ms_fraction"] = sum(float(r["frame_ms"]) > 100 for r in rows) / len(rows)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--executable", type=Path, default=ROOT / ("bin/Release/cudanav_real_gpu_stack_sequence.exe" if os.name == "nt" else "bin/cudanav_real_gpu_stack_sequence"))
    p.add_argument("--maximum-frames", type=int, default=0)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--dll-dir", type=Path)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if a.repeats < 1 or a.maximum_frames < 0:
        p.error("repeats must be positive; maximum-frames must be nonnegative")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    executable = a.out_dir / ("runner.exe" if os.name == "nt" else "runner")
    shutil.copy2(a.executable, executable)
    env = os.environ.copy()
    if a.dll_dir:
        env["PATH"] = str(a.dll_dir.resolve()) + os.pathsep + env.get("PATH", "")
    sources = ["src/gpu_kiss_icp.cu", "include/kiss_icp_spatial.cuh",
               "include/kiss_icp_reduction.cuh", "include/kiss_icp_host_map.hpp",
               "include/kiss_icp_downsample.hpp",
               "include/kiss_icp_order.cuh",
               "include/cudarobotics/kiss_icp_gpu.hpp", "tools/cudanav_real_gpu_stack_sequence.cu"]
    result = {"schema": "cudarobotics.kiss_icp_spatial.v1", "gpu": "NVIDIA consumer GPU",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0,
              "source_hash_normalization": "sha256-bytes",
              "sources_sha256": {s: digest(ROOT / s) for s in sources},
              "executable_sha256": digest(executable), "sequence": a.sequence.name,
              "sequence_sha256": digest(a.sequence), "runs": []}
    for repeat in range(a.repeats):
        modes = list(MODES) if repeat % 2 == 0 else list(reversed(MODES))
        for mode in modes:
            stem = a.out_dir / f"{mode}_{repeat}"
            normal, nn = MODES[mode]
            command = [str(executable.resolve()), "--sequence", str(a.sequence.resolve()),
                       "--json", str(stem.with_suffix(".json").resolve()), "--csv", str(stem.with_suffix(".csv").resolve()),
                       "--kiss-normal-backend", normal, "--kiss-nn-backend", nn,
                       "--kiss-reduction-backend", "atomic", "--kiss-map-backend", "unordered",
                       "--kiss-downsample-backend", "unordered",
                       "--kiss-normal-query-order", "input",
                       "--maximum-ate-rmse-m", "3", "--maximum-final-drift-percent", "5",
                       "--minimum-inliers", "100", "--maximum-all-colliding-evaluations", "6", "--check"]
            if a.maximum_frames:
                command.extend(["--maximum-frames", str(a.maximum_frames), "--minimum-control-evaluations", "1"])
            print(f"Running {mode}, repeat {repeat}", flush=True)
            with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
                status = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if not stem.with_suffix(".json").exists():
                raise RuntimeError(f"{mode} did not write metrics; see {stem}.log")
            metrics = json.loads(stem.with_suffix(".json").read_text())
            with stem.with_suffix(".csv").open(newline="") as stream:
                timing = summarize(list(csv.DictReader(stream)))
            run = {"mode": mode, "repeat": repeat, "returncode": status, "argv": command,
                   "timing": timing, "quality_pass": metrics["quality_pass"],
                   "ate_rmse_m": metrics["ate_rmse_m"], "final_drift_percent": metrics["final_drift_percent"],
                   "inliers_min": metrics["inliers_min"], "mppi": metrics["mppi"]}
            run["odometry_quality_pass"] = run["ate_rmse_m"] <= 3 and run["final_drift_percent"] <= 5 and run["inliers_min"] >= 100
            result["runs"].append(run)
            (a.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"  frame p95={timing['frame_ms']['p95']:.3f} ms; ATE={run['ate_rmse_m']:.4f} m; quality={run['quality_pass']}", flush=True)
    result["paired_checks"] = []
    for repeat in range(a.repeats):
        old = next(r for r in result["runs"] if r["mode"] == "legacy" and r["repeat"] == repeat)
        new = next(r for r in result["runs"] if r["mode"] == "spatial" and r["repeat"] == repeat)
        result["paired_checks"].append({"repeat": repeat,
            "frame_p95_under_100ms": new["timing"]["frame_ms"]["p95"] < 100,
            "ate_within_2cm_of_baseline": new["ate_rmse_m"] <= old["ate_rmse_m"] + .02,
            "drift_within_0_02_percentage_points": new["final_drift_percent"] <= old["final_drift_percent"] + .02})
    (a.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = ["# Exact spatial KISS-ICP comparison", "", "Same real scans, map settings and control stride; one executable, sequential runs.", "",
              "| Method / repeat | Frame p95 ms | Frame p99 ms | Normal mean ms | NN mean ms | ATE m | Drift % | >100ms | Quality |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["runs"]:
        t = r["timing"]
        report.append(f"| {r['mode']} / {r['repeat']} | {t['frame_ms']['p95']:.3f} | {t['frame_ms']['p99']:.3f} | {t['normal_ms']['mean']:.3f} | {t['nn_ms']['mean']:.3f} | {r['ate_rmse_m']:.4f} | {r['final_drift_percent']:.4f} | {t['over_100ms_fraction']:.1%} | {r['quality_pass']} |")
    report.extend(["", "FIFO response/deadline metrics are calculated from measured compute durations and sensor timestamps under serial, lossless replay; they are not observed ROS latency. Input loading and CSV writes are outside frame timing. MPPI evaluates every tenth scan. Commands are not applied; this is shadow execution.", ""])
    (a.out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = list(MODES)
        for ax, field, title in [(axes[0], "frame_ms", "Pipeline p95 (mean of 2 replays)" if a.repeats == 2 else "Pipeline p95 (mean across replays)"), (axes[1], "normal_ms", "Map normal estimation mean")]:
            metric = "p95" if field == "frame_ms" else "mean"
            values = [statistics.mean(r["timing"][field][metric] for r in result["runs"] if r["mode"] == m) for m in labels]
            ax.bar(labels, values, color=["#9e5665", "#d2a34c", "#388b82"])
            ax.set(title=title, ylabel="milliseconds")
            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
            ax.set_ylim(0, max(values) * 1.2)
        axes[0].axhline(100, ls="--", color="gray", label="10 Hz budget")
        axes[0].legend()
        fig.suptitle("Exact queries; same real scans and map resolution")
        fig.tight_layout()
        fig.savefig(a.out_dir / "comparison.png", dpi=160)
        plt.close(fig)
    if any(r["returncode"] != 0 or not r["quality_pass"] for r in result["runs"]):
        raise SystemExit("One or more quality gates failed; failures are retained.")
    if any(not all(v for k, v in c.items() if k != "repeat") for c in result["paired_checks"]):
        raise SystemExit("One or more paired performance/odometry checks failed.")


if __name__ == "__main__":
    main()
