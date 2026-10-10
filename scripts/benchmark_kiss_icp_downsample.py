#!/usr/bin/env python3
"""Paired full-route comparison of exact cached and pooled scan centroids."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from benchmark_kiss_icp_pipeline import SOURCES, source_digest
from benchmark_kiss_icp_spatial import digest, percentile, summarize
from benchmark_kiss_icp_normals import COMPARE_FIELDS, read_rows

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--executable", type=Path, default=ROOT / (
        "bin/Release/cudanav_real_gpu_stack_sequence.exe" if os.name == "nt"
        else "bin/cudanav_real_gpu_stack_sequence"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-executable", type=Path, default=ROOT / (
        "bin/Release/kiss_icp_downsample_benchmark.exe" if os.name == "nt"
        else "bin/kiss_icp_downsample_benchmark"))
    p.add_argument("--maximum-frames", type=int, default=0)
    p.add_argument("--dll-dir", type=Path)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if a.repeats < 1 or a.maximum_frames < 0:
        p.error("repeats must be positive; maximum-frames must be nonnegative")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    exe = a.out_dir / ("runner.exe" if os.name == "nt" else "runner")
    shutil.copy2(a.executable, exe)
    cpu_exe = a.out_dir / ("cpu_runner.exe" if os.name == "nt" else "cpu_runner")
    shutil.copy2(a.cpu_executable, cpu_exe)
    sources = list(dict.fromkeys(SOURCES + ["scripts/benchmark_kiss_icp_downsample.py",
        "scripts/benchmark_kiss_icp_normals.py", "tools/kiss_icp_downsample_benchmark.cpp"]))
    for relative in sources:
        target = a.out_dir / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    env = os.environ.copy()
    if a.dll_dir:
        env["PATH"] = str(a.dll_dir.resolve()) + os.pathsep + env.get("PATH", "")
    result = {"schema": "cudarobotics.kiss_icp_downsample.v1", "gpu": "NVIDIA consumer GPU",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0,
              "source_hash_normalization": "sha256-text-lf",
              "sources_sha256": {s: source_digest(ROOT / s) for s in sources},
              "executable_sha256": digest(exe), "sequence": a.sequence.name,
              "sequence_sha256": digest(a.sequence), "sequence_bytes": a.sequence.stat().st_size,
              "maximum_frames": a.maximum_frames, "process_priority": "normal",
              "runs": [], "paired_checks": []}

    def save() -> None:
        (a.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    cpu_csv = a.out_dir / "cpu.csv"
    cpu_command = [str(cpu_exe.resolve()), "--sequence", str(a.sequence.resolve()),
                   "--csv", str(cpu_csv.resolve()), "--samples", "24", "--repeats", "6"]
    print("Profiling raw scan centroids before GPU timing", flush=True)
    with (a.out_dir / "cpu.log").open("w", encoding="utf-8") as log:
        cpu_status = subprocess.run(cpu_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    result["cpu_probe"] = {"returncode": cpu_status, "argv": cpu_command,
                           "executable_sha256": digest(cpu_exe)}
    save()
    if cpu_status or not cpu_csv.exists():
        raise SystemExit("CPU probe failed; all artifacts retained.")
    cpu_rows = read_rows(cpu_csv)
    probe = result["cpu_probe"]
    probe["all_outputs_byte_exact"] = all(int(r["byte_equal"]) for r in cpu_rows)
    probe["new_slabs_during_timing"] = sum(int(r["new_arena_slabs"]) for r in cpu_rows if r["mode"] == "pooled")
    probe["raw_scans"] = sum(r["mode"] == "profile" for r in cpu_rows)
    probe["timing"] = {}
    for mode in ("cached", "pooled"):
        values = [float(r["total_ms"]) for r in cpu_rows if r["mode"] == mode]
        probe["timing"][mode] = {"mean": statistics.mean(values), "p95": percentile(values, .95), "max": max(values)}
    profiles = [r for r in cpu_rows if r["mode"] == "profile"]
    probe["profile_means"] = {field: statistics.mean(float(r[field]) for r in profiles) for field in (
        "points", "voxels", "keys_ms", "reserve_ms", "aggregate_ms", "emit_ms", "destroy_ms", "allocation_calls", "allocation_bytes")}
    save()

    schedule = [("validate", 0)]
    for repeat in range(a.repeats):
        schedule.extend((mode, repeat) for mode in (
            ("cached", "pooled") if repeat % 2 == 0 else ("pooled", "cached")))
    for mode, repeat in schedule:
        stem = a.out_dir / f"{mode}_{repeat}"
        command = [str(exe.resolve()), "--sequence", str(a.sequence.resolve()),
                   "--json", str(stem.with_suffix(".json").resolve()),
                   "--csv", str(stem.with_suffix(".csv").resolve()),
                   "--kiss-normal-backend", "voxel", "--kiss-nn-backend", "voxel",
                   "--kiss-reduction-backend", "block", "--kiss-map-backend", "dense",
                   "--kiss-downsample-backend", mode, "--kiss-normal-query-order", "cell",
                   "--kiss-normal-update", "full", "--maximum-ate-rmse-m", "3",
                   "--maximum-final-drift-percent", "5", "--minimum-inliers", "100",
                   "--maximum-all-colliding-evaluations", "6", "--check"]
        if a.maximum_frames:
            command.extend(["--maximum-frames", str(a.maximum_frames), "--minimum-control-evaluations", "1"])
        print(f"Running {mode}, repeat {repeat}", flush=True)
        with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        run = {"mode": mode, "repeat": repeat, "returncode": status, "argv": command}
        result["runs"].append(run)
        save()
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".csv").exists():
            raise SystemExit(f"Runner did not write metrics; preserved status and {stem}.log")
        metrics = json.loads(stem.with_suffix(".json").read_text())
        rows = read_rows(stem.with_suffix(".csv"))
        run.update(timing=summarize(rows), quality_pass=metrics["quality_pass"],
                   ate_rmse_m=metrics["ate_rmse_m"], final_drift_percent=metrics["final_drift_percent"],
                   inliers_min=metrics["inliers_min"], mppi=metrics["mppi"],
                   odometry_config=metrics["odometry_config"], duration_s=metrics["duration_s"],
                   downsample_memory=metrics["downsample_memory"],
                   frames_with_new_slabs=sum(int(row["downsample_new_slabs"]) > 0 for row in rows))
        if mode == "validate":
            run["all_centroids_and_order_byte_exact"] = status == 0
        save()
        print(f"  frame p95={run['timing']['frame_ms']['p95']:.3f} ms; downsample mean={run['timing']['downsample_ms']['mean']:.3f} ms; quality={run['quality_pass']}", flush=True)
        if mode == "validate" and (status or not run["quality_pass"]):
            raise SystemExit("Full-route centroid validation failed; all artifacts retained.")
    for repeat in range(a.repeats):
        old, new = (next(r for r in result["runs"] if r["mode"] == m and r["repeat"] == repeat)
                    for m in ("cached", "pooled"))
        old_rows, new_rows = (read_rows(a.out_dir / f"{m}_{repeat}.csv") for m in ("cached", "pooled"))
        equal = len(old_rows) == len(new_rows) and all(
            x[field] == y[field] for x, y in zip(old_rows, new_rows) for field in COMPARE_FIELDS)
        result["paired_checks"].append({"repeat": repeat,
            "downsample_mean_under_5ms": new["timing"]["downsample_ms"]["mean"] < 5,
            "downsample_p95_faster": new["timing"]["downsample_ms"]["p95"] < old["timing"]["downsample_ms"]["p95"],
            "frame_p95_faster": new["timing"]["frame_ms"]["p95"] < old["timing"]["frame_ms"]["p95"],
            "trajectory_and_map_csv_equal": equal,
            "ate_equal": old["ate_rmse_m"] == new["ate_rmse_m"],
            "drift_equal": old["final_drift_percent"] == new["final_drift_percent"]})
    result["compared_csv_fields"] = COMPARE_FIELDS
    save()
    report = ["# Pooled scan centroids", "", "One frozen executable; sequential replays. Validate timing includes both methods and byte comparison.", "",
              "| Mode / repeat | Frame mean ms | p95 ms | p99 ms | Centroids mean ms | p95 ms | ATE m | Quality |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["runs"]:
        t = r["timing"]
        report.append(f"| {r['mode']} / {r['repeat']} | {t['frame_ms']['mean']:.3f} | {t['frame_ms']['p95']:.3f} | {t['frame_ms']['p99']:.3f} | {t['downsample_ms']['mean']:.3f} | {t['downsample_ms']['p95']:.3f} | {r['ate_rmse_m']:.9f} | {r['quality_pass']} |")
    report.extend(["", "FIFO metrics are calculated serial replay, not observed ROS scheduling. Commands are not applied. All failures are retained.", ""])
    (a.out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for i, mode in enumerate(("cached", "pooled")):
            runs = [r for r in result["runs"] if r["mode"] == mode]
            for ax, field in zip(axes, ("frame_ms", "downsample_ms")):
                values = [r["timing"][field]["p95"] for r in runs]
                ax.bar(i, sum(values)/len(values), color=("#a36670", "#338c80")[i])
                ax.plot([i]*len(values), values, "ko", markersize=4)
        for ax, title in zip(axes, ("Whole-frame p95", "Scan centroid p95")):
            ax.set(title=title, ylabel="milliseconds", xticks=range(2), xticklabels=("cached", "pooled"))
        fig.suptitle("Same real scans, point density and centroid order")
        fig.tight_layout()
        fig.savefig(a.out_dir / "comparison.png", dpi=160)
        plt.close(fig)
    if any(r["returncode"] or not r["quality_pass"] for r in result["runs"]):
        raise SystemExit("One or more native quality gates failed; failures retained.")
    if any(not all(v for k, v in c.items() if k != "repeat") for c in result["paired_checks"]):
        raise SystemExit("One or more paired performance/exactness checks failed; failures retained.")


if __name__ == "__main__":
    main()
