#!/usr/bin/env python3
"""Exact incremental map normals: full-route validation and paired timing.

One frozen executable; unchanged scans, density, ICP and mapping/ESDF/MPPI.
Validation also executes full normals, so its frame time is not a speed result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from benchmark_kiss_icp_pipeline import SOURCES, source_digest
from benchmark_kiss_icp_spatial import digest, summarize

ROOT = Path(__file__).resolve().parents[1]
COMPARE_FIELDS = ("frame", "stamp_ns", "estimated_x", "estimated_y", "xy_error_m",
                  "inliers", "map_points", "observed_voxels", "integrated_rays",
                  "unknown_cells")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot(result: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    modes = ("full", "incremental")
    for i, mode in enumerate(modes):
        runs = [r for r in result["runs"] if r["mode"] == mode]
        values = [r["timing"]["frame_ms"]["p95"] for r in runs]
        axes[0].bar(i, sum(values) / len(values), color=("#a36670", "#338c80")[i])
        axes[0].plot([i] * len(values), values, "ko", markersize=4)
        normal = sum(r["timing"]["normal_ms"]["mean"] for r in runs) / len(runs)
        prepare = sum(r["timing"]["normal_cache_prepare_ms"]["mean"] for r in runs) / len(runs)
        axes[1].bar(i, normal, color="#528eb0", label="GPU normals" if i == 0 else None)
        axes[1].bar(i, prepare, bottom=normal, color="#d0a452", label="Cache preparation" if i == 0 else None)
    axes[0].axhline(40, color="gray", ls="--", label="40 ms target")
    axes[0].set(title="Frame p95 (dots: individual replays)", ylabel="milliseconds")
    axes[1].set(title="Normal stage mean", ylabel="milliseconds")
    for ax in axes:
        ax.set_xticks(range(2), modes)
        ax.legend()
    fig.suptitle(f"{result.get('downsample_backend', 'cached').capitalize()} scan centroids; same real scans and exact normal support")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


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
    p.add_argument("--downsample-backend", choices=("pooled", "cached"), default="pooled",
                   help="Pin the same centroid backend in both modes; cached reproduces the earlier comparison")
    p.add_argument("--verify-default", action="store_true",
                   help="Also replay without centroid/normal-update flags and verify pooled/incremental defaults")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--above-normal-priority", action="store_true",
                   help="Windows only: use the same AboveNormal process priority for all runners")
    a = p.parse_args()
    if a.repeats < 1 or a.maximum_frames < 0:
        p.error("repeats must be positive; maximum-frames must be nonnegative")
    if a.above_normal_priority and os.name != "nt":
        p.error("--above-normal-priority is only supported on Windows")
    if a.verify_default and a.downsample_backend != "pooled":
        p.error("--verify-default requires --downsample-backend pooled")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    exe = a.out_dir / ("runner.exe" if os.name == "nt" else "runner")
    shutil.copy2(a.executable, exe)
    sources = list(dict.fromkeys(SOURCES + ["scripts/benchmark_kiss_icp_normals.py",
        "tests/kiss_icp_normal_cache_exact.cu", "tests/kiss_icp_gpu_streaming_smoke.cu", "CMakeLists.txt"]))
    for relative in sources:
        target = a.out_dir / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    env = os.environ.copy()
    if a.dll_dir:
        env["PATH"] = str(a.dll_dir.resolve()) + os.pathsep + env.get("PATH", "")
    result = {"schema": "cudarobotics.kiss_icp_normals.v1", "gpu": "NVIDIA consumer GPU",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0,
              "source_hash_normalization": "sha256-text-lf",
              "sources_sha256": {s: source_digest(ROOT / s) for s in sources},
              "executable_sha256": digest(exe), "sequence": a.sequence.name,
              "sequence_sha256": digest(a.sequence), "sequence_bytes": a.sequence.stat().st_size,
              "maximum_frames": a.maximum_frames,
              "downsample_backend": a.downsample_backend,
              "process_priority": "above_normal" if a.above_normal_priority else "normal",
              "runs": [], "paired_checks": []}

    def save() -> None:
        (a.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    schedule = [("validate", 0)]
    for repeat in range(a.repeats):
        schedule.extend((mode, repeat) for mode in (
            ("full", "incremental") if repeat % 2 == 0 else ("incremental", "full")))
    if a.verify_default:
        schedule.append(("default", 0))
    for mode, repeat in schedule:
        stem = a.out_dir / f"{mode}_{repeat}"
        command = [str(exe.resolve()), "--sequence", str(a.sequence.resolve()),
                   "--json", str(stem.with_suffix(".json").resolve()),
                   "--csv", str(stem.with_suffix(".csv").resolve()),
                   "--kiss-normal-backend", "voxel", "--kiss-nn-backend", "voxel",
                   "--kiss-reduction-backend", "block", "--kiss-map-backend", "dense",
                   "--kiss-normal-query-order", "cell",
                   "--maximum-ate-rmse-m", "3", "--maximum-final-drift-percent", "5",
                   "--minimum-inliers", "100", "--maximum-all-colliding-evaluations", "6", "--check"]
        if mode != "default":
            command.extend(["--kiss-downsample-backend", a.downsample_backend, "--kiss-normal-update", mode])
        if a.maximum_frames:
            command.extend(["--maximum-frames", str(a.maximum_frames), "--minimum-control-evaluations", "1"])
        print(f"Running {mode}, repeat {repeat}", flush=True)
        with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS
                                    if a.above_normal_priority else 0).returncode
        run = {"mode": mode, "repeat": repeat, "returncode": status, "argv": command}
        result["runs"].append(run)
        save()  # Preserve errors even when the runner does not produce metrics.
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".csv").exists():
            raise SystemExit(f"Runner did not write metrics; preserved return code and {stem}.log")
        metrics = json.loads(stem.with_suffix(".json").read_text())
        rows = read_rows(stem.with_suffix(".csv"))
        run.update(timing=summarize(rows), quality_pass=metrics["quality_pass"],
                   ate_rmse_m=metrics["ate_rmse_m"], final_drift_percent=metrics["final_drift_percent"],
                   inliers_min=metrics["inliers_min"], mppi=metrics["mppi"],
                   odometry_config=metrics["odometry_config"], duration_s=metrics["duration_s"],
                   normal_cache=metrics["normal_cache"], downsample_memory=metrics["downsample_memory"],
                   actual_normal_update=metrics["normal_update"], actual_downsample_backend=metrics["downsample_backend"])
        cache = run["normal_cache"]
        total = cache["reused_points"] + cache["recomputed_points"]
        cache["reuse_fraction"] = cache["reused_points"] / total if total else 0.
        if mode == "validate":
            run["all_normals_bit_exact"] = status == 0
        save()
        print(f"  p95={run['timing']['frame_ms']['p95']:.3f} ms; quality={run['quality_pass']}; reuse={cache['reuse_fraction']:.2%}", flush=True)
        if mode == "validate" and (status or not run["quality_pass"]):
            raise SystemExit("Full-route normal validation failed; all artifacts retained.")

    for repeat in range(a.repeats):
        old, new = (next(r for r in result["runs"] if r["mode"] == m and r["repeat"] == repeat)
                    for m in ("full", "incremental"))
        old_rows, new_rows = (read_rows(a.out_dir / f"{m}_{repeat}.csv") for m in ("full", "incremental"))
        equal = len(old_rows) == len(new_rows) and all(
            x[field] == y[field] for x, y in zip(old_rows, new_rows) for field in COMPARE_FIELDS)
        result["paired_checks"].append({"repeat": repeat,
            "frame_p95_under_40ms": new["timing"]["frame_ms"]["p95"] < 40,
            "faster_than_full": new["timing"]["frame_ms"]["p95"] < old["timing"]["frame_ms"]["p95"],
            "trajectory_and_map_csv_equal": equal,
            "ate_equal": old["ate_rmse_m"] == new["ate_rmse_m"],
            "drift_equal": old["final_drift_percent"] == new["final_drift_percent"]})
        # Occupancy update atomics can differ even between two Full executions,
        # including frame zero before normal computation. Retain the difference
        # count, while gating exact odometry outputs and native quality separately.
        result.setdefault("occupancy_csv_different_frames", []).append({"repeat": repeat,
            "full_vs_incremental": sum(x["occupied_cells"] != y["occupied_cells"]
                                       for x, y in zip(old_rows, new_rows))})
    full_zero = read_rows(a.out_dir / "full_0.csv")
    result["occupancy_full_repeat_different_frames"] = [
        {"repeat": repeat, "vs_repeat_zero": sum(x["occupied_cells"] != y["occupied_cells"]
            for x, y in zip(full_zero, read_rows(a.out_dir / f"full_{repeat}.csv")))}
        for repeat in range(1, a.repeats)]
    result["compared_csv_fields"] = COMPARE_FIELDS
    if a.verify_default:
        default = next(r for r in result["runs"] if r["mode"] == "default")
        incremental = next(r for r in result["runs"] if r["mode"] == "incremental" and r["repeat"] == 0)
        default_rows, incremental_rows = (read_rows(a.out_dir / f"{m}_0.csv") for m in ("default", "incremental"))
        result["default_checks"] = {
            "pooled_centroids": default["actual_downsample_backend"] == "pooled",
            "incremental_normals": default["actual_normal_update"] == "incremental",
            "normal_reuse_executed": default["normal_cache"]["reused_points"] > 0,
            "frame_p95_under_40ms": default["timing"]["frame_ms"]["p95"] < 40,
            "trajectory_and_map_csv_equal": len(default_rows) == len(incremental_rows) and all(
                x[field] == y[field] for x, y in zip(default_rows, incremental_rows) for field in COMPARE_FIELDS),
            "ate_equal": default["ate_rmse_m"] == incremental["ate_rmse_m"],
            "drift_equal": default["final_drift_percent"] == incremental["final_drift_percent"]}
    save()
    report = ["# Exact incremental map normals", "", f"One executable, sequential full-route replays; {a.downsample_backend} centroids in both modes. Validation timing includes an extra full-normal kernel.", "",
              "| Mode / repeat | Frame mean ms | p95 ms | p99 ms | GPU normals mean ms | Cache prepare mean ms | Reused | ATE m | Quality |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["runs"]:
        t = r["timing"]
        report.append(f"| {r['mode']} / {r['repeat']} | {t['frame_ms']['mean']:.3f} | {t['frame_ms']['p95']:.3f} | {t['frame_ms']['p99']:.3f} | {t['normal_ms']['mean']:.3f} | {t['normal_cache_prepare_ms']['mean']:.3f} | {r['normal_cache']['reuse_fraction']:.2%} | {r['ate_rmse_m']:.9f} | {r['quality_pass']} |")
    report.extend(["", "FIFO metrics are calculated serial replay, not observed ROS scheduling. MPPI runs every tenth scan. Commands are not applied. All failures are retained.", ""])
    (a.out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    if a.plot:
        plot(result, a.out_dir / "comparison.png")
    if any(r["returncode"] or not r["quality_pass"] for r in result["runs"]):
        raise SystemExit("One or more quality gates failed; failures retained.")
    if any(not all(v for k, v in c.items() if k != "repeat") for c in result["paired_checks"]):
        raise SystemExit("One or more paired performance/exactness checks failed; failures retained.")
    if not all(result.get("default_checks", {}).values()):
        raise SystemExit("One or more default-policy checks failed; failures retained.")


if __name__ == "__main__":
    main()
