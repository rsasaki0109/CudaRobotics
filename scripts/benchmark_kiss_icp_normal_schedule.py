#!/usr/bin/env python3
"""Exact split normal scheduling: bit validation and alternating full-route timing pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from benchmark_kiss_icp_normals import COMPARE_FIELDS, read_rows
from benchmark_kiss_icp_pipeline import SOURCES, source_digest
from benchmark_kiss_icp_spatial import digest, summarize

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (*COMPARE_FIELDS, "normal_reused_points", "normal_recomputed_points")


def equal_rows(left: Path, right: Path) -> bool:
    a, b = read_rows(left), read_rows(right)
    return len(a) == len(b) and all(x[k] == y[k] for x, y in zip(a, b) for k in FIELDS)


def plot(result: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    modes = ("fused", "split")
    for i, mode in enumerate(modes):
        runs = [r for r in result["runs"] if r["mode"] == mode]
        values = [r["timing"]["frame_ms"]["p95"] for r in runs]
        axes[0].bar(i, sum(values) / len(values), color=("#a36670", "#338c80")[i])
        axes[0].plot([i + (j - (len(values) - 1) / 2) * .08 for j in range(len(values))],
                     values, "ko", markersize=4)
        bottom = 0.
        for field, color, label in (("normal_ms", "#338c80", "Normal GPU"),
                                     ("normal_cache_prepare_ms", "#d0a452", "Cache preparation")):
            value = sum(r["timing"][field]["mean"] for r in runs) / len(runs)
            axes[1].bar(i, value, bottom=bottom, color=color, label=label if i == 0 else None)
            bottom += value
    axes[0].set(title="Whole-frame p95", ylabel="milliseconds")
    axes[1].set(title="Normal stage means", ylabel="milliseconds")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(range(2), modes)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.suptitle("Same scans and exact normal bits")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=ROOT / (
        "bin/Release/cudanav_real_gpu_stack_sequence.exe" if os.name == "nt"
        else "bin/cudanav_real_gpu_stack_sequence"))
    parser.add_argument("--dll-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maximum-frames", type=int, default=0)
    parser.add_argument("--verify-default", action="store_true")
    parser.add_argument("--expected-default", choices=("fused", "split"), default="fused")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.maximum_frames < 0:
        parser.error("repeats must be positive; maximum-frames must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    exe = args.out_dir / ("runner.exe" if os.name == "nt" else "runner")
    shutil.copy2(args.executable, exe)
    sources = list(dict.fromkeys(SOURCES + [
        "scripts/benchmark_kiss_icp_normal_schedule.py", "scripts/benchmark_kiss_icp_normals.py",
        "tests/kiss_icp_host_map_exact.cpp", "tests/kiss_icp_gpu_streaming_smoke.cu",
        "tests/kiss_icp_normal_cache_exact.cu", "CMakeLists.txt",
        "src/mppi_gpu.cu", "src/voxel_mapping_gpu.cu", "src/esdf_2d_gpu.cu",
        "include/cuda_mppi_controller/mppi_gpu.hpp", "include/cudarobotics/voxel_mapping_gpu.hpp",
        "include/cudarobotics/esdf_2d_gpu.hpp", "include/cuda_check.cuh"]))
    for relative in sources:
        target = args.out_dir / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    env = os.environ.copy()
    if args.dll_dir:
        env["PATH"] = str(args.dll_dir.resolve()) + os.pathsep + env.get("PATH", "")
    result = {"schema": "cudarobotics.kiss_icp_normal_schedule.v1", "gpu": "NVIDIA consumer GPU",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_worktree_dirty": subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0,
              "source_hash_normalization": "sha256-text-lf",
              "sources_sha256": {s: source_digest(ROOT / s) for s in sources},
              "executable_sha256": digest(exe), "sequence": args.sequence.name,
              "sequence_sha256": digest(args.sequence), "sequence_bytes": args.sequence.stat().st_size,
              "maximum_frames": args.maximum_frames, "process_priority": "normal",
              "expected_default_schedule": args.expected_default,
              "compared_csv_fields": FIELDS, "runs": [], "paired_checks": []}

    def save() -> None:
        (args.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    schedule = [("validate", 0)]
    for repeat in range(args.repeats):
        schedule.extend((mode, repeat) for mode in (
            ("fused", "split") if repeat % 2 == 0 else ("split", "fused")))
    if args.verify_default:
        schedule.append(("default", 0))
    for mode, repeat in schedule:
        stem = args.out_dir / f"{mode}_{repeat}"
        command = [str(exe.resolve()), "--sequence", str(args.sequence.resolve()),
                   "--json", str(stem.with_suffix(".json").resolve()),
                   "--csv", str(stem.with_suffix(".csv").resolve()),
                   "--kiss-downsample-backend", "pooled", "--kiss-map-backend",
                   "validate" if mode == "validate" else "pooled", "--kiss-nn-backend", "voxel",
                   "--kiss-normal-backend", "voxel", "--kiss-reduction-backend", "block",
                   "--kiss-normal-query-order", "cell", "--kiss-normal-update",
                   "validate" if mode == "validate" else "incremental",
                   "--maximum-ate-rmse-m", "3", "--maximum-final-drift-percent", "5",
                   "--minimum-inliers", "100", "--maximum-all-colliding-evaluations", "6", "--check"]
        if mode != "default":
            command.extend(["--kiss-normal-schedule", "split" if mode == "validate" else mode])
        if args.maximum_frames:
            command.extend(["--maximum-frames", str(args.maximum_frames), "--minimum-control-evaluations", "1"])
        print(f"Running {mode}, repeat {repeat}", flush=True)
        with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
        run = {"mode": mode, "repeat": repeat, "returncode": status, "argv": command}
        result["runs"].append(run)
        save()
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".csv").exists():
            raise SystemExit(f"Runner did not write metrics; return code and {stem}.log retained")
        metrics = json.loads(stem.with_suffix(".json").read_text())
        run.update(timing=summarize(read_rows(stem.with_suffix(".csv"))),
                   quality_pass=metrics["quality_pass"], ate_rmse_m=metrics["ate_rmse_m"],
                   final_drift_percent=metrics["final_drift_percent"], inliers_min=metrics["inliers_min"],
                   mppi=metrics["mppi"], odometry_config=metrics["odometry_config"],
                   duration_s=metrics["duration_s"], normal_cache=metrics["normal_cache"],
                   host_map_memory=metrics["host_map_memory"], actual_normal_schedule=metrics["normal_schedule"])
        run["artifacts_sha256"] = {suffix: digest(stem.with_suffix(suffix)) for suffix in (".csv", ".json", ".log")}
        if mode == "validate":
            run["all_map_points_order_and_normals_bit_exact"] = status == 0
        save()
        print(f"  frame p95={run['timing']['frame_ms']['p95']:.3f} ms; "
              f"normal mean={run['timing']['normal_ms']['mean']:.3f} ms; quality={run['quality_pass']}", flush=True)
        if mode == "validate" and (status or not run["quality_pass"]):
            raise SystemExit("Full-route point/order/normal validation failed; all artifacts retained")
    for repeat in range(args.repeats):
        old, new = (next(r for r in result["runs"] if r["mode"] == m and r["repeat"] == repeat)
                    for m in ("fused", "split"))
        result["paired_checks"].append({"repeat": repeat,
            "frame_p95_under_40ms": new["timing"]["frame_ms"]["p95"] < 40,
            "faster_frame_p95": new["timing"]["frame_ms"]["p95"] < old["timing"]["frame_ms"]["p95"],
            "faster_normal_mean": new["timing"]["normal_ms"]["mean"] < old["timing"]["normal_ms"]["mean"],
            "trajectory_map_and_reuse_csv_equal": equal_rows(args.out_dir / f"fused_{repeat}.csv", args.out_dir / f"split_{repeat}.csv"),
            "ate_equal": old["ate_rmse_m"] == new["ate_rmse_m"],
            "drift_equal": old["final_drift_percent"] == new["final_drift_percent"]})
        result.setdefault("p95_saved_ms", []).append(old["timing"]["frame_ms"]["p95"] - new["timing"]["frame_ms"]["p95"])
        a, b = (read_rows(args.out_dir / f"{m}_{repeat}.csv") for m in ("fused", "split"))
        result.setdefault("occupied_cell_different_frames", []).append(sum(x["occupied_cells"] != y["occupied_cells"] for x, y in zip(a, b)))
    result["all_pairs_reach_1ms_target"] = all(v >= 1 for v in result["p95_saved_ms"])
    if args.verify_default:
        run = next(r for r in result["runs"] if r["mode"] == "default")
        result["default_checks"] = {
            "expected_normal_schedule": run["actual_normal_schedule"] == args.expected_default,
            "expected_queue_storage": (run["normal_cache"]["recompute_queue_bytes"] > 0) == (args.expected_default == "split"),
            "frame_p95_under_40ms": run["timing"]["frame_ms"]["p95"] < 40,
            "trajectory_map_and_reuse_csv_equal": equal_rows(args.out_dir / "default_0.csv", args.out_dir / f"{args.expected_default}_0.csv")}
    save()
    report = ["# Exact split normal scheduling comparison", "", "Native shadow execution; pooled centroids/maps and exact incremental normals in both timing modes.", "",
              "| Mode / repeat | Frame mean ms | p95 ms | p99 ms | Normal mean ms | Cache preparation mean ms | Queue MiB | Quality |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for run in result["runs"]:
        t = run["timing"]
        report.append(f"| {run['mode']} / {run['repeat']} | {t['frame_ms']['mean']:.3f} | {t['frame_ms']['p95']:.3f} | {t['frame_ms']['p99']:.3f} | {t['normal_ms']['mean']:.3f} | {t['normal_cache_prepare_ms']['mean']:.3f} | {run['normal_cache']['recompute_queue_bytes'] / 2**20:.2f} | {run['quality_pass']} |")
    report.extend(["", "Validation includes reference map and full-normal work; excluded from timing pairs. MPPI runs every tenth scan; commands are not applied. All failures are retained.", ""])
    (args.out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    if args.plot:
        plot(result, args.out_dir / "comparison.png")
    if any(r["returncode"] or not r["quality_pass"] for r in result["runs"]):
        raise SystemExit("One or more quality gates failed; failures retained")
    if any(not all(v for k, v in c.items() if k != "repeat") for c in result["paired_checks"]):
        raise SystemExit("One or more paired performance/exactness checks failed; failures retained")
    if not all(result.get("default_checks", {}).values()):
        raise SystemExit("One or more default-policy checks failed; failures retained")


if __name__ == "__main__":
    main()
