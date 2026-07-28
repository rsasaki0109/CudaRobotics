#!/usr/bin/env python3
"""Benchmark CudaRobotics registration against external CPU baselines.

The public entry point runs one subprocess per (algorithm, size) cell so that
probreg CPD and other heavy CPU baselines cannot poison the rest of the run.
Each child does one small warmup, then reports median timing/error over the
requested deterministic trials.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")


ALGORITHMS = (
    "cudarobotics_filterreg_gpu",
    "cudarobotics_fgr_gpu",
    "cudarobotics_robust_p2plane_gpu",
    "cudarobotics_robust_treg_gpu",
    "cudarobotics_sinkhorn_gpu",
    "probreg_filterreg_cpu",
    "probreg_cpd_rigid_cpu",
    "open3d_gicp_cpu",
)

SCENARIOS = {
    "lumpy_partial": {
        "overlap": 0.85,
        "noise_sigma": 0.02,
        "outlier_ratio": 0.0,
        "euler": (0.12, -0.18, 0.08),
        "translation": (0.35, -0.25, 0.20),
        "notes": "asymmetric lumpy surface, 85% overlap, 2 cm noise",
    },
    "low_overlap": {
        "overlap": 0.60,
        "noise_sigma": 0.02,
        "outlier_ratio": 0.0,
        "euler": (0.12, -0.18, 0.08),
        "translation": (0.35, -0.25, 0.20),
        "notes": "same surface and transform, 60% source overlap",
    },
    "outlier_partial": {
        "overlap": 0.85,
        "noise_sigma": 0.02,
        "outlier_ratio": 0.10,
        "euler": (0.12, -0.18, 0.08),
        "translation": (0.35, -0.25, 0.20),
        "notes": "85% overlap plus 10% uniform source outliers",
    },
    "large_offset": {
        "overlap": 0.85,
        "noise_sigma": 0.02,
        "outlier_ratio": 0.0,
        "euler": (0.28, -0.32, 0.22),
        "translation": (0.60, -0.45, 0.35),
        "notes": "larger rigid transform under identity initialization",
    },
}

CSV_FIELDS = (
    "scenario",
    "algorithm",
    "size",
    "target_points",
    "source_points_median",
    "trials_requested",
    "trials_completed",
    "status",
    "quality_pass",
    "median_ms",
    "min_ms",
    "max_ms",
    "median_rot_err_deg",
    "median_trans_err_m",
    "median_rmse_m",
    "load_before_1min",
    "load_after_1min",
    "python",
    "numpy_version",
    "cudarobotics_version",
    "probreg_version",
    "open3d_version",
    "notes",
    "error",
    "elapsed_ms_trials",
    "rot_err_deg_trials",
    "trans_err_m_trials",
    "rmse_m_trials",
)


def read_load1() -> float:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return float("nan")


def make_lumpy(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3), dtype=np.float32)
    bumps = np.array(
        [
            [0.8, 0.2, 0.5, 0.9, 0.25],
            [-0.3, 0.9, 0.2, 0.7, 0.30],
            [0.1, -0.6, 0.8, 0.8, 0.22],
            [-0.7, -0.4, -0.5, 1.0, 0.28],
            [0.4, 0.3, -0.85, 0.6, 0.20],
        ],
        dtype=np.float32,
    )
    for i in range(n):
        z = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        r2 = math.sqrt(max(0.0, 1.0 - z * z))
        dx, dy, dz = r2 * math.cos(phi), r2 * math.sin(phi), z
        radius = (
            2.0
            + 0.35 * math.sin(3.0 * phi) * (1.0 - z * z)
            + 0.30 * dz * dx
            + 0.20 * math.cos(2.0 * phi)
        )
        for bx, by, bz, height, width in bumps:
            ang = 1.0 - (dx * bx + dy * by + dz * bz)
            radius += height * math.exp(-(ang * ang) / (2.0 * width * width))
        pts[i] = radius * np.array([dx, dy, dz], dtype=np.float32)
    return pts


def euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    c, s = math.cos, math.sin
    return np.array(
        [
            [c(ry) * c(rz), -c(ry) * s(rz), s(ry)],
            [
                s(rx) * s(ry) * c(rz) + c(rx) * s(rz),
                -s(rx) * s(ry) * s(rz) + c(rx) * c(rz),
                -s(rx) * c(ry),
            ],
            [
                -c(rx) * s(ry) * c(rz) + s(rx) * s(rz),
                c(rx) * s(ry) * s(rz) + s(rx) * c(rz),
                c(rx) * c(ry),
            ],
        ],
        dtype=np.float64,
    )


def make_pair(
    n: int, trial: int, scenario: str = "lumpy_partial"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = SCENARIOS[scenario]
    target = make_lumpy(n, seed=101 + trial)
    gt_rotation = euler_xyz(*spec["euler"])
    gt_translation = np.array(spec["translation"], dtype=np.float64)
    rng = np.random.default_rng(700 + trial)
    keep = rng.uniform(0.0, 1.0, len(target)) <= float(spec["overlap"])
    source = (
        target.astype(np.float64) @ gt_rotation.T
        + gt_translation
        + rng.normal(0.0, float(spec["noise_sigma"]), size=target.shape)
    )
    source = source[keep].astype(np.float32)
    outlier_ratio = float(spec["outlier_ratio"])
    if outlier_ratio > 0.0:
        outlier_count = max(1, int(round(len(source) * outlier_ratio)))
        outliers = rng.uniform(-3.5, 3.5, size=(outlier_count, 3)).astype(np.float32)
        source = np.vstack([source, outliers])
    expected_rotation = gt_rotation.T
    expected_translation = -gt_rotation.T @ gt_translation
    return target, source, expected_rotation, expected_translation


def rotation_error_deg(rotation: np.ndarray, expected: np.ndarray) -> float:
    delta = rotation @ expected.T
    arg = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(arg))


def transform_rmse(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    sample_count: int = 4096,
) -> float:
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return float("nan")
    if len(source) > sample_count:
        idx = np.linspace(0, len(source) - 1, sample_count, dtype=np.int64)
        source_eval = source[idx]
    else:
        source_eval = source
    aligned = source_eval.astype(np.float64) @ rotation.T + translation
    tree = cKDTree(target.astype(np.float64))
    dists, _ = tree.query(aligned, k=1, workers=-1)
    return float(np.sqrt(np.mean(dists * dists)))


def version_info(algorithm: str) -> dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "numpy_version": np.__version__,
        "cudarobotics_version": "",
        "probreg_version": "",
        "open3d_version": "",
    }
    if algorithm.startswith("cudarobotics"):
        import cudarobotics as cr

        info["cudarobotics_version"] = getattr(cr, "__version__", "")
    if algorithm.startswith("probreg"):
        import probreg

        info["probreg_version"] = getattr(probreg, "__version__", "")
    if algorithm.startswith("open3d"):
        import open3d as o3d

        info["open3d_version"] = getattr(o3d, "__version__", "")
    return info


def run_cudarobotics_filterreg(
    target: np.ndarray, source: np.ndarray, maxiter: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cudarobotics as cr

    registrar = cr.registration.FilterReg()
    rotation, translation, info = registrar.register(target, source)
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3), np.asarray(
        translation, dtype=np.float64
    ), dict(info)


def run_cudarobotics_rigid(
    algorithm: str,
    target: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cudarobotics as cr

    registrars = {
        "cudarobotics_fgr_gpu": cr.registration.Fgr,
        "cudarobotics_robust_p2plane_gpu": cr.registration.RobustP2Plane,
        "cudarobotics_robust_treg_gpu": cr.registration.RobustTreg,
        "cudarobotics_sinkhorn_gpu": cr.registration.SinkhornReg,
    }
    registrar = registrars[algorithm]()
    rotation, translation, info = registrar.register(target, source)
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3), np.asarray(
        translation, dtype=np.float64
    ), dict(info)


def run_probreg_filterreg(
    target: np.ndarray, source: np.ndarray, maxiter: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from probreg import filterreg

    transformation, sigma2, q_value = filterreg.registration_filterreg(
        source,
        target,
        update_sigma2=True,
        maxiter=maxiter,
        tol=0.001,
    )
    return (
        np.asarray(transformation.rot, dtype=np.float64).reshape(3, 3),
        np.asarray(transformation.t, dtype=np.float64),
        {"final_sigma2": float(sigma2), "q": float(q_value)},
    )


def run_probreg_cpd(
    target: np.ndarray, source: np.ndarray, maxiter: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from probreg import cpd

    result = cpd.registration_cpd(
        source,
        target,
        tf_type_name="rigid",
        maxiter=maxiter,
        tol=0.001,
        use_cuda=False,
    )
    transformation = result.transformation
    return (
        np.asarray(transformation.rot, dtype=np.float64).reshape(3, 3),
        np.asarray(transformation.t, dtype=np.float64),
        {"final_sigma2": float(result.sigma2), "q": float(result.q)},
    )


def run_open3d_gicp(
    target: np.ndarray, source: np.ndarray, maxiter: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import open3d as o3d

    source_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(source.astype(np.float64))
    )
    target_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(target.astype(np.float64))
    )
    result = o3d.pipelines.registration.registration_generalized_icp(
        source_cloud,
        target_cloud,
        1.0,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=maxiter),
    )
    transform = np.asarray(result.transformation, dtype=np.float64)
    return (
        transform[:3, :3],
        transform[:3, 3],
        {"fitness": float(result.fitness), "inlier_rmse": float(result.inlier_rmse)},
    )


def run_algorithm(
    algorithm: str, target: np.ndarray, source: np.ndarray, maxiter: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if algorithm == "cudarobotics_filterreg_gpu":
        return run_cudarobotics_filterreg(target, source, maxiter)
    if algorithm in {
        "cudarobotics_fgr_gpu",
        "cudarobotics_robust_p2plane_gpu",
        "cudarobotics_robust_treg_gpu",
        "cudarobotics_sinkhorn_gpu",
    }:
        return run_cudarobotics_rigid(algorithm, target, source)
    if algorithm == "probreg_filterreg_cpu":
        return run_probreg_filterreg(target, source, maxiter)
    if algorithm == "probreg_cpd_rigid_cpu":
        return run_probreg_cpd(target, source, maxiter)
    if algorithm == "open3d_gicp_cpu":
        return run_open3d_gicp(target, source, maxiter)
    raise ValueError(f"unknown algorithm: {algorithm}")


def median(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(statistics.median(clean)) if clean else float("nan")


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def joined(values: list[float]) -> str:
    return ";".join(fmt_float(v) for v in values)


def run_child(args: argparse.Namespace) -> int:
    load_before = read_load1()
    row: dict[str, Any] = {
        "scenario": args.scenario,
        "algorithm": args.algorithm,
        "size": args.size,
        "target_points": args.size,
        "source_points_median": "",
        "trials_requested": args.trials,
        "trials_completed": 0,
        "status": "error",
        "quality_pass": "",
        "median_ms": "",
        "min_ms": "",
        "max_ms": "",
        "median_rot_err_deg": "",
        "median_trans_err_m": "",
        "median_rmse_m": "",
        "load_before_1min": fmt_float(load_before),
        "load_after_1min": "",
        "notes": "",
        "error": "",
        "elapsed_ms_trials": "",
        "rot_err_deg_trials": "",
        "trans_err_m_trials": "",
        "rmse_m_trials": "",
    }
    try:
        row.update(version_info(args.algorithm))
        warm_n = min(512, args.size)
        target, source, _, _ = make_pair(warm_n, -1, args.scenario)
        run_algorithm(args.algorithm, target, source, args.maxiter)

        elapsed_ms: list[float] = []
        rot_errors: list[float] = []
        trans_errors: list[float] = []
        rmses: list[float] = []
        source_counts: list[int] = []
        for trial in range(args.trials):
            target, source, expected_rotation, expected_translation = make_pair(
                args.size, trial, args.scenario
            )
            source_counts.append(int(len(source)))
            start = time.perf_counter()
            rotation, translation, info = run_algorithm(
                args.algorithm, target, source, args.maxiter
            )
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
            rot_errors.append(rotation_error_deg(rotation, expected_rotation))
            trans_errors.append(
                float(np.linalg.norm(translation - expected_translation))
            )
            if "final_rmse" in info:
                rmses.append(float(info["final_rmse"]))
            elif "inlier_rmse" in info:
                rmses.append(float(info["inlier_rmse"]))
            else:
                rmses.append(transform_rmse(source, target, rotation, translation))

        row["source_points_median"] = int(statistics.median(source_counts))
        row["trials_completed"] = len(elapsed_ms)
        row["status"] = "ok"
        row["median_ms"] = fmt_float(median(elapsed_ms))
        row["min_ms"] = fmt_float(min(elapsed_ms))
        row["max_ms"] = fmt_float(max(elapsed_ms))
        row["median_rot_err_deg"] = fmt_float(median(rot_errors))
        row["median_trans_err_m"] = fmt_float(median(trans_errors))
        row["median_rmse_m"] = fmt_float(median(rmses))
        row["quality_pass"] = (
            median(rot_errors) <= args.max_rot_error_deg
            and median(trans_errors) <= args.max_trans_error_m
        )
        row["elapsed_ms_trials"] = joined(elapsed_ms)
        row["rot_err_deg_trials"] = joined(rot_errors)
        row["trans_err_m_trials"] = joined(trans_errors)
        row["rmse_m_trials"] = joined(rmses)
        if args.algorithm == "probreg_filterreg_cpu":
            row["notes"] = "probreg update_sigma2=True"
        scenario_note = SCENARIOS[args.scenario]["notes"]
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + scenario_note
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if args.traceback:
            row["error"] += "\\n" + traceback.format_exc()
    finally:
        row["load_after_1min"] = fmt_float(read_load1())
        print(json.dumps(row, sort_keys=True))
    return 0


def wait_for_load(max_load: float, timeout_s: float, sleep_s: float) -> None:
    if max_load <= 0:
        return
    deadline = time.time() + timeout_s
    while read_load1() > max_load and time.time() < deadline:
        time.sleep(sleep_s)


def timeout_row(
    scenario: str,
    algorithm: str,
    size: int,
    trials: int,
    timeout_s: float,
    load_before: float,
    stderr: str,
) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "scenario": scenario,
            "algorithm": algorithm,
            "size": size,
            "target_points": size,
            "trials_requested": trials,
            "trials_completed": 0,
            "status": "timeout",
            "load_before_1min": fmt_float(load_before),
            "load_after_1min": fmt_float(read_load1()),
            "python": sys.version.split()[0],
            "numpy_version": np.__version__,
            "error": f"timeout after {timeout_s:.0f}s",
        }
    )
    if stderr.strip():
        row["error"] += ": " + stderr.strip()[-500:]
    return row


def error_row(
    scenario: str,
    algorithm: str,
    size: int,
    trials: int,
    load_before: float,
    message: str,
) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "scenario": scenario,
            "algorithm": algorithm,
            "size": size,
            "target_points": size,
            "trials_requested": trials,
            "trials_completed": 0,
            "status": "error",
            "load_before_1min": fmt_float(load_before),
            "load_after_1min": fmt_float(read_load1()),
            "python": sys.version.split()[0],
            "numpy_version": np.__version__,
            "error": message,
        }
    )
    return row


def run_parent(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    algorithms = args.algorithms
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for scenario in args.scenarios:
        for size in args.sizes:
            for algorithm in algorithms:
                wait_for_load(args.load_gate, args.load_gate_timeout, args.load_gate_sleep)
                load_before = read_load1()
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run-cell",
                    "--scenario",
                    scenario,
                    "--algorithm",
                    algorithm,
                    "--size",
                    str(size),
                    "--trials",
                    str(args.trials),
                    "--maxiter",
                    str(args.maxiter),
                    "--max-rot-error-deg",
                    str(args.max_rot_error_deg),
                    "--max-trans-error-m",
                    str(args.max_trans_error_m),
                ]
                if args.traceback:
                    cmd.append("--traceback")
                print(
                    f"[cell] scenario={scenario} algorithm={algorithm} size={size} "
                    f"trials={args.trials} load={load_before:.2f}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    completed = subprocess.run(
                        cmd,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=args.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                    row = timeout_row(
                        scenario,
                        algorithm,
                        size,
                        args.trials,
                        args.timeout_seconds,
                        load_before,
                        stderr,
                    )
                    rows.append(row)
                    write_csv(csv_path, rows)
                    print(
                        f"[cell] timeout scenario={scenario} algorithm={algorithm} size={size}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                stdout_lines = [line for line in completed.stdout.splitlines() if line]
                if not stdout_lines:
                    stderr = completed.stderr.strip()
                    message = f"exit code {completed.returncode}"
                    if stderr:
                        message += f"; stderr={stderr[-1000:]}"
                    row = error_row(
                        scenario,
                        algorithm,
                        size,
                        args.trials,
                        load_before,
                        message,
                    )
                else:
                    try:
                        row = json.loads(stdout_lines[-1])
                    except json.JSONDecodeError as exc:
                        row = error_row(
                            scenario,
                            algorithm,
                            size,
                            args.trials,
                            load_before,
                            f"could not parse child JSON: {exc}; stdout={completed.stdout[-500:]}",
                        )
                if completed.returncode != 0 and row.get("status") == "ok":
                    row["status"] = "error"
                    row["error"] = completed.stderr.strip() or f"exit code {completed.returncode}"
                if completed.stderr.strip() and row.get("status") != "ok":
                    err = row.get("error", "")
                    row["error"] = (err + "\n" if err else "") + completed.stderr.strip()[-1000:]
                rows.append(row)
                write_csv(csv_path, rows)
                print(
                    f"[cell] status={row.get('status')} scenario={scenario} "
                    f"algorithm={algorithm} size={size} median_ms={row.get('median_ms', '')}",
                    file=sys.stderr,
                    flush=True,
                )

    write_csv(csv_path, rows)
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_markdown(path: Path, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row.get("status") == "ok"]
    quality_passes = [
        row for row in successful
        if str(row.get("quality_pass", "")).lower() == "true"
    ]
    lines = [
        "# Registration Unified Benchmark",
        "",
        "This report compares CudaRobotics GPU registration implementations and "
        "optional external CPU baselines on identical deterministic point-cloud pairs.",
        "",
        f"- Cells completed: {len(successful)}/{len(rows)}",
        f"- Quality gates passed: {len(quality_passes)}/{len(successful)}",
        f"- Raw CSV: `{csv_path.name}`",
        "",
        "| Scenario | Algorithm | Points | Status | Quality | Median (ms) | "
        "Rotation error (deg) | Translation error (m) | RMSE (m) |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality = row.get("quality_pass", "")
        quality_text = (
            "PASS" if str(quality).lower() == "true"
            else "FAIL" if str(quality).lower() == "false"
            else "n/a"
        )
        lines.append(
            f"| {row.get('scenario', '')} | `{row.get('algorithm', '')}` | "
            f"{row.get('size', '')} | {row.get('status', '')} | {quality_text} | "
            f"{row.get('median_ms', '') or 'n/a'} | "
            f"{row.get('median_rot_err_deg', '') or 'n/a'} | "
            f"{row.get('median_trans_err_m', '') or 'n/a'} | "
            f"{row.get('median_rmse_m', '') or 'n/a'} |"
        )
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        lines += ["", "## Unavailable or Failed Cells", ""]
        for row in errors:
            lines.append(
                f"- `{row.get('algorithm', '')}` / `{row.get('scenario', '')}` / "
                f"{row.get('size', '')}: {row.get('error', 'unknown error')}"
            )
    lines += [
        "",
        "Quality gates use the configured median rotation and translation error "
        "limits. A missing optional dependency is reported as an unavailable cell, "
        "not silently removed from the comparison.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def suite_passed(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        row.get("status") == "ok"
        and str(row.get("quality_pass", "")).lower() == "true"
        for row in rows
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2000, 8000, 32000])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--csv", default="docs/results/registration_external_baselines_2026-06-11.csv")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=["lumpy_partial"],
        help="Synthetic registration data scenarios to benchmark.",
    )
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--maxiter", type=int, default=64)
    parser.add_argument("--max-rot-error-deg", type=float, default=5.0)
    parser.add_argument("--max-trans-error-m", type=float, default=0.20)
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Write a Markdown summary; defaults to the CSV path with .md suffix.",
    )
    parser.add_argument("--load-gate", type=float, default=0.0)
    parser.add_argument("--load-gate-timeout", type=float, default=900.0)
    parser.add_argument("--load-gate-sleep", type=float, default=15.0)
    parser.add_argument("--traceback", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any requested cell fails, times out, or misses quality gates.",
    )
    parser.add_argument("--run-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=ALGORITHMS, help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run_cell and (args.scenario is None or args.algorithm is None or args.size is None):
        parser.error("--run-cell requires --scenario, --algorithm and --size")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.run_cell:
        return run_child(args)
    result = run_parent(args)
    csv_path = Path(args.csv)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    write_markdown(args.markdown or csv_path.with_suffix(".md"), csv_path, rows)
    if args.strict and not suite_passed(rows):
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
