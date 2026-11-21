"""``cudarobotics-cudanav``: run the GPU CudaNav pipeline from the command line.

Subcommands:

* ``sim`` - closed-loop S-course mission (synthetic LiDAR, command-driven plant);
* ``sequence`` - odometry + mapping + ESDF over a recorded ``CRKICP1`` sequence.

Both write ``summary.json``, ``manifest.json``, ``trajectory.csv``,
``occupancy.npy``, ``esdf.npy`` and ``costmap.npy`` (plus ``overview.png`` when
matplotlib is installed). A failed run still writes ``summary.json`` with the
failing stage and exits non-zero.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_QUALITY = 2
EXIT_USAGE = 64


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cudarobotics-cudanav",
        description="GPU KISS-ICP -> voxel map -> ESDF -> CUDA MPPI (CudaNav).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("sim", help="closed-loop S-course mission on synthetic LiDAR")
    sim.add_argument("--output-dir", type=Path, required=True)
    sim.add_argument("--traversals", type=_positive_int, default=1)
    sim.add_argument("--max-steps", type=_positive_int, default=1800)
    sim.add_argument("--minimum-duration-s", type=float, default=0.0)
    sim.add_argument(
        "--scenario",
        choices=("s_course", "crossing"),
        default="s_course",
        help="crossing: two pedestrians cross the corridors (one traversal)",
    )
    sim.add_argument(
        "--dynamic-mode",
        choices=("map", "tracked_static", "predicted"),
        default="predicted",
        help="crossing only: how moving obstacles reach the planner",
    )
    sim.add_argument("--seed", type=int, default=0, help="crossing only: pedestrian schedule seed")
    sim.add_argument("--obs-latency-frames", type=int, default=0)
    sim.add_argument("--obs-position-noise-m", type=float, default=0.0)
    sim.add_argument("--obs-velocity-noise-mps", type=float, default=0.0)
    sim.add_argument("--obs-dropout", type=float, default=0.0)

    seq = sub.add_parser("sequence", help="odometry + mapping + ESDF over a CRKICP1 file")
    seq.add_argument("sequence", type=Path)
    seq.add_argument("--output-dir", type=Path, required=True)
    seq.add_argument("--max-frames", type=int, default=0)
    seq.add_argument("--max-ate-m", type=float, default=5.0)
    seq.add_argument("--max-drift-percent", type=float, default=10.0)
    seq.add_argument("--min-inliers", type=int, default=30)

    for command in (sim, seq):
        command.add_argument(
            "--check",
            action="store_true",
            help="exit 2 when the run completes but its quality gate fails",
        )
        command.add_argument("--no-plot", action="store_true", help="skip overview.png")
        command.add_argument(
            "--stage-chain",
            choices=("device", "host"),
            default="device",
            help="keep occupancy/ESDF/costmap on the GPU (device) or copy them between stages (host)",
        )
    return parser


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_grids(output: Path, step: Any) -> dict[str, str]:
    if step is None:
        return {}
    np.save(output / "occupancy.npy", step.occupancy.data)
    np.save(output / "esdf.npy", step.distance_field.distances)
    np.save(output / "costmap.npy", step.costmap)
    return {"occupancy": "occupancy.npy", "esdf": "esdf.npy", "costmap": "costmap.npy"}


def _plot(output: Path, step: Any, columns: Sequence[str], rows: Sequence[Sequence[Any]], est: tuple[str, str], ref: tuple[str, str]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    data = np.asarray(rows, dtype=np.float64)
    index = {name: i for i, name in enumerate(columns)}
    occ = step.occupancy
    extent = (
        occ.origin[0],
        occ.origin[0] + occ.width * occ.resolution,
        occ.origin[1],
        occ.origin[1] + occ.height * occ.resolution,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(occ.data, origin="lower", extent=extent, cmap="gray_r", vmin=-1, vmax=100)
    axes[1].imshow(step.distance_field.distances, origin="lower", extent=extent, cmap="viridis")
    for axis, title in zip(axes, ("occupancy + trajectory", "ESDF (m)")):
        axis.plot(data[:, index[ref[0]]], data[:, index[ref[1]]], "g-", lw=1.5, label="reference")
        axis.plot(data[:, index[est[0]]], data[:, index[est[1]]], "r--", lw=1.2, label="estimated")
        for obstacle in (0, 1):
            if f"obstacle_{obstacle}_x" in index:
                axis.plot(
                    data[:, index[f"obstacle_{obstacle}_x"]],
                    data[:, index[f"obstacle_{obstacle}_y"]],
                    ":", color="tab:orange", lw=1.0,
                    label="pedestrian" if obstacle == 0 else None,
                )
        axis.set_title(title)
        axis.set_aspect("equal")
    axes[0].legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "overview.png", dpi=110)
    plt.close(figure)
    return "overview.png"


def _finish(
    args: argparse.Namespace,
    argv: Sequence[str],
    scenario: str,
    parameters: dict[str, Any],
    result: Any,
    columns: Sequence[str],
    est: tuple[str, str],
    ref: tuple[str, str],
    inputs: dict[str, Any] | None = None,
) -> int:
    from .evidence import build_manifest, write_json

    output: Path = args.output_dir
    _write_csv(output / "trajectory.csv", columns, result.rows)
    outputs = {"summary": "summary.json", "trajectory": "trajectory.csv"}
    outputs.update(_write_grids(output, result.last_step))
    if not args.no_plot and result.last_step is not None:
        plot = _plot(output, result.last_step, columns, result.rows, est, ref)
        if plot:
            outputs["overview"] = plot
    summary = dict(result.summary)
    summary["stage_chain"] = args.stage_chain
    summary["stage_latency_ms"] = result.stage_latency_ms
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        build_manifest(
            command=["cudarobotics-cudanav", *argv],
            scenario=scenario,
            parameters=parameters,
            outputs=outputs,
            inputs=inputs,
        ),
    )
    status = "PASS" if summary["quality_pass"] else "FAIL"
    print(
        f"{status}: {scenario} frames={summary['frames']} "
        f"drift={summary['odometry_drift_percent']:.3f}% "
        f"ate={summary['odometry_ate_rmse_m']:.3f} m -> {output / 'summary.json'}"
    )
    if args.check and not summary["quality_pass"]:
        return EXIT_QUALITY
    return EXIT_OK


def _run_crossing(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .dynamic_scenario import (
        SCENARIO,
        TRAJECTORY_COLUMNS,
        ObservationModel,
        run_crossing,
    )
    from .pipeline import CudaNavPipeline

    observation = ObservationModel(
        latency_frames=args.obs_latency_frames,
        position_noise_m=args.obs_position_noise_m,
        velocity_noise_mps=args.obs_velocity_noise_mps,
        dropout_probability=args.obs_dropout,
    )
    result = run_crossing(
        dynamic_mode=args.dynamic_mode,
        seed=args.seed,
        observation=observation,
        maximum_steps=min(args.max_steps, 600),
        pipeline=CudaNavPipeline(stage_chain=args.stage_chain),
    )
    parameters = {
        "stage_chain": args.stage_chain,
        "scenario": "crossing",
        "dynamic_mode": args.dynamic_mode,
        "seed": args.seed,
        "observation": result.summary["observation"],
        "max_steps": min(args.max_steps, 600),
    }
    return _finish(
        args, argv, SCENARIO, parameters, result, TRAJECTORY_COLUMNS,
        ("estimated_x", "estimated_y"), ("truth_x", "truth_y"),
    )


def _run_sim(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .closed_loop import SCENARIO, TRAJECTORY_COLUMNS, run_s_course
    from .pipeline import CudaNavPipeline

    if args.scenario == "crossing":
        return _run_crossing(args, argv)

    result = run_s_course(
        traversals=args.traversals,
        maximum_steps=args.max_steps,
        minimum_duration_s=args.minimum_duration_s,
        pipeline=CudaNavPipeline(stage_chain=args.stage_chain),
    )
    parameters = {
        "stage_chain": args.stage_chain,
        "traversals": args.traversals,
        "max_steps": args.max_steps,
        "minimum_duration_s": args.minimum_duration_s,
    }
    return _finish(
        args, argv, SCENARIO, parameters, result, TRAJECTORY_COLUMNS,
        ("estimated_x", "estimated_y"), ("truth_x", "truth_y"),
    )


def _run_sequence(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from .evidence import SequenceGate, sha256_file
    from .pipeline import CudaNavPipeline
    from .sequence import SEQUENCE_COLUMNS, run_sequence, sequence_header

    gate = SequenceGate(
        maximum_ate_rmse_m=args.max_ate_m,
        maximum_final_drift_percent=args.max_drift_percent,
        minimum_inliers=args.min_inliers,
    )
    sequence_header(args.sequence)  # reject malformed input before allocating GPU state
    result = run_sequence(
        args.sequence,
        maximum_frames=args.max_frames,
        gate=gate,
        pipeline=CudaNavPipeline(enable_planner=False, stage_chain=args.stage_chain),
    )
    parameters = {
        "stage_chain": args.stage_chain,
        "max_frames": args.max_frames,
        "max_ate_m": args.max_ate_m,
        "max_drift_percent": args.max_drift_percent,
        "min_inliers": args.min_inliers,
    }
    inputs = {
        "sequence": str(args.sequence),
        "sequence_sha256": sha256_file(args.sequence),
    }
    return _finish(
        args, argv, "cudanav_recorded_sequence", parameters, result, SEQUENCE_COLUMNS,
        ("estimated_x", "estimated_y"), ("reference_x", "reference_y"), inputs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return EXIT_OK if error.code == 0 else EXIT_USAGE
    if args.command == "sim" and args.scenario == "crossing":
        from .dynamic_scenario import ObservationModel

        try:
            ObservationModel(
                latency_frames=args.obs_latency_frames,
                position_noise_m=args.obs_position_noise_m,
                velocity_noise_mps=args.obs_velocity_noise_mps,
                dropout_probability=args.obs_dropout,
            )
        except ValueError as error:
            parser.print_usage(sys.stderr)
            print(f"cudarobotics-cudanav: {error}", file=sys.stderr)
            return EXIT_USAGE
    if args.command == "sequence" and not args.sequence.is_file():
        parser.print_usage(sys.stderr)
        print(f"cudarobotics-cudanav: sequence file not found: {args.sequence}", file=sys.stderr)
        return EXIT_USAGE

    output: Path = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "sim":
        scenario = (
            "cudanav_s_course_crossing" if args.scenario == "crossing" else "cudanav_s_course"
        )
    else:
        scenario = "cudanav_recorded_sequence"
    stage = "initialize"
    try:
        from .evidence import runtime_info

        info = runtime_info()
        if not info.get("device_count"):
            raise RuntimeError(
                "no CUDA-capable GPU is visible: " + str(info.get("error", "device_count=0"))
            )
        stage = "run"
        if args.command == "sim":
            return _run_sim(args, argv)
        return _run_sequence(args, argv)
    except Exception as error:  # noqa: BLE001 - every failure must be recorded
        from .evidence import build_manifest, failure_summary, write_json

        write_json(output / "summary.json", failure_summary(stage, error, scenario=scenario))
        try:
            write_json(
                output / "manifest.json",
                build_manifest(
                    command=["cudarobotics-cudanav", *argv],
                    scenario=scenario,
                    parameters={},
                    outputs={"summary": "summary.json"},
                ),
            )
        except Exception:  # noqa: BLE001 - the summary already records the failure
            pass
        print(f"cudarobotics-cudanav: {stage} failed: {error}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
