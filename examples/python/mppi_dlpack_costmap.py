#!/usr/bin/env python3
"""CUDA DLPack costmap example for cudarobotics.MppiPlanner.

The costmap stays on the GPU as a PyTorch or CuPy uint8 tensor. Path and
footprint inputs remain NumPy host arrays in the current Python API.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import cudarobotics as cr


SIZE_X = 200
SIZE_Y = 200
RESOLUTION = 0.05
DEFAULT_RESULT = Path("build/onboarding/dlpack/dlpack_result.json")


class CudaDLPackOnly:
    """Prefer CUDA DLPack when an array also advertises a CPU buffer facade."""

    def __init__(self, value):
        self._value = value

    def __dlpack__(self, *args, **kwargs):
        return self._value.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._value.__dlpack_device__()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULT,
        help="write the versioned recipe result to this path",
    )
    return parser.parse_args(argv)


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def make_torch_costmap():
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    costmap = torch.zeros((SIZE_Y, SIZE_X), dtype=torch.uint8, device="cuda")
    wx0, wx1 = int(4.9 / RESOLUTION), int(5.1 / RESOLUTION)
    gy0, gy1 = int(4.0 / RESOLUTION), int(6.0 / RESOLUTION)
    costmap[:gy0, wx0:wx1] = 254
    costmap[gy1:, wx0:wx1] = 254
    return costmap, "torch"


def make_cupy_costmap():
    try:
        import cupy as cp
    except ImportError:
        return None, None
    costmap = cp.zeros((SIZE_Y, SIZE_X), dtype=cp.uint8)
    wx0, wx1 = int(4.9 / RESOLUTION), int(5.1 / RESOLUTION)
    gy0, gy1 = int(4.0 / RESOLUTION), int(6.0 / RESOLUTION)
    costmap[:gy0, wx0:wx1] = 254
    costmap[gy1:, wx0:wx1] = 254
    return CudaDLPackOnly(costmap), "cupy"


def make_device_costmap() -> tuple[Any, str]:
    costmap, backend = make_torch_costmap()
    if costmap is not None:
        return costmap, backend
    costmap, backend = make_cupy_costmap()
    if costmap is not None:
        return costmap, backend
    raise RuntimeError(
        "Install PyTorch with CUDA or CuPy including toolkit headers "
        "(for example 'cupy-cuda12x[ctk]') to run this example"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        costmap, backend = make_device_costmap()
    except RuntimeError as exc:
        result = {
            "schema_version": 1,
            "recipe": "dlpack_costmap",
            "passed": False,
            "failure_category": "dependency",
            "message": str(exc),
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
        write_result(args.result_json, result)
        print(str(exc), file=sys.stderr)
        print(f"wrote failure result: {args.result_json}", file=sys.stderr)
        return 1
    path_x = np.arange(1.0, 9.05, 0.1, dtype=np.float32)
    path = np.stack([path_x, np.full_like(path_x, 5.0)], axis=1).astype(np.float32)

    planner = cr.MppiPlanner(
        batch_size=2048,
        time_steps=56,
        model_dt=0.05,
        path_angle_weight=0.25,
        distance_field_weight=12.0,
        distance_field_cutoff=0.8,
    )

    state = np.array([1.0, 5.0, 0.0], dtype=np.float32)
    for step in range(500):
        try:
            v, vy, w, info = planner.compute(
                state,
                costmap,
                path,
                (9.0, 5.0, 0.0),
                resolution=RESOLUTION,
                goal_is_final=True,
            )
        except Exception as exc:
            result = {
                "schema_version": 1,
                "recipe": "dlpack_costmap",
                "passed": False,
                "failure_category": "cuda_runtime",
                "backend": backend,
                "message": str(exc),
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
            write_result(args.result_json, result)
            print(f"DLPack runtime failed: {exc}", file=sys.stderr)
            print(f"wrote failure result: {args.result_json}", file=sys.stderr)
            return 1
        yaw = float(state[2])
        state[0] += 0.05 * (v * math.cos(yaw) - vy * math.sin(yaw))
        state[1] += 0.05 * (v * math.sin(yaw) + vy * math.cos(yaw))
        state[2] = math.atan2(math.sin(yaw + 0.05 * w), math.cos(yaw + 0.05 * w))
        if np.linalg.norm(state[:2] - np.array([9.0, 5.0], dtype=np.float32)) < 0.25:
            result = {
                "schema_version": 1,
                "recipe": "dlpack_costmap",
                "passed": True,
                "failure_category": None,
                "backend": backend,
                "steps": step + 1,
                "final_state": [float(state[0]), float(state[1]), float(state[2])],
                "valid_rollouts": int(info["valid_rollouts"]),
                "sampled_rollouts": int(info["sampled_rollouts"]),
                "valid_rollout_ratio": float(info["valid_rollout_ratio"]),
                "best_cost": float(info["best_cost"]),
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
            write_result(args.result_json, result)
            print(
                "backend={backend} steps={steps} final=({x:.2f}, {y:.2f}) "
                "valid={valid}/{sampled} ratio={ratio:.3f} best_cost={best:.1f}".format(
                    backend=backend,
                    steps=step + 1,
                    x=state[0],
                    y=state[1],
                    valid=info["valid_rollouts"],
                    sampled=info["sampled_rollouts"],
                    ratio=info["valid_rollout_ratio"],
                    best=info["best_cost"],
                )
            )
            print(f"result={args.result_json}")
            return 0

    result = {
        "schema_version": 1,
        "recipe": "dlpack_costmap",
        "passed": False,
        "failure_category": "algorithm_check",
        "steps": 500,
        "final_state": [float(state[0]), float(state[1]), float(state[2])],
        "valid_rollout_ratio": float(info["valid_rollout_ratio"]),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    write_result(args.result_json, result)
    print(
        "goal not reached; final=({:.2f}, {:.2f}) valid_ratio={:.3f}".format(
            state[0], state[1], info["valid_rollout_ratio"]
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
