"""Machine-readable summaries, quality gates, and run manifests for CudaNav."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
MPPI_RNG_SEED = 42  # Fixed cuRAND seed of the CUDA MPPI core (src/mppi_gpu.cu).

# Required by the Docker ``cudanav`` summary contract (docker/entrypoint.sh).
REQUIRED_SUMMARY_FIELDS = (
    "schema_version",
    "success",
    "smoke_pass",
    "collision_count",
    "odometry_drift_percent",
    "command_deadline_miss_rate",
)


@dataclass(frozen=True)
class ClosedLoopGate:
    """Thresholds shared with ``scripts/cudanav_evidence.py`` profiles."""

    name: str
    maximum_drift_percent: float
    maximum_deadline_miss_rate: float
    maximum_goal_distance_m: float
    maximum_collisions: int = 0


SMOKE_GATE = ClosedLoopGate("smoke", 5.0, 0.05, 0.30)
MULTI_TRAVERSAL_GATE = ClosedLoopGate("multi_traversal", 1.0, 0.01, 0.30)


@dataclass(frozen=True)
class SequenceGate:
    """Defaults of ``tools/cudanav_kiss_icp_sequence.cu``."""

    maximum_ate_rmse_m: float = 5.0
    maximum_final_drift_percent: float = 10.0
    minimum_inliers: int = 30


def percentile(values: Iterable[float], fraction: float) -> float:
    """Nearest-rank percentile, matching the native tools."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0.0,
        "mean": float(np.mean(values)) if values else 0.0,
    }


def trajectory_metrics(estimated_xy: Any, reference_xy: Any) -> dict[str, float]:
    """ATE (XY RMSE), final error, path lengths and final drift percent."""
    est = np.asarray(estimated_xy, dtype=np.float64).reshape(-1, 2)
    ref = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    if est.shape != ref.shape or est.shape[0] == 0:
        raise ValueError("estimated and reference trajectories must be non-empty and aligned")
    errors = np.hypot(*(est - ref).T)
    reference_distance = float(np.sum(np.hypot(*np.diff(ref, axis=0).T))) if len(ref) > 1 else 0.0
    estimated_distance = float(np.sum(np.hypot(*np.diff(est, axis=0).T))) if len(est) > 1 else 0.0
    final_error = float(errors[-1])
    return {
        "ate_rmse_m": float(math.sqrt(float(np.mean(errors * errors)))),
        "final_error_m": final_error,
        "reference_distance_m": reference_distance,
        "estimated_distance_m": estimated_distance,
        "final_drift_percent": (
            100.0 * final_error / reference_distance if reference_distance > 1.0e-6 else 0.0
        ),
    }


def missing_summary_fields(summary: Mapping[str, Any]) -> list[str]:
    return sorted(set(REQUIRED_SUMMARY_FIELDS) - set(summary))


def failure_summary(stage: str, error: BaseException | str, *, scenario: str) -> dict[str, Any]:
    """A summary that satisfies the schema while recording why a run failed."""
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "success": False,
        "smoke_pass": False,
        "collision_count": None,
        "odometry_drift_percent": None,
        "command_deadline_miss_rate": None,
        "failure": {
            "stage": stage,
            "type": type(error).__name__ if isinstance(error, BaseException) else "error",
            "message": str(error),
        },
    }


def _provenance() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "_source_provenance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "available": True,
            "source_digest": str(payload["source_digest"]),
            "source_count": len(payload["sources"]),
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def runtime_info() -> dict[str, Any]:
    """CUDA runtime/device metadata; never raises when no GPU is present."""
    try:
        from .._cudarobotics import cuda_runtime_info

        return dict(cuda_runtime_info())
    except Exception as error:  # pragma: no cover - depends on the extension
        return {"device_count": 0, "error": f"{type(error).__name__}: {error}"}


def build_manifest(
    *,
    command: Sequence[str],
    scenario: str,
    parameters: Mapping[str, Any],
    outputs: Mapping[str, str],
    cuda: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .. import __version__

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "cudarobotics.cudanav.run",
        "scenario": scenario,
        "cudarobotics_version": __version__,
        "source_provenance": _provenance(),
        "command": list(command),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cuda": dict(runtime_info() if cuda is None else cuda),
        "seeds": {"mppi_curand": MPPI_RNG_SEED, "simulation": "deterministic (no RNG)"},
        "parameters": dict(parameters),
        "inputs": dict(inputs or {}),
        "outputs": dict(outputs),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
