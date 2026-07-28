"""CPU-reference versus GPU registration consistency tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cudarobotics import registration


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rigid transform mapping paired ``source`` rows to ``target``."""
    source_mean = source.mean(axis=0, dtype=np.float64)
    target_mean = target.mean(axis=0, dtype=np.float64)
    covariance = (
        (source.astype(np.float64) - source_mean).T
        @ (target.astype(np.float64) - target_mean)
    )
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    delta = actual @ expected.T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def paired_clouds() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260728)
    target = rng.normal(size=(384, 3)).astype(np.float32)
    target[:, 0] *= 1.8
    target[:, 1] += 0.2 * target[:, 0] ** 2
    angle = 0.09
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation = np.array([0.08, -0.05, 0.03], dtype=np.float32)
    source = target @ rotation.T + translation
    return target, source.astype(np.float32)


@pytest.mark.parametrize(
    ("registrar_type", "translation_tolerance_m"),
    [
        (registration.FilterReg, 0.03),
        (registration.RobustTreg, 0.03),
        # Point-to-plane has a weaker tangential translation constraint.
        (registration.RobustP2Plane, 0.06),
        (registration.SinkhornReg, 0.03),
    ],
)
def test_gpu_registration_matches_cpu_rigid_reference(
    registrar_type, translation_tolerance_m
):
    target, source = paired_clouds()
    cpu_rotation, cpu_translation = kabsch(source, target)
    cpu_aligned = source @ cpu_rotation.T + cpu_translation
    assert np.sqrt(np.mean((cpu_aligned - target) ** 2)) < 1e-5

    try:
        actual_rotation, actual_translation, info = registrar_type().register(
            target,
            source,
            init_rotation=cpu_rotation.astype(np.float32),
            init_translation=cpu_translation.astype(np.float32),
        )
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "device" in str(exc):
            pytest.skip("CUDA GPU not available in this environment")
        raise

    actual_rotation = np.asarray(actual_rotation, dtype=np.float64).reshape(3, 3)
    actual_translation = np.asarray(actual_translation, dtype=np.float64)
    assert rotation_error_deg(actual_rotation, cpu_rotation) < 1.0
    assert (
        np.linalg.norm(actual_translation - cpu_translation)
        < translation_tolerance_m
    )
    assert np.isfinite(float(info["final_rmse"]))
