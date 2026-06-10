"""Smoke tests for the cudarobotics Python package."""

import numpy as np
import pytest

import cudarobotics as cr
from cudarobotics import registration


def test_version():
    assert isinstance(cr.__version__, str)
    assert cr.__version__


def test_mppi_planner_smoke():
    try:
        planner = cr.MppiPlanner(batch_size=64, time_steps=8, model_dt=0.05)
    except RuntimeError as exc:
        if "no CUDA-capable device" in str(exc):
            pytest.skip("CUDA GPU not available in this environment")
        raise
    costmap = np.zeros((40, 40), dtype=np.uint8)
    path = np.array([[1.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    v, vy, w, info = planner.compute(
        (1.0, 5.0, 0.0), costmap, path, (5.0, 5.0, 0.0), resolution=0.05
    )
    assert isinstance(v, float)
    assert isinstance(vy, float)
    assert isinstance(w, float)
    assert isinstance(info, dict)


@pytest.mark.parametrize(
    "cls",
    [
        registration.FilterReg,
        registration.SinkhornReg,
        registration.Fgr,
        registration.Bcpd,
        registration.RobustTreg,
        registration.RobustP2Plane,
    ],
)
def test_registration_constructors(cls):
    assert cls() is not None
