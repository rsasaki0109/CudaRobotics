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
        planner = cr.MppiPlanner(
            batch_size=64,
            time_steps=8,
            model_dt=0.05,
            path_angle_weight=0.25,
            curvature_speed_weight=8.0,
            curvature_speed_min=0.18,
            distance_field_weight=1.0,
            distance_field_cutoff=0.5,
        )
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
    assert {
        "best_cost",
        "mean_cost",
        "sampled_rollouts",
        "valid_rollouts",
        "valid_rollout_ratio",
        "all_colliding",
        "retreating",
    }.issubset(info)


def test_mppi_planner_cuda_dlpack_costmap_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available for torch DLPack smoke test")
    try:
        planner = cr.MppiPlanner(
            batch_size=64,
            time_steps=8,
            model_dt=0.05,
            path_angle_weight=0.25,
            curvature_speed_weight=8.0,
            curvature_speed_min=0.18,
            distance_field_weight=1.0,
            distance_field_cutoff=0.5,
        )
    except RuntimeError as exc:
        if "no CUDA-capable device" in str(exc):
            pytest.skip("CUDA GPU not available in this environment")
        raise
    costmap = torch.zeros((40, 40), dtype=torch.uint8, device="cuda")
    path = np.array([[1.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    v, vy, w, info = planner.compute(
        (1.0, 5.0, 0.0), costmap, path, (5.0, 5.0, 0.0), resolution=0.05
    )
    assert isinstance(v, float)
    assert isinstance(vy, float)
    assert isinstance(w, float)
    assert isinstance(info, dict)
    assert "valid_rollout_ratio" in info


def test_cuda_array_prefers_dlpack_over_buffer_facade():
    helper = getattr(cr, "_prefer_cuda_dlpack", None)
    if helper is None:
        pytest.skip("installed package predates CUDA DLPack preference helper")

    class FakeCudaArray:
        __cuda_array_interface__ = {"shape": (2, 2)}

        def __dlpack__(self, *args, **kwargs):
            return (args, kwargs)

        def __dlpack_device__(self):
            return (2, 0)

    wrapped = helper(FakeCudaArray())
    assert not hasattr(wrapped, "__cuda_array_interface__")
    assert wrapped.__dlpack__(stream=7) == ((), {"stream": 7})
    assert wrapped.__dlpack_device__() == (2, 0)


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


def test_registration_result_contract():
    result = registration.RegistrationResult(
        rotation=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
        info={"iterations": 0},
    )
    assert result.rotation.shape == (3, 3)
    assert result.translation.shape == (3,)
    assert result.info["iterations"] == 0
    with pytest.raises(Exception):
        result.translation = np.ones(3, dtype=np.float32)
