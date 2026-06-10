from ._cudarobotics import (
    BcpdParams,
    FilterRegParams,
    FgrParams,
    MotionModel,
    MppiParams,
    MppiResult,
    RobustP2PlaneParams,
    RobustTregParams,
    SinkhornRegParams,
    _Bcpd,
    _FilterReg,
    _Fgr,
    _MppiPlanner,
    _RobustP2Plane,
    _RobustTreg,
    _SinkhornReg,
    __version__,
)

_MOTION_MODELS = {
    "diff": MotionModel.DiffDrive,
    "diff_drive": MotionModel.DiffDrive,
    "diffdrive": MotionModel.DiffDrive,
    "ackermann": MotionModel.Ackermann,
    "omni": MotionModel.Omni,
}


def _coerce_motion_model(value):
    if isinstance(value, MotionModel):
        return value
    key = str(value).strip().replace("-", "_").lower()
    try:
        return _MOTION_MODELS[key]
    except KeyError as exc:
        raise ValueError("motion_model must be DiffDrive, Ackermann, or Omni") from exc


class MppiPlanner:
    """GPU MPPI planner backed by the CudaRobotics CUDA rollout core."""

    def __init__(self, params=None, **kwargs):
        self.params = MppiParams() if params is None else params
        for key, value in kwargs.items():
            attr = "lambda_" if key in {"lambda", "temperature"} else key
            if attr == "motion_model":
                value = _coerce_motion_model(value)
            if not hasattr(self.params, attr):
                raise TypeError(f"unknown MppiParams field: {key}")
            setattr(self.params, attr, value)
        self._planner = _MppiPlanner(self.params)

    def reset(self):
        self._planner.reset()

    def set_speed_limit(self, v_max):
        self._planner.set_speed_limit(v_max)

    def compute(
        self,
        state,
        costmap,
        path,
        goal,
        *,
        origin=(0.0, 0.0),
        resolution=0.05,
        goal_is_final=False,
        footprint=None,
    ):
        return self._planner.compute(
            state, costmap, path, goal, origin, resolution, goal_is_final, footprint
        )


class FilterReg:
    """GPU FilterReg probabilistic point-cloud registration."""

    def __init__(self, params=None, **kwargs):
        self.params = FilterRegParams() if params is None else params
        for key, value in kwargs.items():
            if not hasattr(self.params, key):
                raise TypeError(f"unknown FilterRegParams field: {key}")
            setattr(self.params, key, value)
        self._registrar = _FilterReg(self.params)

    def register(self, target, source, init_rotation=None, init_translation=None):
        rotation, translation, info = self._registrar.register_clouds(
            target, source, init_rotation, init_translation
        )
        return rotation, translation, info


class SinkhornReg:
    """GPU unbalanced Sinkhorn optimal-transport registration."""

    def __init__(self, params=None, **kwargs):
        self.params = SinkhornRegParams() if params is None else params
        for key, value in kwargs.items():
            if not hasattr(self.params, key):
                raise TypeError(f"unknown SinkhornRegParams field: {key}")
            setattr(self.params, key, value)
        self._registrar = _SinkhornReg(self.params)

    def register(self, target, source, init_rotation=None, init_translation=None):
        return self._registrar.register_clouds(
            target, source, init_rotation, init_translation
        )


class Fgr:
    """GPU Fast Global Registration (FPFH + graduated non-convexity)."""

    def __init__(self, params=None, **kwargs):
        self.params = FgrParams() if params is None else params
        for key, value in kwargs.items():
            if not hasattr(self.params, key):
                raise TypeError(f"unknown FgrParams field: {key}")
            setattr(self.params, key, value)
        self._registrar = _Fgr(self.params)

    def register(self, target, source):
        return self._registrar.register_clouds(target, source)


class RobustTreg:
    """GPU robust Student's-t point-to-point registration."""

    def __init__(self, params=None, **kwargs):
        self.params = RobustTregParams() if params is None else params
        for key, value in kwargs.items():
            if not hasattr(self.params, key):
                raise TypeError(f"unknown RobustTregParams field: {key}")
            setattr(self.params, key, value)
        self._registrar = _RobustTreg(self.params)

    def register(self, target, source, init_rotation=None, init_translation=None):
        return self._registrar.register_clouds(
            target, source, init_rotation, init_translation
        )


class RobustP2Plane:
    """GPU robust Student's-t point-to-plane registration."""

    def __init__(self, params=None, **kwargs):
        self.params = RobustP2PlaneParams() if params is None else params
        for key, value in kwargs.items():
            if not hasattr(self.params, key):
                raise TypeError(f"unknown RobustP2PlaneParams field: {key}")
            setattr(self.params, key, value)
        self._registrar = _RobustP2Plane(self.params)

    def register(self, target, source, init_rotation=None, init_translation=None):
        return self._registrar.register_clouds(
            target, source, init_rotation, init_translation
        )


class Bcpd:
    """GPU BCPD non-rigid point-set registration."""

    def __init__(self, params=None, **kwargs):
        self.params = BcpdParams() if params is None else params
        for key, value in kwargs.items():
            attr = "lambda_" if key == "lambda" else key
            if not hasattr(self.params, attr):
                raise TypeError(f"unknown BcpdParams field: {key}")
            setattr(self.params, attr, value)
        self._registrar = _Bcpd(self.params)

    def register(self, target, source):
        deformed, info = self._registrar.register_clouds(target, source)
        return deformed, info


from . import registration

__all__ = [
    "Bcpd",
    "BcpdParams",
    "FilterReg",
    "FilterRegParams",
    "Fgr",
    "FgrParams",
    "MotionModel",
    "MppiParams",
    "MppiPlanner",
    "MppiResult",
    "RobustP2Plane",
    "RobustP2PlaneParams",
    "RobustTreg",
    "RobustTregParams",
    "SinkhornReg",
    "SinkhornRegParams",
    "__version__",
    "registration",
]
