from ._cudarobotics import MotionModel, MppiParams, MppiResult, _MppiPlanner, __version__

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


__all__ = [
    "MotionModel",
    "MppiParams",
    "MppiPlanner",
    "MppiResult",
    "__version__",
]
