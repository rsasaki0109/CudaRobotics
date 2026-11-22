"""ROS-free geometry helpers for the scan tracker node."""

from __future__ import annotations

import numpy as np


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix of a (normalised) quaternion."""
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if not norm > 0.0:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_points(points: np.ndarray, rotation: np.ndarray, translation) -> np.ndarray:
    """``rotation @ p + translation`` for each row of an ``(N, 3)`` array."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ np.asarray(rotation, dtype=np.float64).T + np.asarray(translation, dtype=np.float64)


def grid_from_message(data, width: int, height: int) -> np.ndarray:
    """Row-major ``(height, width)`` int8 view of ``nav_msgs/OccupancyGrid.data``."""
    grid = np.asarray(data, dtype=np.int8)
    if grid.size != width * height:
        raise ValueError("occupancy data size does not match width * height")
    return grid.reshape(height, width)
