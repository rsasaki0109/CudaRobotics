import math

import numpy as np
import pytest

from cuda_nav_tracking.frames import grid_from_message, quaternion_matrix, transform_points


def test_quaternion_matrix_matches_a_yaw_rotation():
    yaw = 0.7
    rotation = quaternion_matrix(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    expected = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    np.testing.assert_allclose(rotation, expected, atol=1e-12)
    np.testing.assert_allclose(quaternion_matrix(0, 0, 0, 2.0), np.eye(3))  # normalised
    with pytest.raises(ValueError):
        quaternion_matrix(0, 0, 0, 0)


def test_transform_points_and_grid_layout():
    rotation = quaternion_matrix(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    out = transform_points([[1.0, 0.0, 0.5]], rotation, (2.0, 3.0, 0.0))
    np.testing.assert_allclose(out, [[2.0, 4.0, 0.5]], atol=1e-12)
    grid = grid_from_message([0, 100, -1, 0, 0, 100], width=3, height=2)
    assert grid.shape == (2, 3) and grid[0, 1] == 100 and grid[1, 2] == 100
    with pytest.raises(ValueError):
        grid_from_message([0, 0, 0], width=2, height=2)
