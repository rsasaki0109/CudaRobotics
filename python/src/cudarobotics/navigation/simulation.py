"""ROS-free S-course world used by the ``cudarobotics-cudanav sim`` closed loop.

This mirrors ``tools/cudanav_gpu_closed_loop_s_course.cu`` and
``ros2_ws/src/cuda_nav_bringup/cuda_nav_bringup/simulation_geometry.py``: a
walled course with two offset obstacles, a planar LiDAR that emits three
height layers per hit, and a unicycle plant integrated at 10 Hz. It is a
synthetic, command-driven plant: useful for checking the integrated GPU stack,
not real-sensor evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

DT = 0.10
ROBOT_RADIUS = 0.24
SCAN_RAYS = 240
SCAN_MAX_RANGE = 12.0
SCAN_Z_LEVELS = (-0.45, 0.0, 0.45)
PATH_SPACING = 0.12
GOAL_TOLERANCE = 0.30

WAYPOINTS = (
    (0.0, 0.0),
    (1.8, 0.0),
    (2.9, 1.4),
    (5.05, 1.4),
    (5.35, -1.4),
    (7.5, -1.4),
    (9.0, 0.0),
)


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def _rectangle(min_x: float, min_y: float, max_x: float, max_y: float) -> list[tuple[float, float, float, float]]:
    return [
        (min_x, min_y, max_x, min_y),
        (max_x, min_y, max_x, max_y),
        (max_x, max_y, min_x, max_y),
        (min_x, max_y, min_x, min_y),
    ]


def course_segments() -> np.ndarray:
    """Wall segments ``(x0, y0, x1, y1)`` of the S-course, float32 ``(S, 4)``."""
    segments = _rectangle(-1.0, -2.5, 10.0, 2.5)
    segments += _rectangle(3.6, -2.5, 4.3, 0.55)
    segments += _rectangle(6.1, -0.55, 6.8, 2.5)
    return np.asarray(segments, dtype=np.float32)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def raycast(origin_x: float, origin_y: float, angles: np.ndarray, segments: np.ndarray, max_range: float) -> np.ndarray:
    """Distance along each ray to the first segment hit (``max_range`` on miss)."""
    angles = np.asarray(angles, dtype=np.float32)
    dx = np.cos(angles)[:, None]
    dy = np.sin(angles)[:, None]
    x0, y0, x1, y1 = (segments[:, i][None, :] for i in range(4))
    ex = x1 - x0
    ey = y1 - y0
    denominator = dx * ey - dy * ex
    ax = x0 - np.float32(origin_x)
    ay = y0 - np.float32(origin_y)
    valid = np.abs(denominator) >= np.float32(1.0e-12)
    safe = np.where(valid, denominator, np.float32(1.0))
    distance = (ax * ey - ay * ex) / safe
    fraction = (ax * dy - ay * dx) / safe
    hit = valid & (distance >= 0.0) & (fraction >= 0.0) & (fraction <= 1.0)
    distance = np.where(hit, distance, np.float32(max_range))
    return np.minimum(distance.min(axis=1), np.float32(max_range))


def point_segment_distance(x: float, y: float, segments: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = (segments[:, i].astype(np.float64) for i in range(4))
    ex = x1 - x0
    ey = y1 - y0
    length_squared = ex * ex + ey * ey
    fraction = np.clip(
        ((x - x0) * ex + (y - y0) * ey) / np.where(length_squared > 1.0e-18, length_squared, 1.0),
        0.0,
        1.0,
    )
    fraction = np.where(length_squared > 1.0e-18, fraction, 0.0)
    return np.hypot(x - (x0 + fraction * ex), y - (y0 + fraction * ey))


def collides(x: float, y: float, segments: np.ndarray, radius: float = ROBOT_RADIUS) -> bool:
    return bool(np.any(point_segment_distance(x, y, segments) <= radius))


def simulate_scan(truth: Pose2D, segments: np.ndarray) -> np.ndarray:
    """Sensor-frame ``(N, 3)`` float32 points for a 240-ray planar LiDAR."""
    local = (
        np.float32(-math.pi)
        + np.float32(2.0 * math.pi) * np.arange(SCAN_RAYS, dtype=np.float32) / np.float32(SCAN_RAYS)
    )
    distance = raycast(truth.x, truth.y, np.float32(truth.yaw) + local, segments, SCAN_MAX_RANGE)
    hit = distance < np.float32(SCAN_MAX_RANGE)
    local = local[hit]
    distance = distance[hit]
    xs = distance * np.cos(local)
    ys = distance * np.sin(local)
    levels = np.asarray(SCAN_Z_LEVELS, dtype=np.float32)
    points = np.empty((xs.size, levels.size, 3), dtype=np.float32)
    points[:, :, 0] = xs[:, None]
    points[:, :, 1] = ys[:, None]
    points[:, :, 2] = levels[None, :]
    return points.reshape(-1, 3)


def interpolate_path(spacing: float = PATH_SPACING, reverse: bool = False) -> np.ndarray:
    """Mission polyline sampled every ``spacing`` metres, float32 ``(N, 2)``."""
    waypoints = list(reversed(WAYPOINTS)) if reverse else list(WAYPOINTS)
    path: list[tuple[float, float]] = []
    for index in range(len(waypoints) - 1):
        (ax, ay), (bx, by) = waypoints[index], waypoints[index + 1]
        dx, dy = bx - ax, by - ay
        samples = max(1, math.ceil(math.hypot(dx, dy) / spacing))
        for sample in range(samples):
            if index > 0 and sample == 0:
                continue
            fraction = sample / samples
            path.append((ax + fraction * dx, ay + fraction * dy))
    path.append(waypoints[-1])
    return np.asarray(path, dtype=np.float32)


def step_unicycle(pose: Pose2D, v: float, w: float, dt: float = DT) -> Pose2D:
    """Midpoint-yaw unicycle integration."""
    middle_yaw = pose.yaw + 0.5 * w * dt
    return Pose2D(
        pose.x + v * math.cos(middle_yaw) * dt,
        pose.y + v * math.sin(middle_yaw) * dt,
        wrap_angle(pose.yaw + w * dt),
    )


# --- moving obstacles -------------------------------------------------------

OBSTACLE_POLYGON_SIDES = 12


@dataclass(frozen=True)
class MovingObstacle:
    """A disc that walks back and forth between ``start`` and ``end``.

    The motion is a deterministic triangle wave: it starts at ``start`` shifted
    by ``phase`` seconds along the schedule and never reacts to the robot.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    speed: float
    radius: float = 0.25
    phase: float = 0.0

    def state(self, time_s: float) -> tuple[float, float, float, float]:
        """World ``(x, y, vx, vy)`` at ``time_s``."""
        (ax, ay), (bx, by) = self.start, self.end
        length = math.hypot(bx - ax, by - ay)
        if length <= 1.0e-9 or self.speed <= 0.0:
            return ax, ay, 0.0, 0.0
        ux, uy = (bx - ax) / length, (by - ay) / length
        travelled = (time_s + self.phase) * self.speed % (2.0 * length)
        if travelled <= length:
            return ax + ux * travelled, ay + uy * travelled, ux * self.speed, uy * self.speed
        back = travelled - length
        return bx - ux * back, by - uy * back, -ux * self.speed, -uy * self.speed


def crossing_obstacles(phases: tuple[float, float] = (0.0, 0.0)) -> tuple[MovingObstacle, ...]:
    """Two pedestrians sweeping across the S-course's two corridors.

    One crosses the upper corridor beside the first block, the other the lower
    corridor beside the second block, so the robot must time both passages.
    """
    return (
        MovingObstacle(start=(4.0, 0.80), end=(4.0, 2.20), speed=0.5, radius=0.25, phase=phases[0]),
        MovingObstacle(start=(6.45, -2.20), end=(6.45, -0.80), speed=0.4, radius=0.25, phase=phases[1]),
    )


def obstacle_states(obstacles, time_s: float) -> np.ndarray:
    """``(N, 5)`` float32 rows of ``x, y, vx, vy, radius`` at ``time_s``."""
    rows = [(*obstacle.state(time_s), obstacle.radius) for obstacle in obstacles]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def obstacle_segments(states: np.ndarray, sides: int = OBSTACLE_POLYGON_SIDES) -> np.ndarray:
    """Circumscribed polygons of the obstacle discs as ``(N * sides, 4)`` segments."""
    states = np.asarray(states, dtype=np.float32).reshape(-1, 5)
    if states.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * math.pi, sides + 1, dtype=np.float64)
    # Circumscribe so the polygon's edges are not inside the true disc.
    scale = 1.0 / math.cos(math.pi / sides)
    segments = []
    for x, y, _vx, _vy, radius in states.astype(np.float64):
        px = x + radius * scale * np.cos(angles)
        py = y + radius * scale * np.sin(angles)
        segments.append(np.stack([px[:-1], py[:-1], px[1:], py[1:]], axis=1))
    return np.concatenate(segments).astype(np.float32)


def raycast_with_source(
    origin_x: float, origin_y: float, angles: np.ndarray, segments: np.ndarray, max_range: float
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`raycast`, also returning the index of the segment each ray hit (-1 on miss)."""
    angles = np.asarray(angles, dtype=np.float32)
    dx = np.cos(angles)[:, None]
    dy = np.sin(angles)[:, None]
    x0, y0, x1, y1 = (segments[:, i][None, :] for i in range(4))
    ex = x1 - x0
    ey = y1 - y0
    denominator = dx * ey - dy * ex
    ax = x0 - np.float32(origin_x)
    ay = y0 - np.float32(origin_y)
    valid = np.abs(denominator) >= np.float32(1.0e-12)
    safe = np.where(valid, denominator, np.float32(1.0))
    distance = (ax * ey - ay * ex) / safe
    fraction = (ax * dy - ay * dx) / safe
    hit = valid & (distance >= 0.0) & (fraction >= 0.0) & (fraction <= 1.0)
    distance = np.where(hit, distance, np.float32(max_range))
    index = distance.argmin(axis=1)
    nearest = distance[np.arange(distance.shape[0]), index]
    index = np.where(nearest < np.float32(max_range), index, -1)
    return np.minimum(nearest, np.float32(max_range)), index


def simulate_scan_with_obstacles(
    truth: Pose2D, segments: np.ndarray, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sensor-frame points as :func:`simulate_scan`, with moving obstacles.

    Returns ``(points, dynamic)`` where ``dynamic`` is a boolean mask marking
    the points that hit a moving obstacle (what a perfect segmentation would
    remove before mapping).
    """
    dynamic_segments = obstacle_segments(states)
    combined = np.concatenate([segments, dynamic_segments]).astype(np.float32)
    local = (
        np.float32(-math.pi)
        + np.float32(2.0 * math.pi) * np.arange(SCAN_RAYS, dtype=np.float32) / np.float32(SCAN_RAYS)
    )
    distance, source = raycast_with_source(
        truth.x, truth.y, np.float32(truth.yaw) + local, combined, SCAN_MAX_RANGE
    )
    hit = distance < np.float32(SCAN_MAX_RANGE)
    local = local[hit]
    distance = distance[hit]
    on_obstacle = source[hit] >= segments.shape[0]
    xs = distance * np.cos(local)
    ys = distance * np.sin(local)
    levels = np.asarray(SCAN_Z_LEVELS, dtype=np.float32)
    points = np.empty((xs.size, levels.size, 3), dtype=np.float32)
    points[:, :, 0] = xs[:, None]
    points[:, :, 1] = ys[:, None]
    points[:, :, 2] = levels[None, :]
    return points.reshape(-1, 3), np.repeat(on_obstacle, levels.size)


def dynamic_clearance(x: float, y: float, states: np.ndarray, radius: float = ROBOT_RADIUS) -> float:
    """Smallest surface-to-surface distance from the robot to any moving obstacle."""
    states = np.asarray(states, dtype=np.float64).reshape(-1, 5)
    if states.shape[0] == 0:
        return math.inf
    gaps = np.hypot(states[:, 0] - x, states[:, 1] - y) - states[:, 4] - radius
    return float(gaps.min())
