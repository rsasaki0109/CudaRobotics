"""Deterministic 2D geometry used by the CudaNav loopback simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float


def default_segments() -> tuple[Segment, ...]:
    """Return the fixed S-course walls and obstacle rectangles."""
    segments = [
        Segment(-1.0, -2.5, 10.0, -2.5),
        Segment(10.0, -2.5, 10.0, 2.5),
        Segment(10.0, 2.5, -1.0, 2.5),
        Segment(-1.0, 2.5, -1.0, -2.5),
    ]
    segments.extend(rectangle_segments(3.6, -2.5, 4.3, 0.55))
    segments.extend(rectangle_segments(6.1, -0.55, 6.8, 2.5))
    return tuple(segments)


def rectangle_segments(
    min_x: float, min_y: float, max_x: float, max_y: float
) -> tuple[Segment, ...]:
    return (
        Segment(min_x, min_y, max_x, min_y),
        Segment(max_x, min_y, max_x, max_y),
        Segment(max_x, max_y, min_x, max_y),
        Segment(min_x, max_y, min_x, min_y),
    )


def raycast(
    origin_x: float,
    origin_y: float,
    angle: float,
    segments: Iterable[Segment],
    max_range: float,
) -> float:
    """Return the nearest ray/segment intersection distance."""
    dx = math.cos(angle)
    dy = math.sin(angle)
    best = max_range
    for segment in segments:
        ex = segment.x1 - segment.x0
        ey = segment.y1 - segment.y0
        denominator = dx * ey - dy * ex
        if abs(denominator) < 1.0e-12:
            continue
        ax = segment.x0 - origin_x
        ay = segment.y0 - origin_y
        ray_distance = (ax * ey - ay * ex) / denominator
        segment_fraction = (ax * dy - ay * dx) / denominator
        if (
            ray_distance >= 0.0
            and 0.0 <= segment_fraction <= 1.0
            and ray_distance < best
        ):
            best = ray_distance
    return best


def point_segment_distance(
    x: float, y: float, segment: Segment
) -> float:
    ex = segment.x1 - segment.x0
    ey = segment.y1 - segment.y0
    squared_length = ex * ex + ey * ey
    if squared_length <= 1.0e-18:
        return math.hypot(x - segment.x0, y - segment.y0)
    fraction = (
        (x - segment.x0) * ex + (y - segment.y0) * ey
    ) / squared_length
    fraction = min(1.0, max(0.0, fraction))
    closest_x = segment.x0 + fraction * ex
    closest_y = segment.y0 + fraction * ey
    return math.hypot(x - closest_x, y - closest_y)


def collides(
    x: float, y: float, radius: float, segments: Iterable[Segment]
) -> bool:
    return any(
        point_segment_distance(x, y, segment) <= radius
        for segment in segments
    )


def mission_waypoints() -> tuple[tuple[float, float], ...]:
    return (
        (0.0, 0.0),
        (1.8, 0.0),
        (2.9, 1.4),
        (5.05, 1.4),
        (5.35, -1.4),
        (7.5, -1.4),
        (9.0, 0.0),
    )


def interpolate_polyline(
    waypoints: Sequence[tuple[float, float]], spacing: float
) -> list[tuple[float, float, float]]:
    """Sample a polyline as x, y, tangent-yaw poses."""
    if len(waypoints) < 2 or spacing <= 0.0:
        raise ValueError("at least two waypoints and positive spacing are required")
    output: list[tuple[float, float, float]] = []
    for index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            continue
        yaw = math.atan2(dy, dx)
        samples = max(1, int(math.ceil(length / spacing)))
        for sample in range(samples):
            if index > 0 and sample == 0:
                continue
            fraction = sample / samples
            output.append(
                (start[0] + fraction * dx, start[1] + fraction * dy, yaw)
            )
    final = waypoints[-1]
    previous = waypoints[-2]
    output.append(
        (
            final[0],
            final[1],
            math.atan2(final[1] - previous[1], final[0] - previous[0]),
        )
    )
    return output
