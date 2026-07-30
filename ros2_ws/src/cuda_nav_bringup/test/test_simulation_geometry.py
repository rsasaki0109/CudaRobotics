import math

import pytest

from cuda_nav_bringup.simulation_geometry import (
    Segment,
    collides,
    default_segments,
    interpolate_polyline,
    mission_waypoints,
    point_segment_distance,
    raycast,
)


def test_raycast_hits_axis_aligned_wall():
    distance = raycast(
        0.0,
        0.0,
        0.0,
        [Segment(2.0, -1.0, 2.0, 1.0)],
        10.0,
    )
    assert distance == pytest.approx(2.0)


def test_collision_and_course_clearance():
    segments = default_segments()
    assert collides(-0.80, 0.0, 0.24, segments)
    assert not collides(0.0, 0.0, 0.24, segments)
    path = interpolate_polyline(mission_waypoints(), 0.02)
    for x, y, _ in path:
        assert not collides(x, y, 0.24, segments)
    minimum_clearance = min(
        point_segment_distance(x, y, segment)
        for x, y, _ in path
        for segment in segments
    )
    # Keep a numerical margin beyond the configured 0.60 m inflation radius.
    assert minimum_clearance >= 0.80


def test_mission_polyline_is_dense_and_finite():
    samples = interpolate_polyline(mission_waypoints(), 0.12)
    assert len(samples) > 50
    assert samples[0][:2] == mission_waypoints()[0]
    assert samples[-1][:2] == mission_waypoints()[-1]
    assert all(math.isfinite(value) for sample in samples for value in sample)


def test_invalid_polyline_rejected():
    with pytest.raises(ValueError):
        interpolate_polyline([(0.0, 0.0)], 0.1)


def test_release_traversals_exceed_ten_minute_theoretical_floor():
    waypoints = mission_waypoints()
    traversal_distance = sum(
        math.dist(start, end)
        for start, end in zip(waypoints, waypoints[1:])
    )
    assert 30 * traversal_distance / 0.55 > 600.0
