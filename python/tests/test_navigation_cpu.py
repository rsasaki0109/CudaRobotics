"""CudaNav Python API checks that need the extension but no GPU."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from cudarobotics.navigation import (
    OccupancyGrid,
    clear_footprint,
    costmap_from_esdf,
    esdf_2d_cpu_reference,
    local_path_window,
    transform_points,
)
from cudarobotics.navigation import cli, evidence
from cudarobotics.navigation.core import KissIcpOdometry, as_points
from cudarobotics.navigation.sequence import (
    MAGIC,
    SequenceFormatError,
    SequenceFrame,
    iter_sequence,
    write_sequence,
)
from cudarobotics.navigation.simulation import (
    Pose2D,
    collides,
    course_segments,
    interpolate_path,
    raycast,
    simulate_scan,
    step_unicycle,
)


def test_raycast_hits_course_walls():
    segments = course_segments()
    angles = np.asarray([0.0, math.pi / 2, math.pi, -math.pi / 2], dtype=np.float32)
    distances = raycast(0.0, 0.0, angles, segments, 12.0)
    np.testing.assert_allclose(distances, [3.6, 2.5, 1.0, 2.5], atol=1e-5)
    assert raycast(0.0, 0.0, np.asarray([0.0]), segments, 2.0)[0] == pytest.approx(2.0)


def test_collision_uses_robot_radius():
    segments = course_segments()
    assert not collides(0.0, 0.0, segments)
    assert collides(3.45, 0.0, segments)
    assert collides(0.0, 2.3, segments)


def test_simulated_scan_layers_and_frame():
    points = simulate_scan(Pose2D(0.0, 0.0, 0.0), course_segments())
    assert points.dtype == np.float32 and points.shape[1] == 3
    assert points.shape[0] % 3 == 0 and points.shape[0] > 300
    np.testing.assert_allclose(sorted(set(points[:, 2].tolist())), [-0.45, 0.0, 0.45], atol=1e-6)
    ranges = np.hypot(points[:, 0], points[:, 1])
    assert ranges.max() < 12.0


def test_mission_path_and_unicycle():
    forward = interpolate_path()
    reverse = interpolate_path(reverse=True)
    np.testing.assert_allclose(forward[0], [0.0, 0.0])
    np.testing.assert_allclose(forward[-1], [9.0, 0.0])
    np.testing.assert_allclose(reverse[0], [9.0, 0.0])
    steps = np.hypot(*np.diff(forward, axis=0).T)
    assert np.median(steps) <= 0.12 + 1e-6
    # Like the native sampler, interior waypoints themselves are not emitted.
    assert steps.max() <= 2 * 0.12 + 1e-6
    moved = step_unicycle(Pose2D(), 0.5, 0.0, 0.1)
    assert moved.x == pytest.approx(0.05) and moved.y == pytest.approx(0.0)


def test_local_path_window_matches_native_contract():
    path = np.column_stack([np.arange(0.0, 10.0, 0.1), np.zeros(100)]).astype(np.float32)
    local = local_path_window(path, 1.02, 0.3)
    assert local.xy[0, 0] == pytest.approx(1.0)
    assert local.xy[-1, 0] - local.xy[0, 0] == pytest.approx(3.0, abs=0.11)
    assert local.goal[2] == pytest.approx(0.0)
    assert not local.goal_is_final
    tail = local_path_window(path, 9.85, 0.0)
    assert tail.goal_is_final and tail.goal[0] == pytest.approx(9.9)
    with pytest.raises(ValueError):
        local_path_window(path[:1], 0.0, 0.0)


def test_costmap_bands():
    occupancy = np.asarray([[-1, 100, 0, 0, 0, 0, 0, 49]], dtype=np.int8)
    distances = np.asarray([[0.0, 0.0, 0.2, 0.25, 0.5, 0.99, 1.0, 1.5]], dtype=np.float32)
    costmap = costmap_from_esdf(occupancy, distances)
    expected = [255, 254, 253, 253, int(252 * math.exp(-0.75)), int(252 * math.exp(-3 * 0.74)), 0, 0]
    assert costmap.dtype == np.uint8
    assert costmap[0].tolist() == expected
    with pytest.raises(ValueError):
        costmap_from_esdf(occupancy, distances[:, :4])


def test_clear_footprint_only_touches_the_disc():
    grid = OccupancyGrid(np.full((20, 20), 100, dtype=np.int8), 0.1, (-1.0, -1.0))
    cleared = clear_footprint(grid, 0.0, 0.0, 0.30)
    assert grid.data.min() == 100  # input is not modified
    freed = cleared.data == 0
    assert 20 <= freed.sum() <= 32
    xs, ys = cleared.cell_centers()
    assert np.all(np.hypot(xs[freed], ys[freed]) <= 0.30 + 1e-6)


def test_esdf_cpu_reference_distances_and_policies():
    occupancy = np.zeros((9, 9), dtype=np.int8)
    occupancy[4, 4] = 100
    occupancy[0, 0] = -1
    field = esdf_2d_cpu_reference(occupancy, resolution=0.1, max_distance=0.35, unknown_policy="free")
    assert field.distances.shape == (9, 9) and field.distances.dtype == np.float32
    assert field.distances[4, 4] == pytest.approx(0.0)
    assert field.distances[4, 5] == pytest.approx(0.1)
    assert field.distances[5, 5] == pytest.approx(math.sqrt(2) * 0.1, rel=1e-5)
    assert field.distances[0, 8] == pytest.approx(0.35)  # clipped
    assert field.occupied_cells == 1 and field.unknown_cells == 1
    occupied_unknown = esdf_2d_cpu_reference(occupancy, resolution=0.1, max_distance=0.35)
    assert occupied_unknown.distances[0, 0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        esdf_2d_cpu_reference(occupancy, resolution=0.1, unknown_policy="maybe")
    with pytest.raises(ValueError):
        esdf_2d_cpu_reference(occupancy)  # raw arrays need a resolution


def test_point_validation_and_transform():
    with pytest.raises(ValueError):
        as_points(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        as_points([[0.0, np.nan, 0.0]])
    rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    moved = transform_points([[1.0, 0.0, 0.0]], rotation, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(moved, [[1.0, 3.0, 3.0]], atol=1e-6)


def test_unknown_config_field_is_rejected_before_gpu_allocation():
    with pytest.raises(TypeError, match="unknown KissIcpConfig field"):
        KissIcpOdometry(not_a_field=1)


def test_invalid_stage_chain_is_rejected_before_gpu_allocation():
    from cudarobotics.navigation import CudaNavPipeline

    with pytest.raises(ValueError, match="stage_chain"):
        CudaNavPipeline(stage_chain="gpu")


def _frames(count, version):
    rng = np.random.default_rng(7)
    for index in range(count):
        points = rng.normal(size=(40, 3)).astype(np.float32)
        times = np.linspace(0.0, 0.1, 40, dtype=np.float32) if version == 2 else None
        yield SequenceFrame(index, 1000 + index, (0.1 * index, 0.0, 0.0, 0.0), points, times)


@pytest.mark.parametrize("version", [1, 2])
def test_sequence_roundtrip(tmp_path, version):
    path = tmp_path / "seq.crkicp"
    assert write_sequence(path, _frames(3, version), version=version) == 3
    frames = list(iter_sequence(path))
    assert [frame.stamp_ns for frame in frames] == [1000, 1001, 1002]
    assert frames[2].reference[0] == pytest.approx(0.2)
    assert frames[0].points.shape == (40, 3) and frames[0].points.dtype == np.float32
    if version == 2:
        assert frames[0].point_times.shape == (40,)
        assert frames[0].scan_end == pytest.approx(0.1)
    else:
        assert frames[0].point_times is None
    assert len(list(iter_sequence(path, maximum_frames=2))) == 2


def test_sequence_rejects_malformed_files(tmp_path):
    good = tmp_path / "good.crkicp"
    write_sequence(good, _frames(2, 1))
    data = good.read_bytes()
    cases = {
        "magic": b"NOTMAGIC" + data[8:],
        "truncated": data[:-5],
        "trailing": data + b"\0",
        "header": MAGIC + struct.pack("<II", 3, 2) + data[16:],
    }
    for name, payload in cases.items():
        path = tmp_path / f"{name}.crkicp"
        path.write_bytes(payload)
        with pytest.raises(SequenceFormatError):
            list(iter_sequence(path))
    with pytest.raises(ValueError):
        write_sequence(tmp_path / "few.crkicp", [SequenceFrame(0, 0, (0, 0, 0, 0), np.zeros((5, 3)))])


def test_trajectory_metrics_and_percentile():
    reference = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    estimated = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.2)]
    metrics = evidence.trajectory_metrics(estimated, reference)
    assert metrics["reference_distance_m"] == pytest.approx(2.0)
    assert metrics["final_error_m"] == pytest.approx(0.2)
    assert metrics["final_drift_percent"] == pytest.approx(10.0)
    assert metrics["ate_rmse_m"] == pytest.approx(math.sqrt((0.01 + 0.04) / 3))
    assert evidence.percentile([5, 1, 3, 2, 4], 0.95) == 5
    assert evidence.percentile([], 0.5) == 0.0
    with pytest.raises(ValueError):
        evidence.trajectory_metrics(estimated, reference[:2])


def test_failure_summary_satisfies_the_docker_contract():
    summary = evidence.failure_summary("run", RuntimeError("boom"), scenario="cudanav_s_course")
    assert evidence.missing_summary_fields(summary) == []
    assert summary["success"] is False and summary["smoke_pass"] is False
    assert summary["failure"] == {"stage": "run", "type": "RuntimeError", "message": "boom"}


def test_manifest_records_version_seed_and_cuda():
    manifest = evidence.build_manifest(
        command=["cudarobotics-cudanav", "sim"],
        scenario="cudanav_s_course",
        parameters={"traversals": 1},
        outputs={"summary": "summary.json"},
        cuda={"device_count": 0},
    )
    import cudarobotics

    assert manifest["cudarobotics_version"] == cudarobotics.__version__
    assert manifest["seeds"]["mppi_curand"] == 42
    assert manifest["source_provenance"]["available"] is True
    assert manifest["source_provenance"]["source_count"] > 0
    json.dumps(manifest, allow_nan=False)
    info = evidence.runtime_info()
    assert "device_count" in info


def test_cli_usage_errors(tmp_path, capsys):
    assert cli.main(["sim"]) == cli.EXIT_USAGE
    assert cli.main(["sequence", str(tmp_path / "missing.crkicp"), "--output-dir", str(tmp_path)]) == cli.EXIT_USAGE
    assert cli.main(["--help"]) == cli.EXIT_OK
    assert cli.main(["sim", "--output-dir", str(tmp_path), "--stage-chain", "gpu"]) == cli.EXIT_USAGE
    capsys.readouterr()


def test_cli_records_missing_gpu_as_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evidence, "runtime_info", lambda: {"device_count": 0, "error": "no CUDA-capable device"})
    output = tmp_path / "out"
    assert cli.main(["sim", "--output-dir", str(output)]) == cli.EXIT_FAILED
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failure"]["stage"] == "initialize"
    assert "no CUDA-capable" in summary["failure"]["message"]
    assert evidence.missing_summary_fields(summary) == []
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"][:2] == ["cudarobotics-cudanav", "sim"]
    capsys.readouterr()


def test_cli_records_malformed_sequence_as_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evidence, "runtime_info", lambda: {"device_count": 1})
    bad = tmp_path / "bad.crkicp"
    bad.write_bytes(b"NOTMAGIC" + b"\0" * 16)
    output = tmp_path / "out"
    assert cli.main(["sequence", str(bad), "--output-dir", str(output)]) == cli.EXIT_FAILED
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failure"]["stage"] == "run"
    assert summary["failure"]["type"] == "SequenceFormatError"
    capsys.readouterr()


# --- moving obstacles (crossing scenario) -----------------------------------


def test_moving_obstacle_walks_a_triangle_wave():
    from cudarobotics.navigation.simulation import MovingObstacle

    walker = MovingObstacle(start=(0.0, 0.0), end=(2.0, 0.0), speed=0.5, radius=0.25)
    assert walker.state(0.0) == pytest.approx((0.0, 0.0, 0.5, 0.0))
    assert walker.state(2.0) == pytest.approx((1.0, 0.0, 0.5, 0.0))
    assert walker.state(6.0) == pytest.approx((1.0, 0.0, -0.5, 0.0))  # on the way back
    assert walker.state(8.0)[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    shifted = MovingObstacle(start=(0.0, 0.0), end=(2.0, 0.0), speed=0.5, phase=2.0)
    assert shifted.state(0.0) == pytest.approx(walker.state(2.0))


def test_obstacles_are_seen_by_the_scan_and_masked():
    from cudarobotics.navigation.simulation import (
        obstacle_segments,
        simulate_scan_with_obstacles,
    )

    segments = course_segments()
    states = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.25]], dtype=np.float32)
    assert obstacle_segments(states).shape == (12, 4)
    points, dynamic = simulate_scan_with_obstacles(Pose2D(), segments, states)
    plain = simulate_scan(Pose2D(), segments)
    assert points.shape == plain.shape and dynamic.shape == (points.shape[0],)
    assert 0 < dynamic.sum() < points.shape[0]
    ahead = points[dynamic]
    # Hits lie on the circumscribed 12-gon of the disc centred 1.0 m ahead.
    from_centre = np.hypot(ahead[:, 0] - 1.0, ahead[:, 1])
    assert np.all((from_centre > 0.249) & (from_centre < 0.25 / math.cos(math.pi / 12) + 1e-3))
    assert np.all(ahead[:, 0] > 0.0)
    empty_points, empty_mask = simulate_scan_with_obstacles(
        Pose2D(), segments, np.zeros((0, 5), dtype=np.float32)
    )
    np.testing.assert_array_equal(empty_points, plain)
    assert not empty_mask.any()


def test_dynamic_clearance_and_phases_are_deterministic():
    from cudarobotics.navigation.dynamic_scenario import crossing_phases
    from cudarobotics.navigation.simulation import dynamic_clearance

    states = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.25]], dtype=np.float32)
    assert dynamic_clearance(0.0, 0.0, states) == pytest.approx(1.0 - 0.25 - 0.24)
    assert math.isinf(dynamic_clearance(0.0, 0.0, np.zeros((0, 5))))
    assert crossing_phases(3) == crossing_phases(3)
    assert crossing_phases(3) != crossing_phases(4)


def test_observation_model_delays_and_perturbs_deterministically():
    from cudarobotics.navigation.dynamic_scenario import ObservationModel, observe

    history = [np.full((2, 5), float(index), dtype=np.float32) for index in range(5)]
    exact = observe(history, ObservationModel(), np.random.default_rng(0))
    np.testing.assert_array_equal(exact, history[-1])
    delayed = observe(history, ObservationModel(latency_frames=2), np.random.default_rng(0))
    np.testing.assert_array_equal(delayed, history[2])
    noisy = ObservationModel(position_noise_m=0.1, velocity_noise_mps=0.2)
    first = observe(history, noisy, np.random.default_rng(7))
    second = observe(history, noisy, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:, 4] == history[-1][:, 4])  # radius is not perturbed
    dropped = observe(history, ObservationModel(dropout_probability=0.99), np.random.default_rng(1))
    assert dropped.shape[0] <= 2
    with pytest.raises(ValueError):
        ObservationModel(latency_frames=-1)
    with pytest.raises(ValueError):
        ObservationModel(dropout_probability=1.0)


def test_pedestrians_yield_instead_of_walking_into_the_robot():
    from cudarobotics.navigation.dynamic_scenario import PedestrianClock
    from cudarobotics.navigation.simulation import MovingObstacle

    walker = MovingObstacle(start=(0.0, 0.0), end=(4.0, 0.0), speed=1.0, radius=0.25)
    clock = PedestrianClock([walker])
    robot = Pose2D(1.2, 0.0, 0.0)
    for _ in range(20):
        clock.advance(robot, 0.1)
    x = clock.states()[0, 0]
    assert x < robot.x - 0.25 - 0.24  # stopped short of the robot
    assert clock.paused[0] and np.all(clock.states()[0, 2:4] == 0.0)
    clock.advance(Pose2D(3.5, 3.0, 0.0), 0.1)  # robot moved away: walking resumes
    assert not clock.paused[0]


def test_impatient_pedestrians_turn_around():
    from cudarobotics.navigation.dynamic_scenario import PedestrianClock
    from cudarobotics.navigation.simulation import MovingObstacle

    walker = MovingObstacle(start=(0.0, 0.0), end=(4.0, 0.0), speed=1.0, radius=0.25)
    clock = PedestrianClock([walker], patience_s=1.0)
    robot = Pose2D(1.2, 0.0, 0.0)
    for _ in range(20):  # walks up to the robot, then waits 1 s
        clock.advance(robot, 0.1)
    assert clock.turnarounds == 1 and clock.direction[0] == -1.0
    before = clock.states()[0, 0]
    for _ in range(2):
        clock.advance(robot, 0.1)
    after = clock.states()
    assert after[0, 0] < before and after[0, 2] < 0.0  # walking away, velocity reversed


def test_polyline_walker_and_layouts():
    from cudarobotics.navigation.dynamic_scenario import LAYOUTS, layout_obstacles
    from cudarobotics.navigation.simulation import PolylineWalker, crowd_obstacles

    walker = PolylineWalker(points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), speed=0.5)
    assert walker.state(1.0) == pytest.approx((0.5, 0.0, 0.5, 0.0))
    assert walker.state(3.0) == pytest.approx((1.0, 0.5, 0.0, 0.5))
    assert walker.state(5.0) == pytest.approx((1.0, 0.5, 0.0, -0.5))  # on the way back
    segments = course_segments()
    crowd = crowd_obstacles(4)
    assert len(crowd) == 5 and crowd == crowd_obstacles(4)
    for obstacle in crowd:
        for t in np.linspace(0.0, 1.0, 20):
            x = obstacle.start[0] + t * (obstacle.end[0] - obstacle.start[0])
            y = obstacle.start[1] + t * (obstacle.end[1] - obstacle.start[1])
            assert not collides(x, y, segments, obstacle.radius)
    counts = {layout: len(layout_obstacles(layout, 0)[0]) for layout in LAYOUTS}
    assert counts == {"crossing": 2, "head_on": 2, "overtake": 3, "crowd": 5}
    assert math.isinf(layout_obstacles("crossing", 0)[1])
    with pytest.raises(ValueError):
        layout_obstacles("stampede", 0)


def test_grid_hashed_pairs_match_brute_force():
    from cudarobotics.navigation.tracking import _close_pairs, _clusters

    rng = np.random.default_rng(3)
    for size in (0, 1, 5, 200):
        xy = rng.uniform(-2.0, 3.0, size=(size, 2)).astype(np.float32)
        # Some points exactly one distance apart on the grid and duplicates.
        xy = np.concatenate([xy, np.round(xy[: size // 2] / 0.3) * np.float32(0.3), xy[:3]])
        first, second = _close_pairs(xy, 0.3)
        found = {(int(i), int(j)) for i, j in zip(first, second)}
        delta = xy[:, None, :] - xy[None, :, :]
        close = (delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]) <= np.float32(0.09)
        expected = {(i, j) for i, j in zip(*np.nonzero(np.triu(close, k=1)))}
        assert found == expected
        components = _clusters(xy, 0.3)
        assert sorted(np.concatenate(components).tolist()) == list(range(xy.shape[0])) if xy.size else components == []
        assert [int(part[0]) for part in components] == sorted(int(part[0]) for part in components)


def test_clearing_discs_spare_long_standing_structure():
    from cudarobotics.navigation.pipeline import STRUCTURE_PROTECT_FRAMES, _spare_structure
    from cudarobotics.navigation.tracking import OccupancyAge

    data = np.zeros((40, 40), dtype=np.int8)
    data[:, 30] = 100  # a wall at x = 1.525
    age = OccupancyAge()
    grid = OccupancyGrid(data=data, resolution=0.05, origin=(0.0, 0.0))
    disc = [(1.2, 1.0, 0.4)]
    for _ in range(STRUCTURE_PROTECT_FRAMES - 1):
        age.update(grid)
    assert _spare_structure(disc, age) == disc  # not structure yet
    age.update(grid)
    (x, y, radius), = _spare_structure(disc, age)
    assert (x, y) == (1.2, 1.0) and radius < 1.525 - 1.2
    far = [(0.3, 1.0, 0.4)]
    assert _spare_structure(far, age) == far


def test_crossing_cli_usage_errors(tmp_path, capsys):
    base = ["sim", "--output-dir", str(tmp_path), "--scenario", "crossing"]
    assert cli.main(base + ["--layout", "stampede"]) == cli.EXIT_USAGE
    assert cli.main(base + ["--dynamic-mode", "oracle"]) == cli.EXIT_USAGE
    assert cli.main(base + ["--obs-dropout", "1.5"]) == cli.EXIT_USAGE
    assert cli.main(base + ["--obs-latency-frames", "-1"]) == cli.EXIT_USAGE
    capsys.readouterr()


# --- scan-based moving-object tracker ----------------------------------------


def _blob(cx, cy, radius=0.25, rays=9, sensor=(0.0, 0.0)):
    """Near-side arc of a disc as seen from ``sensor`` (three height layers)."""
    facing = math.atan2(sensor[1] - cy, sensor[0] - cx)
    angles = facing + np.linspace(-1.2, 1.2, rays)
    xy = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    return np.repeat(xy, 3, axis=0)


def test_tracker_confirms_a_walking_object_with_its_velocity():
    from cudarobotics.navigation.tracking import ScanObjectTracker

    tracker = ScanObjectTracker()
    for frame in range(12):
        xy = _blob(3.0 + 0.05 * frame, 1.0)
        result = tracker.update(xy, np.ones(xy.shape[0], bool), 0.1 * frame, sensor_xy=(0.0, 0.0))
    assert result.tracks.shape == (1, 5)
    x, y, vx, vy, radius = result.tracks[0]
    assert x == pytest.approx(3.55, abs=0.08) and y == pytest.approx(1.0, abs=0.08)
    assert vx == pytest.approx(0.5, abs=0.15) and abs(vy) < 0.15
    assert 0.2 <= radius <= 0.35
    assert result.dynamic_mask.all()
    assert result.history.shape[1] == 3 and result.history.shape[0] >= 1


def test_tracker_releases_still_objects_and_rejects_walls():
    from cudarobotics.navigation.tracking import ScanObjectTracker, TrackerConfig

    tracker = ScanObjectTracker(TrackerConfig(commit_seconds=1.0))
    wall = np.repeat(np.stack([np.linspace(0.0, 2.0, 21), np.full(21, -1.0)], axis=1), 3, axis=0)
    for frame in range(15):
        still = _blob(2.0, 1.0)
        xy = np.concatenate([still, wall])
        result = tracker.update(xy, np.ones(xy.shape[0], bool), 0.1 * frame)
        if frame == 5:
            # Seen three times but not moving: reported as a static object and
            # kept out of the map while it might still start moving.
            assert result.tracks.shape[0] == 0 and result.static_objects.shape[0] == 1
            assert result.dynamic_mask[: still.shape[0]].all()
    # After commit_seconds it is static: released to the map, no longer reported.
    assert result.tracks.shape[0] == 0 and result.static_objects.shape[0] == 0
    assert not result.dynamic_mask.any()  # the 2 m wall is never a detection


def test_tracker_deletes_a_lost_track_and_validates_input():
    from cudarobotics.navigation.tracking import ScanObjectTracker, TrackerConfig

    tracker = ScanObjectTracker()
    for frame in range(10):
        xy = _blob(1.0, 0.05 * frame - 2.0)
        result = tracker.update(xy, np.ones(xy.shape[0], bool), 0.1 * frame)
    assert result.tracks.shape[0] == 1
    empty = np.zeros((0, 2))
    for frame in range(10, 17):
        result = tracker.update(empty, np.zeros(0, bool), 0.1 * frame)
    assert result.tracks.shape[0] == 0 and tracker.live_tracks == 0
    with pytest.raises(ValueError):
        tracker.update(np.zeros((3, 2)), np.ones(2, bool), 2.0)
    with pytest.raises(ValueError):
        tracker.update(np.zeros((0, 2)), np.zeros(0, bool), 0.5)  # time went backwards
    with pytest.raises(ValueError):
        TrackerConfig(gate=0.0)


def test_tracker_corrects_the_near_side_bias_of_the_centre():
    from cudarobotics.navigation.tracking import ScanObjectTracker

    tracker = ScanObjectTracker()
    xy = _blob(4.0, 0.0)
    for frame in range(4):
        tracker.update(xy, np.ones(xy.shape[0], bool), 0.1 * frame, sensor_xy=(0.0, 0.0))
    # Not confirmed (it does not move), but the static object sits near the
    # true centre rather than on the visible arc 0.25 m closer to the sensor.
    result = tracker.update(xy, np.ones(xy.shape[0], bool), 0.4, sensor_xy=(0.0, 0.0))
    assert result.static_objects[0, 0] == pytest.approx(4.0, abs=0.07)


def test_occupancy_age_separates_structure_from_new_objects():
    from cudarobotics.navigation.tracking import OccupancyAge

    data = np.zeros((20, 20), dtype=np.int8)
    data[10, :] = 100  # a wall along row 10 (y in [1.0, 1.1))
    grid = OccupancyGrid(data=data, resolution=0.1, origin=(0.0, 0.0))
    age = OccupancyAge()
    points = np.asarray([[0.55, 1.05], [0.55, 0.98], [0.55, 0.45]], dtype=np.float32)
    candidates, structure = age.classify(points, 20)
    assert candidates.all() and not structure.any()  # before any map: all candidates
    for _ in range(5):
        age.update(grid)
    candidates, structure = age.classify(points, 20)
    assert candidates.all()  # the wall is still young
    for _ in range(20):
        age.update(grid)
    candidates, structure = age.classify(points, 20)
    assert structure.tolist() == [True, True, False]  # wall and the cell in front of it
    assert candidates.tolist() == [False, False, True]
    # Rolling the grid by one cell keeps each cell's age.
    shifted = OccupancyGrid(data=np.roll(data, -1, axis=1), resolution=0.1, origin=(0.1, 0.0))
    age.update(shifted)
    assert age.classify(points, 20)[1].tolist() == [True, True, False]
