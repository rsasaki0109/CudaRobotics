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
