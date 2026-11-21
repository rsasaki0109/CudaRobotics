"""CudaNav GPU checks: the real CUDA cores, the pipeline, and both CLI modes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cudarobotics.navigation import (
    CudaNavPipeline,
    Esdf2D,
    KissIcpOdometry,
    VoxelMap,
    cli,
    cuda_runtime_info,
    default_esdf_config,
    default_kiss_icp_config,
    default_voxel_mapping_config,
    esdf_2d_cpu_reference,
    transform_points,
)
from cudarobotics.navigation.closed_loop import TRAJECTORY_COLUMNS, rows_as_array, run_s_course
from cudarobotics.navigation.evidence import missing_summary_fields
from cudarobotics.navigation.sequence import SequenceFrame, run_sequence, write_sequence
from cudarobotics.navigation.simulation import (
    Pose2D,
    course_segments,
    interpolate_path,
    simulate_scan,
)

pytestmark = pytest.mark.gpu_navigation


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not cuda_runtime_info().get("device_count"):
        pytest.skip("no CUDA-capable device")


def _straight_line_scans(count=12, step=0.1):
    segments = course_segments()
    poses = [Pose2D(step * index, 0.0, 0.0) for index in range(count)]
    return poses, [simulate_scan(pose, segments) for pose in poses]


def test_kiss_icp_tracks_a_straight_motion():
    odometry = KissIcpOdometry(default_kiss_icp_config())
    poses, scans = _straight_line_scans()
    frames = [odometry.register(scan) for scan in scans]
    assert odometry.frame_count == len(scans)
    assert frames[-1].map_initialized and frames[-1].inliers >= 30
    assert frames[-1].x == pytest.approx(poses[-1].x, abs=0.05)
    assert abs(frames[-1].y) < 0.05 and abs(frames[-1].yaw) < 0.02
    assert odometry.map_points().shape[1] == 3
    assert odometry.timing()["icp_ms"] > 0.0
    with pytest.raises(ValueError):
        odometry.register(np.zeros((10, 2), dtype=np.float32))


def test_invalid_core_configuration_raises():
    with pytest.raises(ValueError):
        KissIcpOdometry(map_voxel_size=-1.0)
    with pytest.raises(ValueError):
        VoxelMap(width=0)


def test_voxel_map_and_gpu_esdf_match_the_cpu_oracle():
    voxel_config = default_voxel_mapping_config()
    voxel_map = VoxelMap(voxel_config)
    _, scans = _straight_line_scans(count=1)
    stats = voxel_map.integrate(scans[0], (0.0, 0.0, 0.0))
    assert stats["integrated_rays"] > 0 and stats["observed_voxels"] > 500
    grid = voxel_map.occupancy()
    assert grid.data.shape == (voxel_config.height, voxel_config.width)
    assert grid.data.dtype == np.int8 and grid.occupied_cells >= 10
    assert set(np.unique(grid.data)).issubset({-1, 0, 100})

    gpu = Esdf2D(default_esdf_config(voxel_config)).compute(grid, max_distance=2.0)
    cpu = esdf_2d_cpu_reference(grid, max_distance=2.0, unknown_policy="free")
    assert gpu.origin == grid.origin
    np.testing.assert_allclose(gpu.distances, cpu.distances, atol=1e-5)
    assert gpu.occupied_cells == cpu.occupied_cells


def test_pipeline_step_products():
    nav = CudaNavPipeline()
    path = interpolate_path()
    _, scans = _straight_line_scans(count=3)
    for scan in scans:
        step = nav.step(scan, path=path)
    assert step.costmap.dtype == np.uint8
    assert step.costmap.shape == step.occupancy.data.shape == step.distance_field.distances.shape
    v, w = step.command
    assert 0.0 <= v <= nav.mppi_params.v_max + 1e-5 and abs(w) <= nav.mppi_params.w_max + 1e-5
    assert step.planner["valid_rollout_ratio"] > 0.0
    assert {"odometry", "mapping", "esdf", "costmap", "mppi", "total"} <= set(step.timings_ms)
    mapping_only = CudaNavPipeline(enable_planner=False).step(scans[0])
    assert mapping_only.command is None and mapping_only.planner is None


def test_device_and_host_stage_chains_agree():
    path = interpolate_path()
    _, scans = _straight_line_scans(count=4)
    device = CudaNavPipeline(stage_chain="device")
    host = CudaNavPipeline(stage_chain="host")
    for scan in scans:
        on_device = device.step(scan, path=path)
        on_host = host.step(scan, path=path)
        assert on_device.stage_chain == "device" and on_host.stage_chain == "host"
        # The odometry and voxel cores see identical input, so the grids agree;
        # the costmap's exponential band may differ by one step (expf rounding).
        np.testing.assert_array_equal(on_device.occupancy.data, on_host.occupancy.data)
        np.testing.assert_array_equal(
            on_device.distance_field.distances, on_host.distance_field.distances
        )
        difference = np.abs(on_device.costmap.astype(int) - on_host.costmap.astype(int))
        assert difference.max() <= 1
        assert on_device.occupied_cells == on_host.occupied_cells == on_host.occupancy.occupied_cells
        assert on_device.command is not None and all(np.isfinite(on_device.command))


def test_device_step_grids_expire_after_the_next_step():
    nav = CudaNavPipeline(enable_planner=False)
    _, scans = _straight_line_scans(count=3)
    first = nav.step(scans[0])
    kept = nav.step(scans[1]).materialize()
    nav.step(scans[2])
    with pytest.raises(RuntimeError, match="overwrote"):
        first.occupancy
    assert kept.costmap.dtype == np.uint8 and kept.occupancy.data.dtype == np.int8


def test_closed_loop_passes_the_smoke_gate_with_both_stage_chains():
    first = run_s_course(pipeline=CudaNavPipeline(stage_chain="device"))
    second = run_s_course(pipeline=CudaNavPipeline(stage_chain="host"))
    for result in (first, second):
        summary = result.summary
        assert summary["smoke_pass"], {k: v for k, v in summary["checks"].items() if not v}
        assert summary["collision_count"] == 0
        assert summary["odometry_drift_percent"] < 5.0
        assert missing_summary_fields(summary) == []
    # Voxel log-odds use clamped float atomics, so frames are not bit-identical
    # across runs; the mission outcome must still agree closely.
    assert abs(first.summary["frames"] - second.summary["frames"]) <= 0.1 * first.summary["frames"]


def _write_synthetic_sequence(path):
    """Record the closed loop's ground-truth motion as a CRKICP1 sequence."""
    segments = course_segments()
    rows = rows_as_array(run_s_course().rows)
    columns = [TRAJECTORY_COLUMNS.index(name) for name in ("truth_x", "truth_y", "truth_yaw")]
    frames = []
    for index, (x, y, yaw) in enumerate(rows[:, columns]):
        pose = Pose2D(float(x), float(y), float(yaw))
        # The plant starts at the origin, so references are already relative.
        frames.append(
            SequenceFrame(index, index * 100_000_000, (pose.x, pose.y, 0.0, pose.yaw), simulate_scan(pose, segments))
        )
    write_sequence(path, frames)
    return len(frames)


def test_recorded_sequence_and_cli(tmp_path, capsys):
    sequence = tmp_path / "s_course.crkicp"
    frame_count = _write_synthetic_sequence(sequence)
    result = run_sequence(sequence)
    assert result.summary["frames"] == frame_count >= 100
    assert result.summary["smoke_pass"], result.summary["checks"]
    assert result.summary["odometry_ate_rmse_m"] < 0.2

    output = tmp_path / "out"
    assert cli.main(["sequence", str(sequence), "--output-dir", str(output), "--check", "--no-plot"]) == cli.EXIT_OK
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert summary["scenario"] == "cudanav_recorded_sequence"
    assert len(manifest["inputs"]["sequence_sha256"]) == 64
    assert manifest["cuda"]["device_count"] >= 1
    for name in ("trajectory.csv", "occupancy.npy", "esdf.npy", "costmap.npy"):
        assert (output / name).is_file()
    assert np.load(output / "costmap.npy").dtype == np.uint8
    capsys.readouterr()


def test_cli_sim_writes_the_complete_run(tmp_path, capsys):
    output = tmp_path / "sim"
    assert cli.main(["sim", "--output-dir", str(output), "--check", "--no-plot"]) == cli.EXIT_OK
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["smoke_pass"] and summary["claims"]["real_data"] is False
    assert summary["stage_chain"] == "device"
    assert summary["stage_latency_ms"]["frame"]["p95"] > 0.0
    lines = (output / "trajectory.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("step,traversal,time_s,truth_x") and len(lines) == summary["frames"] + 1
    capsys.readouterr()


def test_transform_points_matches_odometry_frame():
    odometry = KissIcpOdometry(default_kiss_icp_config())
    _, scans = _straight_line_scans(count=2)
    frame = odometry.register(scans[0])
    world = transform_points(scans[0], frame.rotation, frame.translation)
    np.testing.assert_allclose(world, scans[0], atol=1e-5)  # first frame is the origin
