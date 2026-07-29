#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from inspect_pointcloud2_timing import inspect_timing, point_field_statistics
from run_cudanav_kiss_icp_real import timing_admission_matches
from test_cudanav_kiss_icp_real import sequence_database


class PointCloud2TimingAdmissionTest(unittest.TestCase):
    def test_numpy_statistics_match_dependency_free_fallback_with_row_padding(
        self,
    ) -> None:
        payload = bytearray(56)
        times = [0.0, 0.1, 0.05, 0.02, 0.08, 0.1]
        rings = [0, 1, 2, 3, 4, 5]
        for index, (stamp, ring) in enumerate(zip(times, rings)):
            row, column = divmod(index, 3)
            offset = row * 28 + column * 8
            struct.pack_into("<fH", payload, offset, stamp, ring)
        cloud = {
            "height": 2,
            "width": 3,
            "row_step": 28,
            "point_step": 8,
            "is_bigendian": False,
            "data": bytes(payload),
            "fields": {
                "time": {"offset": 0, "datatype": 7, "count": 1},
                "ring": {"offset": 4, "datatype": 4, "count": 1},
            },
        }
        for field, unique in (("time", False), ("ring", True)):
            accelerated = point_field_statistics(
                cloud, field, collect_unique=unique
            )
            fallback = point_field_statistics(
                cloud,
                field,
                collect_unique=unique,
                numpy_acceleration=False,
            )
            self.assertEqual(accelerated, fallback)

    def inspect(
        self,
        database: Path,
        *,
        unit: str = "seconds",
        ring_field: str | None = "ring",
        require_ring: bool = True,
    ):
        return inspect_timing(
            database,
            pointcloud_topic="/points",
            point_time_field="time",
            point_time_datatype=7,
            point_time_unit=unit,
            ring_field=ring_field,
            ring_datatype=4 if ring_field else None,
            require_ring=require_ring,
            start_offset_s=0.0,
            maximum_duration_s=5.0,
            maximum_frames=3,
            minimum_frames=3,
            minimum_scan_span_s=0.05,
            maximum_scan_span_s=0.15,
            require_unambiguous_unit=True,
        )

    def test_admits_stable_timed_pointcloud_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "timed.db3"
            sequence_database(database, timed=True)
            report = self.inspect(database)
            self.assertTrue(report["valid"])
            self.assertEqual(report["selection"]["frames"], 3)
            self.assertEqual(
                report["point_time"]["plausible_units"], ["seconds"]
            )
            self.assertEqual(report["ring"]["minimum"], 0)
            self.assertEqual(report["ring"]["maximum"], 15)
            self.assertEqual(report["ring"]["distinct_values"], 16)
            self.assertTrue(all(report["checks"].values()))
            export_report = {
                "sequence_version": 2,
                "frames": 3,
                "pointcloud_topic": "/points",
                "point_time": {"field": "time", "unit": "seconds"},
                "ring": {"field": "ring"},
                "point_fields": {
                    "time": {"datatype": 7},
                    "ring": {"datatype": 4},
                },
            }
            self.assertTrue(
                timing_admission_matches(
                    report,
                    export_report,
                    expected_database_sha256=report["database"]["sha256"],
                    minimum_frames=3,
                )
            )
            self.assertFalse(
                timing_admission_matches(
                    report,
                    export_report,
                    expected_database_sha256="0" * 64,
                    minimum_frames=3,
                )
            )

    def test_rejects_wrong_declared_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "timed.db3"
            sequence_database(database, timed=True)
            report = self.inspect(database, unit="milliseconds")
            self.assertFalse(report["valid"])
            self.assertFalse(report["checks"]["declared_unit_plausible"])
            self.assertFalse(report["checks"]["unit_unambiguous"])

    def test_rejects_ambiguous_physical_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "timed.db3"
            sequence_database(database, timed=True)
            report = inspect_timing(
                database,
                pointcloud_topic="/points",
                point_time_field="time",
                point_time_datatype=7,
                point_time_unit="seconds",
                ring_field="ring",
                ring_datatype=4,
                require_ring=True,
                start_offset_s=0.0,
                maximum_duration_s=5.0,
                maximum_frames=3,
                minimum_frames=3,
                minimum_scan_span_s=1e-5,
                maximum_scan_span_s=0.2,
                require_unambiguous_unit=True,
            )
            self.assertFalse(report["valid"])
            self.assertEqual(
                report["point_time"]["plausible_units"],
                ["seconds", "milliseconds"],
            )
            self.assertTrue(report["checks"]["declared_unit_plausible"])
            self.assertFalse(report["checks"]["unit_unambiguous"])

    def test_rejects_missing_time_or_required_ring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "xyz.db3"
            sequence_database(database)
            with self.assertRaisesRegex(ValueError, "scalar field: time"):
                self.inspect(database)

            timed = Path(directory) / "timed.db3"
            sequence_database(timed, timed=True)
            with self.assertRaisesRegex(
                ValueError, "scalar field: missing_ring"
            ):
                inspect_timing(
                    timed,
                    pointcloud_topic="/points",
                    point_time_field="time",
                    point_time_datatype=7,
                    point_time_unit="seconds",
                    ring_field="missing_ring",
                    ring_datatype=4,
                    require_ring=True,
                    start_offset_s=0.0,
                    maximum_duration_s=5.0,
                    maximum_frames=3,
                    minimum_frames=3,
                    minimum_scan_span_s=0.05,
                    maximum_scan_span_s=0.15,
                    require_unambiguous_unit=True,
                )


if __name__ == "__main__":
    unittest.main()
