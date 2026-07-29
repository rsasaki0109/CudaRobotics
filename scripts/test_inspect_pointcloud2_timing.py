#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from inspect_pointcloud2_timing import inspect_timing
from run_cudanav_kiss_icp_real import timing_admission_matches
from test_cudanav_kiss_icp_real import sequence_database


class PointCloud2TimingAdmissionTest(unittest.TestCase):
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
