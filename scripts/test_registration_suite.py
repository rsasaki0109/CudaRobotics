#!/usr/bin/env python3
"""CPU-only checks for the unified registration benchmark plumbing."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("benchmark_registration_external.py")
SPEC = importlib.util.spec_from_file_location("registration_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    target, source, rotation, translation = MODULE.make_pair(
        256, 0, "outlier_partial"
    )
    assert target.shape == (256, 3)
    assert source.shape[1] == 3
    assert rotation.shape == (3, 3)
    assert translation.shape == (3,)
    assert float(
        MODULE.np.max(MODULE.np.abs(rotation @ rotation.T - MODULE.np.eye(3)))
    ) < 1e-10

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        csv_path = output / "suite.csv"
        md_path = output / "suite.md"
        row = {field: "" for field in MODULE.CSV_FIELDS}
        row.update(
            {
                "scenario": "lumpy_partial",
                "algorithm": "cudarobotics_filterreg_gpu",
                "size": 256,
                "status": "ok",
                "quality_pass": True,
                "median_ms": "1.25",
                "median_rot_err_deg": "0.2",
                "median_trans_err_m": "0.01",
                "median_rmse_m": "0.02",
            }
        )
        MODULE.write_csv(csv_path, [row])
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        MODULE.write_markdown(md_path, csv_path, rows)
        text = md_path.read_text(encoding="utf-8")
        assert "Registration Unified Benchmark" in text
        assert "Quality gates passed: 1/1" in text
        assert "PASS" in text
        assert MODULE.suite_passed(rows)
        rows[0]["quality_pass"] = "False"
        assert not MODULE.suite_passed(rows)
    print("registration suite checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
