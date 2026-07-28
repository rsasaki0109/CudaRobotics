#!/usr/bin/env python3
"""Verify that the CUDA sources bundled in the Python sdist are current."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

CUDA_SOURCES = (
    "mppi_gpu.cu",
    "filterreg_gpu.cu",
    "sinkhorn_gpu.cu",
    "fgr_gpu.cu",
    "bcpd_gpu.cu",
    "robust_treg_gpu.cu",
    "robust_p2plane_gpu.cu",
)


def assert_same(source: Path, bundled: Path) -> None:
    assert bundled.is_file(), f"bundled Python core file is missing: {bundled}"
    source_bytes = source.read_bytes().replace(b"\r\n", b"\n").replace(
        b"\r", b"\n"
    )
    bundled_bytes = bundled.read_bytes().replace(b"\r\n", b"\n").replace(
        b"\r", b"\n"
    )
    assert source_bytes == bundled_bytes, (
        f"bundled Python core is stale: {bundled} differs from {source}; "
        "run ./scripts/sync_python_core.sh"
    )


def main() -> int:
    for filename in CUDA_SOURCES:
        assert_same(
            REPO / "src" / filename,
            REPO / "python" / "core" / "src" / filename,
        )

    assert_same(
        REPO / "include" / "cuda_check.cuh",
        REPO / "python" / "core" / "include" / "cuda_check.cuh",
    )
    assert_same(
        REPO / "include" / "cuda_mppi_controller" / "mppi_gpu.hpp",
        REPO
        / "python"
        / "core"
        / "include"
        / "cuda_mppi_controller"
        / "mppi_gpu.hpp",
    )

    source_headers = sorted(
        (REPO / "include" / "cudarobotics").glob("*.hpp"),
        key=lambda path: path.name,
    )
    bundled_header_dir = (
        REPO / "python" / "core" / "include" / "cudarobotics"
    )
    bundled_headers = sorted(
        bundled_header_dir.glob("*.hpp"),
        key=lambda path: path.name,
    )
    assert [path.name for path in source_headers] == [
        path.name for path in bundled_headers
    ], (
        "bundled cudarobotics header inventory is stale; "
        "run ./scripts/sync_python_core.sh"
    )
    for source in source_headers:
        assert_same(source, bundled_header_dir / source.name)

    print(
        "Python bundled CUDA core is synchronized: "
        f"{len(CUDA_SOURCES)} sources, {len(source_headers) + 2} headers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
