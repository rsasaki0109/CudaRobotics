#!/usr/bin/env python3
"""Regression tests for the Python release-artifact verifier."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import verify_python_release_artifacts as verifier


VERSION = "0.2.0"
ROOT = f"cudarobotics-{VERSION}"


def required_sdist_names() -> set[str]:
    names = {
        f"{ROOT}/CMakeLists.txt",
        f"{ROOT}/pyproject.toml",
        f"{ROOT}/README.md",
        f"{ROOT}/src/cudarobotics/__init__.py",
        f"{ROOT}/src/cudarobotics/bindings.cpp",
        f"{ROOT}/src/cudarobotics/registration/__init__.py",
        f"{ROOT}/core/include/cuda_check.cuh",
        f"{ROOT}/core/include/cuda_mppi_controller/mppi_gpu.hpp",
    }
    names.update(
        f"{ROOT}/core/src/{name}" for name in verifier.CUDA_SOURCES
    )
    names.update(
        f"{ROOT}/core/include/cudarobotics/{path.name}"
        for path in (
            verifier.REPO / "include" / "cudarobotics"
        ).glob("*.hpp")
    )
    return names


def write_sdist(path: Path, names: set[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(names):
            contents = b"test\n"
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))


def write_wheel(path: Path, *, include_extension: bool = True) -> None:
    dist_info = f"cudarobotics-{VERSION}.dist-info"
    files = {
        "cudarobotics/__init__.py": b"",
        "cudarobotics/registration/__init__.py": b"",
        f"{dist_info}/METADATA": (
            f"Name: cudarobotics\nVersion: {VERSION}\n".encode()
        ),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\n",
        f"{dist_info}/RECORD": b"",
    }
    if include_extension:
        files["cudarobotics/_cudarobotics.test.so"] = b"native"
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)


class ArtifactVerifierTests(unittest.TestCase):
    def test_valid_sdist_and_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdist = root / f"cudarobotics-{VERSION}.tar.gz"
            wheel = root / f"cudarobotics-{VERSION}-py3-none-any.whl"
            write_sdist(sdist, required_sdist_names())
            write_wheel(wheel)
            verifier.validate_sdist(sdist, VERSION)
            verifier.validate_wheel(wheel, VERSION)

    def test_sdist_rejects_cache_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"cudarobotics-{VERSION}.tar.gz"
            names = required_sdist_names()
            names.add(f"{ROOT}/.pytest_cache/README.md")
            write_sdist(path, names)
            with self.assertRaisesRegex(
                AssertionError, "cache/build artifacts"
            ):
                verifier.validate_sdist(path, VERSION)

    def test_wheel_requires_native_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / f"cudarobotics-{VERSION}-py3-none-any.whl"
            )
            write_wheel(path, include_extension=False)
            with self.assertRaisesRegex(
                AssertionError, "exactly one native extension"
            ):
                verifier.validate_wheel(path, VERSION)


if __name__ == "__main__":
    unittest.main()
