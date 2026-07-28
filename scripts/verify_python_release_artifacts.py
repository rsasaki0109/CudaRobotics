#!/usr/bin/env python3
"""Validate Python release archives and emit a reproducible artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tarfile
import zipfile
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


def package_version() -> str:
    text = (REPO / "python" / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if match is None:
        raise AssertionError("package version not found in python/pyproject.toml")
    return match.group(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        encoding="utf-8",
    ).strip()


def validate_sdist(path: Path, version: str) -> None:
    root = f"cudarobotics-{version}"
    required = {
        f"{root}/CMakeLists.txt",
        f"{root}/pyproject.toml",
        f"{root}/README.md",
        f"{root}/src/cudarobotics/__init__.py",
        f"{root}/src/cudarobotics/bindings.cpp",
        f"{root}/src/cudarobotics/registration/__init__.py",
        f"{root}/core/include/cuda_check.cuh",
        f"{root}/core/include/cuda_mppi_controller/mppi_gpu.hpp",
    }
    required.update(f"{root}/core/src/{name}" for name in CUDA_SOURCES)
    required.update(
        f"{root}/core/include/cudarobotics/{path.name}"
        for path in (REPO / "include" / "cudarobotics").glob("*.hpp")
    )

    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        missing = sorted(required - names)
        assert not missing, f"{path.name} is missing required files: {missing}"
        forbidden = sorted(
            name
            for name in names
            if any(
                part in name.split("/")
                for part in (".pytest_cache", "__pycache__", "dist", "build")
            )
            or name.endswith((".pyc", ".pyo"))
        )
        assert not forbidden, (
            f"{path.name} contains cache/build artifacts: {forbidden}"
        )


def validate_wheel(path: Path, version: str) -> None:
    dist_info = f"cudarobotics-{version}.dist-info"
    required = {
        "cudarobotics/__init__.py",
        "cudarobotics/registration/__init__.py",
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/RECORD",
    }
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        assert not missing, f"{path.name} is missing required files: {missing}"
        extensions = sorted(
            name
            for name in names
            if name.startswith("cudarobotics/_cudarobotics.")
            and name.endswith((".so", ".pyd"))
        )
        assert len(extensions) == 1, (
            f"{path.name} must contain exactly one native extension, got "
            f"{extensions}"
        )
        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        assert re.search(r"(?m)^Name: cudarobotics$", metadata)
        assert re.search(
            rf"(?m)^Version: {re.escape(version)}$", metadata
        ), f"{path.name} metadata version does not match {version}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=REPO / "python" / "dist",
        help="directory containing the sdist and one or more wheels",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="optional output path for the artifact manifest",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail when the repository has uncommitted changes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = package_version()
    dist_dir = args.dist_dir.resolve()
    sdist = dist_dir / f"cudarobotics-{version}.tar.gz"
    assert sdist.is_file(), f"sdist not found: {sdist}"
    wheels = sorted(dist_dir.glob(f"cudarobotics-{version}-*.whl"))
    assert wheels, f"no wheels found in {dist_dir}"

    validate_sdist(sdist, version)
    for wheel in wheels:
        validate_wheel(wheel, version)

    dirty = bool(git_output("status", "--porcelain"))
    if args.require_clean:
        assert not dirty, "release artifacts must be verified from a clean checkout"

    artifacts = [sdist, *wheels]
    manifest = {
        "schema_version": 1,
        "package": "cudarobotics",
        "package_version": version,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "artifacts": [
            {
                "name": path.name,
                "kind": "sdist" if path == sdist else "wheel",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    print(
        f"validated cudarobotics {version}: "
        f"1 sdist, {len(wheels)} wheel(s)"
    )
    for artifact in manifest["artifacts"]:
        print(
            f"  {artifact['name']}  {artifact['bytes']} bytes  "
            f"sha256={artifact['sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
