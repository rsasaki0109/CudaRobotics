#!/usr/bin/env python3
"""Validate cross-surface support claims for the v1.0 release target."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from v1_release_attestation import MODES, load_reference


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs" / "v1_support_matrix.json"

ROS_PACKAGES = (
    "cuda_robotics_msgs",
    "cuda_robotics_common",
    "cuda_kiss_icp",
    "cuda_voxel_mapping",
    "cuda_esdf",
    "cuda_voxel_costmap_layer",
    "cuda_mppi_controller",
    "cuda_nav_bringup",
)


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def normalized_shell(text: str) -> str:
    return " ".join(text.replace("\\\n", " ").split())


def python_version() -> str:
    text = read("python/pyproject.toml")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if match is None:
        raise ValueError("Python package version is missing")
    return match.group(1)


def ros_versions() -> dict[str, str]:
    versions = {}
    for package in ROS_PACKAGES:
        root = ET.parse(
            ROOT / "ros2_ws" / "src" / package / "package.xml"
        ).getroot()
        value = root.findtext("version")
        if value is None:
            raise ValueError(f"ROS package version is missing: {package}")
        versions[package] = value
    return versions


def attestations_share_release_commit(
    gates: dict[str, dict[str, Any]],
) -> bool:
    passed = [gate for gate in gates.values() if gate.get("passed") is True]
    return len(passed) == len(MODES) and len(
        {gate.get("git_commit") for gate in passed}
    ) == 1


def evaluate(
    matrix: dict[str, Any],
    *,
    attestation_root: Path = ROOT,
    readiness_evidence: dict[str, Any] | None = None,
    expected_release_commit: str | None = None,
) -> dict[str, Any]:
    surfaces = matrix.get("surfaces", {})
    main = matrix.get("main_demo", {})
    readiness = matrix.get("release_readiness", {})
    if not isinstance(readiness, dict):
        readiness = {}
    evidence = (
        readiness_evidence
        if isinstance(readiness_evidence, dict)
        else readiness
    )
    target = matrix.get("target_version")
    source_version = matrix.get("source_version")
    source_tag = matrix.get("source_tag")
    actual_python = python_version()
    actual_ros = ros_versions()
    declared_ros = surfaces.get("ros2", {}).get("package_versions")
    documentation = surfaces.get("documentation", {})
    command = main.get("run_command")
    build_command = main.get("build_command")
    status = matrix.get("status")
    published_version = readiness.get("published_version")
    install_page = read("docs/site/install.html")
    index_page = read("docs/site/index.html")
    command_surfaces = (
        "README.md",
        "docker/README.md",
        "docs/site/install.html",
        "docs/site/nav2.html",
        "ros2_ws/src/cuda_nav_bringup/README.md",
    )
    checks = {
        "schema": matrix.get("schema_version") == 1,
        "status": status in {"development", "release"},
        "target_version": target == "1.0.0",
        "target_tag": matrix.get("target_tag") == "v1.0.0",
        "source_version": source_version == actual_python
        and source_version != target,
        "source_tag": source_tag == f"v{source_version}",
        "surface_table": isinstance(surfaces, dict)
        and {
            "python_source",
            "python_wheels",
            "ros2",
            "docker_source",
            "colab",
            "documentation",
        }
        <= set(surfaces),
        "time_budget": main.get("time_budget_seconds") == 900,
        "main_surface": main.get("surface") == "docker_source",
        "main_result": main.get("result") == "out/cudanav_closed_loop.json"
        and main.get("gate") == "smoke_pass",
        "main_command": isinstance(command, str)
        and bool(command)
        and all(
            normalized_shell(command) in normalized_shell(read(relative))
            for relative in command_surfaces
        ),
        "main_build_command": isinstance(build_command, str)
        and bool(build_command)
        and all(
            normalized_shell(build_command)
            in normalized_shell(read(relative))
            for relative in command_surfaces
        ),
        "python_version": surfaces.get("python_source", {}).get("version")
        == actual_python
        and surfaces.get("python_wheels", {}).get("version")
        == actual_python,
        "python_requirement": surfaces.get("python_source", {}).get("python")
        == ">=3.9"
        and 'requires-python = ">=3.9"' in read("python/pyproject.toml"),
        "python_cuda": surfaces.get("python_source", {}).get("cuda_toolkit")
        == ">=12.0"
        and "CUDA Toolkit >= 12.0" in read("python/README.md"),
        "wheel_matrix": surfaces.get("python_wheels", {}).get("python")
        == ["cp310", "cp312"]
        and 'CIBW_BUILD: cp310-* cp312-*'
        in read(".github/workflows/python-package.yml"),
        "ros_versions": declared_ros == actual_ros,
        "ros_platform": surfaces.get("ros2", {}).get("distribution")
        == "jazzy"
        and surfaces.get("ros2", {}).get("platform") == "Ubuntu 24.04"
        and surfaces.get("ros2", {}).get("cuda_toolkit") == "12.6"
        and "ubuntu-24.04" in read(".github/workflows/ros2_cuda_mppi.yml")
        and "--cuda-toolkit 12.6"
        in read(".github/workflows/ros2_cuda_mppi.yml"),
        "ros_command": surfaces.get("ros2", {}).get("command")
        in read("ros2_ws/src/cuda_nav_bringup/README.md"),
        "docker_runtime": surfaces.get("docker_source", {}).get("cuda_runtime")
        == "12.6"
        and "ARG CUDA_VER=12-6" in read("docker/Dockerfile")
        and "driver (>= 525" in read("docker/README.md"),
        "colab": surfaces.get("colab", {}).get("requires_gpu_runtime") is True
        and surfaces.get("colab", {}).get("url") in read("README.md")
        and (
            f"git clone --depth 1 --branch {source_tag} "
            in read("examples/colab/cudarobotics_quickstart.ipynb")
        )
        and (
            f"/tree/{source_tag}/"
            in read("examples/colab/cudarobotics_quickstart.ipynb")
        ),
        "documentation": documentation.get("install_page")
        == "docs/site/install.html"
        and documentation.get("nav2_page") == "docs/site/nav2.html"
        and documentation.get("release_notes")
        == "docs/releases/v1.0.0_notes.md"
        and documentation.get("release_checklist")
        == "docs/releases/v1.0.0_release_checklist.md"
        and f"# v{target} " in read(str(documentation.get("release_notes")))
        and f"# v{target} Release Checklist"
        in read(str(documentation.get("release_checklist"))),
        "published_version": (
            status == "development"
            and isinstance(published_version, str)
            and re.fullmatch(r"\d+\.\d+\.\d+", published_version) is not None
            and published_version != target
            and f"published v{published_version}" in install_page.lower()
            and f"v{published_version}" in index_page
            and (
                f"releases/tag/v{published_version}"
                in install_page
            )
        )
        or (
            status == "release"
            and published_version == target
            and str(target) in install_page
            and str(target) in index_page
            and f"releases/tag/{matrix.get('target_tag')}" in install_page
        ),
    }
    target_tag = str(matrix.get("target_tag", ""))
    attestation_gates = {
        key: load_reference(
            evidence.get(key),
            repo_root=attestation_root,
            key=key,
            target_version=str(target),
            target_tag=target_tag,
        )
        for key in MODES
    }
    attestation_commits = {
        gate.get("git_commit")
        for gate in attestation_gates.values()
        if gate.get("passed") is True
    }
    readiness_checks = {
        "contract_valid": all(checks.values()),
        "release_status": status == "release",
        "python_at_target": actual_python == target,
        "ros_at_target": set(actual_ros.values()) == {target},
        "published_target": published_version == target,
        "quickstart_evidence": attestation_gates[
            "quickstart_15_minute_evidence"
        ]["passed"],
        "cudanav_release_evidence": attestation_gates[
            "cudanav_release_evidence"
        ]["passed"],
        "docker_gpu_evidence": attestation_gates[
            "docker_gpu_evidence"
        ]["passed"],
        "documentation_deployment": attestation_gates[
            "documentation_deployment"
        ]["passed"],
        "same_release_commit": attestations_share_release_commit(
            attestation_gates
        ),
        "release_commit_binding": expected_release_commit is not None
        and attestation_commits == {expected_release_commit},
        "colab_target_ref": (
            f"/blob/{matrix.get('target_tag')}/"
            in str(surfaces.get("colab", {}).get("url", ""))
        ),
    }
    return {
        "valid": all(checks.values()),
        "ready": all(readiness_checks.values()),
        "checks": checks,
        "readiness": readiness_checks,
        "attestations": attestation_gates,
        "actual": {
            "python_version": actual_python,
            "ros_package_versions": actual_ros,
        },
    }


def load(path: Path = MATRIX_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("support matrix root must be an object")
    return payload
