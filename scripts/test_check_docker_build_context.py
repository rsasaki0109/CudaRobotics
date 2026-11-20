#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_docker_build_context import REPO, check, parse_build_stage

PACKAGE_CMAKE = """
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
add_library(core SHARED "${REPO_ROOT}/src/core.cu")
target_include_directories(core PUBLIC $<BUILD_INTERFACE:${REPO_ROOT}/include>)
"""

DOCKERFILE = """
FROM ros:jazzy AS build
COPY src/core.cu src/
COPY {headers}
COPY ros2_ws/src/pkg ros2_ws/src/pkg
COPY ros2_ws/src/py_pkg ros2_ws/src/py_pkg
RUN cd ros2_ws && colcon build \\
      --packages-select pkg py_pkg \\
      --cmake-args -DCMAKE_BUILD_TYPE=Release
FROM ros:jazzy-ros-core
COPY --from=build /ws/ros2_ws/install /ws/install
"""


def make_repo(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "core.cu").write_text('#include "core.cuh"\n#include <cub/cub.cuh>\n')
    (root / "include").mkdir()
    (root / "include" / "core.cuh").write_text('#include "private.cuh"\n')
    (root / "include" / "private.cuh").write_text("// private header\n")
    pkg = root / "ros2_ws" / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "CMakeLists.txt").write_text(PACKAGE_CMAKE)
    py_pkg = root / "ros2_ws" / "src" / "py_pkg"
    py_pkg.mkdir(parents=True)
    (py_pkg / "setup.py").write_text("")


class CheckDockerBuildContextTest(unittest.TestCase):
    def run_check(self, headers: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_repo(root)
            dockerfile = root / "Dockerfile"
            dockerfile.write_text(DOCKERFILE.format(headers=headers))
            return check(root, dockerfile)

    def test_parses_only_the_build_stage(self) -> None:
        copied, packages = parse_build_stage(DOCKERFILE.format(headers="include include"))
        self.assertEqual(packages, ["pkg", "py_pkg"])
        self.assertIn("include", copied)
        self.assertNotIn("/ws/ros2_ws/install", copied)

    def test_whole_include_tree_passes(self) -> None:
        self.assertEqual(self.run_check("include include"), [])

    def test_missing_transitive_header_fails(self) -> None:
        errors = self.run_check("include/core.cuh include/core.cuh")
        self.assertEqual(
            errors,
            ["include/core.cuh includes include/private.cuh, which the image does not copy"],
        )

    def test_repository_dockerfile_is_complete(self) -> None:
        self.assertEqual(check(REPO, REPO / "docker" / "Dockerfile"), [])


if __name__ == "__main__":
    unittest.main()
