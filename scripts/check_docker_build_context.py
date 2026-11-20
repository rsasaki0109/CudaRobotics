#!/usr/bin/env python3
"""Check that the Docker build stage copies every repository file it compiles.

The ROS 2 packages compile shared cores from the repository root
(``${REPO_ROOT}/src/*.cu`` and ``${REPO_ROOT}/include``). The Docker image
copies only part of the repository, so a new private header can build in CI
(full checkout) yet break the tagged image. This check resolves, without
Docker or CUDA:

* every ``${REPO_ROOT}``/``${CUDAROBOTICS_ROOT}`` path referenced by the
  CMakeLists of the packages built in the image, and
* the transitive quoted ``#include`` closure of every compiled source,

and fails if a repository file needed by the build is not copied.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
ROOT_VARS = ("REPO_ROOT", "CUDAROBOTICS_ROOT")
ROOT_REF = re.compile(r"\$\{(?:%s)\}/([A-Za-z0-9_./-]+)" % "|".join(ROOT_VARS))
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}


def _logical_lines(text: str) -> list[str]:
    return [line.strip() for line in re.sub(r"\\\n", " ", text).splitlines()]


def parse_build_stage(dockerfile: str) -> tuple[list[str], list[str]]:
    """Return (copied repo paths, colcon packages) of the first build stage."""
    copied: list[str] = []
    packages: list[str] = []
    stages = 0
    for line in _logical_lines(dockerfile):
        if not line or line.startswith("#"):
            continue
        keyword = line.split(None, 1)[0].upper()
        if keyword == "FROM":
            stages += 1
            if stages > 1:
                break
        elif keyword == "COPY":
            tokens = shlex.split(line)[1:]
            if any(t.startswith("--from") for t in tokens):
                continue
            args = [t for t in tokens if not t.startswith("--")]
            copied.extend(PurePosixPath(a).as_posix().rstrip("/") for a in args[:-1])
        elif keyword == "RUN" and "--packages-select" in line:
            tail = line.split("--packages-select", 1)[1]
            for token in tail.split():
                if token.startswith("-"):
                    break
                packages.append(token)
    return copied, packages


def _covered(rel: str, copied: list[str]) -> bool:
    return any(rel == c or rel.startswith(c + "/") or c == "." for c in copied)


def _rel(path: Path, repo: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None


def check(repo: Path, dockerfile: Path) -> list[str]:
    copied, packages = parse_build_stage(dockerfile.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not packages:
        return [f"{dockerfile}: no `colcon build --packages-select` list found"]

    pending: list[Path] = []
    include_dirs: list[Path] = []
    for package in packages:
        package_dir = repo / "ros2_ws" / "src" / package
        if not _covered(f"ros2_ws/src/{package}", copied):
            errors.append(f"package {package} is built but not copied")
            continue
        cmake = package_dir / "CMakeLists.txt"
        if not cmake.is_file():
            if not (package_dir / "setup.py").is_file():
                errors.append(f"package {package} has neither CMakeLists.txt nor setup.py")
            continue  # ament_python packages compile nothing from the repo root
        include_dirs.append(package_dir / "include")
        for ref in sorted(set(ROOT_REF.findall(cmake.read_text(encoding="utf-8")))):
            ref = ref.rstrip("/")
            if (repo / ref).is_dir():
                # Include roots may be partially copied; the include closure
                # below checks the headers that are actually used.
                include_dirs.append(repo / ref)
            elif (repo / ref).is_file():
                if not _covered(ref, copied):
                    errors.append(f"{package}: CMake uses {ref}, which is not copied")
                if (repo / ref).suffix in SOURCE_SUFFIXES:
                    pending.append(repo / ref)
        pending.extend(
            p for p in package_dir.rglob("*") if p.suffix in SOURCE_SUFFIXES and p.is_file()
        )

    seen: set[Path] = set()
    while pending:
        source = pending.pop().resolve()
        if source in seen:
            continue
        seen.add(source)
        for name in INCLUDE.findall(source.read_text(encoding="utf-8", errors="replace")):
            candidates = [source.parent / name] + [d / name for d in include_dirs]
            found = next((c for c in candidates if c.is_file()), None)
            if found is None:
                continue  # system, ROS, or generated header
            rel = _rel(found, repo)
            if rel is None:
                continue
            if not _covered(rel, copied):
                errors.append(
                    f"{_rel(source, repo)} includes {rel}, which the image does not copy"
                )
            pending.append(found)
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--dockerfile", type=Path, default=None)
    args = parser.parse_args(argv)
    dockerfile = args.dockerfile or args.repo / "docker" / "Dockerfile"
    errors = check(args.repo, dockerfile)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print(f"{dockerfile.name}: build context covers every repository source and header")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
