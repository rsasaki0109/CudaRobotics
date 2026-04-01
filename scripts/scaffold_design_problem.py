#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


INTERFACE_TEMPLATE = """from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class {problem_class}Input:
    key: str
    payload: float


@dataclass(frozen=True)
class {problem_class}Request:
    key: str


@dataclass(frozen=True)
class {problem_class}Recommendation:
    variant: str
    key: str
    choice: str
    score: float
    rationale: str


class {problem_class}Explorer(Protocol):
    name: str
    paradigm: str

    def recommend(
        self,
        rows: Sequence[{problem_class}Input],
        request: {problem_class}Request,
    ) -> {problem_class}Recommendation:
        ...
"""


INIT_TEMPLATE = """from .functional_variant import FunctionalVariant
from .oop_variant import OOPVariant
from .pipeline_variant import PipelineVariant

PROBLEM_KIND = "{problem_kind}"
INTERFACE_FILE = "{interface_file}"
TITLE = "{problem_title}"
DESCRIPTION_LINES = [
    "replace with a concrete problem statement",
    "replace with a second line describing what changes between variants",
    "replace with a third line describing the evaluation focus",
]
REQUEST_SUMMARY = "replace with the request summary for this problem"
METRIC_NOTES = [
    "replace with the primary regret or utility metric",
    "replace with the primary exact-match or constraint metric",
]


def build_requests(rows):
    raise NotImplementedError("Define build_requests(rows) before enabling this problem in the experiment harness")


def build_report(rows, iterations):
    raise NotImplementedError("Define build_report(rows, iterations) before enabling this problem in the experiment harness")


def build_variants():
    return [
        FunctionalVariant(),
        OOPVariant(),
        PipelineVariant(),
    ]
"""


FUNCTIONAL_TEMPLATE = """from {interface_import} import {problem_class}Explorer, {problem_class}Input, {problem_class}Recommendation, {problem_class}Request


class FunctionalVariant({problem_class}Explorer):
    name = "functional_variant"
    paradigm = "functional"

    def recommend(self, rows, request):
        candidates = [row for row in rows if row.key == request.key]
        if not candidates:
            raise ValueError(f"No candidates for {{request.key}}")
        best = max(candidates, key=lambda row: row.payload)
        return {problem_class}Recommendation(
            variant=self.name,
            key=request.key,
            choice="functional-placeholder",
            score=best.payload,
            rationale="replace with a real functional rule set",
        )
"""


OOP_TEMPLATE = """from dataclasses import dataclass

from {interface_import} import {problem_class}Explorer, {problem_class}Input, {problem_class}Recommendation, {problem_class}Request


@dataclass(frozen=True)
class Objective:
    name: str


class OOPVariant({problem_class}Explorer):
    name = "oop_variant"
    paradigm = "oop"

    def __init__(self):
        self.objectives = [Objective(name="placeholder")]

    def recommend(self, rows, request):
        candidates = [row for row in rows if row.key == request.key]
        if not candidates:
            raise ValueError(f"No candidates for {{request.key}}")
        best = max(candidates, key=lambda row: row.payload)
        return {problem_class}Recommendation(
            variant=self.name,
            key=request.key,
            choice="oop-placeholder",
            score=best.payload,
            rationale="replace with a real object-based rule set",
        )
"""


PIPELINE_TEMPLATE = """from {interface_import} import {problem_class}Explorer, {problem_class}Input, {problem_class}Recommendation, {problem_class}Request


class PipelineVariant({problem_class}Explorer):
    name = "pipeline_variant"
    paradigm = "pipeline"

    def recommend(self, rows, request):
        candidates = [row for row in rows if row.key == request.key]
        if not candidates:
            raise ValueError(f"No candidates for {{request.key}}")
        stage = sorted(candidates, key=lambda row: row.payload, reverse=True)
        best = stage[0]
        return {problem_class}Recommendation(
            variant=self.name,
            key=request.key,
            choice="pipeline-placeholder",
            score=best.payload,
            rationale="replace with a real staged selection pipeline",
        )
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold a new experiment-first design problem with 3 variant stubs.")
    parser.add_argument("problem", help="Problem name, for example planner_selection or cache_policy")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Print the files that would be created without writing them")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Target repository root. Defaults to the current repository.",
    )
    return parser.parse_args()


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    if not slug:
        raise ValueError("Problem name must contain at least one alphanumeric character")
    return slug


def to_class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


def to_title(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("_"))


def write_file(path: Path, content: str, force: bool, dry_run: bool) -> None:
    if path.exists() and not force:
        return
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    args = parse_args()
    slug = slugify(args.problem)
    problem_class = to_class_name(slug)
    interface_import = f"core.{slug}_interface"
    root = Path(args.root).resolve()

    files = {
        root / "core" / "__init__.py": "",
        root / "experiments" / "__init__.py": "",
        root / "core" / f"{slug}_interface.py": INTERFACE_TEMPLATE.format(problem_class=problem_class),
        root / "experiments" / slug / "__init__.py": INIT_TEMPLATE.format(
            problem_kind=slug,
            interface_file=f"{slug}_interface.py",
            problem_title=to_title(slug),
        ),
        root / "experiments" / slug / "functional_variant.py": FUNCTIONAL_TEMPLATE.format(
            interface_import=interface_import, problem_class=problem_class
        ),
        root / "experiments" / slug / "oop_variant.py": OOP_TEMPLATE.format(
            interface_import=interface_import, problem_class=problem_class
        ),
        root / "experiments" / slug / "pipeline_variant.py": PIPELINE_TEMPLATE.format(
            interface_import=interface_import, problem_class=problem_class
        ),
    }

    for path, content in files.items():
        write_file(path, content, args.force, args.dry_run)
        print(path.relative_to(root))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
