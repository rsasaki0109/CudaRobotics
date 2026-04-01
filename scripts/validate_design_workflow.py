#!/usr/bin/env python3

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_DOCS = [
    ROOT / "docs" / "experiments.md",
    ROOT / "docs" / "decisions.md",
    ROOT / "docs" / "interfaces.md",
]

REQUIRED_MODULE_ATTRIBUTES = [
    "PROBLEM_KIND",
    "INTERFACE_FILE",
    "TITLE",
    "DESCRIPTION_LINES",
    "REQUEST_SUMMARY",
    "METRIC_NOTES",
    "build_requests",
    "build_report",
]

SUPPORTED_PROBLEM_KINDS = {
    "planner_selection",
    "time_budget_selection",
}


def experiment_modules() -> list[str]:
    modules: list[str] = []
    for path in sorted((ROOT / "experiments").iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("__") or path.name == "data":
            continue
        if (path / "__init__.py").exists():
            modules.append(path.name)
    return modules


def validate_variant_module(module_name: str) -> None:
    module = importlib.import_module(f"experiments.{module_name}")
    for attribute in REQUIRED_MODULE_ATTRIBUTES:
        if not hasattr(module, attribute):
            raise RuntimeError(f"experiments.{module_name} is missing {attribute}")

    if not hasattr(module, "build_variants"):
        raise RuntimeError(f"experiments.{module_name} is missing build_variants()")

    if module.PROBLEM_KIND not in SUPPORTED_PROBLEM_KINDS:
        raise RuntimeError(f"experiments.{module_name} has unsupported PROBLEM_KIND {module.PROBLEM_KIND}")

    variants = module.build_variants()
    if len(variants) < 3:
        raise RuntimeError(f"experiments.{module_name} must expose at least 3 variants")

    seen_names: set[str] = set()
    seen_paradigms: set[str] = set()
    for variant in variants:
        if not getattr(variant, "name", None):
            raise RuntimeError(f"experiments.{module_name} contains a variant without name")
        if not getattr(variant, "paradigm", None):
            raise RuntimeError(f"experiments.{module_name} contains a variant without paradigm")
        if variant.name in seen_names:
            raise RuntimeError(f"experiments.{module_name} has duplicate variant name {variant.name}")
        seen_names.add(variant.name)
        seen_paradigms.add(variant.paradigm)

    if len(seen_paradigms) < 3:
        raise RuntimeError(f"experiments.{module_name} must keep at least 3 distinct paradigms alive")

    expected_interface = ROOT / "core" / module.INTERFACE_FILE
    if not expected_interface.exists():
        raise RuntimeError(f"Missing matching core interface: {expected_interface.relative_to(ROOT)}")


def validate_docs() -> None:
    for path in REQUIRED_DOCS:
        if not path.exists():
            raise RuntimeError(f"Missing required doc: {path.relative_to(ROOT)}")


def validate_generated_experiments(modules: list[str]) -> None:
    out_dir = ROOT / "build" / "design_docs_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "scripts/run_design_experiments.py", "--docs-dir", str(out_dir.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
    )
    generated = out_dir / "experiments.md"
    if not generated.exists():
        raise RuntimeError("run_design_experiments.py did not generate experiments.md in validation output")
    generated_text = generated.read_text()
    for module_name in modules:
        if module_name not in generated_text:
            raise RuntimeError(f"Generated experiments.md is missing the {module_name} section")


def main() -> int:
    validate_docs()
    modules = experiment_modules()
    if not modules:
        raise RuntimeError("No experiment modules found under experiments/")
    for module_name in modules:
        validate_variant_module(module_name)
    validate_generated_experiments(modules)
    print("Design workflow validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
