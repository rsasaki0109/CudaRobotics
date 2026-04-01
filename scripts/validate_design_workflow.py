#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_DOCS = [
    ROOT / "docs" / "experiments.md",
    ROOT / "docs" / "experiments_history.md",
    ROOT / "docs" / "decisions.md",
    ROOT / "docs" / "interfaces.md",
]
FIXTURE_DIR = ROOT / "experiments" / "data"
FIXTURE_MANIFEST = FIXTURE_DIR / "manifest.json"
HISTORY_DIR = ROOT / "experiments" / "history"

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

    if not isinstance(module.PROBLEM_KIND, str) or not re.fullmatch(r"[a-z0-9_]+", module.PROBLEM_KIND):
        raise RuntimeError(
            f"experiments.{module_name} must expose a slug-like PROBLEM_KIND, got {module.PROBLEM_KIND!r}"
        )

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


def validate_fixture_manifest() -> None:
    if not FIXTURE_MANIFEST.exists():
        raise RuntimeError(f"Missing fixture manifest: {FIXTURE_MANIFEST.relative_to(ROOT)}")

    data = json.loads(FIXTURE_MANIFEST.read_text())
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise RuntimeError("experiments/data/manifest.json must contain a non-empty fixtures list")

    manifest_filenames: list[str] = []
    seen: set[str] = set()
    for entry in fixtures:
        if not isinstance(entry, dict):
            raise RuntimeError("Fixture manifest entries must be objects")
        filename = entry.get("filename")
        source = entry.get("source")
        if not isinstance(filename, str) or not filename.endswith(".csv"):
            raise RuntimeError(f"Invalid fixture filename in manifest: {filename!r}")
        if filename in seen:
            raise RuntimeError(f"Duplicate fixture filename in manifest: {filename}")
        seen.add(filename)
        manifest_filenames.append(filename)
        if not isinstance(source, str) or not source.endswith(".csv"):
            raise RuntimeError(f"Invalid fixture source in manifest for {filename}: {source!r}")
        path = FIXTURE_DIR / filename
        if not path.exists():
            raise RuntimeError(f"Missing fixture listed in manifest: {path.relative_to(ROOT)}")

    discovered = sorted(path.name for path in FIXTURE_DIR.glob("*.csv"))
    if sorted(manifest_filenames) != discovered:
        raise RuntimeError(
            "Fixture manifest and experiments/data/*.csv are out of sync. "
            "Refresh the manifest or fixture files so both lists match."
        )


def validate_history_snapshots() -> None:
    if not HISTORY_DIR.exists():
        raise RuntimeError(f"Missing design history directory: {HISTORY_DIR.relative_to(ROOT)}")

    snapshots = sorted(HISTORY_DIR.glob("*.json"))
    if not snapshots:
        raise RuntimeError("No design history snapshots found under experiments/history/")

    required_snapshot_keys = {
        "schema_version",
        "snapshot_id",
        "created_at",
        "label",
        "inputs",
        "benchmark_iterations",
        "problems",
    }
    required_problem_keys = {
        "slug",
        "title",
        "description_lines",
        "request_summary",
        "metric_notes",
        "request_count",
        "aggregate_table",
        "case_tables",
    }

    for path in snapshots:
        data = json.loads(path.read_text())
        missing = required_snapshot_keys - set(data)
        if missing:
            raise RuntimeError(f"{path.relative_to(ROOT)} is missing snapshot keys: {sorted(missing)}")
        if data["schema_version"] != 1:
            raise RuntimeError(f"{path.relative_to(ROOT)} has unsupported schema_version {data['schema_version']}")
        if not isinstance(data["inputs"], list) or not data["inputs"]:
            raise RuntimeError(f"{path.relative_to(ROOT)} must contain a non-empty inputs list")
        if not isinstance(data["problems"], list) or not data["problems"]:
            raise RuntimeError(f"{path.relative_to(ROOT)} must contain a non-empty problems list")
        for problem in data["problems"]:
            missing_problem = required_problem_keys - set(problem)
            if missing_problem:
                raise RuntimeError(
                    f"{path.relative_to(ROOT)} contains a problem entry missing keys: {sorted(missing_problem)}"
                )
            aggregate_table = problem["aggregate_table"]
            if not isinstance(aggregate_table, dict):
                raise RuntimeError(f"{path.relative_to(ROOT)} aggregate_table must be an object")
            if not isinstance(aggregate_table.get("headers"), list) or not isinstance(aggregate_table.get("rows"), list):
                raise RuntimeError(f"{path.relative_to(ROOT)} aggregate_table must contain headers and rows lists")


def normalize_generated_doc(text: str) -> str:
    lines = text.splitlines()
    normalized: list[str] = []
    runtime_column: int | None = None
    in_runtime_table = False

    for line in lines:
        if line.startswith("|") and "Runtime ms/request" in line:
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            runtime_column = headers.index("Runtime ms/request")
            in_runtime_table = True
            normalized.append(line)
            continue

        if in_runtime_table and line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            is_separator = all(re.fullmatch(r":?-+:?", cell) for cell in cells)
            if not is_separator and runtime_column is not None and runtime_column < len(cells):
                cells[runtime_column] = "<runtime>"
                line = "| " + " | ".join(cells) + " |"
            normalized.append(line)
            continue

        if in_runtime_table and not line.startswith("|"):
            in_runtime_table = False
            runtime_column = None

        normalized.append(line)

    return "\n".join(normalized)


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

    checked_in = ROOT / "docs" / "experiments.md"
    if not checked_in.exists():
        raise RuntimeError("docs/experiments.md is missing")
    if normalize_generated_doc(checked_in.read_text()) != normalize_generated_doc(generated_text):
        raise RuntimeError(
            "docs/experiments.md is stale. Run `python3 scripts/run_design_experiments.py` and commit the refreshed doc."
        )


def validate_generated_history() -> None:
    out_dir = ROOT / "build" / "design_docs_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/snapshot_design_experiments.py",
            "--render-only",
            "--history-dir",
            str((ROOT / "experiments" / "history").relative_to(ROOT)),
            "--docs-path",
            str((out_dir / "experiments_history.md").relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
    )
    generated = out_dir / "experiments_history.md"
    if not generated.exists():
        raise RuntimeError("snapshot_design_experiments.py did not generate experiments_history.md in validation output")
    checked_in = ROOT / "docs" / "experiments_history.md"
    if checked_in.read_text() != generated.read_text():
        raise RuntimeError(
            "docs/experiments_history.md is stale. "
            "Run `python3 scripts/snapshot_design_experiments.py --render-only` and commit the refreshed doc."
        )


def validate_snapshot_compare() -> None:
    snapshots = sorted(HISTORY_DIR.glob("*.json"))
    if len(snapshots) < 2:
        return

    out_dir = ROOT / "build" / "design_docs_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_output = out_dir / "snapshot_compare.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_design_snapshots.py",
            "--output",
            str(compare_output.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
    )
    if not compare_output.exists():
        raise RuntimeError("compare_design_snapshots.py did not generate its validation output")
    if "# Snapshot Comparison" not in compare_output.read_text():
        raise RuntimeError("compare_design_snapshots.py produced an unexpected output format")


def main() -> int:
    validate_docs()
    validate_fixture_manifest()
    validate_history_snapshots()
    modules = experiment_modules()
    if not modules:
        raise RuntimeError("No experiment modules found under experiments/")
    for module_name in modules:
        validate_variant_module(module_name)
    validate_generated_experiments(modules)
    validate_generated_history()
    validate_snapshot_compare()
    print("Design workflow validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
