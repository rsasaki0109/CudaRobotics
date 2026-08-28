import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize_adoption_results.py"


def load_script():
    spec = importlib.util.spec_from_file_location("summarize_adoption_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_result(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summarizes_activation_failures_and_continuation_proxy(tmp_path):
    module = load_script()
    paths = [
        write_result(
            tmp_path / "python_pass.json",
            {
                "schema_version": 1,
                "surface": "python_quickstart",
                "passed": True,
                "duration_seconds": 60,
            },
        ),
        write_result(
            tmp_path / "python_fail.json",
            {
                "schema_version": 1,
                "surface": "python_quickstart",
                "passed": False,
                "failure_category": "version_mismatch",
                "duration_seconds": 0.01,
            },
        ),
        write_result(
            tmp_path / "dlpack.json",
            {
                "schema_version": 1,
                "recipe": "dlpack_costmap",
                "passed": True,
                "duration_seconds": 42,
            },
        ),
        write_result(
            tmp_path / "variant.json",
            {
                "schema_version": 1,
                "surface": "python_quickstart",
                "recipe": "planning_variant",
                "passed": True,
                "duration_seconds": 58,
            },
        ),
    ]
    summary = module.summarize(paths, "2026-W35")
    assert summary["initial_activation_rate"] == 0.5
    assert summary["known_failure_classification_rate"] == 1.0
    assert summary["continuation_recipe_completion_proxy"] == 2
    assert summary["surfaces"]["python_planning_variant"]["passed"] == 1
    assert summary["surfaces"]["python_quickstart"]["duration_seconds_p90"] == 60
    assert "not a user retention rate" in module.render_markdown(summary)


def test_derives_ros_duration_from_timestamps(tmp_path):
    module = load_script()
    result = write_result(
        tmp_path / "ros.json",
        {
            "schema_version": 1,
            "profile": "smoke",
            "summary_gate": {"passed": True},
            "passed": True,
            "started_at": "2026-08-28T00:00:00+00:00",
            "finished_at": "2026-08-28T00:02:30+00:00",
        },
    )
    summary = module.summarize([result], "2026-W35")
    assert summary["surfaces"]["ros2_cudanav"]["duration_seconds_median"] == 150
    assert summary["continuation_recipe_completion_proxy"] == 0
    assert summary["integration_completion_proxy"] == 1


def test_rejects_unknown_schema(tmp_path):
    module = load_script()
    result = write_result(
        tmp_path / "unknown.json",
        {"schema_version": 1, "passed": True},
    )
    try:
        module.summarize([result], "2026-W35")
    except ValueError as exc:
        assert "unrecognized" in str(exc)
    else:
        raise AssertionError("unknown schema was accepted")
