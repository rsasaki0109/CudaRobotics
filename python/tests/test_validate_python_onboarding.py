import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_python_onboarding.py"


def load_script():
    spec = importlib.util.spec_from_file_location("validate_python_onboarding", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def passing_fixture(tmp_path: Path) -> Path:
    (tmp_path / "mppi.log").write_text("mppi passed\n", encoding="utf-8")
    (tmp_path / "registration.log").write_text("registration passed\n", encoding="utf-8")
    (tmp_path / "mppi_quickstart.gif").write_bytes(b"GIF89a")
    result_path = tmp_path / "python_quickstart_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "surface": "python_quickstart",
                "passed": True,
                "duration_seconds": 1.2,
                "steps": [
                    {"name": "mppi", "passed": True, "returncode": 0, "log": "mppi.log"},
                    {"name": "registration", "passed": True, "returncode": 0, "log": "registration.log"},
                ],
                "artifacts": {
                    "mppi_gif": "mppi_quickstart.gif",
                    "result": "python_quickstart_result.json",
                },
            }
        ),
        encoding="utf-8",
    )
    return result_path


def test_accepts_bound_passing_artifacts(tmp_path):
    module = load_script()
    assert module.validate(passing_fixture(tmp_path)) == []


def test_rejects_missing_gif(tmp_path):
    module = load_script()
    result_path = passing_fixture(tmp_path)
    (tmp_path / "mppi_quickstart.gif").unlink()
    assert "MPPI GIF is missing or empty" in module.validate(result_path)


def test_rejects_failed_result(tmp_path):
    module = load_script()
    result_path = passing_fixture(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["passed"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert "passed must be true" in module.validate(result_path)
