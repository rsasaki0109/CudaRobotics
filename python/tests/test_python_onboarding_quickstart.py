import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "python" / "onboarding_quickstart.py"


def load_script():
    spec = importlib.util.spec_from_file_location("onboarding_quickstart", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_failure_classification():
    module = load_script()
    assert module.classify_failure("ModuleNotFoundError: No module named 'cudarobotics'") == "import"
    assert module.classify_failure("CUDA driver version is insufficient") == "cuda_runtime"
    assert module.classify_failure("goal not reached") == "algorithm_check"
    assert module.classify_failure("unexpected child failure") == "unknown"


def test_default_output_stays_in_build_tree():
    module = load_script()
    args = module.parse_args([])
    assert args.output_dir == ROOT / "build" / "onboarding" / "python"
    assert args.recipe == "initial"


def test_source_version_matches_package_metadata():
    module = load_script()
    assert module.source_version() == "0.3.0"


def test_missing_gif_becomes_classified_artifact_failure(tmp_path):
    module = load_script()
    step = {"passed": True}
    module.require_mppi_gif(step, tmp_path / "missing.gif")
    assert step["passed"] is False
    assert step["failure_category"] == "artifact"


def test_documented_next_steps_exist():
    source = SCRIPT.read_text(encoding="utf-8")
    for relative in (
        "examples/python/mppi_dlpack_costmap.py",
        "docs/onboarding_recipes.md",
        "ros2_ws/src/cuda_mppi_controller/",
    ):
        assert relative in source
