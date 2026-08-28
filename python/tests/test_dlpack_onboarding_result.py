import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "python" / "mppi_dlpack_costmap.py"


def load_script():
    spec = importlib.util.spec_from_file_location("mppi_dlpack_costmap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dependency_failure_writes_classified_result(tmp_path):
    module = load_script()

    def unavailable():
        raise RuntimeError("install a CUDA array dependency")

    module.make_device_costmap = unavailable
    result_path = tmp_path / "dlpack_result.json"
    assert module.main(["--result-json", str(result_path)]) == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["recipe"] == "dlpack_costmap"
    assert result["passed"] is False
    assert result["failure_category"] == "dependency"
    assert result["message"] == "install a CUDA array dependency"
    assert result["duration_seconds"] >= 0


def test_default_result_stays_in_build_tree():
    module = load_script()
    assert module.parse_args([]).result_json == Path(
        "build/onboarding/dlpack/dlpack_result.json"
    )


def test_cuda_dlpack_proxy_hides_buffer_and_forwards_protocol():
    module = load_script()

    class FakeCudaArray:
        def __dlpack__(self, *args, **kwargs):
            return (args, kwargs)

        def __dlpack_device__(self):
            return (2, 0)

    proxy = module.CudaDLPackOnly(FakeCudaArray())
    assert proxy.__dlpack__(stream=1) == ((), {"stream": 1})
    assert proxy.__dlpack_device__() == (2, 0)
    assert not hasattr(proxy, "__cuda_array_interface__")


def test_runtime_failure_writes_classified_result(tmp_path, monkeypatch):
    module = load_script()
    module.make_device_costmap = lambda: (object(), "cupy")

    class FailingPlanner:
        def compute(self, *args, **kwargs):
            raise TypeError("DLPack bridge failed")

    monkeypatch.setattr(module.cr, "MppiPlanner", lambda **kwargs: FailingPlanner())
    result_path = tmp_path / "runtime_failure.json"
    assert module.main(["--result-json", str(result_path)]) == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["failure_category"] == "cuda_runtime"
    assert result["backend"] == "cupy"
    assert "DLPack bridge failed" in result["message"]
