import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "install_python_wheel.py"


def load_script():
    spec = importlib.util.spec_from_file_location("install_python_wheel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selects_published_cp310_and_cp312_assets():
    module = load_script()
    contract = module.load_contract()
    for minor in (10, 12):
        version, requirement = module.wheel_requirement(
            contract,
            system="Linux",
            machine="x86_64",
            implementation="cpython",
            major=3,
            minor=minor,
        )
        tag = f"cp3{minor}"
        assert version == "0.3.0"
        assert f"cudarobotics-0.3.0-{tag}-{tag}" in requirement
        assert requirement.startswith("cudarobotics[examples] @ https://github.com/")


def test_rejects_unsupported_interpreter_and_platform():
    module = load_script()
    contract = module.load_contract()
    try:
        module.wheel_requirement(
            contract,
            system="Linux",
            machine="x86_64",
            implementation="cpython",
            major=3,
            minor=11,
        )
    except ValueError as exc:
        assert "CPython" in str(exc)
    else:
        raise AssertionError("CPython 3.11 was accepted")
    try:
        module.wheel_requirement(
            contract,
            system="Windows",
            machine="AMD64",
            implementation="cpython",
            major=3,
            minor=12,
        )
    except ValueError as exc:
        assert "Linux x86_64" in str(exc)
    else:
        raise AssertionError("Windows was accepted")
