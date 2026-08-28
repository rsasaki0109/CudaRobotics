import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "colab" / "cudarobotics_quickstart.ipynb"


def _cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _cell_index(cells, marker):
    matches = [index for index, source in enumerate(cells) if marker in source]
    assert len(matches) == 1, f"expected one cell containing {marker!r}, got {matches}"
    return matches[0]


def test_colab_preflight_precedes_source_build():
    cells = _cells()
    assert _cell_index(cells, "GPU runtime not detected") < _cell_index(
        cells, "git clone --depth 1 --branch v0.3.0"
    )


def test_colab_completion_runs_after_both_demos():
    cells = _cells()
    completion = _cell_index(cells, 'print("\\n✅ Quickstart complete")')
    assert _cell_index(cells, "trajectory = [state[:2].copy()]") < completion
    assert _cell_index(cells, "rot_err_deg = np.degrees") < completion
    assert completion < _cell_index(cells, "## Choose your next step")


def test_colab_completion_artifact_has_required_checks():
    cells = _cells()
    completion = cells[_cell_index(cells, 'print("\\n✅ Quickstart complete")')]
    for marker in (
        '"schema_version": 1',
        '"surface": "colab_quickstart"',
        '"release": "v0.3.0"',
        '"mppi"',
        '"registration"',
        '"artifacts"',
        'Path("cudarobotics_quickstart_result.json")',
        'quickstart_result["passed"]',
        'mppi_gif_bytes',
        '"duration_seconds"',
    ):
        assert marker in completion
