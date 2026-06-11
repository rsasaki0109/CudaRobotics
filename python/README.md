# cudarobotics (Python)

Experimental Python bindings for the CudaRobotics GPU MPPI planner and
point-cloud registration algorithms.

## Requirements

- Linux x86_64 with an NVIDIA GPU (compute capability >= 3.5)
- CUDA Toolkit >= 12.0 (nvcc on `PATH` or set via `CUDACXX`)
- CMake >= 3.18, C++17 compiler

## Install

Editable install from a clone (recommended for development):

```bash
pip install -e python/
pip install -e 'python/[examples]'
```

Install from source distribution (PyPI-style sdist; compiles against local CUDA):

```bash
pip install cudarobotics  # once published
# or from a built sdist:
pip install dist/cudarobotics-*.tar.gz
```

Pre-built manylinux wheels are built on every `master` push via GitHub Actions
(`cibuildwheel` job in `.github/workflows/python-package.yml`) and uploaded as
the `cudarobotics-manylinux-wheels` artifact. They still require a compatible
NVIDIA driver at runtime.

## Quick test

```bash
python -c "import cudarobotics as cr; print(cr.__version__)"
python examples/python/mppi_quickstart.py
python examples/python/registration_quickstart.py
```

## DLPack Costmaps

`MppiPlanner.compute()` accepts CUDA DLPack producers for the `costmap`
argument, so PyTorch or CuPy costmaps can stay on the GPU. NumPy and other CPU
buffer-protocol arrays continue to use the existing host path. Set
`distance_field_weight > 0` to enable the optional GPU distance-field
clearance critic built from the same costmap.

```python
import numpy as np
import torch
import cudarobotics as cr

planner = cr.MppiPlanner(
    batch_size=2048,
    time_steps=56,
    model_dt=0.05,
    path_angle_weight=0.25,
    distance_field_weight=12.0,
    distance_field_cutoff=0.8,
)
costmap = torch.zeros((200, 200), dtype=torch.uint8, device="cuda")
path = np.array([[1.0, 5.0], [5.0, 5.0]], dtype=np.float32)
v, vy, w, info = planner.compute(
    (1.0, 5.0, 0.0), costmap, path, (5.0, 5.0, 0.0), resolution=0.05
)
```

## Layout

- `src/cudarobotics/` — pure Python package + nanobind module
- `core/` — bundled CUDA sources (`include/` + registration/MPPI `.cu` files).
  In a git checkout these are symlinks into the repo root; sdist/wheels copy
  the resolved files.

See the repository root [`readme.md`](../readme.md) for algorithm docs and demos.
