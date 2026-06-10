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

## Layout

- `src/cudarobotics/` — pure Python package + nanobind module
- `core/` — bundled CUDA sources (`include/` + registration/MPPI `.cu` files).
  In a git checkout these are symlinks into the repo root; sdist/wheels copy
  the resolved files.

See the repository root [`readme.md`](../readme.md) for algorithm docs and demos.
