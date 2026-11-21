# cudarobotics (Python)

Experimental Python bindings for the CudaRobotics GPU MPPI planner and
point-cloud registration algorithms.

## Requirements

- Linux x86_64 with an NVIDIA GPU (compute capability >= 3.5)
- CUDA Toolkit >= 12.0 (nvcc on `PATH` or set via `CUDACXX`)
- CMake >= 3.18, C++17 compiler

## Install

From a repository clone on supported Linux x86_64 and CPython 3.10/3.12, the
shortest published-wheel path is:

```bash
python scripts/install_python_wheel.py
```

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
python examples/python/onboarding_quickstart.py
```

The onboarding command runs the MPPI and registration quickstarts together
and writes a versioned success or failure result to
`build/onboarding/python/python_quickstart_result.json`. To run the examples
individually:

```bash
python examples/python/mppi_quickstart.py
python examples/python/mppi_dlpack_costmap.py  # requires CUDA PyTorch or CuPy
python examples/python/registration_quickstart.py
```

## CudaNav Navigation (development)

`cudarobotics.navigation` exposes the GPU KISS-ICP odometry, rolling voxel
map, exact 2-D ESDF, and CUDA MPPI cores that the ROS 2 CudaNav packages use,
plus one pipeline object and a CLI:

```bash
cudarobotics-cudanav sim --output-dir build/cudanav_sim --check
cudarobotics-cudanav sequence recording.crkicp --output-dir build/cudanav_seq
```

```python
from cudarobotics.navigation import CudaNavPipeline

nav = CudaNavPipeline()
step = nav.step(points_xyz, path=path_xy)  # (N, 3) scan, (M, 2) path
v, w = step.command
```

By default the occupancy, ESDF and costmap stay on the GPU from the voxel
projection to MPPI (`stage_chain="device"`). The step's grids are copied to
NumPy only when they are read.

Tracked moving objects can be passed as `step(..., dynamic_obstacles=rows)`,
with rows of `(x, y, vx, vy, radius)`, and MPPI then avoids their predicted
trajectories (`MppiPlanner.set_dynamic_obstacles` does the same for the
planner alone). `CudaNavPipeline(tracker=ScanObjectTracker())` instead finds
and tracks moving objects in the scan itself.
`cudarobotics-cudanav sim --scenario crossing` evaluates this
with pedestrians crossing the course; see
[`docs/cudanav_dynamic_obstacles.md`](../docs/cudanav_dynamic_obstacles.md).

`sim` is a synthetic, command-driven closed loop, not real-sensor evidence.
See [`docs/python_cudanav.md`](../docs/python_cudanav.md) for the API, the
output files, and current limits. This surface is not part of a published
release yet.

## Registration Result API

v0.2 keeps the tuple API and adds a normalized dataclass result:

```python
registrar = cr.registration.FilterReg()
result = registrar.register_result(target_xyz, source_xyz)
print(result.rotation.shape, result.translation.shape, result.info)
```

Rigid registrars that support an initial transform accept either a `(3, 3)`
rotation matrix or a flat length-9 array. `register()` continues to return
`(rotation, translation, info)` for compatibility.

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
    curvature_speed_weight=0.0,
    curvature_speed_min=0.18,
    distance_field_weight=12.0,
    distance_field_cutoff=0.8,
)
costmap = torch.zeros((200, 200), dtype=torch.uint8, device="cuda")
path = np.array([[1.0, 5.0], [5.0, 5.0]], dtype=np.float32)
v, vy, w, info = planner.compute(
    (1.0, 5.0, 0.0), costmap, path, (5.0, 5.0, 0.0), resolution=0.05
)
```

Example script:
[`examples/python/mppi_dlpack_costmap.py`](../examples/python/mppi_dlpack_costmap.py).
It tries CUDA PyTorch first, then CuPy, and prints the rollout validity
diagnostics returned by `compute()`.

## MPPI Diagnostics Info

`MppiPlanner.compute()` returns `(v, vy, w, info)`. The `info` dictionary is
intended for controller tuning and failure diagnosis:

| field | meaning |
|---|---|
| `best_cost` | lowest sampled rollout cost in the final iteration |
| `mean_cost` | mean sampled rollout cost in the final iteration |
| `sampled_rollouts` | number of sampled trajectories |
| `valid_rollouts` | sampled trajectories that avoided collision-cost hits |
| `valid_rollout_ratio` | `valid_rollouts / sampled_rollouts` |
| `all_colliding` | all sampled rollouts collided before retreat handling |
| `retreating` | command came from the recovery back-out sequence |

## Layout

- `src/cudarobotics/` — pure Python package + nanobind module
- `core/` — bundled CUDA sources (`include/` + registration/MPPI `.cu` files).
  In a git checkout these are symlinks into the repo root; sdist/wheels copy
  the resolved files.

See the repository root [`readme.md`](../readme.md) for algorithm docs and demos.
