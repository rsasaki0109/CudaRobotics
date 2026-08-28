# Onboarding Recipes

Start with the smallest successful run, then choose one next step. Each level
has an observable completion condition so a user does not have to infer
whether the example worked.

## Level 0: First local success

Fast-path requirements: Linux x86_64, an NVIDIA driver and GPU, and CPython
3.10 or 3.12. The published wheel does not require a local CUDA compiler.

```bash
python scripts/install_python_wheel.py
python examples/python/onboarding_quickstart.py
```

For an editable source build, use `pip install -e 'python/[examples]'`; that
path supports Python 3.9+ and additionally needs CUDA Toolkit 12.x, CMake
3.18+, and a C++17 compiler.

Success means the command exits zero and writes:

- `build/onboarding/python/python_quickstart_result.json` with
  `"passed": true`;
- `build/onboarding/python/mppi_quickstart.gif`;
- separate MPPI and registration logs in the same directory.

Validate the artifact binding without rerunning the GPU workload:

```bash
python scripts/validate_python_onboarding.py \
  build/onboarding/python/python_quickstart_result.json
```

If a step fails, the JSON retains `failed_step` and `failure_category`. Start
with that category instead of retrying the complete build without a diagnosis.

## Level 1: Change one planning input

Open `examples/python/mppi_quickstart.py` and change one value only:

- move the wall gap by changing `gy0` and `gy1`;
- change `batch_size=2048` to compare rollout counts;
- move the start or goal pose.

Then retain the baseline and variant separately:

```bash
python examples/python/onboarding_quickstart.py \
  --output-dir build/onboarding/python_variant \
  --recipe planning_variant
```

Completion means both result JSON files pass and the new GIF visibly differs
from the baseline. Record the changed input beside the variant result; timings
from different machines are not directly comparable benchmark claims.

## Level 2: Keep a learning costmap on the GPU

Use an existing CUDA-enabled PyTorch or CuPy environment when possible, then
run the zero-copy DLPack path:

```bash
# Only when neither CUDA PyTorch nor a complete CuPy runtime is installed:
# this CuPy toolkit bundle may download more than 1 GB.
pip install 'cupy-cuda12x[ctk]'
python examples/python/mppi_dlpack_costmap.py
```

Completion means the command exits zero, reports `backend=torch` or
`backend=cupy`, reaches the goal, and writes
`build/onboarding/dlpack/dlpack_result.json` with a non-zero valid rollout
ratio. This is the shortest route from the NumPy quickstart toward a
perception or learning pipeline.

## Level 3: Move to ROS 2 and Nav2

Use the deterministic CudaNav bringup rather than assembling individual nodes
for the first ROS run:

```bash
python3 scripts/run_cudanav_closed_loop.py \
  --profile smoke --output-dir build/cudanav_closed_loop
```

The supported ROS 2 environment and launch contracts are documented in
[`cudanav_closed_loop.md`](cudanav_closed_loop.md) and
[`cudanav_architecture.md`](cudanav_architecture.md). Completion means the
command exits zero, `build/cudanav_closed_loop/manifest.json` has
`"passed": true`, and its retained `mission_summary.json` has
`"smoke_pass": true`. A smoke run is an onboarding result, not the retained
10-minute release claim.

## Choose by goal

| Goal | Recommended next action |
|---|---|
| Tune MPPI behavior | Level 1, then inspect `info` diagnostics in `python/README.md` |
| Connect PyTorch or CuPy | Level 2 |
| Compare registration methods | Run `examples/python/registration_quickstart.py`, then open `docs/registration_benchmark.md` |
| Integrate a robot stack | Level 3 |
| Reproduce a published result | Choose a fixed-seed suite from `docs/reproducibility.md` |

These recipes produce local artifacts only. CudaRobotics does not upload
runtime results or identifiers by default.
