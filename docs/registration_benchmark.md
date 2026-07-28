# Registration Unified Benchmark

The v0.2 registration suite compares CudaRobotics GPU registrars and optional
external CPU baselines on identical deterministic point-cloud pairs.

## Included Algorithms

- CudaRobotics FilterReg, FGR, robust point-to-plane, robust point-to-point,
  and Sinkhorn rigid registration
- probreg FilterReg and rigid CPD when `probreg` is installed
- Open3D GICP when `open3d` is installed

Every scenario/size/algorithm cell runs in its own process. A failed optional
dependency or timeout is recorded in the CSV and Markdown outputs instead of
invalidating completed cells.

## Run

Quick CudaRobotics-only run:

```bash
python3 scripts/run_registration_suite.py \
  --algorithms cudarobotics_filterreg_gpu cudarobotics_fgr_gpu \
    cudarobotics_robust_p2plane_gpu cudarobotics_robust_treg_gpu \
    cudarobotics_sinkhorn_gpu \
  --scenarios lumpy_partial low_overlap outlier_partial large_offset \
  --sizes 2000 8000 --trials 3 --strict
```

Full run, including installed CPU baselines:

```bash
python3 scripts/run_registration_suite.py \
  --scenarios lumpy_partial low_overlap outlier_partial large_offset \
  --sizes 2000 8000 32000 --trials 3 \
  --csv build/registration_suite/registration_suite.csv
```

The command writes `registration_suite.csv` and `registration_suite.md`.
Default quality gates require median rotation error at or below 5 degrees and
median translation error at or below 0.20 m. Override them with
`--max-rot-error-deg` and `--max-trans-error-m`.
Pass `--strict` in CI and release checks to return a nonzero status for an
unavailable, timed-out, or quality-failing requested cell.

## Python Result API

The tuple-returning `register()` methods remain available. v0.2 also provides a
normalized result object:

```python
import cudarobotics as cr

result = cr.registration.FilterReg().register_result(target_xyz, source_xyz)
print(result.rotation.shape)       # (3, 3)
print(result.translation.shape)    # (3,)
print(result.info["final_rmse"])
```

`RegistrationResult` is a frozen dataclass containing float32 NumPy arrays and
an immutable result binding. The `info` mapping is copied from the native
implementation.
