# Registration External Baselines (2026-06-11)

Fixed-seed point-cloud registration comparison against external CPU baselines.
This run is meant as a reproducible distribution artifact, not a universal
registration leaderboard.

## Setup

- Data: asymmetric lumpy closed surface, transformed by a fixed rigid motion,
  85% source overlap, Gaussian noise sigma = 0.02 m.
- Initialization: identity for every backend.
- Trials: median of 3 deterministic trials per method/size after one warmup.
- Isolation: each method/size cell runs in a separate subprocess. This keeps
  heavy CPU baselines from poisoning the rest of the run.
- Iterations: 64 maximum iterations for all methods.
- Software: CudaRobotics 0.1.0, probreg 0.3.8, Open3D 0.19.0.
- Load: 1-minute load before cells was recorded in the CSV. The full run used
  a load gate of 12; several CPU-heavy cells still left the machine with load
  near or above that after completion.

## Results

Median wall time and median rigid-transform error. Translation error is shown
in millimeters. Lower is better.

| size | method | median time | rot err | trans err | status |
| ---:| --- | ---:| ---:| ---:| --- |
| 2,000 | CudaRobotics FilterReg (GPU) | 545.1 ms | 0.070 deg | 1.46 mm | ok |
| 2,000 | probreg FilterReg (CPU) | 716.9 ms | 0.000 deg | 1.05 mm | ok |
| 2,000 | probreg CPD rigid (CPU) | 4.57 s | 0.037 deg | 0.81 mm | ok |
| 2,000 | Open3D GICP (CPU) | 180.5 ms | 0.060 deg | 1.65 mm | ok |
| 8,000 | CudaRobotics FilterReg (GPU) | 562.7 ms | 0.055 deg | 0.71 mm | ok |
| 8,000 | probreg FilterReg (CPU) | 2.70 s | 0.000 deg | 0.63 mm | ok |
| 8,000 | probreg CPD rigid (CPU) | 69.3 s | 0.020 deg | 0.41 mm | ok |
| 8,000 | Open3D GICP (CPU) | 134.2 ms | 0.055 deg | 0.48 mm | ok |
| 32,000 | CudaRobotics FilterReg (GPU) | 699.7 ms | 0.142 deg | 0.86 mm | ok |
| 32,000 | probreg FilterReg (CPU) | 7.94 s | 0.000 deg | 0.22 mm | ok |
| 32,000 | probreg CPD rigid (CPU) | - | - | - | exit code -9 |
| 32,000 | Open3D GICP (CPU) | 241.9 ms | 0.020 deg | 0.20 mm | ok |

Same-algorithm FilterReg comparison:

| size | CudaRobotics GPU | probreg CPU | CPU/GPU time ratio |
| ---:| ---:| ---:| ---:|
| 2,000 | 545.1 ms | 716.9 ms | 1.3x |
| 8,000 | 562.7 ms | 2.70 s | 4.8x |
| 32,000 | 699.7 ms | 7.94 s | 11.3x |

## Reading The Numbers

- The cleanest headline is same-algorithm: CudaRobotics FilterReg stays under
  0.70 s from 2k to 32k points, while probreg FilterReg grows from 0.72 s to
  7.94 s in this run.
- probreg CPD is accurate but scales poorly here: 4.57 s at 2k, 69.3 s at 8k,
  and the 32k child process was killed with exit code -9.
- Open3D GICP is the strongest CPU baseline in this identity-init benchmark.
  It is faster than CudaRobotics FilterReg for all three sizes in this run, so
  this artifact should not be read as a universal GPU-over-CPU claim.
- Accuracy is good for all successful methods. The CudaRobotics FilterReg
  median rotation error stays below 0.15 deg and median translation error stays
  below 1.5 mm across the tested sizes.
- probreg FilterReg is run with `update_sigma2=True` (recorded in the CSV as
  `probreg update_sigma2=True`). Without that setting it is not a fair
  baseline on these noisy partial-overlap pairs.

## Reproduce

```bash
python3 -m venv --system-site-packages /path/to/regbench_venv
/path/to/regbench_venv/bin/python -m pip install probreg==0.3.8 -e python/
/path/to/regbench_venv/bin/python scripts/benchmark_registration_external.py \
  --sizes 2000 8000 32000 \
  --trials 3 \
  --scenarios lumpy_partial \
  --timeout-seconds 360 \
  --load-gate 12 \
  --csv docs/results/registration_external_baselines_2026-06-11.csv
```

Open3D 0.19.0 was already available in the system site-packages for this run.
If it is not installed, install `open3d==0.19.0` into the benchmark environment.

Raw aggregate CSV:
[`registration_external_baselines_2026-06-11.csv`](registration_external_baselines_2026-06-11.csv).

Script note: `scripts/benchmark_registration_external.py` now keeps this
scenario as `lumpy_partial` and can also generate follow-up stress rows with
`--scenarios low_overlap outlier_partial large_offset`. Those additional rows
are not part of the 2026-06-11 checked-in numbers above.
