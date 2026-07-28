# Registration Unified GPU Smoke

Date: 2026-07-28

This is a one-trial release smoke, not a stable performance comparison. It
verifies that every rigid-registration Python binding runs on the same
deterministic partial-overlap input and satisfies the default accuracy gate.

Environment:

- GPU: NVIDIA GeForce GTX 1660 Ti
- Driver: 596.36
- CUDA Toolkit: 12.8
- Python: 3.12.10
- NumPy: 2.4.3
- CudaRobotics: 0.2.0
- Input: 512 target points, 448 source points, 85% overlap, 2 cm noise

| Algorithm | Quality | Median (ms) | Rotation error (deg) | Translation error (m) | RMSE (m) |
|---|---|---:|---:|---:|---:|
| FilterReg | PASS | 1680.68 | 0.050 | 0.0018 | 0.0369 |
| FGR | PASS | 90.08 | 0.113 | 0.0054 | 0.0410 |
| Robust point-to-plane | PASS | 40.87 | 0.763 | 0.0039 | 0.0306 |
| Robust point-to-point | PASS | 51.12 | 0.128 | 0.0023 | 0.0234 |
| Sinkhorn | PASS | 856.48 | 0.319 | 0.0060 | 0.1881 |

All 5/5 cells passed the default median error limits of 5 degrees and 0.20 m.
The raw rows are in
[`registration_unified_smoke_2026-07-28.csv`](registration_unified_smoke_2026-07-28.csv).
Use multiple trials and larger sizes before making performance claims.
