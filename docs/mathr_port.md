# MathematicalRobotics native migration

This document records the native migration from
[scomup/MathematicalRobotics](https://github.com/scomup/MathematicalRobotics).
The upstream project is MIT-licensed. The port keeps the upstream attribution
in the relevant headers and is covered by the repository's existing
[`LICENSE.md`](../LICENSE.md). The redistributed upstream notice is kept in
[`mathr_license.md`](mathr_license.md).

The complete path-by-path ledger is
[`mathr_migration_matrix.md`](mathr_migration_matrix.md). Every upstream path
is classified there as a native implementation, an explicitly adapted local
backend, or a documented non-algorithmic exclusion.

## Native destinations

| CudaRobotics component | Upstream source | Purpose |
|---|---|---|
| [`include/cudarobotics/lie_group_math.cuh`](../include/cudarobotics/lie_group_math.cuh) | `mathR/utilities/math_tools.py` | Host/device SO(2), SE(2), SO(3), and SE(3) exponential/logarithm primitives and Jacobians |
| [`include/cudarobotics/math_tools.hpp`](../include/cudarobotics/math_tools.hpp), [`include/cudarobotics/numerical_derivative.hpp`](../include/cudarobotics/numerical_derivative.hpp) | `mathR/utilities/math_tools.py` | v2m/p2m/makeRt/transform adapters, HSO3/dLogSO3, numerical derivatives |
| [`include/cudarobotics/imu_preintegration.hpp`](../include/cudarobotics/imu_preintegration.hpp) | `mathR/imu_preintegration/preintegration.py` | Fixed-size IMU preintegration, calibration, lever arm, bias correction, prediction, and factor linearization |
| [`include/cudarobotics/imu_graph.hpp`](../include/cudarobotics/imu_graph.hpp) | `mathR/imu_preintegration/imu_factor.py` | Bias/NavState/position-velocity/transition factors, 15DoF NavState+bias graph assembly, and CUDA block linearization |
| [`include/cudarobotics/robust_loss.cuh`](../include/cudarobotics/robust_loss.cuh) | `mathR/utilities/robust_kernel.py` | Host/device L2, Huber, pseudo-Huber, and Cauchy rho coefficients |
| [`include/cudarobotics/g2o_io.hpp`](../include/cudarobotics/g2o_io.hpp) | `mathR/utilities/g2o_io.py` | Dependency-free SE(2)/SE(3) g2o reader and quaternion conversion |
| [`include/cudarobotics/graph_optimization.hpp`](../include/cudarobotics/graph_optimization.hpp) | `mathR/graph_optimization/graph_solver.py` | Right-retracted SE(2)/SE(3) graph GN solvers with robust losses |
| [`include/cudarobotics/gauss_newton.hpp`](../include/cudarobotics/gauss_newton.hpp) | `mathR/optimization/gauss_newton.py` | Residual/Jacobian block contract, damping, robust weighting, and manifold plus callback |
| [`include/cudarobotics/kinematics.hpp`](../include/cudarobotics/kinematics.hpp) | `mathR/kinematics/*.py` | 2D/3D velocity and IMU frame transforms, 12DoF input model, and state/pose adapters |
| [`include/cudarobotics/geometry.hpp`](../include/cudarobotics/geometry.hpp) | `mathR/robot_geometry/basic_geometry.py` | PCA line fit, plane fit, point-line and point-plane factors |
| [`include/cudarobotics/imls.hpp`](../include/cudarobotics/imls.hpp) | `mathR/imls/imls.py` | Deterministic local normal estimation and IMLS surface query |
| [`include/cudarobotics/polygon.hpp`](../include/cudarobotics/polygon.hpp) | `mathR/utilities/polygon.py` | Point containment and signed threshold residual |
| [`include/cudarobotics/projection.hpp`](../include/cudarobotics/projection.hpp) | `mathR/slam/projection.py` | T_cw/T_wc camera transforms, reprojection/Jacobians, body-camera composition, camera/point factors, and pose plus/minus |
| [`include/cudarobotics/bal_io.hpp`](../include/cudarobotics/bal_io.hpp), [`include/cudarobotics/bundle_adjustment.hpp`](../include/cudarobotics/bundle_adjustment.hpp) | `mathR/slam/load_ba_datasets.py`, `demo_bundle_adjustment.py` | BAL loader and deterministic 3D camera-point BA reference; large CUDA BA remains the scalable backend |
| [`include/cudarobotics/filters.hpp`](../include/cudarobotics/filters.hpp) | `mathR/filter/ekf.py`, `particle_filter.py` | State2D, odometry EKF, GPS correction, and deterministic particle filter |

The migration keeps CudaRobotics-native storage and GPU backends instead of
copying NumPy/SciPy object graphs. Fixed-size Lie, IMU, kinematics, and
projection operations are host/device; the existing GPU pose-graph and BA
executables provide scalable CUDA paths, while the headers and CTest cases
provide deterministic references.

## Conventions

- Matrices are row-major fixed-size arrays; no Eigen or SciPy types cross the
  core API boundary.
- `SE(3)` exponential-map tangents use `[rho_x, rho_y, rho_z, omega_x, omega_y, omega_z]`.
- The `p2m` convenience adapter follows MathematicalRobotics and stores direct
  translation, while `se3_exp` uses the Lie exponential's `V(rho)` translation.
- An IMU `NavState` stores `R` (body to navigation frame), navigation-frame
  position `p`, and navigation-frame velocity `v`.
- IMU bias correction is evaluated around the preintegrator's stored
  linearization bias. Calibration rotation and lever-arm centripetal
  correction are handled before the integration step.
- `linearize_imu_factor` returns a 9-vector residual, 9x9 state Jacobians, and
  a 9x6 bias Jacobian. `linearize_imu_factor_15` packs these into source
  `[state(9), bias(6)]` and target `[state(9), 0 bias]` blocks.

## SLAM and IMU integration

The GPU 3D pose graph remains a 6DoF pose backend and uses a pose-only edge
bridge for its synthetic odometry benchmark. The full factor is no longer a
future item: `ImuFactorGraph15` owns 15DoF vertices (9DoF NavState plus 6DoF
bias), assembles the analytic factor Jacobians, and is covered by CPU
convergence and CUDA block-linearization parity tests. Both backends use the
same preintegration implementation.

## Verification

From a configured build directory:

```text
cmake --build build --target test_lie_group_math test_imu_preintegration \
  test_robust_loss test_g2o_io test_graph_optimization test_graph_g2o \
  test_mathr_native test_imu_graph --config Release
ctest --test-dir build -C Release -R \
  "test_lie_group_math|test_imu_preintegration|test_robust_loss|test_g2o_io|test_graph_optimization|test_graph_g2o|test_mathr_native|test_imu_graph" \
  --output-on-failure
cmake --build build --target test_imu_preintegration_gpu test_projection_gpu \
  --config Release
ctest --test-dir build -C Release -R \
  "test_imu_preintegration_gpu|test_projection_gpu" --output-on-failure
```

The GPU tests construct preintegration/factor blocks and camera projection
inside CUDA kernels and compare complete outputs with the CPU references.

For a runtime CPU/GPU comparison, run `bin/Release/gpu_pose_graph_slam_3d.exe`.
The reference run on 2026-08-02 used 384 poses and 575 edges and reported
GPU 825.243 ms versus CPU 653.698 ms, with final translation/rotation RMSE of
0.4391 m / 2.9724 deg (GPU) and 0.4398 m / 2.9750 deg (CPU). The small
synthetic graph is intentionally a correctness and integration example; larger
graphs are the intended workload for the CUDA backend, so these timings are
not a performance guarantee.
