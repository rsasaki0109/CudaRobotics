# MathematicalRobotics migration matrix

This is the completion ledger for the full migration effort. The upstream
snapshot is `scomup/MathematicalRobotics` `main` at
`79600010f0c86179905a6960e5fce2bb7cc85d77` (checked on 2026-08-02).

The rule is simple: every upstream algorithmic module must end as `DONE`,
`ADAPT`, or `EXCLUDE` with a reason. `PLANNED` means work remains and does not
count as migration completion.

## Status definitions

| Status | Meaning |
|---|---|
| `DONE` | Native CudaRobotics implementation exists and has a matching test or executable. |
| `ADAPT` | The algorithm exists locally, but its API, numerical convention, or backend integration still has to be made equivalent and verified. |
| `PLANNED` | The algorithm is not yet implemented or audited. |
| `EXCLUDE` | It is a GUI, plotting, static asset, dataset, packaging wrapper, or a duplicate with a documented local replacement; it is not an unclassified algorithm. |

## Algorithm and infrastructure ledger

| Upstream path(s) | CudaRobotics destination | Status | Completion evidence / remaining work |
|---|---|---|---|
| `mathR/utilities/math_tools.py` | `include/cudarobotics/lie_group_math.cuh`, `math_tools.hpp`, `numerical_derivative.hpp` | `DONE` | Exp/log, left/right Jacobians, HSO3/dLogSO3/dHinvSO3, quaternion conversion, array adapters, transforms, and numerical derivatives are covered by `test_lie_group_math` and `test_mathr_native`. |
| `mathR/utilities/robust_kernel.py` | `include/cudarobotics/robust_loss.cuh` | `DONE` | L2, Huber, pseudo-Huber, and Cauchy are covered by `test_robust_loss`. |
| `mathR/utilities/g2o_io.py` | `include/cudarobotics/g2o_io.hpp` | `DONE` | Dependency-free SE(2)/SE(3) reader, upper-triangle information expansion, and quaternion-to-transform conversion are covered by `test_g2o_io` and `test_graph_g2o`. |
| `mathR/utilities/polygon.py` | `include/cudarobotics/polygon.hpp` | `DONE` | Non-convex containment, edge inclusion, and threshold residual semantics are covered by `test_mathr_native`. |
| `mathR/utilities/pcd_io.py` | Existing point-cloud I/O and registration loaders | `EXCLUDE` | File-format plumbing is not a robotics algorithm; native point-cloud loaders already exist where needed. Document any required format adapter in the benchmark. |
| `mathR/utilities/plot_tools.py`, `gl_objects.py` | CudaRobotics video/demo utilities | `EXCLUDE` | Visualization helpers are replaced by existing OpenCV/CUDA video paths. |
| `mathR/utilities/test.py` | CTest and focused CPU/GPU tests | `DONE` | Deterministic CTest replacements cover every native primitive and CPU/GPU parity boundary. |
| `mathR/graph_optimization/graph_solver.py` | `include/cudarobotics/graph_optimization.hpp` plus GPU block assembly/PCG | `DONE` | Native SE(2)/SE(3) vertex/edge ownership, right retraction, robust weighting, damping, and GN iteration control are covered by `test_graph_optimization`; GPU scaling remains in the existing pose graph targets. |
| `mathR/graph_optimization/demo_*.py`, `plot_pose.py` | `src/gpu_pose_graph_slam*.cu`, `test_graph_g2o` | `DONE` | Native GPU pose-graph benchmarks and deterministic SE(2)/SE(3) g2o replays cover the numerical demo paths; plotting is replaced by existing video utilities. |
| `mathR/imu_preintegration/preintegration.py` | `include/cudarobotics/imu_preintegration.hpp` | `DONE` | Fixed-size preintegration, calibration, lever arm, bias correction, prediction, and CPU/GPU parity are tested. |
| `mathR/imu_preintegration/imu_factor.py` | `include/cudarobotics/imu_graph.hpp` | `DONE` | IMU preintegration, bias prior/change, NavState prior/transition, position-velocity, 2-D projection, and `ImuFactorGraph15` state+bias blocks are covered by `test_imu_graph`; CUDA block-linearization parity is covered by `test_imu_preintegration_gpu`. |
| `mathR/imu_preintegration/demo_*.py`, `test_imu_predict.py` | `test_imu_graph`, `test_imu_preintegration_gpu` | `DONE` | Deterministic CPU replay, CUDA factor replay, and 15DoF graph convergence replace plot-driven demos. |
| `mathR/filter/ekf.py` | `include/cudarobotics/filters.hpp`, `src/extended_kalman_filter.cpp` | `DONE` | State2D, odometry Jacobians, GPS correction, covariance propagation, and a deterministic correction gate are covered by `test_mathr_native`. |
| `mathR/filter/particle_filter.py` | `include/cudarobotics/filters.hpp`, `src/particle_filter.cpp`, `src/particle_filter.cu` | `DONE` | Gaussian measurement weighting, normalization, ESS, systematic resampling, and deterministic estimate coverage are in `test_mathr_native`; existing PF/AMCL executables remain the scalable paths. |
| `mathR/filter/config.py`, `draw_cov.py`, `plt_tools.py` | Test fixtures and CudaRobotics visualization | `EXCLUDE` | Configuration/plotting wrappers are not algorithmic; retain only parameters needed by the parity tests. |
| `mathR/filter/demo_ekf.py`, `demo_pf.py` | Existing EKF/PF executables and `test_mathr_native` | `DONE` | Numerical demo cases are represented by deterministic native filter regression tests and existing executables. |
| `mathR/optimization/gauss_newton.py` | `include/cudarobotics/gauss_newton.hpp`, graph/BA backends | `DONE` | Reusable residual/Jacobian block assembly, damping, robust weighting, and manifold plus callback are covered by `test_mathr_native`. |
| `mathR/optimization/demo_*.py` | CudaRobotics benchmark targets | `EXCLUDE` | Demo entrypoints are replaced by CTest/benchmark targets after the common GN core is verified. |
| `mathR/kinematics/transfrom_imu.py`, `transfrom_velocity.py` | `include/cudarobotics/kinematics.hpp` | `DONE` | Frame, velocity, angular-rate, IMU lever-arm, 12DoF input model, pose-matrix adapter, and state-retraction conventions are covered by rigid-motion tests in `test_mathr_native`. |
| `mathR/kinematics/demo_*.py` | Kinematics smoke executable | `EXCLUDE` | Presentation demos are not copied; their numerical cases become tests for the native API. |
| `mathR/robot_geometry/basic_geometry.py` | `include/cudarobotics/geometry.hpp` | `DONE` | PCA line fit, least-squares plane fit, point-line and point-plane residuals are covered by `test_mathr_native`; existing ICP/GICP/NDT consume the same row-major math conventions. |
| `mathR/robot_geometry/demo_p2line_matching.py`, `demo_p2plane_matching.py` | `test_mathr_native`, existing registration targets | `DONE` | Numerical matching cases are represented by deterministic geometry tests and native registration executables. |
| `mathR/robot_geometry/demo_plane_cross_cube.py`, `geometry_plot.py` | Geometry tests/video | `EXCLUDE` | Convert the mathematical cases to tests; plotting/UI code is not ported. |
| `mathR/imls/imls.py` | `include/cudarobotics/imls.hpp` | `DONE` | Local normal PCA and weighted nearest-neighbor IMLS query are validated on a fixed scan line in `test_mathr_native`. |
| `mathR/slam/projection.py` | `include/cudarobotics/projection.hpp` / BA path | `DONE` | T_cw/T_wc camera transform, reprojection, body-camera composition, camera/point prior and between factors, pose plus/minus, undistortion, analytic Jacobians, and CUDA parity are covered by CPU/native and `test_projection_gpu`. |
| `mathR/slam/demo_bundle_adjustment.py`, `load_ba_datasets.py` | `include/cudarobotics/bal_io.hpp`, `include/cudarobotics/bundle_adjustment.hpp`, `src/gpu_bundle_adjustment.cu` | `DONE` | BAL parsing, 3D camera-point reprojection BA reference, and scalable CUDA Schur BA are covered by `test_mathr_native` and existing GPU BA target. |
| `mathR/slam/test_predict.py` | CTest projection/BA regression | `DONE` | Deterministic BAL, reprojection, and solver regressions replace the Python test script. |
| `mathR/slam/gui.py`, `demo_*.py` | Native demo/benchmark targets | `EXCLUDE` | GUI is not a reusable CUDA algorithm; numerical demos are represented by executable benchmarks. |
| `mathR/lie_group/cube_rotation.py`, `small_rotation.py` | Lie-group tests/examples | `DONE` | Small-angle and near-pi rotation cases are retained in `test_lie_group_math` and CUDA Lie/IMU smoke coverage. |
| `mathR/lie_group/demo_se3.py`, `gui.py` | Existing Lie demo/video | `EXCLUDE` | UI/visualization wrapper only; the underlying SE(3) behavior is covered by the common math tests. |
| `mathR/data/**` | Test fixtures / external-data adapters | `EXCLUDE` | Do not duplicate upstream datasets into the library; use reproducible fixtures or documented download paths for parity benchmarks. |
| `docs/*.md`, `docs/*.pdf`, `imgs/**` | `docs/mathr_port.md`, migration matrix, native benchmark docs | `DONE` | Conventions, destination APIs, exclusions, and MIT attribution are documented; static PDFs/GIFs are intentionally not runtime code. |
| `setup.py`, `requirements.txt`, package `__init__.py` files | CMake targets and CudaRobotics headers | `EXCLUDE` | Python packaging metadata is replaced by the native build system. |

## Completion gate

This effort is not complete while any row above is `PLANNED`, or while an
`ADAPT` row lacks a parity test and a documented destination. The final audit
will also run the upstream-path inventory against this table so newly added
algorithm files cannot silently escape classification.

For the fixed upstream snapshot above, the ledger currently contains no
`PLANNED` or `ADAPT` rows.

The final inventory check re-read all 111 upstream blobs (55 algorithmic
`mathR` blobs, excluding `mathR/data`). Every algorithm root was present in this ledger;
the only root-level exception was `mathR/__init__.py`, covered by the
packaging row above.
