# CudaRobotics: A GPU-Parallel Suite for Robotics Algorithm Benchmarking

> Status note (2026-07-29): this is a legacy breadth-oriented draft and is not
> submission-ready. Several counts and benchmark claims below predate the
> CudaNav end-to-end stack and must be regenerated before reuse. The
> authoritative current claim ledger is
> [`artifacts/cudarobotics_systems.json`](artifacts/cudarobotics_systems.json);
> it now includes a passing native all-GPU S-course closed loop, but
> intentionally reports the paper as not ready until ROS 2 Jazzy,
> release-profile closed-loop, real-rosbag, and physical multi-GPU evidence is
> attached. The target paper title is now “CudaNav: A Reproducible End-to-End
> GPU Autonomy Stack.”

## Abstract

We present CudaRobotics, an open-source C++/CUDA repository containing 97 GPU-parallel implementations spanning localization, path planning, navigation, mapping, point-cloud processing, differentiable control, neuroevolution, and swarm optimization. Each algorithm provides a CPU vs GPU comparison with GIF-based visualization, enabling direct visual assessment of quality differences from GPU parallelism (e.g., 100 vs 10,000 particles for particle filtering, 50 vs 50,000 samples for DWA). The repository includes research-style extensions: a differentiable MPPI controller with matched-time evaluation across 8 baselines and a 7-DOF manipulator benchmark, neural SDF navigation, a GPU CartPole RL environment, and GPU point-cloud processing with up to 599x speedup. All implementations run on a single consumer GPU with no external dependencies beyond CUDA, OpenCV, and Eigen.

## I. Introduction

GPU acceleration has transformed robotics computation, but most GPU robotics libraries focus on a single domain — simulation [Isaac Gym], point clouds [PCL-GPU], or planning [MPPI-Generic]. Educational and benchmarking resources that span the full robotics stack on GPU remain scarce.

CudaRobotics fills this gap with 97 self-contained CUDA implementations across 8 robotics domains, each with CPU vs GPU comparisons and animated visualizations. The repository serves three purposes:

1. **Benchmarking**: quantitative CPU vs GPU speedup measurements for common robotics algorithms
2. **Education**: visual demonstrations of how GPU parallelism improves algorithm quality (not just speed)
3. **Research**: novel extensions including differentiable MPPI, neural SDF navigation, and GPU neuroevolution

## II. Algorithm Coverage

### Localization (8 implementations)
- Extended Kalman Filter
- Particle Filter (CPU 100 vs CUDA 10,000 particles)
- FastSLAM 1.0
- AMCL (Adaptive Monte Carlo Localization)
- Expansion Reset MCL (independent CUDA reimplementation of Ueda IROS 2004)
- Particle Filter on Episode (PFoE)
- Graph SLAM
- Differentiable Particle Filter (soft-resampling + reparameterized motion noise; end-to-end autodiff via dual numbers — 57% RMSE reduction vs hand-tuned baseline on the 8-landmark tracking task; also supports learnable MLP observation models, including finite-difference tracking-loss fine-tuning to 0.88x handcrafted RMSE on clean Gaussian observations; calibration-learned surrogate observation training reaches 0.72x under range outliers and distance-dependent range bias, 0.84x under occlusion plus hidden kidnap, and range-reset particle injection reduces kidnap-only RMSE to 0.14x the no-injection Gaussian baseline, and an injection-rate / ESS-trigger ablation shows recovery saturating near 6% fixed rate with an ESS-triggered 12% policy reaching 0.09x while firing on only 77 of 240 post-jump steps; see `paper/diff_pf_observation_scenarios.md`)

### Path Planning (16 implementations)
- A*, Dijkstra (grid-based)
- RRT, RRT*, Informed RRT*, RRT-Connect (sampling-based)
- RRT* 3D (drone planning)
- RRT* Reeds-Shepp
- PRM (Probabilistic Roadmap)
- Hybrid A*
- State Lattice Planner
- Voronoi Road Map
- Potential Field (2D and 3D)

### Navigation and Control (12 implementations)
- Dynamic Window Approach (CPU 50 vs CUDA 50,000 samples)
- Frenet Optimal Trajectory
- MPPI (Model Predictive Path Integral)
- Diff-MPPI (with 8 feedback baselines, 7-DOF arm benchmark)
- STOMP (Stochastic Trajectory Optimization)
- LQR Speed-Steer Control
- Multi-Robot Planner (500 robots collision avoidance)
- ORCA (Optimal Reciprocal Collision Avoidance)
- Social Force Model

### Mapping and Perception (10 implementations)
- Occupancy Grid Mapping
- ICP (Iterative Closest Point)
- NDT (Normal Distributions Transform)
- Value Iteration
- ESDF via Jump Flooding (2D Euclidean distance transform; 640K cells in 0.193 ms)
- ESDF 3D via Jump Flooding (3D distance transform with 26-neighbour propagation; 1.05M voxels in 0.854 ms)
- 3D Voxel Map (log-odds occupancy with 3D DDA raycasting; 65K rays/scan in 0.61 ms)
- Massive Parallel Collision Checker (2D DDA segment check; 1M candidate segments/scan in 0.49 ms)

### Point-Cloud Processing (5 implementations)
- Voxel Grid Filter
- Statistical Outlier Removal (492x speedup at 2K points)
- Normal Estimation (599x speedup at 2K points)
- RANSAC Plane Detection (136x speedup at 20K points)
- GICP (Generalized ICP)

### GPU Learning (4 implementations)
- GPU Neuroevolution (4096 parallel policies)
- MiniIsaacGym (parallel CartPole simulation)
- GPU REINFORCE training
- Neural SDF learning + navigation

### Swarm Optimization (4 implementations)
- Particle Swarm Optimization
- Differential Evolution
- CMA-ES
- Ant Colony Optimization (TSP)

### Research Extensions (8 implementations)
- Forward-mode autodiff engine (dual numbers)
- GPU MLP training/inference
- Differentiable MPPI with matched-time evaluation
- 7-DOF manipulator benchmark with parallelized gradient
- Mechanism analysis (gradient freshness)
- Pareto frontier analysis
- Neural SDF for MPPI planning
- ESDF-MPPI (JFA distance field + bilinear lookup cost; 0.335 ms / rollout iter with K=4096, T=30)
- Comparison visualization framework

## III. Design Principles

1. **Self-contained**: each algorithm is a single .cu file with minimal shared headers
2. **CPU vs GPU comparison**: every implementation includes a CPU baseline using the same algorithm
3. **Visual output**: OpenCV-based GIF generation for direct quality comparison
4. **No external robotics dependencies**: only CUDA, OpenCV, and Eigen required
5. **Single GPU**: all benchmarks run on one consumer desktop GPU

## IV. Representative Results

### Speedup Highlights

| Domain | Operation | CPU | GPU | Speedup |
|---|---|---|---|---|
| Point Cloud | Normal Estimation (2K pts) | 574 ms | 0.96 ms | 599x |
| Point Cloud | Statistical Filter (2K pts) | 389 ms | 0.79 ms | 492x |
| Point Cloud | RANSAC Plane (20K pts) | 142 ms | 1.04 ms | 136x |
| Mapping | ESDF (per cell, JFA vs brute force) | 16.13 us | 0.30 ns | 53,404x |
| Mapping | ESDF 3D (per voxel, JFA-3D vs brute force) | 70.58 us | 0.81 ns | 86,613x |
| Mapping | 3D Voxel Map (per ray, DDA log-odds) | 547 ns | 9.3 ns | 58x |
| Mapping | Collision Check (per candidate, 2D DDA) | 596 ns | 0.47 ns | 1,277x |
| Planning | RRT* Rewire (per node, parallel vs sequential) | 26.84 us | 0.43 us | 62x |
| Navigation | DWA (50K samples) | — | real-time | 1000x sample count |
| Localization | Particle Filter (10K particles) | — | real-time | 100x particle count |
| Control | Diff-MPPI gradient (7-DOF) | 13.6 ms | 0.79 ms | 17x |

### Quality Improvements from GPU Parallelism

GPU enables qualitatively different results, not just faster computation:
- Particle filter: 10,000 particles produce smooth, accurate tracking where 100 particles fail
- DWA: 50,000 samples enable smooth obstacle avoidance where 50 samples produce jerky motion
- Multi-robot: 500 robots with collision avoidance (infeasible on CPU at real-time rates)

### Diff-MPPI Research Results

On `dynamic_slalom` at matched 1.0 ms: diff_mppi_3 is the only successful method across 6 non-hybrid baselines. On 7-DOF manipulation: diff_mppi_3 success=1.00 at 0.84 ms vs feedback_mppi_ref 0.75 at 4.01 ms.

## V. Related Open-Source GPU Robotics

| Project | Focus | Algorithms |
|---|---|---|
| MPPI-Generic | MPPI variants | 3 (MPPI, Tube-MPPI, RMPPI) |
| cuNRTO | Trajectory optimization | 1 (SQP-based NRTO) |
| Isaac Gym | Physics simulation | Environments, not algorithms |
| PCL-GPU | Point cloud | Selected GPU accelerations |
| **CudaRobotics** | **Full robotics stack** | **97 algorithms, 8 domains** |

## VI. Availability

Repository: https://github.com/rsasaki0109/CudaRobotics
Gallery: https://rsasaki0109.github.io/CudaRobotics/
License: [TBD]

Build: CMake >= 3.18, CUDA >= 11.0, OpenCV 3.x/4.x, Eigen 3.
