# CudaRobotics

CUDA-accelerated C++ implementations of robotics algorithms, based on [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) and [CppRobotics](https://github.com/onlytailei/CppRobotics).

Each algorithm leverages GPU parallelism for significant speedup over CPU-only implementations.

## Why CUDA? — Visual Quality Difference

GPU enables orders-of-magnitude more particles/samples, resulting in visually better results:

| | |
|---|---|
| **Multi-Robot: CPU 5 robots vs CUDA 500 robots** | **Particle Filter: CPU 100 vs CUDA 10,000 particles** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_multi_robot_visual.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_pf_visual.gif" width="400"/> |
| **DWA: CPU 50 vs CUDA 50,000 samples** | **emcl2: Standard MCL (fails) vs Expansion Reset (recovers)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_dwa_visual.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_emcl2.gif" width="400"/> |
| **Value Iteration: CPU vs CUDA convergence** | **Particle Filter on Episode (PFoE)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_value_iteration.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/pf_on_episode.gif" width="400"/> |

<details>
<summary>All CPU vs CUDA speed comparisons (click to expand)</summary>

| | |
|---|---|
| **500 Robots Collision Avoidance** | **Particle Filter** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_multi_robot.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_pf.gif" width="400"/> |
| **Dynamic Window Approach** | **Frenet Optimal Trajectory** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_dwa.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_frenet.gif" width="400"/> |
| **RRT** | **RRT*** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrt.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrtstar.gif" width="400"/> |
| **A*** | **Dijkstra** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_astar.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_dijkstra.gif" width="400"/> |
| **Potential Field** | **Voronoi Road Map** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_potential_field.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_voronoi.gif" width="400"/> |
| **3D RRT* (Drone)** | **Occupancy Grid Mapping** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrt3d.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_occupancy_grid.gif" width="400"/> |
| **FastSLAM 1.0** | **AMCL** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_fastslam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_amcl.gif" width="400"/> |

</details>

## Requirements
- CMake >= 3.18
- CUDA Toolkit >= 11.0
- OpenCV 3.x / 4.x
- Eigen 3

## Build
```bash
mkdir build
cd build
cmake ../
make -j8
```

Executables are in `bin/`.

### Docker
```bash
docker build -t cuda-robotics .
docker run --gpus all cuda-robotics ./bin/benchmark_pf
```
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Algorithms

### Localization

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| Particle Filter | `pf` | 1000 particles: predict + weight update + resampling |
| Extended Kalman Filter | *(CPU only)* | 4x4 matrices - no GPU benefit |
| **AMCL** | `amcl` | **Adaptive particle count + GPU likelihood field + KLD-sampling** |
| **FastSLAM 1.0** | `fastslam1` | **Particle x Landmark parallel EKF update (SLAM)** |
| **Graph SLAM** | `graph_slam` | **GPU pose graph optimization with CG solver (SLAM)** |

#### Particle Filter
Each particle's motion prediction and observation likelihood computation runs as an independent GPU thread. Systematic resampling uses parallel binary search.

<img src="https://rsasaki0109.github.io/CudaRobotics/pf.gif" alt="pf" width="400"/>

#### FastSLAM 1.0
Combines particle filter (for robot pose) with per-particle EKF (for landmark positions). Each particle independently runs EKF updates for all observed landmarks on GPU. All 2x2 matrix operations (Jacobian, Kalman gain, covariance update) are inline — no Eigen on device.

#### Extended Kalman Filter
<img src="https://rsasaki0109.github.io/CudaRobotics/ekf.gif" alt="ekf" width="400"/>

### Path Planning

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| A* | `astar_cuda` | Obstacle map construction (grid cells in parallel) |
| Dijkstra | `dijkstra_cuda` | Obstacle map construction (grid cells in parallel) |
| RRT | `rrt_cuda` | Nearest neighbor search + collision checking |
| RRT* | `rrtstar_cuda` | Nearest neighbor + near nodes + rewiring + collision |
| **RRT* Reeds-Shepp** | `rrtstar_rs_cuda` | **Batch RS path computation + collision check (nonholonomic)** |
| **Informed RRT*** | `informed_rrtstar_cuda` | **Ellipsoidal sampling + parallel NN/rewiring** |
| **3D RRT*** | `rrtstar_3d_cuda` | **3D nearest neighbor + 3D collision (drone/UAV)** |
| Dynamic Window Approach | `dwa` | ~120K velocity samples evaluated in parallel |
| Frenet Optimal Trajectory | `frenet` | ~140 candidate paths: polynomial solve + spline + collision |
| State Lattice Planner | `slp_cuda` | Parallel lookup table search + trajectory optimization |
| Potential Field | `potential_field` | Grid-parallel potential computation (attractive + repulsive) |
| **3D Potential Field** | `potential_field_3d` | **3D grid-parallel potential (216K+ cells, drone/UAV)** |
| PRM | `prm_cuda` | Parallel collision check + k-NN + edge collision |
| Voronoi Road Map | `voronoi_road_map` | Jump Flooding Algorithm for parallel Voronoi diagram |

#### A*
Obstacle map is constructed on GPU where each grid cell checks distance to all obstacles in parallel. Search uses CPU priority queue.

<img src="https://rsasaki0109.github.io/CudaRobotics/astar.gif" alt="a_star" width="400"/>

#### Dijkstra
<img src="https://rsasaki0109.github.io/CudaRobotics/dijkstra.gif" alt="dijkstra" width="400"/>

#### RRT
GPU-accelerated nearest neighbor search with shared-memory reduction. Collision checking also runs on GPU.

<img src="https://rsasaki0109.github.io/CudaRobotics/rrt.gif" alt="rrt" width="400"/>

#### RRT* Reeds-Shepp
Extends RRT* with car-like kinematics (forward/reverse driving). The key GPU kernel evaluates Reeds-Shepp paths to all candidate parent nodes in parallel — each thread computes the analytical RS path (48 path types: CSC + CCC families), discretizes it, and checks collision along the entire path.

#### Informed RRT*
Extends RRT* with ellipsoidal focused sampling. Once an initial path is found, samples are drawn from an ellipse defined by start, goal, and current best cost — the ellipse shrinks as better paths are found, accelerating convergence. GPU handles parallel NN search, radius search, and collision checking.

#### 3D RRT* (Drone/UAV)
Full 3D extension of RRT* for aerial navigation. Nodes are (x,y,z), obstacles are spheres. GPU kernels handle 3D nearest neighbor search, 3D radius search, and batch 3D collision checking. Visualization shows XY (top) and XZ (side) projections.

#### Dynamic Window Approach
All (velocity, yaw_rate) combinations in the dynamic window are evaluated simultaneously on GPU. Each thread simulates a full trajectory and computes goal/speed/obstacle costs. Parallel reduction finds the optimal control.

<img src="https://rsasaki0109.github.io/CudaRobotics/dwa.gif" alt="dwa" width="400"/>

#### Frenet Optimal Trajectory
Each candidate path runs as one GPU thread: quintic/quartic polynomial coefficients solved via Cramer's rule (no Eigen on device), cubic spline evaluation with binary search, collision checking, and cost computation - all fused in a single kernel.

<img src="https://rsasaki0109.github.io/CudaRobotics/frenet.gif" alt="frenet" width="400"/>

#### State Lattice Planner
Multiple target states are optimized simultaneously on GPU. Lookup table search and trajectory optimization (Newton's method with numerical Jacobian) run in parallel.

<img src="https://rsasaki0109.github.io/CudaRobotics/slp.gif" alt="slp" width="400"/>

#### Potential Field
GPU computes the entire potential field in one kernel launch: each thread calculates one grid cell's attractive potential (toward goal) and repulsive potential (from all obstacles). Path following uses gradient descent on CPU.

#### 3D Potential Field (Drone/UAV)
Extends potential field to 3D with spherical obstacles. GPU computes 216,000+ grid cells (60x60x60) in parallel. Each cell: 3D attractive potential + 3D repulsive potential from all spheres. Gradient descent over 26 neighbors (3^3 - 1). Visualization shows XY and XZ slice heatmaps.

#### PRM (Probabilistic Road Map)
Three GPU kernels: (1) parallel collision checking of N=500 random samples, (2) parallel k-NN search for roadmap construction, (3) parallel edge collision checking. Dijkstra path search on CPU.

#### Voronoi Road Map
Uses the Jump Flooding Algorithm (JFA) on GPU to construct a Voronoi diagram in O(log N) fully-parallel passes. Each pass, every grid cell checks neighbors at decreasing step sizes and adopts the nearest seed. Road map extracted from Voronoi edges, path found with Dijkstra.

### Mapping

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| Occupancy Grid | `occupancy_grid` | Ray-parallel lidar update (360 threads/scan) |

#### Occupancy Grid Mapping
Each lidar ray is processed by one GPU thread using DDA line walking. Log-odds occupancy probability updated along each ray with atomicAdd.

### Multi-Robot

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| Multi-Robot Planner | `multi_robot_planner` | N robots: force computation in parallel |

#### Multi-Robot Collision Avoidance
Each robot computes attractive/repulsive forces from goals, obstacles, and other robots on GPU. Scales to 500+ robots.

### Path Tracking

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| LQR Steering Control | *(CPU only)* | Sequential control loop |
| LQR Speed+Steering | *(CPU only)* | Sequential control loop |
| MPC | *(CPU only)* | Requires IPOPT solver |

#### LQR Steering Control
<img src="https://rsasaki0109.github.io/CudaRobotics/lqr_steering.gif" alt="lqr_steering" width="400"/>

#### LQR Speed and Steering Control
<img src="https://rsasaki0109.github.io/CudaRobotics/lqr_full.gif" alt="lqr_full" width="400"/>

#### MPC Speed and Steering Control
Requires [CppAD](https://www.coin-or.org/CppAD/Doc/install.htm) and [IPOPT](https://coin-or.github.io/Ipopt/). Uncomment related lines in CMakeLists.txt to build.

<img src="https://rsasaki0109.github.io/CudaRobotics/mpc.gif" alt="mpc" width="400"/>

## Benchmark: CPU vs CUDA

### Particle Filter (`bin/benchmark_pf`)
100 steps (SIM_TIME=10s):

| Particles | CPU | CUDA | Speedup |
|---|---|---|---|
| 100 | 84 ms | 3.4 ms | **25x** |
| 1,000 | 1,410 ms | 6.9 ms | **204x** |
| 5,000 | 19,417 ms | 12.2 ms | **1,592x** |
| 10,000 | 75,618 ms | 27.2 ms | **2,776x** |

### Dynamic Window Approach (`bin/benchmark_dwa`)
100 iterations per resolution:

| Samples | CPU | CUDA | Speedup |
|---|---|---|---|
| 9 | 1.1 ms | 1.3 ms | 0.9x |
| 405 | 54 ms | 1.4 ms | **40x** |
| 1,449 | 197 ms | 1.4 ms | **140x** |
| 8,421 | 1,205 ms | 1.7 ms | **705x** |

Run `bin/benchmark_pf` to reproduce.

## CUDA Implementation Patterns

| Pattern | Used In |
|---|---|
| 1 sample = 1 thread (embarrassingly parallel) | PF, DWA, Frenet, State Lattice |
| Shared-memory reduction | PF (weight normalize/mean), DWA (min cost), Frenet (min cost) |
| GPU obstacle map / potential field | A*, Dijkstra, Potential Field |
| GPU nearest neighbor search | RRT, RRT*, PRM |
| Jump Flooding Algorithm (JFA) | Voronoi Road Map |
| Inline linear algebra (Cramer's rule) | Frenet (quintic/quartic solve) |
| cuRAND device-side RNG | PF |

## References
- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- [Probabilistic Robotics](http://www.probabilistic-robotics.org/)
- [The Dynamic Window Approach to Collision Avoidance](https://ieeexplore.ieee.org/document/580977)
- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/publication/224156269)
- [State Space Sampling of Feasible Motions for High-Performance Mobile Robot Navigation](https://www.ri.cmu.edu/pub_files/pub4/howard_thomas_2008_1/howard_thomas_2008_1.pdf)
