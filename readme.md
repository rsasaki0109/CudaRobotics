# CudaRobotics

CUDA-accelerated C++ implementations of robotics algorithms, based on [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) and [CppRobotics](https://github.com/onlytailei/CppRobotics).

Each algorithm leverages GPU parallelism for significant speedup over CPU-only implementations.

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

## Algorithms

### Localization

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| Particle Filter | `pf` | 1000 particles: predict + weight update + resampling |
| Extended Kalman Filter | *(CPU only)* | 4x4 matrices - no GPU benefit |

#### Particle Filter
Each particle's motion prediction and observation likelihood computation runs as an independent GPU thread. Systematic resampling uses parallel binary search.

<img src="https://ram-lab.com/file/tailei/gif/pf.gif" alt="pf" width="400"/>

#### Extended Kalman Filter
<img src="https://ram-lab.com/file/tailei/gif/ekf.gif" alt="ekf" width="400"/>

### Path Planning

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| A* | `astar_cuda` | Obstacle map construction (grid cells in parallel) |
| Dijkstra | `dijkstra_cuda` | Obstacle map construction (grid cells in parallel) |
| RRT | `rrt_cuda` | Nearest neighbor search + collision checking |
| RRT* | `rrtstar_cuda` | Nearest neighbor + near nodes + rewiring + collision |
| Dynamic Window Approach | `dwa` | ~120K velocity samples evaluated in parallel |
| Frenet Optimal Trajectory | `frenet` | ~140 candidate paths: polynomial solve + spline + collision |
| State Lattice Planner | `slp_cuda` | Parallel lookup table search + trajectory optimization |
| Potential Field | `potential_field` | Grid-parallel potential computation (attractive + repulsive) |
| PRM | `prm_cuda` | Parallel collision check + k-NN + edge collision |
| Voronoi Road Map | `voronoi_road_map` | Jump Flooding Algorithm for parallel Voronoi diagram |

#### A*
Obstacle map is constructed on GPU where each grid cell checks distance to all obstacles in parallel. Search uses CPU priority queue.

<img src="https://ram-lab.com/file/tailei/gif/a_star.gif" alt="a_star" width="400"/>

#### Dijkstra
<img src="https://ram-lab.com/file/tailei/gif/dijkstra.gif" alt="dijkstra" width="400"/>

#### RRT
GPU-accelerated nearest neighbor search with shared-memory reduction. Collision checking also runs on GPU.

<img src="https://ram-lab.com/file/tailei/gif/rrt.gif" alt="rrt" width="400"/>

#### Dynamic Window Approach
All (velocity, yaw_rate) combinations in the dynamic window are evaluated simultaneously on GPU. Each thread simulates a full trajectory and computes goal/speed/obstacle costs. Parallel reduction finds the optimal control.

<img src="https://ram-lab.com/file/tailei/gif/dwa.gif" alt="dwa" width="400"/>

#### Frenet Optimal Trajectory
Each candidate path runs as one GPU thread: quintic/quartic polynomial coefficients solved via Cramer's rule (no Eigen on device), cubic spline evaluation with binary search, collision checking, and cost computation - all fused in a single kernel.

<img src="https://ram-lab.com/file/tailei/gif/frenet.gif" alt="frenet" width="400"/>

#### State Lattice Planner
Multiple target states are optimized simultaneously on GPU. Lookup table search and trajectory optimization (Newton's method with numerical Jacobian) run in parallel.

<img src="https://ram-lab.com/file/tailei/gif/slp.gif" alt="slp" width="400"/>

#### Potential Field
GPU computes the entire potential field in one kernel launch: each thread calculates one grid cell's attractive potential (toward goal) and repulsive potential (from all obstacles). Path following uses gradient descent on CPU.

#### PRM (Probabilistic Road Map)
Three GPU kernels: (1) parallel collision checking of N=500 random samples, (2) parallel k-NN search for roadmap construction, (3) parallel edge collision checking. Dijkstra path search on CPU.

#### Voronoi Road Map
Uses the Jump Flooding Algorithm (JFA) on GPU to construct a Voronoi diagram in O(log N) fully-parallel passes. Each pass, every grid cell checks neighbors at decreasing step sizes and adopts the nearest seed. Road map extracted from Voronoi edges, path found with Dijkstra.

### Path Tracking

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| LQR Steering Control | *(CPU only)* | Sequential control loop |
| LQR Speed+Steering | *(CPU only)* | Sequential control loop |
| MPC | *(CPU only)* | Requires IPOPT solver |

#### LQR Steering Control
<img src="https://ram-lab.com/file/tailei/gif/lqr_steering.gif" alt="lqr_steering" width="400"/>

#### LQR Speed and Steering Control
<img src="https://ram-lab.com/file/tailei/gif/lqr_full.gif" alt="lqr_full" width="400"/>

#### MPC Speed and Steering Control
Requires [CppAD](https://www.coin-or.org/CppAD/Doc/install.htm) and [IPOPT](https://coin-or.github.io/Ipopt/). Uncomment related lines in CMakeLists.txt to build.

<img src="https://ram-lab.com/file/tailei/gif/mpc.gif" alt="mpc" width="400"/>

## Benchmark: CPU vs CUDA

Particle Filter, 100 steps (SIM_TIME=10s):

| Particles | CPU | CUDA | Speedup |
|---|---|---|---|
| 100 | 84 ms | 3.4 ms | **25x** |
| 1,000 | 1,410 ms | 6.9 ms | **204x** |
| 5,000 | 19,417 ms | 12.2 ms | **1,592x** |
| 10,000 | 75,618 ms | 27.2 ms | **2,776x** |

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
