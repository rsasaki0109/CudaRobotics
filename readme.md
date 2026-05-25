# CudaRobotics

CUDA-accelerated robotics algorithms (C++/CUDA), based on [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) and [CppRobotics](https://github.com/onlytailei/CppRobotics) plus differentiable extensions.

## Why CUDA?

Same algorithm on CPU and GPU — GPU enables orders of magnitude more particles / samples / rays:

| | |
|---|---|
| **Particle Filter: CPU 100 vs CUDA 10,000** | **Expansion Reset MCL: kidnap recovery (10,000 particles)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_pf_visual.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif" width="400"/> |
| **MegaParticles-style Stein MCL: 1M range particles, hidden kidnap recovery** | **GPU Global Localization MCL: sensor-reset kidnap recovery (32,768 particles)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_megaparticles_stein_mcl.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_global_localization_mcl.gif" width="400"/> |
| **PF + ESDF observation lookup (10,000 particles)** | **Multi-Robot: CPU 5 vs CUDA 500** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/pf_esdf.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_multi_robot_visual.gif" width="400"/> |
| **DWA: CPU 50 vs CUDA 50,000 samples** | **3D LiDAR Sim: CPU 16x512 vs CUDA 64x2048 rays** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_dwa_visual.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_lidar3d_sim.gif" width="400"/> |
| **Reeds-Shepp Fan: 1M candidate paths / frame** | |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_reeds_shepp_fan.gif" width="400"/> | |

## Capability matrix

| Capability | Demo | GPU scale | Headline |
|---|---|---|---|
| Occupancy grid | `comparison_occupancy_grid` | 256x256 | log-odds raycast |
| Collision check | `comparison_collision_check` | 1M segments/scan | 1,277x per candidate |
| Scan matching | `comparison_icp`, `comparison_ndt`, `gpu_ndt_3d_multires`, `gicp` | 10K+ points | parallel correspondences |
| Pose-graph SLAM | `gpu_pose_graph_slam`, `gpu_pose_graph_slam_3d`, `gpu_pose_graph_slam_3d_robust` | 2D 200 poses / 3D 384 poses | robust 3D rejects 36/36 false loops, 6.95→0.28 m |
| Particle filter | `comparison_pf`, `gpu_global_localization_mcl`, `gpu_megaparticles_stein_mcl`, `diff_pf`, `diff_pf_mlp` | 10K-1M particles | MegaParticles-style range SPF: 14.61 m bootstrap vs 0.097 m recovery |
| RRT family | `comparison_rrt*`, `comparison_rrtstar_rewire` | 1M paths / 200K nodes | 5,000x per-path; 62x rewire |
| Crowd / swarm | `gpu_crowd_swarm` | 10,000 boids with uniform-grid neighbours | 105x vs CPU |
| Assignment / tracking | `gpu_hungarian_assignment`, `gpu_assignment_tracking` | 512 x 64x64 assignment / 128 tracking scenes | 158x Hungarian; 14.0x tracking |
| Interaction graph risk | `gpu_interaction_graph_risk` | 2048 agents x 10 message passes | 76.3x vs CPU |
| SfM / multi-view | `gpu_sfm_mini` | 2048 features x 4 views | 217.0x match + BA vs CPU |
| Sparse linear solvers | `gpu_pcg_solver` | 262K unknowns / 1.31M CSR nnz | 13.4x Jacobi-PCG vs CPU |
| Clustering / graph ML | `gpu_em_gmm`, `gpu_spectral_clustering` | 262K GMM points / 3K dense RBF graph | 90.2x EM; 193x spectral |
| Black-box optimization | `gpu_cma_es` | 3 x 32,768 candidates x 10D | 1,254x objective eval |
| Monte Carlo planning | `gpu_mcts_planner` | 64 scenes x 4096 rollouts x 48 horizon | 712x vs CPU |
| Learning-based planning | `gpu_diffusion_planner`, `gpu_diffusion_policy` | 512 x 64 trajectories / 768 BC samples | analytic score → behavior-cloned denoising policy |
| Voxel map (3D) | `comparison_voxel_map` | 256x256x32 | 58x per ray |
| ESDF (2D/3D) | `comparison_esdf`, `comparison_esdf_3d` | 640K cells / 1.05M voxels | 53,404x / 86,613x |
| LiDAR sim | `comparison_lidar_sim`, `comparison_lidar3d_sim`, `comparison_lidar3d_realistic` | 1M 2D / 131K 3D rays | + 5 physical effects (realistic) |

## SLAM / Multi-view geometry

| | |
|---|---|
| **GPU Bundle Adjustment (1000 poses × 8000 LM, 60k obs, 0.5 ms/iter)** | **GPU LiDAR SLAM frontend (scan-to-scan ICP, 0.68 ms/frame)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_bundle_adjustment.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_lidar_slam.gif" width="400"/> |
| **GPU Pose-Graph SLAM backend (2D GN+Jacobi-PCG, RMSE 4.88→0.56 m)** | **GPU 3D Pose-Graph SLAM v2 (384 poses, finite-difference SE(3) Jacobians, RMSE 1.64→0.28 m)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pose_graph_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pose_graph_slam_3d.gif" width="400"/> |
| **GPU robust 3D Pose-Graph SLAM (36 false loops, switch gate rejects 36/36, plain 6.95 m → robust 0.28 m)** | **GPU online SLAM (sliding-window W=60 + iSAM-style global pass on loop, 1.7 ms/step, 3.0 → 0.4 m RMSE)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pose_graph_slam_3d_robust.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_online_slam.gif" width="400"/> |
| **GPU NeRF-style volumetric renderer (720×480, 128 samples/ray, 0.83 ms/frame)** | **GPU SfM mini (2048 features × 4 views, descriptor match + triangulate + point BA, 217.0x vs CPU)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_nerf_volume.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_sfm_mini.gif" width="400"/> |
| **GPU 3D Gaussian Splatting renderer (~1k Gaussians, 0.94 ms/frame)** | |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting.gif" width="400"/> | |

## Solver infrastructure

| | |
|---|---|
| **GPU Jacobi-PCG sparse SPD solver (262K unknowns, 1.31M CSR nnz, 33 iterations, 13.4x vs CPU)** | |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pcg_solver.gif" width="400"/> | |

## Planning / Control

| | |
|---|---|
| **Visibility-aware MPPI (baseline vs −W·V(x,y) visibility)** | **ESDF-MPPI (JFA ESDF + bilinear lookup cost)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/visibility_mppi.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/esdf_mppi.gif" width="400"/> |
| **GPU Multi-Robot Planner (200 robots, parallel BF distance fields)** | **Massive Collision Check (1M segments, 1,277x)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_collision_check.gif" width="400"/> |
| **Massive RRT* Rewire (CPU 2K vs CUDA 200K nodes)** | **3D ESDF (32³ CPU vs 128²×64 CUDA, 86,613x)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrtstar_rewire.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_esdf_3d.gif" width="400"/> |
| **GPU diffusion policy (768-sample BC MLP prior + diffusion refinement, 512×64 paths)** | **GPU diffusion planner (512 trajectories × 64 waypoints, 120 Langevin steps, 0.03 ms/step)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diffusion_policy.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diffusion_planner.gif" width="400"/> |
| **GPU Hungarian-class assignment (512 × 64x64 dense assignments, 0.082 ms/batch, 158x vs CPU Hungarian)** | **GPU CMA-ES black-box optimization (3 x 32,768 candidates x 10D, 0.025 ms/generation eval, 1,254x objective eval)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_hungarian_assignment.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_cma_es.gif" width="400"/> |
| **GPU MCTS planner (64 scenes x 4096 rollouts x 48 horizon, 1.8 ms/plan, 712x vs CPU)** | **GPU assignment tracking (128 scenes × 48 tracks × 72 detections, gated clutter/miss association, 0.093 ms/update, 14.0x vs CPU)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mcts_planner.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_assignment_tracking.gif" width="400"/> |
| **GPU crowd swarm (10,000 boids, uniform-grid neighbours, 0.275 ms/step, 105x vs CPU)** | **GPU interaction-graph risk propagation (2048 agents, 10 message passes, 76.3x vs CPU)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_crowd_swarm.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_interaction_graph_risk.gif" width="400"/> |

## Differentiable / learning

| | |
|---|---|
| **Differentiable MPPI** | **Differentiable Particle Filter (3 panels)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_mppi.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_pf.gif" width="400"/> |
| **DPF MLP likelihood (3 panels: Gaussian / supervised / tuned)** | **DPF realistic obs (Gaussian / Cauchy / learned MLP)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_pf_mlp.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/pf_realistic_obs.gif" width="400"/> |
| **PF + ESDF observation model** | **Differentiable end-to-end SLAM (Adam-tuned σ)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/pf_esdf.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/diff_e2e_slam.gif" width="400"/> |
| **Neural SDF MPPI** | **Neuroevolution: CPU 100 vs CUDA 4096** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/sdf_mppi.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_neuroevo.gif" width="400"/> |

## Sensors / perception

| | |
|---|---|
| **3D LiDAR Realistic** (noise + divergence + multi-path + reflectivity + rolling shutter) | **3D Voxel Map (log-odds, 256³ scale)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_lidar3d_realistic.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_voxel_map.gif" width="400"/> |
| **Massive 2D LiDAR Sim (1M rays/scan)** | **ESDF JFA (640K cells, 53,404x)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_lidar_sim.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_esdf.gif" width="400"/> |
| **GPU multi-resolution NDT 3D (8x8x4 -> 16x16x6, coarse-to-fine SE(3), 9.5 ms/scenario, 0.016 m avg)** | **GPU NDT 3D point cloud registration (16³ voxel NDT + 6-DOF GN on SE(3), 6.7 ms/scenario, ~0.03 m typical)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_ndt_3d_multires.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_ndt_3d.gif" width="400"/> |
| **GPU NDT 2D scan matching (Newton on NDT grid, 0.54 ms/scenario, ~0.02 m typical)** | **GPU GICP 2D scan matching (per-point cov + nearest-neighbour match, 1.9 ms/scenario, ~0.08 m typical)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_ndt_2d.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gicp_2d.gif" width="400"/> |
| **GPU GICP 3D point cloud registration (per-point cov via Cardano eigendecomp + 6-DOF GN on SE(3), 4.7 ms/scenario, ~1 mm typical)** | |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gicp_3d.gif" width="400"/> | |
| **GPU EM GMM clustering (262K points × 5 full-cov Gaussians, 42 EM iterations, 90.2x vs CPU)** | **GPU spectral clustering (3072-point dense RBF graph, 40 subspace iterations, 193x vs CPU)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_em_gmm.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_spectral_clustering.gif" width="400"/> |

<details>
<summary>More classical-algorithm GIFs</summary>

| | |
|---|---|
| RRT | RRT* |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrt.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrtstar.gif" width="400"/> |
| A* | Dijkstra |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_astar.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_dijkstra.gif" width="400"/> |
| Potential Field | Voronoi Road Map |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_potential_field.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_voronoi.gif" width="400"/> |
| 3D RRT* (drone) | Occupancy Grid Mapping |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrt3d.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_occupancy_grid.gif" width="400"/> |
| FastSLAM 1.0 | AMCL |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_fastslam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_amcl.gif" width="400"/> |
| Value Iteration | PF on Episode |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_value_iteration.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/pf_on_episode.gif" width="400"/> |
| Dynamic Window | Frenet Optimal Trajectory |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_dwa.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_frenet.gif" width="400"/> |
| 500-robot multi-robot | Particle Filter |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_multi_robot.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_pf.gif" width="400"/> |

</details>

## Build

```bash
mkdir build && cd build && cmake .. && make -j$(nproc)
```

Requires CMake ≥ 3.18, CUDA Toolkit ≥ 12.0, OpenCV ≥ 4.5, Eigen 3. Executables go to `bin/`.

ROS2 (optional):

```bash
cd ros2_ws && colcon build --packages-select cuda_robotics
```

## Headline benchmarks

| Domain | Best result |
|---|---|
| Particle Filter (10K) | CPU 75 s → CUDA 27 ms — **2,776x** |
| Dynamic Window (8K samples) | CPU 1.2 s → CUDA 1.7 ms — **705x** |
| Global Localization MCL | 32,768 particles, hidden kidnap; local-only post RMSE 20.24 m → sensor-reset recovery 0.022 m |
| MegaParticles-style Stein MCL | 1,048,576 range particles; local bootstrap post RMSE 14.61 m → Stein/bucket posterior recovery 0.097 m |
| 2D ESDF (640K cells) | **53,404x** per cell (JFA) |
| 3D ESDF (1M voxels) | **86,613x** per voxel (JFA-3D) |
| Massive collision check | **1,277x** per candidate (2D DDA) |
| Normal estimation (10K pts) | **3,171x** (PCA, one thread per point) |
| Pose-graph SLAM (200 nodes) | ~200 ms total, RMSE 4.88 → 0.56 m |
| 3D Pose-graph SLAM | 384 poses / 575 edges, finite-difference SE(3) Jacobians, RMSE 1.64 → 0.28 m |
| Robust 3D Pose-graph SLAM | 384 poses / 611 edges, 36 false loop closures, switch gate rejects 36/36; plain 6.95 m → robust 0.28 m |
| 3D Gaussian Splatting (~1k Gaussians, 720x480) | **0.94 ms / frame** |
| GPU diffusion policy | 768-sample behavior cloning MLP + 512 x 64 learned denoising trajectories |
| GPU CMA-ES objective evaluation | 3 x 32,768 candidates x 10D, **1,254x** vs CPU eval |
| GPU MCTS kinodynamic planning | 64 scenes x 4096 rollouts x 48 horizon, **712x** vs CPU |
| GPU assignment tracking | 128 scenes x 48 tracks x 72 detections, **14.0x** vs CPU |
| GPU crowd swarm | 10,000 agents, uniform-grid neighbours, **105x** vs CPU |
| GPU interaction graph risk | 2048 agents x 10 message-passing steps, **76.3x** vs CPU |
| GPU SfM mini | 2048 features x 4 views, match + point BA, **217.0x** vs CPU |
| GPU Jacobi-PCG sparse solver | 262K unknowns / 1.31M CSR nnz, **13.4x** vs CPU |
| GPU EM GMM clustering | 262K points x 5 full-cov Gaussians, **90.2x** vs CPU |
| GPU spectral clustering | 3072-point dense RBF graph, 40 subspace iterations, **193x** vs CPU |

## References

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- [Probabilistic Robotics](http://www.probabilistic-robotics.org/)
- Koide et al., [MegaParticles: Range-based 6-DoF Monte Carlo Localization](https://arxiv.org/abs/2404.16370)
- Diff-MPPI write-up: `paper/`, ablations: `paper/diff_mppi_*_followup.md`
- GitHub Pages gallery: <https://rsasaki0109.github.io/CudaRobotics/>
