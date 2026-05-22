# CudaRobotics

CUDA-accelerated robotics algorithms (C++/CUDA), based on [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) and [CppRobotics](https://github.com/onlytailei/CppRobotics) plus differentiable extensions.

## Why CUDA?

Same algorithm on CPU and GPU — GPU enables orders of magnitude more particles / samples / rays:

| | |
|---|---|
| **Particle Filter: CPU 100 vs CUDA 10,000** | **Expansion Reset MCL: kidnap recovery (10,000 particles)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_pf_visual.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif" width="400"/> |
| **Multi-Robot: CPU 5 vs CUDA 500** | **DWA: CPU 50 vs CUDA 50,000 samples** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_multi_robot_visual.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_dwa_visual.gif" width="400"/> |
| **3D LiDAR Sim: CPU 16x512 vs CUDA 64x2048 rays** | **Reeds-Shepp Fan: 1M candidate paths / frame** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_lidar3d_sim.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_reeds_shepp_fan.gif" width="400"/> |

## Capability matrix

| Capability | Demo | GPU scale | Headline |
|---|---|---|---|
| Occupancy grid | `comparison_occupancy_grid` | 256x256 | log-odds raycast |
| Collision check | `comparison_collision_check` | 1M segments/scan | 1,277x per candidate |
| Scan matching | `comparison_icp`, `comparison_ndt`, `gicp` | 10K+ points | parallel correspondences |
| Particle filter | `comparison_pf`, `diff_pf`, `diff_pf_mlp` | 10K particles | end-to-end differentiable |
| RRT family | `comparison_rrt*`, `comparison_rrtstar_rewire` | 1M paths / 200K nodes | 5,000x per-path; 62x rewire |
| Voxel map (3D) | `comparison_voxel_map` | 256x256x32 | 58x per ray |
| ESDF (2D/3D) | `comparison_esdf`, `comparison_esdf_3d` | 640K cells / 1.05M voxels | 53,404x / 86,613x |
| LiDAR sim | `comparison_lidar_sim`, `comparison_lidar3d_sim`, `comparison_lidar3d_realistic` | 1M 2D / 131K 3D rays | + 5 physical effects (realistic) |

## SLAM / Multi-view geometry

| | |
|---|---|
| **GPU Bundle Adjustment (1000 poses × 8000 LM, 60k obs, 0.5 ms/iter)** | **GPU LiDAR SLAM frontend (scan-to-scan ICP, 0.68 ms/frame)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_bundle_adjustment.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_lidar_slam.gif" width="400"/> |
| **GPU Pose-Graph SLAM backend (GN+Jacobi-PCG, RMSE 4.88→0.56 m)** | **GPU 3D Gaussian Splatting renderer (~1k Gaussians, 0.94 ms/frame)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pose_graph_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting.gif" width="400"/> |
| **GPU online SLAM (sliding-window W=60 + iSAM-style global pass on loop, 1.7 ms/step, 3.0 → 0.4 m RMSE)** | **GPU NeRF-style volumetric renderer (720×480, 128 samples/ray, 0.83 ms/frame)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_online_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_nerf_volume.gif" width="400"/> |

## Planning / Control

| | |
|---|---|
| **Visibility-aware MPPI (baseline vs −W·V(x,y) visibility)** | **ESDF-MPPI (JFA ESDF + bilinear lookup cost)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/visibility_mppi.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/esdf_mppi.gif" width="400"/> |
| **GPU Multi-Robot Planner (200 robots, parallel BF distance fields)** | **Massive Collision Check (1M segments, 1,277x)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_collision_check.gif" width="400"/> |
| **Massive RRT* Rewire (CPU 2K vs CUDA 200K nodes)** | **3D ESDF (32³ CPU vs 128²×64 CUDA, 86,613x)** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_rrtstar_rewire.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_esdf_3d.gif" width="400"/> |
| **GPU diffusion planner (512 trajectories × 64 waypoints, 120 Langevin steps, 0.03 ms/step)** | |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diffusion_planner.gif" width="400"/> | |

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
| 2D ESDF (640K cells) | **53,404x** per cell (JFA) |
| 3D ESDF (1M voxels) | **86,613x** per voxel (JFA-3D) |
| Massive collision check | **1,277x** per candidate (2D DDA) |
| Normal estimation (10K pts) | **3,171x** (PCA, one thread per point) |
| Pose-graph SLAM (200 nodes) | ~200 ms total, RMSE 4.88 → 0.56 m |
| 3D Gaussian Splatting (~1k Gaussians, 720x480) | **0.94 ms / frame** |

## References

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
- [Probabilistic Robotics](http://www.probabilistic-robotics.org/)
- Diff-MPPI write-up: `paper/`, ablations: `paper/diff_mppi_*_followup.md`
- GitHub Pages gallery: <https://rsasaki0109.github.io/CudaRobotics/>
