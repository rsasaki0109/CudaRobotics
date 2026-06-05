# CudaRobotics

[![GitHub stars](https://img.shields.io/github/stars/rsasaki0109/CudaRobotics?style=social)](https://github.com/rsasaki0109/CudaRobotics/stargazers)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Gallery](https://img.shields.io/badge/gallery-animated_demos-blue)](https://rsasaki0109.github.io/CudaRobotics/)

<p align="center">
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif">
    <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif" alt="CUDA MPPI racing demo" width="270"/>
  </a>
  <a href="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif">
    <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif" alt="Expansion-reset MCL demo" width="270"/>
  </a>
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif">
    <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif" alt="CUDA multi-robot planner demo" width="270"/>
  </a>
</p>

<p align="center">
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif">MPPI racing</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif">Expansion-reset MCL</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif">Multi-robot planner</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif">Gaussian-Splatting SLAM</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/">full gallery</a>
</p>

CUDA Robotics is a GPU-first robotics playground and benchmark suite for SLAM,
mapping, perception, planning, MPPI control, point-cloud registration, and
learning demos in C++/CUDA.

If you are looking for a reason to star it: this repo turns robotics algorithms
into small, runnable CUDA examples, then records both the speedups and the
failure cases.

The core pattern is simple: keep a CPU reference or robotics baseline where it
helps, then expose the parallel work as one CUDA thread per particle, ray,
candidate pose, graph node, rollout, voxel, feature, or grid cell.

Full animated gallery: https://rsasaki0109.github.io/CudaRobotics/

## Start Here

| Want to see | Open |
|---|---|
| Visual demos | [Full animated gallery](https://rsasaki0109.github.io/CudaRobotics/) |
| Latest fixed-seed MPPI result | [`docs/results/mppi_zoo_smoke_2026-06-05.md`](docs/results/mppi_zoo_smoke_2026-06-05.md) |
| MPPI paper reproduction zoo | [`docs/mppi_reproduction_zoo.md`](docs/mppi_reproduction_zoo.md) |
| Reproducibility suites | [`docs/reproducibility.md`](docs/reproducibility.md) |
| Diff-MPPI paper material | [`paper/`](paper/) |
| Contributing a demo or reproduction | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Current roadmap snapshot | [`docs/next_actions.md`](docs/next_actions.md) |

## Latest Fixed-Seed Result

The checked-in MPPI zoo smoke result was generated on 2026-06-05 with
`dynamic_crossing,narrow_passage`, `K=64,128`, and 3 seeds per
scenario/planner/K cell. It is a smoke benchmark, not a paper-faithful claim,
but it gives a concrete negative control and a reproducible comparison target.

<img src="docs/results/mppi_zoo_smoke_2026-06-05.svg" alt="MPPI Zoo fixed-seed smoke chart" width="900"/>

| Scenario | K | Baseline MPPI | Strongest signal |
|---|---:|---|---|
| `dynamic_crossing` | 64 | success 0.00, final distance 3.21 | `ducct_mppi_smooth` success 1.00, final distance 1.91 |
| `dynamic_crossing` | 128 | success 0.00, final distance 3.39 | `step_mppi_smooth` success 1.00, final distance 1.88 |
| `narrow_passage` | 64 | success 1.00, 253.3 steps | `tsallis_mppi_smooth` success 1.00, 228.3 steps |
| `narrow_passage` | 128 | success 1.00, 251.7 steps | `tsallis_mppi_smooth` success 1.00, 228.0 steps |

Full report and CSV:
[`docs/results/mppi_zoo_smoke_2026-06-05.md`](docs/results/mppi_zoo_smoke_2026-06-05.md)
and
[`docs/results/mppi_zoo_smoke_2026-06-05.csv`](docs/results/mppi_zoo_smoke_2026-06-05.csv).

## Docker Smoke Test

Requires NVIDIA Container Toolkit and a CUDA-capable GPU.

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_smoke.py --bin ./bin/benchmark_diff_mppi --out-dir build/mppi_zoo'
```

## What Makes It Different

- Self-contained C++/CUDA demos instead of framework-heavy examples.
- GPU kernels are shaped around robotics workloads: rollouts, particles, rays,
  voxels, grid cells, graph nodes, feature matches, and candidate poses.
- Reproduction docs include negative results and limitations, not only wins.
- The MPPI stack includes a growing benchmarked zoo of paper-inspired variants.

## MPPI Reproduction Zoo

The MPPI work is now indexed as a reproducible research backlog. Each entry is
a lightweight CUDA implementation plus notes on where the result works, where it
does not, and what would be required for a paper-faithful reproduction.

| Family | What to open first |
|---|---|
| Step-MPPI | [`docs/step_mppi_reproduction.md`](docs/step_mppi_reproduction.md) |
| Tsallis-MPPI | [`docs/tsallis_mppi_reproduction.md`](docs/tsallis_mppi_reproduction.md) |
| DRA-MPPI | [`docs/dra_mppi_reproduction.md`](docs/dra_mppi_reproduction.md) |
| C2U-MPPI | [`docs/c2u_mppi_reproduction.md`](docs/c2u_mppi_reproduction.md) |
| DUCCT-MPPI | [`docs/ducct_mppi_reproduction.md`](docs/ducct_mppi_reproduction.md) |
| DBaS-Log-MPPI | [`docs/dbas_log_mppi_reproduction.md`](docs/dbas_log_mppi_reproduction.md) |
| PA-MPPI | [`docs/pa_mppi_reproduction.md`](docs/pa_mppi_reproduction.md) |
| Full index | [`docs/mppi_reproduction_zoo.md`](docs/mppi_reproduction_zoo.md) |

## Highlights

| Demo | What it shows |
|---|---|
| `gpu_bnb_loop_closure_slam` | Branch-and-bound loop search scores about 957x fewer candidates than brute force while returning the same relpose on 51/51 attempts. |
| `gpu_gaussian_splatting_slam` | RGB-D Gaussian-Splatting SLAM with GPU ray-cast sensor, point-to-plane ICP tracking, and incremental Gaussian map fusion. |
| `gpu_mppi_racing` | MPPI autonomous racing with 2048 x 40 rollouts per control step on the GPU. |
| `gpu_kdtree_nn` | Exact KD-tree nearest-neighbour search for 40k queries, matching brute force while running much faster. |
| `gpu_sgm_stereo` | Semi-Global Matching stereo with CUDA census and path aggregation. |
| `gpu_wavefront_planner` | Bellman-Ford-style cost-to-go relaxation over a 384x384 planning grid. |

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_bnb_loop_closure_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif" width="400"/> |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_sgm_stereo.gif" width="400"/> |

## Gallery

A representative slice per category. The
[full animated gallery](https://rsasaki0109.github.io/CudaRobotics/) has the rest.

### GPU highlights

The most visually striking GPU demos, where massive parallelism really shows.

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_nerf_volume.gif" width="400"/> |
| RGB-D Gaussian-Splatting SLAM | NeRF volume rendering |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_sfm_mini.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_tsdf_fusion.gif" width="400"/> |
| Structure-from-motion (mini) | TSDF fusion |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_crowd_swarm.gif" width="400"/> |
| MPPI autonomous racing (2048 x 40 rollouts) | Crowd / swarm simulation |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_marching_cubes.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_spectral_clustering.gif" width="400"/> |
| Marching Cubes | Spectral clustering |

### SLAM & scan matching

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pose_graph_slam_3d.gif" width="400"/> |
| Gaussian-Splatting SLAM (RGB-D) | 3D pose-graph SLAM |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_bnb_loop_closure_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_correlative_scan_matching.gif" width="400"/> |
| Branch-and-bound loop closure | Correlative scan matching |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_online_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_csm_submap_slam.gif" width="400"/> |
| Online SLAM | Submap loop-closure SLAM |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_lidar_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_bundle_adjustment.gif" width="400"/> |
| LiDAR SLAM | Bundle adjustment |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_kiss_icp.gif" width="400"/> | |
| KISS-ICP-style LiDAR odometry (0.02% drift from scans alone) | |

### Localization & filtering

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_kld_amcl.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_megaparticles_lsh.gif" width="400"/> |
| KLD-AMCL | MegaParticles global localization (LSH) |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_pf_mlp.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif" width="400"/> |
| Differentiable particle filter (MLP) | Expansion-reset MCL |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_global_localization_mcl.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_megaparticles_stein_mcl.gif" width="400"/> |
| Global localization MCL | MegaParticles Stein MCL |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_amcl.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_megaparticles_6dof.gif" width="400"/> |
| AMCL (CPU vs GPU) | MegaParticles 6-DOF |

### Planning & control

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_mppi.gif" width="400"/> |
| MPPI autonomous racing | MPPI vs Diff-MPPI |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_wavefront_planner.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diffusion_planner.gif" width="400"/> |
| Wavefront planner | Diffusion planner |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_batched_ilqr.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/sdf_mppi.gif" width="400"/> |
| Batched iLQR | SDF-MPPI |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mcts_planner.gif" width="400"/> |
| Multi-robot planner | MCTS planner |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mpc_qp.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_constrained_mpc.gif" width="400"/> |
| Convex MPC: 1024 batched box-QPs via ADMM (OSQP-style) | Constrained nonlinear MPC: 400 robots, AL-iLQR with hard obstacle limits |

### Perception & mapping

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_sgm_stereo.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_tsdf_fusion.gif" width="400"/> |
| SGM stereo | TSDF fusion |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_kdtree_nn.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_direct_vo.gif" width="400"/> |
| KD-tree nearest-neighbour | Direct visual odometry |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_marching_cubes.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_lk_optical_flow.gif" width="400"/> |
| Marching Cubes | Lucas-Kanade optical flow |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_jfa_edt.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_ndt_3d.gif" width="400"/> |
| Jump-flood EDT | NDT 3D scan matching |

### Probabilistic point-cloud registration

Modern probabilistic registration in the spirit of `probreg`, spanning the main
paradigms: filtered EM, Bayesian non-rigid, optimal transport, heavy-tailed
robust EM, and a point-to-plane filtered EM. Each demo recovers a known
transform / warp and is verified in-binary.

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_filterreg.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_bcpd.gif" width="400"/> |
| FilterReg (filtered-EM rigid) | BCPD (Bayesian non-rigid) |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_sinkhorn_reg.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_robust_treg.gif" width="400"/> |
| Sinkhorn-OT (unbalanced optimal transport) | Robust Student's-t (2x outlier tolerance vs Gaussian) |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_filterreg_p2plane.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_robust_p2plane_reg.gif" width="400"/> |
| FilterReg point-to-plane (removes soft-mean curvature bias; 43x lower error at coarse sigma) | Flagship: robust Student's-t x point-to-plane (best under outliers x curvature) |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_real_bunny_reg.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_fgr.gif" width="400"/> |
| Real data: Stanford bunny scan, known SE(3) recovered to 0.1 deg | Fast Global Registration: FPFH + GNC recovers 72 deg from no initial guess |

### Learning & optimization

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/mini_isaac_rl.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_neuroevo.gif" width="400"/> |
| Parallel CartPole RL (REINFORCE) | GPU neuroevolution |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_sdf_nav.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_cma_es.gif" width="400"/> |
| Neural SDF navigation | GPU CMA-ES |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diffusion_policy.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_neural_astar_traversability.gif" width="400"/> |
| Diffusion policy | Neural A* traversability |
| <img src="https://rsasaki0109.github.io/CudaRobotics/pso.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_em_gmm.gif" width="400"/> |
| PSO swarm optimization | EM / GMM clustering |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diff_contact_push.gif" width="400"/> | |
| Differentiable contact: autodiff-through-contact pushing to a target pose | |

### Graph-neural & multi-agent MPPI

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_graph_guided_neural_mppi.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_noregret_game_graph_mppi.gif" width="400"/> |
| Graph-guided neural MPPI | No-regret game graph MPPI |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_interaction_graph_neural_mppi.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_reciprocal_risk_planner.gif" width="400"/> |
| Interaction-graph neural MPPI | Reciprocal-risk planner |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gnn_swarm_controller.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_crowd_swarm.gif" width="400"/> |
| GNN swarm controller | Crowd / swarm simulation |

## What's Inside

- SLAM and scan matching: pose-graph SLAM, online SLAM, correlative scan
  matching, branch-and-bound CSM, submap loop closure, Gaussian-Splatting SLAM,
  KISS-ICP-style LiDAR odometry.
- Localization and filtering: particle filters, KLD-AMCL, MegaParticles-style
  global localization, LSH neighbour consensus, robust smoothers.
- Planning and control: MPPI, Diff-MPPI, graph-neural MPPI, no-regret game
  planners, DWA, RRT family, value iteration, wavefront planning, batched and
  parallel-in-time (associative-scan) iLQR, convex MPC (batched ADMM box-QP),
  constrained nonlinear MPC (augmented-Lagrangian iLQR).
- Perception and mapping: LiDAR simulation, occupancy grids, ESDF/JFA, TSDF,
  Marching Cubes, SGM stereo, optical flow, direct visual odometry, KD-tree NN.
- Point-cloud registration: global front-end (FPFH + Fast Global Registration)
  plus local probabilistic refiners (FilterReg filtered-EM and its point-to-plane
  variant, BCPD Bayesian non-rigid, Sinkhorn unbalanced optimal transport, robust
  Student's-t mixture), NDT, GICP, ICP.
- Learning and optimization: differentiable value iteration, neural A*, GNN/GAT
  policies, diffusion planners, CMA-ES, MCTS, EM/GMM, graph CRF.

## Layout

| Path | Purpose |
|---|---|
| `src/` | Self-contained C++/CUDA demos and benchmarks. |
| `include/` | Shared CUDA helpers. |
| `docs/` | Per-demo notes and reproducibility docs. |
| `scripts/` | Summary, plotting, and repro-suite tooling. |
| `gif/` | Local generated media and benchmark artifacts. |
| `paper/` | Diff-MPPI draft material and experiment notes. |

## Build

Requirements: CMake >= 3.18, CUDA Toolkit >= 12.0, OpenCV >= 4.5, Eigen 3.

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

Executables are written to `bin/`.

Build and run one demo:

```bash
cmake --build build --target gpu_mppi_racing -j$(nproc)
./bin/gpu_mppi_racing
```

## Reproducibility

```bash
python3 scripts/run_repro_suite.py --dry-run --suite smoke
python3 scripts/run_repro_suite.py --build --suite diff-mppi
```

The runner writes CSVs, summaries, logs, `manifest.json`, and a human-readable
`report.md` under `build/repro_suite/`. See
[`docs/reproducibility.md`](docs/reproducibility.md) for suite details.

## Useful Entry Points

- Gallery: https://rsasaki0109.github.io/CudaRobotics/
- MPPI reproduction zoo: [`docs/mppi_reproduction_zoo.md`](docs/mppi_reproduction_zoo.md)
- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Repro suite docs: [`docs/reproducibility.md`](docs/reproducibility.md)
- Next-actions snapshot: [`docs/next_actions.md`](docs/next_actions.md)
- Diff-MPPI paper draft material: [`paper/`](paper/)
- Long-form agent handoff: [`plan.md`](plan.md)

## License

See [`LICENSE.md`](LICENSE.md).
