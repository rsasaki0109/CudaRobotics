# CudaRobotics

CUDA-accelerated robotics algorithms in C++/CUDA, inspired by
[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) and
[CppRobotics](https://github.com/onlytailei/CppRobotics), with additional
differentiable-control, SLAM, mapping, perception, and learning demos.

The core pattern is simple: keep a CPU reference or robotics baseline where it
helps, then expose the parallel work as one CUDA thread per particle, ray,
candidate pose, graph node, rollout, voxel, feature, or grid cell.

Full animated gallery: https://rsasaki0109.github.io/CudaRobotics/

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

### SLAM & scan matching

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pose_graph_slam_3d.gif" width="400"/> |
| Gaussian-Splatting SLAM (RGB-D) | 3D pose-graph SLAM |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_bnb_loop_closure_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_correlative_scan_matching.gif" width="400"/> |
| Branch-and-bound loop closure | Correlative scan matching |

### Localization & filtering

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_kld_amcl.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_megaparticles_lsh.gif" width="400"/> |
| KLD-AMCL | MegaParticles global localization (LSH) |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_pf_mlp.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif" width="400"/> |
| Differentiable particle filter (MLP) | Expansion-reset MCL |

### Planning & control

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_mppi.gif" width="400"/> |
| MPPI autonomous racing | MPPI vs Diff-MPPI |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_wavefront_planner.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diffusion_planner.gif" width="400"/> |
| Wavefront planner | Diffusion planner |

### Perception & mapping

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_sgm_stereo.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_tsdf_fusion.gif" width="400"/> |
| SGM stereo | TSDF fusion |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_kdtree_nn.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_direct_vo.gif" width="400"/> |
| KD-tree nearest-neighbour | Direct visual odometry |

### Learning & optimization

| | |
|---|---|
| <img src="https://rsasaki0109.github.io/CudaRobotics/mini_isaac_rl.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_neuroevo.gif" width="400"/> |
| Parallel CartPole RL (REINFORCE) | GPU neuroevolution |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_sdf_nav.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_cma_es.gif" width="400"/> |
| Neural SDF navigation | GPU CMA-ES |

## What's Inside

- SLAM and scan matching: pose-graph SLAM, online SLAM, correlative scan
  matching, branch-and-bound CSM, submap loop closure, Gaussian-Splatting SLAM.
- Localization and filtering: particle filters, KLD-AMCL, MegaParticles-style
  global localization, LSH neighbour consensus, robust smoothers.
- Planning and control: MPPI, Diff-MPPI, graph-neural MPPI, no-regret game
  planners, DWA, RRT family, value iteration, wavefront planning.
- Perception and mapping: LiDAR simulation, occupancy grids, ESDF/JFA, TSDF,
  Marching Cubes, SGM stereo, optical flow, direct visual odometry, KD-tree NN.
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
- Repro suite docs: [`docs/reproducibility.md`](docs/reproducibility.md)
- Next-actions snapshot: [`docs/next_actions.md`](docs/next_actions.md)
- Diff-MPPI paper draft material: [`paper/`](paper/)
- Long-form agent handoff: [`plan.md`](plan.md)

## License

See [`LICENSE.md`](LICENSE.md).
