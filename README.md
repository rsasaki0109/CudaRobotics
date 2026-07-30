# CudaRobotics

[![GitHub stars](https://img.shields.io/github/stars/rsasaki0109/CudaRobotics?style=social)](https://github.com/rsasaki0109/CudaRobotics/stargazers)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Docs](https://img.shields.io/badge/docs-v0.3.0-1f6f64)](https://rsasaki0109.github.io/CudaRobotics/docs/)
[![Gallery](https://img.shields.io/badge/gallery-animated_demos-blue)](https://rsasaki0109.github.io/CudaRobotics/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rsasaki0109/CudaRobotics/blob/v0.3.0/examples/colab/cudarobotics_quickstart.ipynb)

<p align="center">
  <a href="gif/cudanav_gpu_closed_loop_release.gif">
    <img src="gif/cudanav_gpu_closed_loop_release.gif" alt="CudaNav native all-GPU 30-traversal closed-loop release" width="180"/>
  </a>
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif">
    <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif" alt="CUDA MPPI racing demo" width="180"/>
  </a>
  <a href="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif">
    <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif" alt="Expansion-reset MCL demo" width="180"/>
  </a>
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif">
    <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif" alt="CUDA multi-robot planner demo" width="180"/>
  </a>
  <a href="gif/gpu_multi_robot_place_graph_slam.gif">
    <img src="gif/gpu_multi_robot_place_graph_slam.gif" alt="CUDA multi-robot place graph SLAM demo" width="180"/>
  </a>
</p>

<p align="center">
  <a href="gif/cudanav_gpu_closed_loop_release.gif">CudaNav closed loop</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif">MPPI racing</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/comparison_expansion_reset_mcl.gif">Expansion-reset MCL</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_multi_robot_planner.gif">Multi-robot planner</a>
  /
  <a href="gif/gpu_multi_robot_place_graph_slam.gif">Multi-robot place graph SLAM</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif">Gaussian-Splatting SLAM</a>
  /
  <a href="https://rsasaki0109.github.io/CudaRobotics/">full gallery</a>
</p>

CUDA Robotics is a GPU-first robotics playground and benchmark suite for SLAM,
mapping, perception, planning, MPPI control, point-cloud registration, and
learning demos in C++/CUDA.

Project goal: make this repo a reproducible OSS lab for GPU-accelerated robot
planning, control, registration, and learning interfaces. New work should move
at least one of those tracks from demo code toward runnable benchmarks,
documented results, Python/ROS usability, or real-data validation.

If you are looking for a reason to star it: this repo turns robotics algorithms
into small, runnable CUDA examples, then records both the speedups and the
failure cases.

The core pattern is simple: keep a CPU reference or robotics baseline where it
helps, then expose the parallel work as one CUDA thread per particle, ray,
candidate pose, graph node, rollout, voxel, feature, or grid cell.

Full animated gallery: https://rsasaki0109.github.io/CudaRobotics/

## Quickstart

[Open in Colab](https://colab.research.google.com/github/rsasaki0109/CudaRobotics/blob/v0.3.0/examples/colab/cudarobotics_quickstart.ipynb)
· [Documentation](https://rsasaki0109.github.io/CudaRobotics/docs/)
· [Nav2 CUDA MPPI](ros2_ws/src/cuda_mppi_controller/)
· [Full animated gallery](https://rsasaki0109.github.io/CudaRobotics/)

```bash
pip install -e python/
python examples/python/mppi_quickstart.py
python examples/python/registration_quickstart.py
```

<details>
<summary>More setup, benchmark, and research details</summary>

## Start Here

| Want to see | Open |
|---|---|
| **Install / API / Nav2 docs** | [CudaRobotics docs site](https://rsasaki0109.github.io/CudaRobotics/docs/) |
| **Try it in your browser (free Colab GPU)** | [Colab quickstart notebook](https://colab.research.google.com/github/rsasaki0109/CudaRobotics/blob/v0.3.0/examples/colab/cudarobotics_quickstart.ipynb) |
| Visual demos | [Full animated gallery](https://rsasaki0109.github.io/CudaRobotics/) |
| **GPU MPPI controller plugin for Nav2** | [`ros2_ws/src/cuda_mppi_controller/`](ros2_ws/src/cuda_mppi_controller/) |
| **CudaNav voxel mapping, typed ESDF, and Nav2 bridge** | [`docs/cudanav_architecture.md`](docs/cudanav_architecture.md), [`docs/cuda_voxel_costmap_layer.md`](docs/cuda_voxel_costmap_layer.md) |
| **CudaNav deterministic closed-loop bringup** | [`docs/cudanav_closed_loop.md`](docs/cudanav_closed_loop.md) |
| **CudaNav native all-GPU 30-traversal release** | [`docs/results/cudanav_gpu_closed_loop_release_2026-07-29.md`](docs/results/cudanav_gpu_closed_loop_release_2026-07-29.md) |
| **CudaNav multi-GPU reproducibility matrix** | [`docs/cudanav_multi_gpu.md`](docs/cudanav_multi_gpu.md) |
| **CudaNav physical GPU matrix — GTX 1660 Ti node** | [`docs/results/cudanav_gpu_closed_loop_release_gtx1660ti_2026-07-29.md`](docs/results/cudanav_gpu_closed_loop_release_gtx1660ti_2026-07-29.md) |
| **CudaNav complete autonomy evidence suite** | [`docs/cudanav_autonomy_suite.md`](docs/cudanav_autonomy_suite.md) |
| Nav2 CPU vs CUDA MPPI head-to-head | [`docs/results/cuda_mppi_vs_nav2_2026-06-10.md`](docs/results/cuda_mppi_vs_nav2_2026-06-10.md) |
| CUDA MPPI extended controller scenarios | [`docs/results/cuda_mppi_extended_scenarios_2026-06-12.md`](docs/results/cuda_mppi_extended_scenarios_2026-06-12.md) |
| CUDA MPPI bag / real-data evaluation harness | [`docs/cuda_mppi_bag_eval.md`](docs/cuda_mppi_bag_eval.md) |
| CUDA MPPI curvature speed critic | [`docs/results/cuda_mppi_curvature_speed_2026-06-12.md`](docs/results/cuda_mppi_curvature_speed_2026-06-12.md) |
| CUDA MPPI path-angle critic | [`docs/results/cuda_mppi_path_angle_2026-06-12.md`](docs/results/cuda_mppi_path_angle_2026-06-12.md) |
| CUDA MPPI ESDF clearance critic | [`docs/results/cuda_mppi_esdf_2026-06-11.md`](docs/results/cuda_mppi_esdf_2026-06-11.md) |
| Registration external baselines | [`docs/results/registration_external_baselines_2026-06-11.md`](docs/results/registration_external_baselines_2026-06-11.md) |
| Registration unified benchmark | [`docs/registration_benchmark.md`](docs/registration_benchmark.md) |
| Reproducible real-rosbag CUDA MPPI shadow gate | [`docs/cuda_mppi_bag_eval.md`](docs/cuda_mppi_bag_eval.md) |
| Latest registration GPU smoke | [`docs/results/registration_unified_smoke_2026-07-28.md`](docs/results/registration_unified_smoke_2026-07-28.md) |
| Latest real-rosbag evaluation | [`docs/results/mppi_real_rosbag_erl_prueba2_2026-07-28.md`](docs/results/mppi_real_rosbag_erl_prueba2_2026-07-28.md) |
| Latest fixed-seed MPPI result | [`docs/results/mppi_zoo_suite_2026-06-10.md`](docs/results/mppi_zoo_suite_2026-06-10.md) |
| Quick MPPI smoke result | [`docs/results/mppi_zoo_smoke_2026-06-05.md`](docs/results/mppi_zoo_smoke_2026-06-05.md) |
| MPPI paper reproduction zoo | [`docs/mppi_reproduction_zoo.md`](docs/mppi_reproduction_zoo.md) |
| Reproducibility suites | [`docs/reproducibility.md`](docs/reproducibility.md) |
| Diff-MPPI paper material | [`paper/`](paper/) |
| Paper claim/evidence readiness | [`paper/artifacts/`](paper/artifacts/) |
| Contact-rich Diff-MPPI robustness suite | [`docs/contact_diff_mppi_robustness.md`](docs/contact_diff_mppi_robustness.md) |
| Deadline-matched contact control | [`docs/contact_matched_compute.md`](docs/contact_matched_compute.md) |
| Contributing a demo or reproduction | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| CudaNav end-to-end architecture contract | [`docs/cudanav_architecture.md`](docs/cudanav_architecture.md) |
| Reusable GPU KISS-ICP streaming API | [`docs/kiss_icp_gpu_core.md`](docs/kiss_icp_gpu_core.md) |
| Real-bag GPU KISS-ICP evidence | [`docs/cudanav_kiss_icp_real.md`](docs/cudanav_kiss_icp_real.md) |
| Real-data KISS-ICP → voxel → ESDF → MPPI shadow | [`docs/cudanav_real_gpu_stack.md`](docs/cudanav_real_gpu_stack.md) |
| ROS 2 lifecycle GPU KISS-ICP component | [`ros2_ws/src/cuda_kiss_icp/`](ros2_ws/src/cuda_kiss_icp/) |
| Rolling GPU voxel mapping core | [`docs/voxel_mapping_gpu_core.md`](docs/voxel_mapping_gpu_core.md) |
| ROS 2 lifecycle voxel mapper | [`ros2_ws/src/cuda_voxel_mapping/`](ros2_ws/src/cuda_voxel_mapping/) |
| Exact GPU ESDF core and CPU reference | [`docs/esdf_2d_gpu_core.md`](docs/esdf_2d_gpu_core.md) |
| Typed ROS 2 lifecycle ESDF component | [`ros2_ws/src/cuda_esdf/`](ros2_ws/src/cuda_esdf/) |
| CudaRobotics 1.0 long-term roadmap | [`docs/cudarobotics_1_0_roadmap.md`](docs/cudarobotics_1_0_roadmap.md) |
| Current roadmap snapshot | [`docs/next_actions.md`](docs/next_actions.md) |
| v1.0 cross-surface support contract | [`docs/v1_support_matrix.md`](docs/v1_support_matrix.md) |

## Python MPPI Quickstart

The reusable GPU MPPI core is available as an experimental Python package.
Build requirements: Linux x86_64, CUDA Toolkit >= 12.0, CMake >= 3.18.

**No local GPU?** Run the
[Colab quickstart notebook](https://colab.research.google.com/github/rsasaki0109/CudaRobotics/blob/v0.3.0/examples/colab/cudarobotics_quickstart.ipynb)
— it builds the package on a free Colab GPU and runs the MPPI + registration demos
in your browser.

**Development install** (uses repo-root CUDA sources directly):

```bash
pip install -e python/
pip install -e 'python/[examples]'  # optional: GIF rendering dependencies
python examples/python/mppi_quickstart.py
python examples/python/mppi_dlpack_costmap.py  # optional: CUDA PyTorch or CuPy costmap
```

**Install from source distribution** (self-contained sdist; compiles against local CUDA):

```bash
./scripts/sync_python_core.sh   # maintainers: refresh bundled core before release
cd python && python -m pip install build && python -m build
pip install python/dist/cudarobotics-*.tar.gz
pip install 'python/dist/cudarobotics-*.tar.gz[test]'
pytest python/tests
```

CI attaches `linux_x86_64` wheels for Python 3.10/3.12 as workflow artifacts.
On pushes to `master`, a separate `cibuildwheel` job also builds manylinux
wheels (see `.github/workflows/python-package.yml`). They require a compatible
NVIDIA driver at runtime.

Minimal use:

```python
import numpy as np
import cudarobotics as cr

planner = cr.MppiPlanner(batch_size=2048, time_steps=56, model_dt=0.05)
costmap = np.zeros((200, 200), dtype=np.uint8)
path = np.array([[x, 5.0] for x in np.arange(1.0, 9.1, 0.1)], dtype=np.float32)
v, vy, w, info = planner.compute(
    (1.0, 5.0, 0.0), costmap, path, (9.0, 5.0, 0.0), resolution=0.05
)
```

For learning stacks, `costmap` may also be a CUDA DLPack producer such as a
PyTorch or CuPy tensor. In that case the MPPI core consumes the device pointer
directly instead of staging the costmap through host memory.
See [`examples/python/mppi_dlpack_costmap.py`](examples/python/mppi_dlpack_costmap.py)
for a runnable torch/CuPy example and `info` diagnostics readout.

## Python Registration Quickstart

Rigid and non-rigid registration live under `cudarobotics.registration`:

```bash
pip install -e python/
# or from sdist: pip install python/dist/cudarobotics-*.tar.gz
python examples/python/registration_quickstart.py
```

```python
import cudarobotics as cr

robust = cr.registration.RobustP2Plane()  # Student's-t + point-to-plane
rotation, translation, info = robust.register(target_xyz, source_xyz)

sinkhorn = cr.registration.SinkhornReg()
rotation, translation, info = sinkhorn.register(target_xyz, source_xyz)
```

FilterReg, FGR, BCPD, and `RobustTreg` are also available (see
`examples/python/registration_quickstart.py`).

## Latest Fixed-Seed Result

The checked-in MPPI zoo suite was generated on 2026-06-10 with five navigation
scenarios, ten curated planners (including `soppi` / `soppi_fast`), `K=64,128`,
and 3 seeds per scenario/planner/K cell. It is a fixed-seed benchmark, not a
paper-faithful claim, but it adds stress scenes beyond the earlier smoke pair
and keeps the failures visible.

<img src="docs/results/mppi_zoo_suite_2026-06-10.svg" alt="MPPI Zoo fixed-seed suite chart" width="900"/>

Side-by-side rollout on `dynamic_crossing` (`K=128`): vanilla `mppi` stalls short
of the goal while `step_mppi_smooth` reaches it.

<img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_zoo_dynamic_crossing.gif" alt="MPPI zoo dynamic crossing comparison" width="840"/>

| Scenario | Signal in this suite |
|---|---|
| `dynamic_crossing` | Vanilla `mppi` fails; curated zoo variants solve all cells. |
| `model_mismatch_crossing` | Vanilla `mppi` fails; `step_mppi_smooth` / `tsallis_mppi_smooth` reach 1.00 at `K=128`. |
| `dynamic_pincer` | Vanilla `mppi` fails with large final distance; zoo variants succeed. |
| `uncertain_crossing` | Same pattern as dynamic crossing: vanilla `mppi` fails, zoo variants succeed. |
| `narrow_passage` | All smooth zoo planners succeed; `soppi` also clears both K cells here. |

Full report and CSV:
[`docs/results/mppi_zoo_suite_2026-06-10.md`](docs/results/mppi_zoo_suite_2026-06-10.md)
and
[`docs/results/mppi_zoo_suite_2026-06-10.csv`](docs/results/mppi_zoo_suite_2026-06-10.csv).

Suite leaders (5 scenarios × 2 K values, 3 seeds per cell):

| Planner | Solved | Success | Avg ms | Notes |
|---|---|---|---|---|
| `step_mppi_smooth` | 9/10 | 0.97 | 0.116 | Fastest curated planner in the suite |
| `tsallis_mppi_smooth` | 9/10 | 0.97 | 0.175 | Tied for best solve rate |
| `sc_mppi_smooth` | 9/10 | 0.97 | 0.198 | Strong safety-controlled baseline |
| `soppi` / `soppi_fast` | 2/10 | 0.20 | 0.30 / 0.25 | Navigation negative control; wins only on `narrow_passage` |
| `mppi` | 2/10 | 0.20 | 0.126 | Baseline negative control |

The eight-planner suite from 2026-06-09 remains at
[`docs/results/mppi_zoo_suite_2026-06-09.md`](docs/results/mppi_zoo_suite_2026-06-09.md).
The smaller two-scenario smoke artifact from 2026-06-05 remains at
[`docs/results/mppi_zoo_smoke_2026-06-05.md`](docs/results/mppi_zoo_smoke_2026-06-05.md).

## Docker MPPI Benchmark

Requires NVIDIA Container Toolkit and a CUDA-capable GPU.

End-to-end CudaNav smoke (GPU KISS-ICP, voxel map, ESDF, Nav2 CUDA MPPI,
and command-driven simulator):

```bash
docker build --pull --no-cache -f docker/Dockerfile -t cudarobotics .
docker run --rm --gpus all -v "$PWD/out:/out" cudarobotics cudanav
```

The command exits non-zero unless `/out/cudanav_closed_loop.json` passes the
short integration gate. The 10-minute retained release run remains a separate
v1.0 evidence requirement.

Quick smoke:

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_smoke.py --bin ./bin/benchmark_diff_mppi --out-dir build/mppi_zoo'
```

Expanded fixed-seed suite:

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_suite.py --bin ./bin/benchmark_diff_mppi && python3 scripts/render_mppi_zoo_suite_chart.py'
```

Comparison GIF (`dynamic_crossing`, vanilla `mppi` vs `step_mppi_smooth`):

```bash
cmake --build build --target benchmark_diff_mppi -j$(nproc)
python3 scripts/render_mppi_zoo_gif.py --bin bin/benchmark_diff_mppi
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

| Family | Suite signal (2026-06-09) | What to open first |
|---|---|---|
| Tsallis-MPPI | 10/10 solved; best overall | [`docs/tsallis_mppi_reproduction.md`](docs/tsallis_mppi_reproduction.md) |
| Step-MPPI | 9/10 solved; fastest curated planner | [`docs/step_mppi_reproduction.md`](docs/step_mppi_reproduction.md) |
| SC-MPPI | 9/10 solved; safety-controlled baseline | [`docs/sc_mppi_reproduction.md`](docs/sc_mppi_reproduction.md) |
| DRA-MPPI | 8/10 solved; strong on `dynamic_pincer` | [`docs/dra_mppi_reproduction.md`](docs/dra_mppi_reproduction.md) |
| C2U-MPPI | 8/10 solved | [`docs/c2u_mppi_reproduction.md`](docs/c2u_mppi_reproduction.md) |
| DUCCT-MPPI | 8/10 solved | [`docs/ducct_mppi_reproduction.md`](docs/ducct_mppi_reproduction.md) |
| LP-MPPI | 8/10 solved | [`docs/lp_mppi_reproduction.md`](docs/lp_mppi_reproduction.md) |
| DBaS-Log-MPPI | not in suite; smoke benchmark only | [`docs/dbas_log_mppi_reproduction.md`](docs/dbas_log_mppi_reproduction.md) |
| PA-MPPI | not in suite; narrow-passage smoke | [`docs/pa_mppi_reproduction.md`](docs/pa_mppi_reproduction.md) |
| SOPPI | 2/10 nav; `box_swivel` 1.00; `box_align_contact_arc` 1.00; strict `box_align_contact_loss` `soppi_fast` 1.00 vs MPPI 0.00 | [`docs/soppi_reproduction.md`](docs/soppi_reproduction.md) |
| Full index + CSV | [`docs/results/mppi_zoo_suite_2026-06-10.csv`](docs/results/mppi_zoo_suite_2026-06-10.csv) | [`docs/mppi_reproduction_zoo.md`](docs/mppi_reproduction_zoo.md) |

## Highlights

| Demo | What it shows |
|---|---|
| `cuda_mppi_controller` | Drop-in GPU MPPI for Nav2 — 65k rollouts in ~10 ms; DiffDrive / Ackermann / Omni motion models verified. |
| `gpu_multi_robot_place_graph_slam` | Multi-robot place recognition scores 60,516 descriptor pairs on the GPU, adds inter-robot loop edges, and cuts pose-graph RMSE from 7.59 m to 3.33 m. |
| `gpu_bnb_loop_closure_slam` | Branch-and-bound loop search scores about 957x fewer candidates than brute force while returning the same relpose on 51/51 attempts. |
| `gpu_gaussian_splatting_slam` | RGB-D Gaussian-Splatting SLAM with GPU ray-cast sensor, point-to-plane ICP tracking, and incremental Gaussian map fusion. |
| `gpu_nerf_volume` | NeRF volume rendering with GPU ray marching. |
| `gpu_ndt_3d_multires` | Multi-resolution NDT 3D scan matching on the GPU. |
| `gpu_gicp_3d` | GICP 3D point-cloud registration with GPU parallel correspondence search. |
| `gpu_hungarian_assignment` | GPU Hungarian assignment for multi-target data association. |
| `gpu_mppi_racing` | MPPI autonomous racing with 2048 x 40 rollouts per control step on the GPU. |
| `gpu_kdtree_nn` | Exact KD-tree nearest-neighbour search for 40k queries, matching brute force while running much faster. |
| `gpu_sgm_stereo` | Semi-Global Matching stereo with CUDA census and path aggregation. |
| `gpu_wavefront_planner` | Bellman-Ford-style cost-to-go relaxation over a 384x384 planning grid. |
| `gpu_pcg_solver` | GPU preconditioned conjugate-gradient linear solver benchmark. |
| `gpu_sfm_mini` | Structure-from-motion mini pipeline with GPU triangulation. |
| `gpu_diffusion_planner` | Diffusion-based motion planner with GPU rollout scoring. |
| `gpu_assignment_tracking` | GPU assignment + multi-object tracking pipeline. |
| `gpu_frontier_exploration` | GPU frontier exploration with parallel ray casting over an occupancy grid. |
| `gpu_diff_contact_push` | Differentiable contact pushing with GPU rollout scoring. |
| `gpu_constrained_mpc` | Constrained nonlinear MPC (AL-iLQR) for multi-robot obstacle avoidance. |
| [`gpu_kiss_icp`](docs/gpu_kiss_icp.md) | KISS-ICP-style LiDAR odometry using the [reusable GPU streaming core](docs/kiss_icp_gpu_core.md), exact voxel-hash correspondences, accuracy gates, and JSON metrics. |

| | |
|---|---|
| <img src="gif/gpu_multi_robot_place_graph_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif" width="400"/> |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_sgm_stereo.gif" width="400"/> |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_frontier_exploration.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_diff_contact_push.gif" width="400"/> |

</details>

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
| <img src="gif/gpu_multi_robot_place_graph_slam.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_gaussian_splatting_slam.gif" width="400"/> |
| Multi-robot place graph SLAM | Gaussian-Splatting SLAM (RGB-D) |
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_pose_graph_slam_3d.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_correlative_scan_matching.gif" width="400"/> |
| 3D pose-graph SLAM | Correlative scan matching |
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
| <img src="https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_zoo_dynamic_crossing.gif" width="400"/> | |
| MPPI zoo: vanilla vs `step_mppi_smooth` on `dynamic_crossing` | |
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

- SLAM and scan matching: multi-robot place-graph SLAM, pose-graph SLAM,
  online SLAM, correlative scan matching, branch-and-bound CSM, submap loop
  closure, Gaussian-Splatting SLAM, KISS-ICP-style LiDAR odometry.
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
