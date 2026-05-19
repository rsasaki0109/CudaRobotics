# CudaRobotics

CUDA-accelerated C++ implementations of robotics algorithms, based on [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) and [CppRobotics](https://github.com/onlytailei/CppRobotics).

Each algorithm leverages GPU parallelism for significant speedup over CPU-only implementations.

## Why CUDA? — Visual Quality Difference

GPU enables orders-of-magnitude more particles/samples, resulting in visually better results. All comparisons below use the **same algorithm** on CPU and GPU — the only difference is sample/particle count enabled by GPU parallelism:

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

## Novel Research Extensions

Recent additions push the repository beyond direct CUDA ports of classic robotics algorithms into differentiable building blocks, GPU-native learning systems, point-cloud processing, and large-scale swarm optimization.

| Project | Binaries | Highlights |
|---|---|---|
| Autodiff + GPU MLP foundation | `test_autodiff`, `test_gpu_mlp` | Dual-number forward-mode autodiff and a compact GPU MLP training/inference engine used as the base for later research-style experiments. |
| Differentiable MPPI | `diff_mppi`, `comparison_diff_mppi`, `benchmark_diff_mppi`, `benchmark_diff_mppi_cartpole`, `benchmark_diff_mppi_dynamic_bicycle`, `benchmark_diff_mppi_manipulator`, `benchmark_diff_mppi_manipulator_7dof` | Augments MPPI with a short autodiff refinement stage. Evaluated on 2D dynamic-obstacle navigation, CartPole, dynamic-bicycle, 2-link planar arm, and 7-DOF serial arm. Includes strong in-repo feedback baselines, matched-time tuning, mechanism analysis, MuJoCo transfer checks, and uncertainty follow-ups. On the hard `dynamic_slalom` task, the hybrid controller is the only method that succeeds in the submission-critical `1.0 ms` exact-time table, and the gap survives broader matched-time robustness sweeps. |
| Neural SDF Navigation | `neural_sdf`, `sdf_potential_field`, `sdf_mppi`, `comparison_sdf_nav` | Learns 2D signed distance fields with a GPU MLP, then uses them for potential-field planning and MPPI on non-circular obstacle layouts. |
| Neuroevolution for Cart-Pole | `neuroevo`, `comparison_neuroevo` | Evolves 4096 neural policies in parallel on GPU and compares them against a CPU baseline with side-by-side learning curves. |
| MiniIsaacGym | `mini_isaac`, `mini_isaac_rl` | Runs thousands of CartPole environments in parallel on GPU and trains a compact policy with GPU-side REINFORCE updates. |
| CudaPointCloud | `voxel_grid_filter`, `statistical_filter`, `normal_estimation`, `gicp`, `ransac_plane`, `benchmark_pointcloud` | GPU voxel filtering, outlier removal, PCA normals, plane extraction, and GICP registration. Supports PLY/KITTI/XYZ file input. Normal estimation reaches 3,171x speedup at 10K points. |
| Swarm Optimization | `pso_cuda`, `differential_evolution`, `cma_es`, `aco_tsp`, `comparison_swarm` | Large-scale PSO, DE, CMA-ES, and ACO implementations with animated convergence comparisons. |

| | |
|---|---|
| **MPPI vs Differentiable MPPI** | **Differentiable MPPI trajectory rollouts** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_diff_mppi.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/diff_mppi.gif" width="400"/> |
| **Diff-MPPI exact-time Pareto** | **Diff-MPPI gradient freshness** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/diff_mppi_pareto.png" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/diff_mppi_mechanism.png" width="400"/> |
| **Diff-MPPI 7-DOF benchmark** | **Diff-MPPI ablation figure** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/diff_mppi_7dof.png" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/diff_mppi_ablation.png" width="400"/> |
| **Diff-MPPI task scenarios** | |
| <img src="https://rsasaki0109.github.io/CudaRobotics/diff_mppi_scenarios.png" width="400"/> | |
| **Neural SDF vs true field** | **Neural SDF MPPI vs circle approximation** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/neural_sdf.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_sdf_nav.gif" width="400"/> |
| **Neural SDF potential-field navigation** | **Neural SDF MPPI rollout** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/sdf_potential_field.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/sdf_mppi.gif" width="400"/> |
| **Neuroevolution: CPU 100 vs CUDA 4096 individuals** | **Swarm Optimization: PSO vs DE vs CMA-ES** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_neuroevo.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/comparison_swarm.gif" width="400"/> |
| **GPU Neuroevolution Cart-Pole replay** | **Particle Swarm Optimization** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/neuroevo.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/pso.gif" width="400"/> |
| **4096-way CartPole simulation** | **MiniIsaacGym REINFORCE training** |
| <img src="https://rsasaki0109.github.io/CudaRobotics/mini_isaac.gif" width="400"/> | <img src="https://rsasaki0109.github.io/CudaRobotics/mini_isaac_rl.gif" width="400"/> |
| **Ant Colony Optimization for TSP** | |
| <img src="https://rsasaki0109.github.io/CudaRobotics/aco_tsp.gif" width="400"/> | |

Text-first highlights for modules without public GIFs yet:

| Module | Why it matters |
|---|---|
| CudaPointCloud | Large benchmarked speedups without approximation-heavy CPU baselines: normal estimation reaches **3,171x** at 10K points and RANSAC plane reaches **547x** at 100K. |
| MuJoCo transfer checks | `InvertedPendulum-v4` and `Reacher` pilots show the Diff-MPPI stack ports to standardized tasks, even though they are not the main win condition. |

Autodiff + GPU MLP foundation snapshot:

<img src="https://rsasaki0109.github.io/CudaRobotics/autodiff_gpu_mlp_summary.png" alt="autodiff_gpu_mlp_summary" width="900"/>

Generated from the existing test binaries with:

```bash
python3 scripts/render_autodiff_gpu_mlp_summary.py
```

## Research Results Snapshot

Recent research-style additions are summarized on the GitHub Pages gallery:

- https://rsasaki0109.github.io/CudaRobotics/

Reproducible benchmark entry point:

```bash
python3 scripts/run_repro_suite.py --dry-run --suite smoke
python3 scripts/run_repro_suite.py --build --suite diff-mppi
```

The runner records the exact benchmark, summary, and optional plotting commands in `build/repro_suite/manifest.json`.
See `docs/reproducibility.md` for the suite catalog and output layout.

Concise highlights:

| Area | Key result |
|---|---|
| **Diff-MPPI, dynamic navigation** | On `dynamic_slalom`, the submission-critical exact-time table at `1.0 ms` still has **diff_mppi_3 as the only successful controller** (dist `1.90`), while `feedback_mppi_fused` reaches `10.33`, `feedback_mppi_paper` `11.61`, and `mppi` `14.15`. The harder family-level matched-time sweep keeps the same qualitative split: the best non-hybrid family still fails while the best Diff family succeeds. |
| **Diff-MPPI, 7-DOF manipulator** | At `K=512` on `7dof_dynamic_avoid`: `diff_mppi_3` reaches **success=1.00 at 0.84 ms**, while `feedback_mppi_ref` reaches 0.75 at 4.01 ms. The hybrid controller is 4.8x faster and more reliable. |
| Diff-MPPI, 2-link manipulator | On `arm_static_shelf` at `K=256`: `feedback_mppi_ref` and `feedback_mppi_cov` both reach `success=1.00` at `0.15` final distance, while vanilla MPPI stays at `0.00`. |
| Diff-MPPI, faithful baseline | Both `feedback_mppi_paper` (covariance-regression + LQR, every-step) and `feedback_mppi_faithful` (two-rate, current-action-only) fail on dynamic tasks, confirming that gradient refinement provides complementary value no feedback architecture can replicate. |
| Diff-MPPI, ablation | `step_mppi` (learned sampling bias) performs at vanilla MPPI level (dist `14.25` vs `14.23`), showing that improved sampling alone cannot solve dynamic obstacle tasks—gradient refinement is the key mechanism. |
| Neural SDF navigation | Learned 2D SDFs with potential-field planning and MPPI rollouts on non-circular obstacle layouts. |
| MiniIsaacGym RL | GPU REINFORCE CartPole: average survival `82.6` to `180.4` steps in `160` generations. |
| CudaPointCloud | Normal estimation reaches **3,171x** speedup at 10K points, RANSAC plane **547x** at 100K. Supports `--ply`/`--kitti`/`--xyz` file input. |
| Swarm / neuroevolution | GPU PSO, DE, CMA-ES, ACO, and `4096`-way neuroevolution with animated comparisons. |

### Diff-MPPI experiment workflow

<details>
<summary>Full experiment commands and detailed results (click to expand)</summary>

Fixed rollout budget:

```bash
./bin/benchmark_diff_mppi --quick
python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi.csv
python3 scripts/plot_diff_mppi.py --csv build/benchmark_diff_mppi.csv --out-dir build/plots
```

Cap-based wall-clock sweep:

```bash
./bin/benchmark_diff_mppi --k-values 256,512,1024,2048,4096,6144,8192 --csv build/benchmark_diff_mppi_wall_clock.csv
python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi_wall_clock.csv --time-caps 1.1,1.5,2.0
python3 scripts/plot_diff_mppi.py --csv build/benchmark_diff_mppi_wall_clock.csv --out-dir build/plots --time-caps 1.1,1.5,2.0
```

The benchmark writes per-episode CSV metrics; the summarizer emits Markdown / LaTeX tables for fixed-budget, cap-based wall-clock, and equal-time target comparisons; the plotter writes PNG/PDF figures under `build/plots/`. Reader-facing summary is in the Highlights table above and on GitHub Pages; detailed working notes live under `paper/`.

Exact matched-time tuning (`K` per planner tuned to shared controller-time targets):

```bash
python3 scripts/tune_diff_mppi_time_targets.py --preset dynamic_nav
```

Mechanism analysis (records per-step sampled / refined controls and local gradients; current `dynamic_slalom` trace shows the autodiff stage front-loads corrections — mean early-horizon `0.025` vs. late-horizon `0.001` for `diff_mppi_3`):

```bash
./bin/benchmark_diff_mppi --scenarios dynamic_slalom --planners mppi,feedback_mppi,diff_mppi_1,diff_mppi_3 --seed-count 1 --k-values 1024 --csv build/benchmark_diff_mppi_mechanism.csv --trace-csv build/benchmark_diff_mppi_mechanism_trace.csv --trace-max-steps 80
python3 scripts/plot_diff_mppi_mechanism.py --trace-csv build/benchmark_diff_mppi_mechanism_trace.csv --benchmark-csv build/benchmark_diff_mppi_feedback_dynamic_pair.csv --scenario dynamic_slalom --out-dir build/plots_mechanism
```

Dynamic-obstacle follow-up (seven feedback baselines + Diff-MPPI on two moving-obstacle tasks; `dynamic_slalom` keeps the hard-task split where only Diff-MPPI succeeds). Full numbers and exact-time tuning notes: `paper/diff_mppi_novelty_followup.md`; ICRA/IROS gap list: `paper/icra_iros_gap_list.md`.

```bash
./bin/benchmark_diff_mppi --scenarios dynamic_crossing,dynamic_slalom --k-values 256,512,1024,2048,4096,6144,8192 --csv build/benchmark_diff_mppi_feedback_dynamic_pair.csv
python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi_feedback_dynamic_pair.csv --time-caps 1.0,1.5 --time-targets 1.0,1.5
python3 scripts/plot_diff_mppi.py --csv build/benchmark_diff_mppi_feedback_dynamic_pair.csv --out-dir build/plots_feedback_dynamic_pair --time-caps 1.0,1.5 --time-targets 1.0,1.5
```

Uncertainty follow-up (seed-dependent perturbed obstacle trajectory under nominal-model planning; `mppi` fails both, `feedback_mppi` recovers crossing but fails slalom, Diff-MPPI succeeds on both). Write-up: `paper/diff_mppi_uncertainty_followup.md`.

```bash
./bin/benchmark_diff_mppi --scenarios uncertain_crossing,uncertain_slalom --planners mppi,feedback_mppi,diff_mppi_1,diff_mppi_3 --seed-count 4 --k-values 256,512,1024,2048,4096,6144,8192 --csv build/benchmark_diff_mppi_uncertain.csv
python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi_uncertain.csv --markdown-out build/benchmark_diff_mppi_uncertain_summary.md --latex-out build/benchmark_diff_mppi_uncertain_summary.tex --time-caps 1.0,1.5 --time-targets 1.0,1.5
python3 scripts/tune_diff_mppi_time_targets.py --preset uncertain_dynamic_nav
```

Hybrid-vs-gradient-only ablation (`grad_only_3` improves `corner_turn` slightly but fails `dynamic_crossing` — local gradients alone don't explain the gains):

```bash
./bin/benchmark_diff_mppi --scenarios corner_turn,dynamic_crossing --seed-count 4 --k-values 256,512,1024,2048,4096,6144,8192 --csv build/benchmark_diff_mppi_ablation.csv
python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi_ablation.csv --time-caps 1.0,1.5 --time-targets 1.0,1.5
```

CartPole, dynamic-bicycle, planar-manipulator, and 7-DOF manipulator pilots — out-of-domain transfer checks. Detailed write-ups under `paper/diff_mppi_{cartpole,dynamic_bicycle,manipulator,7dof}_followup.md`.

```bash
./bin/benchmark_diff_mppi_cartpole --csv build/benchmark_diff_mppi_cartpole.csv
./bin/benchmark_diff_mppi_dynamic_bicycle --csv build/benchmark_diff_mppi_dynamic_bicycle.csv
./bin/benchmark_diff_mppi_manipulator --seed-count 4 --k-values 256,512 --csv build/benchmark_diff_mppi_manipulator.csv
./bin/benchmark_diff_mppi_manipulator_7dof --seed-count 4 --k-values 256,512,1024 --csv build/benchmark_diff_mppi_manipulator_7dof.csv
```

Headline transfer results: `arm_static_shelf` K=256 — `feedback_mppi_cov`/`feedback_mppi_ref` reach `success=1.00` at `0.15` while `mppi` stays at `0.00 / 0.23`. `7dof_dynamic_avoid` K=512 — `diff_mppi_3` reaches `success=1.00 at 0.84 ms`, while `feedback_mppi_ref` reaches `0.75 at 4.01 ms`. `dynbike_slalom` K=32 — `diff_mppi_1` lifts `success=0.75 / 12.60` to `1.00 / 2.24`.

</details>

### Local planner cross-comparison

`benchmark_diff_mppi` now hosts a 12-planner sweep across 3 dynamic scenarios x 5 speed-scales x 2 radius-scales x 4 seeds, surfacing where DWA, STOMP, Diff-MPPI, and the Hybrid A* family each win. Detailed report: `docs/local_planner_comparison.md`. Hybrid A* design notes: `docs/hybrid_astar_baseline.md`.

Headline on the hard half (`dyn_speed_scale >= 1.5`):

| planner             | family    | hard solved | mean coll | mean ms |
|---------------------|-----------|------------:|----------:|--------:|
| dwa_med / dwa_fine  | DWA       |       12/12 |      0.00 |    0.06 |
| hybrid_astar_dwa    | Hybrid-A* |       12/12 |      0.00 |    0.07 |
| hybrid_astar_mppi   | Hybrid-A* |       11/12 |      0.00 |    0.56 |
| stomp_3_smooth      | STOMP     |        6/12 |      0.00 |    1.41 |
| diff_mppi_3_early8  | Diff-MPPI |        5/12 |      1.40 |    0.66 |
| hybrid_astar_pp     | Hybrid-A* |        3/12 |     15.58 |    0.02 |
| hybrid_astar_dyn_pp | Hybrid-A* |        2/12 |     15.92 |    0.02 |

Findings:
- **DWA wins decisively** on this benchmark (argmin + tuned w_terminal=20).
- **Global + local hybrid closes the paradigm gap** for both DWA and MPPI locals; the pattern is paradigm-agnostic.
- **Dyn-aware global search alone is brittle** -- the constant-speed search vs. accelerate-from-rest sim timing mismatch makes linearised obstacle prediction worse than blind on hard cells.

### Point-cloud benchmark snapshot

`bin/benchmark_pointcloud` compares CPU vs GPU implementations of voxel-grid filtering, statistical outlier removal, normal estimation, RANSAC plane fitting, and GICP registration. Both CPU and GPU use the same brute-force algorithms (no KD-trees). Supports `--ply`, `--kitti`, `--xyz` flags for external point cloud files.

Use it as a small point-cloud CLI demo:

```bash
./bin/benchmark_pointcloud --quick
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op voxel --leaf-size 0.8 --out build/sample_room_voxel.ply
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op ransac --plane-threshold 0.05 --out build/sample_room_plane.ply
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op normals --k 12 --out build/sample_room_normals.ply
```

`--op` can select `voxel`, `statistical`, `normals`, `ransac`, `gicp`, or `all`. `--out` writes the selected external-input result as `.ply` or `.xyz`; the `ransac` output contains plane inliers, and normals `.xyz` output uses `x y z nx ny nz` rows.

Multi-scale results (synthetic room, both CPU and GPU use same brute-force algorithms):

| Points | Operation | CPU | GPU | Speedup |
|---|---|---:|---:|---:|
| 1,000 | Voxel Grid | 0.67 ms | 1.76 ms | 0.4x (GPU loses) |
| 2,000 | Statistical Filter | 339 ms | 0.82 ms | **412x** |
| 5,000 | Normal Estimation | 4,024 ms | 2.08 ms | **1,933x** |
| 10,000 | Normal Estimation | 15,487 ms | 4.88 ms | **3,171x** |
| 50,000 | RANSAC Plane | 1,518 ms | 2.78 ms | **546x** |
| 100,000 | RANSAC Plane | 3,077 ms | 5.62 ms | **547x** |

Speedups scale with point count because the CPU baseline is O(n^2) for k-NN operations. At small n (<2K), GPU kernel launch overhead exceeds the computation, so CPU wins on simple operations like voxel grid.

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

## Experiment-First Development

This repository now treats some design work as `experiment -> convergence`, not `abstract design -> implementation`.

Current process split:
- `core/`: only the minimum interfaces that multiple variants already share
- `experiments/`: discardable concrete variants with different design styles
- `docs/experiments.md`: generated comparison results
- `docs/decisions.md`: why something is kept, rejected, or not yet promoted
- `docs/interfaces.md`: the current minimum stable contract

Concrete entrypoint:

```bash
python3 scripts/run_design_experiments.py
```

One-command local repair path:

```bash
python3 scripts/design_doctor.py
```

Create a new history snapshot while running the same repair path:

```bash
python3 scripts/design_doctor.py --snapshot-label local_check
```

Render a targeted comparison between the latest two snapshots:

```bash
python3 scripts/compare_design_snapshots.py
```

Check that the latest snapshot did not regress beyond the declared policy:

```bash
python3 scripts/check_design_regressions.py
```

Render convergence signals from the snapshot history:

```bash
python3 scripts/render_design_convergence.py
```

Render the next suggested process moves from those convergence signals:

```bash
python3 scripts/render_design_actions.py
```

Render the helper-promotion watchlist from current shared helper usage:

```bash
python3 scripts/render_helper_promotion.py
```

Refresh the checked-in design docs:

```bash
python3 scripts/refresh_design_docs.py
```

Record a new design snapshot and regenerate the history doc:

```bash
python3 scripts/snapshot_design_experiments.py --label local_check
```

Refresh the version-controlled fixture CSVs from the selected build outputs:

```bash
python3 scripts/refresh_design_fixtures.py
```

Check whether the checked-in fixtures still match the configured build outputs:

```bash
python3 scripts/refresh_design_fixtures.py --check-sync
```

Scaffold a new concrete problem with 3 disposable variants:

```bash
python3 scripts/scaffold_design_problem.py cache_policy --dry-run
```

Validate that the experiment-first guardrails still hold:

```bash
python3 scripts/validate_design_workflow.py
```

Check that the scaffolder still emits the current workflow contract:

```bash
python3 scripts/check_scaffold_design_problem.py
```

Current concrete problems:
- `planner_selection`: choose one planner configuration per dataset/scenario pair
- `fixture_promotion`: choose which benchmark fixture datasets survive into the lightweight experiment corpus
- `time_budget_selection`: choose one planner configuration per dataset/scenario/time-budget request
- `horizon_selection`: choose one Diff-MPPI gradient-update horizon per dataset/scenario pair

Each problem is implemented three different ways:
- functional scoring
- OOP / lexicographic policy objects
- staged pipeline filters

All variants consume the same aggregated input rows, answer the same request type for their problem, and are scored under the same benchmark, readability, and extensibility proxies. The process uses version-controlled fixture CSVs in `experiments/data/`, so design comparisons are reproducible without regenerating the heavy benchmark suite. `scripts/validate_design_workflow.py` now also checks that every experiment module appears in generated docs, which keeps the process state externalized instead of hiding it in code only. Nothing in `experiments/` is assumed to be permanent.

The workflow is now module-driven rather than import-driven:
- each `experiments/<problem>/__init__.py` package declares its own slug-like metadata and request builder
- each problem package also owns its own report builder
- `scripts/run_design_experiments.py` discovers those modules automatically
- `scripts/design_doctor.py` is the promoted local entrypoint for refresh-and-validate maintenance
- `scripts/run_design_experiments.py` also discovers fixture CSVs automatically from `experiments/data/`
- `experiments/data/manifest.json` defines which benchmark CSVs are promoted into the lightweight fixture set
- `scripts/refresh_design_fixtures.py --check-sync` catches drift between checked-in fixtures and available build outputs
- `scripts/snapshot_design_experiments.py` records aggregate design states into `experiments/history/` and regenerates `docs/experiments_history.md`
- `experiments/history/policy.json` defines which metrics are allowed to regress, and by how much
- `experiments/history/actions_policy.json` defines when the process should `hold`, `diversify`, or watch for promotion
- `scripts/check_design_regressions.py` compares the latest two snapshots against that policy
- `scripts/compare_design_snapshots.py` renders the latest or selected snapshot delta without editing checked-in docs
- `scripts/render_design_convergence.py` summarizes which quality signals have started to survive across snapshots
- `scripts/render_design_actions.py` turns those survival signals into explicit next-step advice
- `scripts/render_helper_promotion.py` turns repeated helper reuse into an explicit promotion watchlist
- repeated helper extraction happens in `experiments/support.py` before any implementation is considered for promotion
- `scripts/validate_design_workflow.py` fails if a discovered module is missing from generated docs or if `docs/experiments.md` is stale; the runtime column is normalized during that check because it is machine-dependent

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
| PF on Episode | `pf_on_episode` | Particle-filter localization over full trajectory episodes |

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
| MPPI | `mppi` | 4096-sample path-integral control on GPU |
| **Hybrid A* family** | `benchmark_diff_mppi --planners hybrid_astar_{pp,dwa,dyn_pp,mppi}` | **Forward-only Hybrid A* global path + four local controllers (pure pursuit, DWA, dyn-aware-search + pp, MPPI). Demonstrates the paradigm gap: blind global + pp solves 3/12 hard cells, global + DWA/MPPI local closes the gap to 11-12/12** |
| Differentiable MPPI | `diff_mppi`, `comparison_diff_mppi`, `benchmark_diff_mppi`, `benchmark_diff_mppi_cartpole`, `benchmark_diff_mppi_dynamic_bicycle`, `benchmark_diff_mppi_manipulator` | MPPI sampling update + autodiff control-gradient refinement + multi-scenario CSV benchmarking under fixed sample and wall-clock caps, plus nominal-linearization / rollout-sensitivity / covariance-regression / fused-feedback / high-frequency-feedback baselines, uncertain-dynamic follow-up, CartPole, dynamic-bicycle, and planar-manipulator pilots outside the base kinematic suite |
| Neural SDF Navigation | `neural_sdf`, `sdf_potential_field`, `sdf_mppi`, `comparison_sdf_nav` | Learned implicit obstacle fields for heatmap visualization, potential fields, and MPPI |
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

### Registration / Point Clouds

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| ICP | `icp` | GPU nearest-neighbor correspondences + batch transform updates |
| NDT | `ndt` | Voxelized normal-distribution matching kernels |
| GICP | `gicp` | GPU correspondences + point-to-plane system accumulation |
| Voxel Grid Filter | `voxel_grid_filter` | Point-wise voxel assignment + centroid accumulation |
| Statistical Outlier Removal | `benchmark_pointcloud` | Brute-force GPU k-NN mean-distance filtering |
| Normal Estimation | `benchmark_pointcloud` | PCA normal estimation with one thread per point |
| RANSAC Plane | `ransac_plane` | One RANSAC hypothesis per thread with device-side RNG |

#### GICP
Generalized ICP uses GPU nearest-neighbor search and point-to-plane system accumulation, then solves the 6x6 update on the host. The same infrastructure is reused by `bin/benchmark_pointcloud` to report CPU vs GPU registration throughput.

#### CudaPointCloud Snapshot
The benchmark room cloud now has a rotating visual summary that shows the same synthetic scene as raw input, after statistical filtering, with the dominant plane highlighted, and with local PCA normals:

<img src="https://rsasaki0109.github.io/CudaRobotics/pointcloud_processing.gif" alt="pointcloud_processing" width="720"/>

Generated from the local benchmark dataset with:

```bash
python3 scripts/render_pointcloud_processing_gif.py
```

### Learning / Optimization

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| Neuroevolution | `neuroevo` | One policy evaluation per individual, 4096 individuals in parallel |
| Neuroevolution Comparison | `comparison_neuroevo` | CPU sequential evolution vs GPU population-scale evolution |
| PSO | `pso_cuda` | 100K particles updated in parallel |
| Differential Evolution | `differential_evolution` | Population-wide mutation, crossover, and selection on GPU |
| CMA-ES | `cma_es` | GPU candidate evaluation and covariance-guided search |
| ACO for TSP | `aco_tsp` | Thousands of ants concurrently construct tours |
| Swarm Comparison | `comparison_swarm` | Side-by-side convergence visualization for PSO, DE, and CMA-ES |

### Simulation / RL

| Algorithm | Binary | CUDA Parallelization |
|---|---|---|
| MiniIsaacGym CartPole | `mini_isaac` | 4096 environments stepped in parallel with GPU-side action generation |
| MiniIsaacGym REINFORCE | `mini_isaac_rl` | GPU rollout buffer, return computation, policy-gradient construction, and MLP updates |

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
