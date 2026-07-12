# CUDA Acceleration Static Audit

Scanned 213 CUDA translation units. Scores are triage signals, not measured speedup claims.

| Rank | File | Score | Kernels | Kernel loops | Nested | Round trips | Syncs | Atomics | Signals |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `src/benchmark_diff_mppi.cu` | 809 | 56 | 176 | 31 | 1 | 6 | 0 | nested kernel loops; 176 kernel loops; D2H+H2D round trip; 6 explicit sync; 131 transcendental sites |
| 2 | `src/benchmark_diff_mppi_manipulator_7dof.cu` | 320 | 10 | 44 | 14 | 1 | 2 | 0 | nested kernel loops; 44 kernel loops; D2H+H2D round trip; 2 explicit sync; 19 symbolic local arrays |
| 3 | `src/gpu_constrained_mpc.cu` | 273 | 1 | 45 | 16 | 1 | 0 | 0 | nested kernel loops; 45 kernel loops; D2H+H2D round trip |
| 4 | `src/graph_slam.cu` | 190 | 9 | 29 | 8 | 1 | 3 | 6 | nested kernel loops; 29 kernel loops; D2H+H2D round trip; 3 explicit sync; 6 atomics; 16 transcendental sites |
| 5 | `src/benchmark_diff_mppi_manipulator.cu` | 181 | 12 | 33 | 8 | 1 | 2 | 0 | nested kernel loops; 33 kernel loops; D2H+H2D round trip; 2 explicit sync |
| 6 | `src/gpu_filterreg_p2plane.cu` | 163 | 9 | 22 | 6 | 1 | 0 | 15 | nested kernel loops; 22 kernel loops; D2H+H2D round trip; 15 atomics; 2 symbolic local arrays |
| 7 | `src/benchmark_diff_mppi_mujoco_reacher.cu` | 153 | 10 | 23 | 5 | 1 | 3 | 0 | nested kernel loops; 23 kernel loops; D2H+H2D round trip; 3 explicit sync; 8 symbolic local arrays |
| 8 | `src/gpu_online_slam_3d_switchable.cu` | 147 | 9 | 22 | 7 | 1 | 0 | 6 | nested kernel loops; 22 kernel loops; D2H+H2D round trip; 6 atomics |
| 9 | `src/benchmark_diff_mppi_dynamic_bicycle.cu` | 140 | 10 | 26 | 5 | 1 | 2 | 0 | nested kernel loops; 26 kernel loops; D2H+H2D round trip; 2 explicit sync; 5 transcendental sites |
| 10 | `src/gpu_fgr.cu` | 140 | 6 | 22 | 5 | 1 | 0 | 4 | nested kernel loops; 22 kernel loops; D2H+H2D round trip; 4 atomics; 3 symbolic local arrays; 4 transcendental sites |
| 11 | `src/fgr_gpu.cu` | 132 | 6 | 22 | 4 | 1 | 0 | 4 | nested kernel loops; 22 kernel loops; D2H+H2D round trip; 4 atomics; 3 symbolic local arrays; 4 transcendental sites |
| 12 | `src/gpu_megaparticles_6dof.cu` | 127 | 21 | 11 | 2 | 1 | 4 | 21 | nested kernel loops; 11 kernel loops; D2H+H2D round trip; 4 explicit sync; 21 atomics; 4 symbolic local arrays |
| 13 | `src/gpu_pose_graph_slam_3d.cu` | 113 | 7 | 16 | 5 | 1 | 0 | 6 | nested kernel loops; 16 kernel loops; D2H+H2D round trip; 6 atomics |
| 14 | `src/gpu_pose_graph_slam_3d_switchable.cu` | 113 | 8 | 16 | 5 | 1 | 0 | 6 | nested kernel loops; 16 kernel loops; D2H+H2D round trip; 6 atomics |
| 15 | `src/filterreg_gpu.cu` | 108 | 7 | 13 | 5 | 1 | 0 | 8 | nested kernel loops; 13 kernel loops; D2H+H2D round trip; 8 atomics |
| 16 | `src/gpu_real_bunny_reg.cu` | 107 | 5 | 16 | 3 | 1 | 0 | 8 | nested kernel loops; 16 kernel loops; D2H+H2D round trip; 8 atomics; 2 symbolic local arrays |
| 17 | `src/gpu_robust_p2plane_reg.cu` | 107 | 5 | 16 | 3 | 1 | 0 | 8 | nested kernel loops; 16 kernel loops; D2H+H2D round trip; 8 atomics; 2 symbolic local arrays |
| 18 | `src/benchmark_diff_mppi_cartpole.cu` | 105 | 13 | 20 | 3 | 1 | 2 | 0 | nested kernel loops; 20 kernel loops; D2H+H2D round trip; 2 explicit sync; 4 transcendental sites |
| 19 | `src/gpu_online_slam.cu` | 100 | 10 | 15 | 3 | 1 | 0 | 7 | nested kernel loops; 15 kernel loops; D2H+H2D round trip; 7 atomics; 4 transcendental sites |
| 20 | `src/gpu_megaparticles_lsh.cu` | 96 | 15 | 5 | 0 | 1 | 3 | 27 | 5 kernel loops; D2H+H2D round trip; 3 explicit sync; 27 atomics; 19 transcendental sites |
| 21 | `src/comparison_swarm.cu` | 91 | 13 | 12 | 1 | 1 | 15 | 2 | nested kernel loops; 12 kernel loops; D2H+H2D round trip; 15 explicit sync; 2 atomics |
| 22 | `src/gpu_bundle_adjustment.cu` | 88 | 14 | 9 | 3 | 1 | 1 | 10 | nested kernel loops; 9 kernel loops; D2H+H2D round trip; 1 explicit sync; 10 atomics |
| 23 | `src/benchmark_diff_mppi_mujoco.cu` | 87 | 8 | 13 | 2 | 1 | 3 | 0 | nested kernel loops; 13 kernel loops; D2H+H2D round trip; 3 explicit sync; 4 symbolic local arrays |
| 24 | `src/gpu_megaparticles_stein_mcl.cu` | 86 | 20 | 7 | 0 | 1 | 3 | 16 | 7 kernel loops; D2H+H2D round trip; 3 explicit sync; 16 atomics; 2 symbolic local arrays; 14 transcendental sites |
| 25 | `src/benchmark_diff_mppi_pushing_box.cu` | 85 | 14 | 15 | 1 | 1 | 4 | 0 | nested kernel loops; 15 kernel loops; D2H+H2D round trip; 4 explicit sync; 2 symbolic local arrays; 5 transcendental sites |
| 26 | `src/gpu_megaparticles_gicp_mcl.cu` | 82 | 13 | 5 | 1 | 1 | 3 | 16 | nested kernel loops; 5 kernel loops; D2H+H2D round trip; 3 explicit sync; 16 atomics; 13 transcendental sites |
| 27 | `src/comparison_diff_mppi.cu` | 74 | 7 | 12 | 2 | 1 | 2 | 0 | nested kernel loops; 12 kernel loops; D2H+H2D round trip; 2 explicit sync; 5 transcendental sites |
| 28 | `src/diff_mppi.cu` | 74 | 7 | 12 | 2 | 1 | 2 | 0 | nested kernel loops; 12 kernel loops; D2H+H2D round trip; 2 explicit sync; 5 transcendental sites |
| 29 | `src/gpu_diff_contact_push.cu` | 74 | 1 | 8 | 2 | 1 | 1 | 0 | nested kernel loops; 8 kernel loops; D2H+H2D round trip; 1 explicit sync; 5 symbolic local arrays; 7 transcendental sites |
| 30 | `src/benchmark_diff_mppi_pushing.cu` | 69 | 10 | 11 | 1 | 1 | 2 | 0 | nested kernel loops; 11 kernel loops; D2H+H2D round trip; 2 explicit sync; 2 symbolic local arrays; 5 transcendental sites |
| 31 | `src/comparison_stomp.cu` | 68 | 4 | 7 | 3 | 1 | 3 | 0 | nested kernel loops; 7 kernel loops; D2H+H2D round trip; 3 explicit sync; 2 symbolic local arrays |
| 32 | `src/robust_p2plane_gpu.cu` | 68 | 5 | 11 | 1 | 1 | 0 | 4 | nested kernel loops; 11 kernel loops; D2H+H2D round trip; 4 atomics; 2 symbolic local arrays |
| 33 | `src/stomp.cu` | 68 | 4 | 7 | 3 | 1 | 3 | 0 | nested kernel loops; 7 kernel loops; D2H+H2D round trip; 3 explicit sync; 2 symbolic local arrays |
| 34 | `src/amcl.cu` | 67 | 9 | 8 | 1 | 1 | 7 | 0 | nested kernel loops; 8 kernel loops; D2H+H2D round trip; 7 explicit sync; 17 transcendental sites |
| 35 | `src/comparison_amcl.cu` | 67 | 9 | 8 | 1 | 1 | 7 | 0 | nested kernel loops; 8 kernel loops; D2H+H2D round trip; 7 explicit sync; 17 transcendental sites |
| 36 | `src/comparison_expansion_reset_mcl.cu` | 67 | 11 | 10 | 1 | 1 | 4 | 0 | nested kernel loops; 10 kernel loops; D2H+H2D round trip; 4 explicit sync; 16 transcendental sites |
| 37 | `src/gpu_sinkhorn_reg.cu` | 67 | 5 | 10 | 2 | 1 | 0 | 4 | nested kernel loops; 10 kernel loops; D2H+H2D round trip; 4 atomics |
| 38 | `src/sinkhorn_gpu.cu` | 67 | 5 | 10 | 2 | 1 | 0 | 4 | nested kernel loops; 10 kernel loops; D2H+H2D round trip; 4 atomics |
| 39 | `src/expansion_reset_mcl.cu` | 65 | 11 | 10 | 1 | 1 | 3 | 0 | nested kernel loops; 10 kernel loops; D2H+H2D round trip; 3 explicit sync; 16 transcendental sites |
| 40 | `src/gpu_gicp_2d.cu` | 65 | 4 | 7 | 0 | 0 | 0 | 11 | 7 kernel loops; 11 atomics; 2 symbolic local arrays; 8 transcendental sites |
| 41 | `src/gpu_kiss_icp.cu` | 65 | 4 | 10 | 1 | 1 | 0 | 4 | nested kernel loops; 10 kernel loops; D2H+H2D round trip; 4 atomics; 2 symbolic local arrays |
| 42 | `src/gpu_kld_amcl.cu` | 65 | 13 | 8 | 2 | 1 | 2 | 0 | nested kernel loops; 8 kernel loops; D2H+H2D round trip; 2 explicit sync; 8 transcendental sites |
| 43 | `src/gpu_lidar_slam.cu` | 64 | 8 | 3 | 0 | 1 | 4 | 13 | 3 kernel loops; D2H+H2D round trip; 4 explicit sync; 13 atomics; 10 transcendental sites |
| 44 | `src/gpu_pose_graph_slam.cu` | 64 | 6 | 9 | 1 | 1 | 0 | 6 | nested kernel loops; 9 kernel loops; D2H+H2D round trip; 6 atomics; 4 transcendental sites |
| 45 | `src/rrt_star_3d.cu` | 64 | 4 | 7 | 2 | 1 | 4 | 0 | nested kernel loops; 7 kernel loops; D2H+H2D round trip; 4 explicit sync; 6 transcendental sites |
| 46 | `src/visibility_mppi.cu` | 63 | 8 | 8 | 2 | 1 | 3 | 0 | nested kernel loops; 8 kernel loops; D2H+H2D round trip; 3 explicit sync; 4 transcendental sites |
| 47 | `src/comparison_fastslam.cu` | 62 | 8 | 9 | 1 | 1 | 3 | 0 | nested kernel loops; 9 kernel loops; D2H+H2D round trip; 3 explicit sync; 8 transcendental sites |
| 48 | `src/gpu_em_gmm.cu` | 62 | 2 | 5 | 1 | 1 | 1 | 8 | nested kernel loops; 5 kernel loops; D2H+H2D round trip; 1 explicit sync; 8 atomics; 1 symbolic local arrays; 5 transcendental sites |
| 49 | `src/gpu_gaussian_splatting_slam.cu` | 62 | 4 | 9 | 2 | 1 | 0 | 2 | nested kernel loops; 9 kernel loops; D2H+H2D round trip; 2 atomics |
| 50 | `src/esdf_mppi.cu` | 60 | 7 | 7 | 2 | 1 | 3 | 0 | nested kernel loops; 7 kernel loops; D2H+H2D round trip; 3 explicit sync; 4 transcendental sites |

Interpretation:

- Nested or many per-thread kernel loops suggest algorithmic parallelism is still serial.
- D2H+H2D and explicit synchronization suggest pipeline fusion or device-resident iteration.
- Atomics require contention measurement; their presence alone does not imply a bottleneck.
- Allocation counts are inventory only; source review must confirm whether allocation occurs in a hot loop.

## Manually validated remaining opportunities

These are source-reviewed hypotheses, not benchmark results. `High` means the code contains a serial or asymptotically expensive structure that can be replaced; it does not guarantee a particular speedup.

| Priority | Algorithms | Source evidence | Optimization direction | Multi-x confidence |
|---|---|---|---|---|
| A | FGR (`gpu_fgr.cu`, `fgr_gpu.cu`) | KNN and feature matching scan every target point/feature, giving dense O(N^2) work | Spatial index or hash; shared-memory tiling/GEMM for descriptors | High at large N |
| A | Constrained MPC (`gpu_constrained_mpc.cu`) | One CUDA thread performs long nested horizon/iLQR loops for a whole problem | Block-per-problem horizon parallelism and batched small-matrix operations | High for long horizons or few problems |
| A | MPC-QP (`gpu_mpc_qp.cu`) | One thread serializes ADMM and forward/back substitution for each agent | Warp/block-per-agent solver or batched triangular solves | High as horizon M grows |
| A | MegaParticles 6DoF (`gpu_megaparticles_6dof.cu`) | Cumulative sum is a one-thread O(N) kernel; likelihood also loops over every scan per particle | CUB DeviceScan plus scan-major/coalesced likelihood evaluation | High for the resampling stage |
| A | Pose graph 3D / bundle adjustment | PCG copies scalar reductions to the host in every iteration; normal equations use many atomics | Device-resident PCG/convergence and block-aggregated assembly | Medium-high for solver-heavy cases |
| A | FilterReg / point-to-plane registration | Dense nested correspondence loops and contended atomic normal-equation updates | Spatial pruning, tiled correspondence, block reductions | Medium-high at large point counts |
| B | Graph/online SLAM family | Repeated synchronization, host round trips, and atomic assembly | Keep iterations device-resident; fuse reductions and aggregate atomics | Medium; profile by graph size |
| B | Diff-MPPI variants and STOMP | Per-thread trajectory loops and synchronization remain, but workload is already broadly parallel | Transposed layouts, fusion, graphs, and selective recomputation | Workload-dependent |

## Already demonstrated multi-x acceleration

Repository benchmarks already report large GPU gains for several families, so they are not the first targets for another algorithmic rewrite:

| Algorithm | Reported result | Evidence |
|---|---:|---|
| Batched iLQR | about 140x | `docs/gpu_batched_ilqr.md` |
| KD-tree nearest neighbor | about 175x vs CPU KD-tree; 10,500x vs brute force | `docs/gpu_kdtree_nn.md` |
| SGM stereo | about 46x | `docs/gpu_sgm_stereo.md` |
| Gaussian splatting renderer | 1,381x in the documented comparison | `docs/gaussian_splatting_renderer.md` |
| MPPI control update | up to 6.27x in the controlled benchmark | `docs/results/mppi_control_update_2026-07-12.md` |

## Conclusion

Not every algorithm has another multi-x gain available: small inputs, memory-bandwidth limits, and already parallel implementations often cap low-risk tuning near 1.1-2x. The Priority A items are the strongest remaining multi-x candidates because they expose serial O(N), O(N^2), or host-synchronized iterative work. Benchmarking each proposed replacement against identical inputs is required before claiming a speedup.
