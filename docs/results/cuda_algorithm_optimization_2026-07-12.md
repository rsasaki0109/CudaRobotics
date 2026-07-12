# CUDA Algorithm Optimization Results

Date: 2026-07-12

## Method

Release builds from the modified tree and commit `c0cf016` were run on the same
machine, GPU, inputs, and executable defaults. The baseline was built in a
detached worktree with the same CUDA 12.8, Visual Studio 2022, and OpenCV 4.12
toolchain. Reported improvements are end-to-end timings printed by each demo;
no speedup is inferred from source inspection alone.

## Accepted changes

| Algorithm | Baseline | Optimized | Speedup | Validation |
|---|---:|---:|---:|---|
| FGR, 4,239 x 6,046 FPFH matching/registration | 403.2 ms | 304.1 ms | 1.33x | 0.04 deg / 0.0004 translation error, PASS |
| MegaParticles local bootstrap step | 3.256 ms | 0.989 ms | 3.29x | relocalization run completed; final Mega error 0.109 m / 1.44 deg |
| 3D pose graph, 384 poses / 575 edges | 1131.082 ms | 593.365 ms | 1.91x | final cost 555.65; GPU/CPU RMSE agreement maintained |

FGR tiles 33-dimensional target descriptors in shared memory, reducing repeated
global loads while retaining exact nearest-descriptor matching. MegaParticles
replaces the one-thread O(N) cumulative sum with CUB `DeviceScan::InclusiveSum`.
The pose-graph PCG loop keeps its reductions and scalar recurrence on device,
eliminating three device-to-host scalar copies per iteration.

The million-particle MegaParticles path remains likelihood-dominated: its full
step changed only from 222.811 to 221.880 ms. The 3.29x result applies to the
bootstrap/resampling path containing the replaced scan, not the complete
million-particle pipeline.

## Rejected experiments

| Experiment | Baseline | Candidate | Decision |
|---|---:|---:|---|
| MPC-QP warp-per-agent ADMM, M=40 | 25.717 ms/step | 27.540 ms/step | Rejected: 7% slower; triangular dependency synchronization dominates |
| Bundle Adjustment device-scalar PCG | 1.11 ms/LM iteration | 6.31 ms/LM iteration | Rejected: 5.7x slower; fixed 80 iterations lose effective host-driven early exit |
| FilterReg block-aggregated normal-equation atomics | 1010.0 ms | 1007.9 ms | Rejected: no material end-to-end gain |

All rejected implementations were removed. The unchanged MPC-QP and FilterReg
executables still pass their original numerical checks. Constrained MPC already
batches 400 independent problems and passes its collision, reachability, and
control-limit checks; its backward Riccati recursion is sequential in horizon,
so a safe block rewrite requires a different parallel solver rather than a
low-risk kernel substitution.

## Environment notes

OpenCV 4.12 was installed under `E:/tools/vcpkg`. Configure with:

```powershell
cmake -S . -B build/cuda_all_opt_vcpkg -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=E:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Windows builds exposed two pre-existing portability issues: floating-point
`M_PI` was not available to NVCC/MSVC, and the pose-graph video path used the
Unix-only `mkdir -p`. The source now uses a literal pi constant and a
platform-specific directory command. Missing `ffmpeg` affects GIF conversion
only and does not affect CUDA timing or numerical validation.
