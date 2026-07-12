# MPPI control-update parallel reduction (2026-07-12)

The production MPPI control update previously launched one thread per control
element (`T * 3`, normally 168 threads), with every thread serially summing all
K rollouts. The replacement assigns a tile of 32 adjacent controls to one CUDA
block. Warp lanes retain coalesced reads across adjacent controls while eight
warps divide the K dimension and reduce through shared memory.

## Microbenchmark

- GPU: NVIDIA GeForce GTX 1660 Ti, 6 GB
- CUDA: 12.8 compiler; driver 596.36
- Shape: `T=56`, three controls per step
- Timing: 20 warmups, 200 iterations, CUDA events
- Input: deterministic perturbations and uniform normalized weights

| K | Legacy (ms) | Parallel (ms) | Speedup | Maximum absolute output error |
|---:|---:|---:|---:|---:|
| 2,048 | 0.044448 | 0.030125 | 1.48x | 1.49e-7 |
| 8,192 | 0.262562 | 0.041866 | 6.27x | 7.60e-7 |
| 16,384 | 0.517504 | 0.071277 | 7.26x | 9.54e-7 |
| 65,536 | 1.777296 | 0.286189 | 6.21x | 6.66e-6 |

Run:

```bash
cmake --build build --target benchmark_mppi_control_update
./bin/benchmark_mppi_control_update
```

## Production smoke

The ROS-independent wall-gap closed-loop test passed with the optimized
production core:

| K | Result | Sim time to goal | Mean solve | Max solve | Min valid rollout ratio |
|---:|---|---:|---:|---:|---:|
| 2,048 | pass | 18.1 s | 0.42 ms | 0.77 ms | 100% |
| 65,536 | pass | 16.1 s | 4.44 ms | 5.85 ms | 100% |

The production numbers include rollout, host softmin, transfers, control
update, and costmap upload. They are a correctness smoke on this machine, not a
before/after claim; the speedup table above is the controlled same-process
comparison. Reduction order changes introduce small floating-point differences
but no change in closed-loop success in this test.

## Device-only softmin follow-up

The next optimization removes the per-iteration device-to-host cost copy,
host exponential/normalization loop, and host-to-device weight copy. CUB device
reductions compute the minimum and weight sum; CUDA kernels generate and
normalize weights. The cost vector is copied once after the final optimizer
iteration for diagnostics.

| K | Host pipeline (ms) | GPU pipeline (ms) | Speedup | Maximum weight error |
|---:|---:|---:|---:|---:|
| 2,048 | 0.061009 | 0.048271 | 1.26x | 1.16e-10 |
| 8,192 | 0.154051 | 0.068651 | 2.24x | 2.91e-11 |
| 16,384 | 0.303727 | 0.063111 | 4.81x | 1.46e-11 |
| 65,536 | 1.053354 | 0.069814 | 15.09x | 3.64e-12 |

Production wall-gap smoke after both optimizations:

| K | Result | Sim time to goal | Mean solve | Max solve |
|---:|---|---:|---:|---:|
| 2,048 | pass | 19.8 s | 0.37 ms | 1.71 ms |
| 65,536 | pass | 16.1 s | 3.74 ms | 4.26 ms |

Compared with the immediately preceding control-update-only run on the same
machine, mean end-to-end solve time changed from 0.42 to 0.37 ms at K=2,048
and from 4.44 to 3.74 ms at K=65,536. The K=2,048 trajectory took 1.7 more
simulated seconds to reach the goal. MPPI is sensitive to small floating-point
changes in reduction order, so this is retained as an explicit quality caveat;
both tested runs remained collision-free with a 100% minimum valid-rollout
ratio.

## Device-side diagnostics follow-up

Final rollout diagnostics now reduce on the GPU. Instead of copying K costs and
looping on the CPU, the implementation computes minimum, cost sum, and valid
rollout count on-device and copies one 12-byte result structure.

| K | Host diagnostics (ms) | GPU diagnostics (ms) | Speedup | Min error | Mean error | Valid-count error |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.020475 | 0.073198 | 0.28x | 0 | 0.00390625 | 0 |
| 8,192 | 0.055190 | 0.058780 | 0.94x | 0 | 0.00390625 | 0 |
| 16,384 | 0.091296 | 0.049460 | 1.85x | 0 | 0.00390625 | 0 |
| 65,536 | 0.291283 | 0.042912 | 6.79x | 0 | 0.00390625 | 0 |

Small K does not amortize the extra reduction kernels in isolation. In the
production smoke, however, removing the K-element transfer changed mean solve
time from 0.37 to 0.34 ms at K=2,048 and from 3.74 to 3.35 ms at K=65,536.
Both runs reproduced the preceding trajectories exactly (19.8 and 16.1
simulated seconds to goal), remained collision-free, and retained a 100%
minimum valid-rollout ratio. The small mean-cost difference is float reduction
order at approximately million-scale collision values; minimum and valid count
matched exactly.

## Fused normalization follow-up

Softmin weights now remain unnormalized in device memory. The tiled control
update divides its final reduced value by the device weight sum, removing one
K-element normalization write and one kernel launch. Production smoke remained
successful and collision-free:

| K | Sim time to goal | Mean solve | Max solve |
|---:|---:|---:|---:|
| 2,048 | 18.5 s | 0.34 ms | 0.69 ms |
| 65,536 | 16.1 s | 3.38 ms | 3.77 ms |

This was performance-neutral in end-to-end timing versus the immediately prior
0.34/3.35 ms run, within run-to-run noise. It is retained because it removes a
full-array write and launch without reducing success or clearance in the smoke
test; no additional speedup is claimed.
