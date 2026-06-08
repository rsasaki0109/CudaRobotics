# GPU Batched iLQR Trajectory Optimization

`gpu_batched_ilqr` adds the **gradient-based** optimal-control counterpart to the
repo's sampling-based controllers (the MPPI family, STOMP, CMA-ES, MCTS).
iterative LQR (iLQR / DDP) optimizes a single trajectory with a backward Riccati
sweep and a forward line-search rollout, repeated to convergence — a sequential
algorithm that does **not** parallelize across one trajectory.

The parallelism is across **problems**. A motion planner answering many
start/goal queries in the same map solves thousands of independent iLQR
instances, which maps cleanly onto the repo's canonical idiom: **one thread =
one optimal-control problem**.

## Setup

- Shared 2D obstacle field, `4096` random start/goal queries.
- Unicycle dynamics, state `x = [px, py, theta]`, control `u = [v, omega]`,
  horizon `T = 40`, `25` iLQR iterations.
- Costs: quadratic control, a stage goal pull, a heavy terminal goal term, and
  soft circular-obstacle penalties (Gauss-Newton Hessian, so `l_xx` stays PSD).
- The solver is a single `__host__ __device__` routine called **both** by a
  serial CPU loop and by the batch CUDA kernel, so the two paths run the same
  arithmetic.

## Correctness on a non-convex problem (honest framing)

iLQR has data-dependent discrete branches (which line-search step `alpha` is
accepted, the PD guard on `Q_uu`). On this non-convex field many queries have
two competing local optima (go above vs. below an obstacle), so per-problem
bit-matching between host and device is **neither expected nor claimed** — tiny
floating-point rounding (FMA contraction is already disabled with `--fmad=false`)
tips a few knife-edge queries into the other, equally valid, optimum. The
demo therefore reports the agreement *distribution* and the *solution quality*,
not a single MAE:

- **median** per-problem cost difference `7.6e-6`,
- **88.5%** of queries reach the identical local optimum (within 1% of the CPU
  cost),
- the remaining ~11% settle into an alternative local optimum, but the **mean
  achieved cost matches CPU to ~1.1%** — equal-quality solutions.

This is the efficiency statement made honestly: the GPU is not solving a better
iLQR, it is solving the identical iLQR on every query at once. The win is
throughput on the batch.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_batched_ilqr -j$(nproc)
./bin/gpu_batched_ilqr
```

Generated files:

- `tmp/gpu_batched_ilqr.avi`
- `gif/gpu_batched_ilqr.gif`

## Output

The GIF animates the iLQR convergence of 8 representative queries over the
shared obstacle field, with an info panel tracking the iteration, the mean cost,
and the batch headline (CPU vs GPU timing, speedup, and the CPU/GPU agreement).

Latest local run:

- `4096` problems (`T = 40`, `25` iLQR iterations).
- CPU serial batch `462 ms`, GPU batch kernel `3.3 ms` — about **140x**.
- Per-problem throughput: CPU `112.8 us` vs GPU `0.81 us`.
- Same local optimum on `88.5%` of queries; mean cost CPU `49.96` vs GPU `49.39`
  (`1.1%` apart).
