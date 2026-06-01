# GPU Parallel Wavefront Planner

`src/gpu_wavefront_planner.cu`

A Bellman-Ford-style cost-to-go relaxation over an occupancy grid that yields
the exact shortest-path field from a goal, then extracts a path by greedy
descent. This is the throughput-oriented parallel counterpart to the serial
Dijkstra wavefront used in classic grid planners, complementing the repo's
A*/Dijkstra demos. Maps onto the canonical 2D idiom: **one thread = one cell**.

![demo](https://rsasaki0109.github.io/CudaRobotics/gpu_wavefront_planner.gif)

## Pipeline (CPU and GPU reach the SAME fixpoint)

Each sweep, every free cell pulls its cost-to-go down to the cheapest
neighbour-plus-edge:

```
D(p) = min( D(p), min_{q in N8(p), free} D(q) + w(p,q) )
```

with integer edge weights (10 orthogonal, 14 diagonal ~ 10·√2). The goal is
pinned to 0, obstacles to INF. Iterating to a fixpoint gives the exact
single-source shortest-path field (label-correcting Bellman-Ford). The GPU runs
Jacobi sweeps (`atomicMin` into the shared field) and batches several sweeps
between host sync checks so the per-iteration changed-flag round-trip does not
dominate; the path is then read off by following decreasing cost from the start.

## Correctness — bit-identical fixpoint

The shortest-path cost field is unique, so the GPU's Jacobi sweeps and the CPU's
in-place Gauss-Seidel sweeps converge to the exact same integer field (they take
a different number of sweeps to get there, which is expected). Integer arithmetic
plus a deterministic min means the fields — and the extracted path — match
exactly.

| metric | value |
|---|---|
| cost-field cell mismatches (CPU vs GPU) | `0 / 147456` |
| extracted path (CPU vs GPU) | identical (`723` cells, goal cost `8420`) |

## Result (this machine)

| | scale | time | note |
|---|---|---|---|
| CPU serial (Gauss-Seidel) | `384×384`, 561 sweeps | `~900 ms` | reference |
| **GPU (Jacobi, batched sync)** | one thread / cell, 768 sweeps | **`~4.6 ms`** | **~195×** |

The relaxation does more total work than a priority-queue Dijkstra, but it is
embarrassingly parallel — every cell relaxes independently each sweep — so the
GPU turns the extra work into a large wall-clock win.

## Reproduce

```bash
cd build && cmake .. && make gpu_wavefront_planner -j$(nproc)
cd .. && ./bin/gpu_wavefront_planner
```

Prints the timing + bit-identity table and writes
`gif/gpu_wavefront_planner.gif` (the cost-to-go wavefront expanding from the
goal through the corridors, then the extracted shortest path).

## Notes

- One demo = one `.cu`; reuses `include/cuda_check.cuh` and `include/cuda_video.h`.
- 8-connected grid; integer 10/14 edge weights keep the field exact and
  CPU/GPU bit-identical (no floating-point √2).
- Batched-sync lesson reused from the frontier-exploration demo: checking the
  changed flag every sweep makes the host round-trip dominate; batching 16
  sweeps between checks keeps the result identical (monotone relaxation) while
  removing the overhead.
- `--fmad=false`; GIF served from gh-pages (not committed to the repo).
