# GPU Frontier-Based Exploration on an Occupancy Grid

`src/gpu_frontier_exploration.cu`

The classic Yamauchi (1997) "where do I go next?" primitive for autonomous
mapping. A *frontier* is a FREE cell that touches at least one UNKNOWN cell —
the boundary between what the robot has mapped and what it has not. Driving the
robot to frontiers, over and over, is how occupancy-grid SLAM front-ends decide
where to explore. Maps onto the repo's canonical 2D idiom: **one thread = one
cell**.

![demo](https://rsasaki0109.github.io/CudaRobotics/gpu_frontier_exploration.gif)

## Pipeline (CPU and GPU run the SAME integer logic)

1. **Frontier detect** — a cell is a frontier iff `state == FREE` and any of its
   8 neighbours is `UNKNOWN`. Embarrassingly parallel.
2. **Connected components** — `label[i] = i` for frontier cells; each cell pulls
   its label down to the smallest label among its frontier neighbours; iterate
   until no label moves. This is the parallel min-propagation (union-find-lite)
   used by `gpu_dbscan`, here over a grid neighbourhood. The GPU batches several
   sweeps between host sync checks (the per-iteration changed-flag round-trip
   otherwise dominates); over-running a few sweeps past the fixpoint is harmless
   because labels are monotone non-increasing, so the result is unchanged.
3. **Cluster reduction** — `atomicAdd` accumulates `(sum_x, sum_y, count)` per
   label, giving each frontier component a size and a centroid.
4. **Target select** — pick the component maximising `size / distance` to the
   robot (favouring big, nearby openings), ignoring specks below a minimum size,
   then steer toward the frontier cell ahead of the current heading.

The demo runs the loop for several exploration steps: the robot repeatedly drives
to its chosen frontier and re-reveals the world through a line-of-sight sensor,
so you watch the unknown shrink (5.9 % → 31.6 % explored over the run) and the
frontier fragment into multiple components as it wraps around obstacles.

## Correctness — exact by construction

Every stage is exact integer arithmetic with no data-dependent branch that forks
into a different answer, so CPU and GPU produce bit-identical results.

| metric | value |
|---|---|
| frontier cells (CPU == GPU) | `547 / 547`, flag mismatch `0` |
| connected components (CPU vs GPU) | `1 / 1`, per-cell label mismatch `0` (after canonical renumber) |
| next exploration target (CPU vs GPU) | identical cell |

## Result (this machine)

| | scale | time | note |
|---|---|---|---|
| CPU serial | `512×512` cells, 320 CC sweeps | `~100 ms` | reference (memory-bound; varies run to run) |
| **GPU** | one thread / cell, batched CC sweeps | **`~1.2 ms`** | **~80×** |

The connected-components phase is the cost driver: a thin frontier curve has a
large diameter, so the min-label needs hundreds of sweeps to converge. The GPU
does each full-grid sweep in microseconds; the CPU pays a `262 144`-cell pass
per sweep.

## Reproduce

```bash
cd build && cmake .. && make gpu_frontier_exploration -j$(nproc)
cd .. && ./bin/gpu_frontier_exploration
```

Prints the timing + exact-agreement table and writes
`gif/gpu_frontier_exploration.gif` (the explored map growing, frontier
components colour-coded, robot trajectory + chosen target overlaid).

## Notes

- One demo = one `.cu`; reuses `include/cuda_check.cuh` and `include/cuda_video.h`.
- The ground-truth world (walls + obstacles + a divider with a gap) and the
  line-of-sight sensor reveal are the *simulator*; the GPU work being benchmarked
  is the frontier-detect + connected-components + reduce + select pipeline.
- The component centroid is a useless heading when a frontier fully encircles
  open space (centroid == robot), so the robot steers to the frontier cell most
  aligned with its current heading, committing to a consistent sweep direction.
- `--fmad=false`; GIF served from gh-pages (not committed to the repo).
