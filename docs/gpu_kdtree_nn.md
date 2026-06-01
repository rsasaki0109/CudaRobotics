# GPU KD-tree Nearest-Neighbour Search

`src/gpu_kdtree_nn.cu`

The foundational spatial-index query the repo's point-cloud stack (voxel /
statistical filtering, normal estimation, GICP, RANSAC) leans on but never
showed as a primitive. A balanced KD-tree is built once on the host (recursive
median split) and uploaded as flat arrays; the queries are embarrassingly
parallel, so the GPU map is **one thread = one query point**.

![demo](https://rsasaki0109.github.io/CudaRobotics/gpu_kdtree_nn.gif)

## Pipeline

1. **Build** (host) — recursive median split (`std::nth_element` per level)
   into flat node arrays: `node_pt`, `node_axis`, `left`, `right`.
2. **Query** (GPU) — each thread descends to the query's leaf, then backtracks
   with an explicit stack, pruning any subtree whose splitting-plane distance
   already exceeds the best distance found. This is the textbook **exact** NN
   search, no recursion.

## Correctness — exact, matches brute force

The nearest neighbour is an argmin over squared distances, and the KD-tree
prunes only provably-farther subtrees, so it returns the SAME neighbour as an
exhaustive scan. With random float coordinates ties are measure-zero, so the
GPU KD-tree and CPU brute force agree on the neighbour index for every query.

| metric | value |
|---|---|
| GPU KD-tree vs CPU brute force | `0 / 40000` mismatches (`100.0000 %`) |
| GPU KD-tree vs CPU KD-tree | `0` mismatches |

## Result (this machine)

| method | time (40k queries / 40k points) | note |
|---|---|---|
| CPU brute force | `~1150 ms` | exhaustive `O(Q·N)` baseline |
| CPU KD-tree | `~20 ms` | same algorithm, serial |
| **GPU KD-tree** | **`~0.11 ms`** | one thread / query |
| tree build (host, once) | `~21 ms` | `40000` nodes |

Speedups: **~10500× vs brute force**, **~175× vs the CPU KD-tree** (the
apples-to-apples parallelism win — same `O(log N)` traversal per query, just
40000 of them at once).

## Reproduce

```bash
cd build && cmake .. && make gpu_kdtree_nn -j$(nproc)
cd .. && ./bin/gpu_kdtree_nn
```

Prints the build/query timings + exact-match table and writes
`gif/gpu_kdtree_nn.gif` (the KD-tree partition over the point cloud, with a
sweeping query and the exact nearest neighbour it finds each frame).

## Notes

- One demo = one `.cu`; reuses `include/cuda_check.cuh` and `include/cuda_video.h`.
- The CPU and GPU share one `__host__ __device__ nn_search` with an explicit
  64-deep stack, so both run identical traversal logic.
- The tree is built on the host (the build is recursive / serial); the
  demonstrated GPU win is the query phase, which dominates when many points are
  queried against a static index — exactly the point-cloud-registration case.
- The drawn points are subsampled so the partition is visible; the benchmark
  and queries use all `N_PTS` points.
- `--fmad=false`; GIF served from gh-pages (not committed to the repo).
