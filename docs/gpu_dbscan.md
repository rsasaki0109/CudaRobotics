# GPU DBSCAN Point-Cloud Clustering

`gpu_dbscan` adds density-based clustering to the repo's PointCloud section
(voxel / statistical filtering, normal estimation, GICP, RANSAC, label
propagation, GICP-MCL) — clustering was the one obvious gap. DBSCAN finds
clusters as maximal sets of points reachable through chains of points that
each have at least `min_pts` neighbours inside an `eps` ball, and labels
everything else as noise.

The algorithm has a natural parallel structure that matches the repo's
canonical idiom: **one thread = one point**.

## Pipeline

The CPU and GPU paths share the same four-step pipeline:

1. **Neighbour count** — for each point `i`, count `|{j : dist(i,j) < eps}|`.
2. **Core mark** — `core[i] = (n_neigh[i] >= min_pts)`.
3. **Label propagation** — iteratively pull each core point's label down to
   the smallest label among its core neighbours, until no label changes.
   GPU uses `atomicMin`; CPU uses synchronous Jacobi over a label snapshot.
4. **Border assignment** — non-core points within `eps` of any core inherit
   the smallest neighbouring core label; otherwise they remain noise.

We deliberately use **brute-force pairwise neighbour search** in both paths so
the CPU and GPU run identical arithmetic — the only difference is the parallel
layout. At `N = 8192` this is ~67 M pair checks per pass, big enough for the
GPU's win to be clearly visible and small enough to keep the CPU reference
runtime reasonable.

## Setup

- `N = 8192` 2D points in `[0, 30] × [0, 30]`, generated as 6 Gaussian blobs
  + ~10 % uniform background noise, then shuffled.
- `eps = 0.55`, `min_pts = 8`.

## Correctness

| metric | CPU | GPU |
|---|---|---|
| clusters | 4 | 4 |
| noise points | 647 | 647 |
| cluster agreement (CPU label → most-common GPU label) | — | **100.0 %** |

The CPU and GPU produce the **same partition** of the point set into clusters
and noise. (The propagation iteration counts differ — `16` on CPU vs `12` on
GPU — because GPU `atomicMin` lets a thread see within-sweep updates, which is
Gauss-Seidel-like and converges a little faster than the CPU's synchronous
Jacobi sweep. The fixed point is identical.)

## Result (this machine)

| | points | time | note |
|---|---|---|---|
| CPU serial DBSCAN | 8,192 | 1,597 ms | 16 propagation iterations |
| **GPU DBSCAN** | **8,192** | **10.4 ms** | 12 propagation iterations, **~153×** |

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_dbscan -j$(nproc)
./bin/gpu_dbscan
```

Generated files:

- `tmp/gpu_dbscan.avi`
- `gif/gpu_dbscan.gif`

## Output

The GIF walks the pipeline: raw input → core mark → label-propagation iters
(clusters fan out and merge across the point cloud) → border assignment +
noise. The info panel reports the running cluster count and the CPU-vs-GPU
headline (time, iterations, speedup, and the final cluster agreement).
