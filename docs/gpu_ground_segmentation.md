# GPU LiDAR Ground Segmentation

`gpu_ground_segmentation` separates ground from non-ground points in a LiDAR
scan on the GPU. It is the first step of most LiDAR pipelines (obstacle
extraction, clustering, mapping) and complements the repo's KISS-ICP, voxel
mapping and DBSCAN demos.

A single height threshold fails as soon as the ground is not flat: a ramp rises
above it and a curb splits it. The demo implements a concentric-zone ground model
in the spirit of Patchwork.

## Method

1. **Bin.** Each point gets a polar bin (24 rings with geometrically growing width, from 2 m to 60 m, × 72 sectors). Points are stably sorted by bin: thrust on the GPU, `std::stable_sort` on the CPU, giving the same order.
2. **Fit, one warp per bin.**
   - Seeds are the points within 0.25 m of the bin's lowest point.
   - A plane is fitted by PCA (the smallest eigenvector of the covariance, by Jacobi iterations) and refitted twice on the points within 0.12 m of it.
   - The plane counts if it is within 25° of horizontal.
   - The 32 lanes share the bin's points. Near bins hold hundreds to thousands of points, so one thread per bin left a few threads doing most of the work (GPU 1.8 → 0.8-1.2 ms per scan).
3. **Check, one thread per sector.** The sector's rings are walked outward. A plane is kept only if its height at the bin centre is within 0.25 m + 0.18 × (distance from the last kept plane) of that plane, or of the flat ground under the sensor for the first one. This stops a flat car roof from passing for ground.
4. **Label.** A point is ground if its bin's plane is kept and the point lies within 0.12 m of it.

The check and the labelling are `__host__ __device__` routines shared with the
CPU reference. The fit is the same algorithm, with the GPU's sums taken
lane-strided.

## Scene

A synthetic 64 × 1024 LiDAR (-24.8° to +2° elevation, 1.8 m mounting height, 2 cm range noise), ray-cast per beam with exact ground-truth labels.

- **Terrain:** a 6° ramp beyond x = 10 m, a 0.15 m curb and sidewalk beyond y = 6 m, and gentle undulation.
- **Objects:** cars (one on the ramp), a van, a wall, a crate, a bench, poles and pedestrians.
- **Scans:** eight scans from sensor poses on the flat part, on the ramp and on the sidewalk side.

## Results (8 scans, 482695 returns)

| Method | precision | recall | F1 |
|---|---:|---:|---:|
| concentric-zone model | 0.983 | 0.981 | **0.982** |
| height threshold (0.25 m above flat ground under the sensor) | 0.967 | 0.727 | 0.830 |

- **The height threshold collapses once the sensor faces the ramp.** Per-scan F1 falls from 0.94 on flat ground to 0.61-0.66 at x = 16-20 m, because the rising ground is labelled non-ground.
- **The model stays at F1 0.97-0.99 on every scan.**
- **CPU and GPU agree** on 100% of the labels.
- **Time per scan:** CPU 7-13 ms, GPU 0.8-1.2 ms (about 9-16x). The GPU time is dominated by the sort and the kernel launches, which is ample for a 10 Hz LiDAR. The ranges come from a GPU shared with other work.

## Object clustering after ground removal

The second stage clusters the remaining points into objects: Euclidean clustering with 0.5 m connectivity, dropping clusters under 10 points (the same idea as PCL's `EuclideanClusterExtraction`).

**GPU implementation.**
- The points are sorted by 0.5 m grid cell, and neighbouring cells are found by binary search.
- Each point unites itself with every lower-indexed neighbour in a lock-free union-find. The union-find always hooks the larger root under the smaller one (`atomicCAS`), so each component's root is its smallest point index.
- The CPU reference runs a BFS over the same grid and assigns the same labels.
- The two partitions are compared exactly, and they are identical on every scan.

**Scoring.** Clusters are scored against the ground-truth object of every return (an object counts if the scan sees it with at least 20 points). An object is *found* if one cluster holds at least half of the object's points and at least half of that cluster's points belong to the object.

| Ground removal before clustering | objects found | split objects | clusters mostly of ground | clusters |
|---|---:|---:|---:|---:|
| concentric-zone model | **110 / 115** | 15 | 108 | 262 |
| height threshold | 86 / 115 | 12 | 369 | 481 |
| none | 31 / 115 | 13 | 876 | 938 |

- **Ground removal decides the clustering.** Without it, the ground connects or swamps most objects.
- **With the height threshold,** the ramp's ground survives as hundreds of spurious clusters, and objects standing on it merge with that ground.
- **With the model,** 96% of the objects come out as their own cluster.

**Time per scan:** clustering takes 70-110 ms on the CPU and 13-24 ms on the GPU.
- The GPU time is the unite kernel. Near the sensor a point has thousands of neighbours within 0.5 m, and exact point-level clustering has to test them.
- Sorting the points instead of scanning a dense 1.4 M-cell grid, and union-find instead of iterated label propagation, did not change it.
- Voxel-downsampling before clustering would, at the cost of no longer matching the CPU partition exactly.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_ground_segmentation -j$(nproc)
./bin/gpu_ground_segmentation                       # also writes the GIF
./bin/gpu_ground_segmentation --check --no-video    # the CTest gate
```

`--check` exits non-zero unless the model's F1 is at least 0.95, above the
height threshold's, CPU and GPU agree on at least 99.9% of the labels, at least
90% of the objects come out as one cluster, and the CPU and GPU clusterings are
the same partition. CTest
runs it as `gpu_ground_segmentation_gate` (labels `gpu;pointcloud;ground`).

Generated files: `tmp/gpu_ground_segmentation.avi` and `gif/gpu_ground_segmentation.gif`.

## Output

The GIF shows a bird's-eye view (40 m × 40 m) of each scan: the height threshold on the left, the model on the right.

| Colour | Meaning |
|---|---|
| green | ground, labelled ground |
| white | object, labelled object |
| red | object labelled ground |
| yellow | ground missed |
