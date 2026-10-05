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
- **Objects:** cars (one on the ramp), a van, a wall, a crate, a bench, poles and pedestrians. The boxes stand at headings from -15° to 35°, so that the box fitting below has headings to find.
- **Scans:** eight scans from sensor poses on the flat part, on the ramp and on the sidewalk side.

## Results (8 scans, 482643 returns)

| Method | precision | recall | F1 |
|---|---:|---:|---:|
| concentric-zone model | 0.983 | 0.982 | **0.982** |
| height threshold (0.25 m above flat ground under the sensor) | 0.970 | 0.735 | 0.836 |

- **The height threshold collapses once the sensor faces the ramp.** Per-scan F1 falls from 0.94 on flat ground to 0.62-0.66 at x = 16-20 m, because the rising ground is labelled non-ground.
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
| concentric-zone model | **109 / 116** | 14 | 107 | 269 |
| height threshold | 87 / 116 | 11 | 361 | 481 |
| none | 32 / 116 | 12 | 869 | 938 |

- **Ground removal decides the clustering.** Without it, the ground connects or swamps most objects.
- **With the height threshold,** the ramp's ground survives as hundreds of spurious clusters, and objects standing on it merge with that ground.
- **With the model,** 94% of the objects come out as their own cluster.

**Voxel clustering, same partition.** Points in one voxel of side `0.5 m / sqrt(3)` are always within 0.5 m of each other, so a voxel can be a node. Two voxels are connected if any pair of their points is within 0.5 m.

| Step | What it does |
|---|---|
| graph | about 1550 occupied voxels per scan instead of about 50k points |
| unite | one thread per (voxel, neighbour offset): 62 offsets within 2 voxels per axis, each pair once |
| skip | pairs already in one set, and pairs whose voxels' point bounds are more than 0.5 m apart |
| test | point pairs otherwise, stopping at the first one within 0.5 m |
| label | each component takes its smallest point index |

The result is the same partition and the same labels as the CPU's BFS on every scan.

**Time per scan** (GPU: minimum of 5 runs, on a GPU shared with other work):

| Clusterer | time | same partition as the CPU |
|---|---:|---|
| CPU BFS over the points | 66-89 ms | — |
| GPU point-level union-find | 8.3-8.5 ms | yes |
| GPU voxel-level union-find | **3.8-4.2 ms** | yes |

- **Point-level:** near the sensor a point has thousands of neighbours within 0.5 m, and every one is tested.
- **Voxel-level:** collapses those neighbourhoods and stops at the first close pair, about twice as fast as point-level and about 20x faster than the CPU.
- **What did not help:** sorting the points instead of scanning a dense grid, and union-find instead of iterated label propagation, left the point-level clusterer unchanged. One thread per voxel instead of per (voxel, offset) was slower than point-level (too few threads).

## Oriented boxes: L-shape fitting

The third stage fits an oriented box to every cluster of the model's ground
removal, by the search-based L-shape fitting of Zhang et al. (2017) with the
closeness criterion.

- **Search.** The cluster's points are projected on the axes of each of 90 headings in [0°, 90°) (a rectangle repeats every 90°).
- **Score.** Each point's distance to the nearer edge of the bounding rectangle in that frame is clamped below at 1 cm, and the heading scores the sum of the inverse distances. Points hugging two perpendicular edges, the L a LiDAR sees of a car, score high.
- **Box.** The best heading's bounding rectangle is the box.
- **GPU: one warp per (cluster, heading).** Lane *l* takes points *l*, *l* + 32, …, and the partial sums are combined by an xor butterfly. The clusters are formed on the GPU (a stable sort of the voxel clusterer's labels and `reduce_by_key`), then one thread per cluster picks the best heading.
- **CPU: the same arithmetic.** The CPU runs the same 32 lanes and butterfly with the same heading table, and its boxes are bit-identical to the GPU's on every scan.

**Scoring.** Each ground-truth box whose cluster is found (as above) is one
observation, 50 over the 8 scans. The fit is compared with the axis-aligned
box of the same points:

| Box | heading error | BEV IoU | IoU ≥ 0.5 | centre error | long / short side error |
|---|---:|---:|---:|---:|---:|
| L-shape | **1.3°** | **0.47** | **26 / 50** | 1.11 m | 1.42 / 0.70 m |
| axis-aligned | 15.2° | 0.40 | 19 / 50 | 1.19 m | 1.48 / 0.89 m |

| Box (L-shape vs axis-aligned) | observations | heading error | BEV IoU |
|---|---:|---:|---:|
| car, 20° | 7 | 3.4° vs 20.0° | 0.55 vs 0.39 |
| car on the ramp, -15° | 8 | 0.5° vs 15.0° | 0.61 vs 0.46 |
| car, 35° | 8 | 0.9° vs 35.0° | 0.52 vs 0.31 |
| van on the ramp, 10° | 7 | 0.3° vs 10.0° | 0.44 vs 0.38 |
| crate, 30° | 5 | 0.2° vs 30.0° | 0.14 vs 0.17 |
| wall, 0° | 7 | 2.0° vs 0.0° | 0.19 vs 0.29 |
| bench, 0° | 8 | 1.4° vs 0.0° | 0.70 vs 0.70 |

- **The heading is what L-shape fitting buys.** The heading error drops from 15° to about 1°, and on the rotated cars the IoU rises by 0.15-0.2.
- **The extent is limited by what the scan sees.** The box covers only the visible surface: the far sides of a car, the far end of the 12 m wall, and most of the distant low crate are hidden, so the centre and side errors stay near 1 m for both boxes. The size prior below fills in the hidden part.
- **Faces visible.** With one face visible (11 observations) the heading error is 0.9°, and with two (39) it is 1.4°: the closeness criterion also aligns a single wall-like face.
- **Where axis-aligned wins.** On the 0° wall the axis-aligned box is exact by construction, while the L-shape fits are off by 1-2° on it and on the bench.

**Time per scan** (about 33 fits; GPU minimum of 5 runs, measured while the GPU was 40-55% busy with other work):

| Fitter | time |
|---|---:|
| CPU | 54-88 ms |
| GPU, one thread per (cluster, heading) | 7.1 ms |
| GPU, one warp per (cluster, heading) | **1.6-2.1 ms** |

One thread per heading left a few threads looping over the thousands of points
of the largest clusters. A warp per heading spreads them over 32 lanes.

## Completing the boxes with a size prior

The L-shape box covers only what the scan sees. A class size prior fills in
the rest (host side, one step per cluster):

- **Class.** The cluster's height and footprint stand in for a classifier:
  - a *car* (4.5 × 1.8 m) is 1.0-2.0 m tall;
  - a *van* (6.0 × 2.0 m) is 2.0-3.2 m tall and at least 0.8 m wide, which keeps thin walls out;
  - in both cases the footprint must be at least 1.2 m long and within 1.2× the prior.
- **Axes.** The longer observed side is the length, unless neither side exceeds the class width (× 1.2). Then the sensor sees one end, and the length runs along the axis closer to the line of sight.
- **Growth.** A side shorter than the prior grows away from the sensor, keeping the edge the sensor sees. A side the sensor stands across grows about its centre.

| Box | heading error | BEV IoU | IoU ≥ 0.5 | centre error | long / short side error |
|---|---:|---:|---:|---:|---:|
| L-shape | 1.3° | 0.47 | 26 / 50 | 1.11 m | 1.42 / 0.70 m |
| L-shape + size prior | 1.3° | **0.60** | **34 / 50** | **0.82 m** | 0.85 / 0.49 m |
| prior 10% too small | 1.3° | 0.58 | 33 / 50 | 0.87 m | 0.98 / 0.52 m |
| prior 10% too large | 1.3° | 0.57 | 34 / 50 | 0.83 m | 0.97 / 0.54 m |

| Box | L-shape IoU | + size prior |
|---|---:|---:|
| car, 20° | 0.55 | 0.70 |
| car on the ramp, -15° | 0.61 | 0.82 |
| car, 35° | 0.52 | **0.92** |
| van on the ramp, 10° | 0.44 | 0.51 |
| one face visible (11 observations) | 0.41 | 0.68 (centre error 0.85 → 0.29 m) |
| two faces visible (39 observations) | 0.49 | 0.58 |

- **Best case.** The scene's cars match the car prior exactly. With a prior 10% off in both sides, the mean IoU still rises from 0.47 to 0.57-0.58.
- **The end-view rule matters.** Taking the longer observed side as the length every time gives a mean IoU of 0.53. A car seen end-on then grows sideways.
- **The height-based class is the weak link.**
  - 17 of the 23 car observations are classed as cars.
  - Five misses have an observed footprint wider than 1.2× the prior, and one car is mostly hidden (0.5 m of it visible).
  - The van is never classed as a van. Its roof is above the scan's +2° upper beam, so only 1.1-2.1 m of it is seen. It gets the car prior 4 times out of 7, which still helps it, but it stays short of its true length.
  - The wall, crate and bench get no class and keep their L-shape boxes.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_ground_segmentation -j$(nproc)
./bin/gpu_ground_segmentation                       # also writes the GIF
./bin/gpu_ground_segmentation --check --no-video    # the CTest gate
```

`--check` exits non-zero unless the model's F1 is at least 0.95, above the
height threshold's, CPU and GPU agree on at least 99.9% of the labels, at least
90% of the objects come out as one cluster, the CPU and GPU clusterings are
the same partition, the L-shape boxes beat the axis-aligned ones in mean heading
error and mean IoU, the CPU and GPU boxes are identical, and the size prior
raises the mean IoU and lowers the mean centre error of the L-shape boxes. CTest
runs it as `gpu_ground_segmentation_gate` (labels `gpu;pointcloud;ground`).

Generated files: `tmp/gpu_ground_segmentation.avi` and `gif/gpu_ground_segmentation.gif`.

## Output

The GIF shows a bird's-eye view (40 m × 40 m) of each scan: the height threshold on the left, the model on the right. The right panel also shows the true boxes (blue), the L-shape boxes of the clusters (magenta) and, for clusters classed as cars or vans, the boxes completed with the size prior (orange).

| Colour | Meaning |
|---|---|
| green | ground, labelled ground |
| white | object, labelled object |
| red | object labelled ground |
| yellow | ground missed |
