# Exact incremental KISS-ICP map normals

The shared odometry core can reuse map normals when their ordered neighbour
support is unchanged. It keeps scan/map density, voxel resolutions, map radius,
normal neighbour count, correspondence gates and ICP limits unchanged.
[Full-route results](results/kiss_icp_normals_2026-10-10.md) compare full and
incremental updates using one frozen executable, with separate bitwise
validation of every normal.

## Why reuse is exact

The dense map retains the first point inserted into each voxel. A retained
point's coordinates do not change. Each dense slot carries the point's rank
from the preceding normal update. Eviction moves this metadata with the point;
new points receive no predecessor. GPU map ranks can change without changing
point coordinates.

For each computed normal the cache stores its ordered K neighbours and the
squared distance to the Kth neighbour. It requests K+1 neighbours to detect
equal-distance ties inside the support and at its boundary. A tied or incomplete
support always recomputes, because changing point IDs could alter support or
PCA summation order.

A retained normal is copied only if all selected neighbours survive and no new
point has distance less than or equal to its previous support radius. An exact
spatial index over newly retained points answers that presence query, with
conservative cell bounds. Searches spanning more than four cells in any
direction recompute instead. The delta index uses cells eight times the normal
index's cell size; this affects invalidation cost, not which points are accepted.

When these checks pass, neighbour IDs are remapped in the previous distance
order and the normal's three floats are copied. Otherwise the existing exact
spatial KNN and PCA run. Removed points outside the old support cannot change
the normal; newly added distant points cannot enter the support. Reset clears
predecessor metadata. A block reduction counts reused points with one integer
atomic per block.

## Modes and counters

`KissIcpConfig::normal_update` defaults to `Full`. `Incremental` enables the
cache for the dense map and voxel normal backend; other map/normal backends
fall back to full recomputation. `Validate` requires dense/voxel execution
and recomputes every normal into a separate buffer, comparing all float bits
each frame; any mismatch throws. Its whole-frame timing includes that extra
work and must not be used as incremental performance.

The native stack runner accepts `--kiss-normal-update full|incremental|validate`.
It also defaults to `full`. The measurements reduced GPU normal work but did
not consistently improve full-frame p95, so incremental execution remains
opt-in. The 40 ms paired target was missed in three of five timing pairs.
Its JSON records the requested mode; unsupported incremental combinations
still execute the full path. The CSV appends cache preparation duration,
reused-point count and recomputed-point count. Initial map creation computes no
normals, so its counters are zero. Cumulative public counters reset with the
odometry object.

`normal_ms` uses GPU events around invalidation, reused copies, recomputed
KNN/PCA and the block counter. `normal_cache_prepare_ms` measures host remapping,
transfers, the added-point index and host cache commit. Reading the four-byte
reuse counter is outside those two stage timers and inside whole-frame timing.
Map prune/insert timing includes metadata maintenance. Normal/index timings
are inside odometry wall time; do not add them to that parent duration.

At the runner's 200,000-point capacity, K=12 and 524,288 hash slots, incremental
execution adds 38.23 MiB of fixed GPU storage plus CUB scan scratch, and about
3.05 MiB of reserved host integer storage. Validation adds another 2.29 MiB of
GPU scratch. The map, index and upload otherwise remain as before. These
allocations happen at construction, not per normal query. Tie-heavy clouds,
large support radii, or rapidly changing maps can reuse less; the full mode
avoids cache storage and maintenance in those workloads.

## Reproduction and checks

```bash
cmake --build build --target cudanav_real_gpu_stack_sequence \
  kiss_icp_normal_cache_exact kiss_icp_gpu_streaming_smoke \
  kiss_icp_spatial_exact kiss_icp_reduction_exact \
  kiss_icp_host_map_exact kiss_icp_downsample_exact gpu_kiss_icp
ctest --test-dir build --output-on-failure -j 1 -R \
  'kiss_icp_normal_cache_exact|kiss_icp_gpu_streaming_smoke|kiss_icp_spatial_exact|kiss_icp_reduction_exact|kiss_icp_host_map_exact|kiss_icp_downsample_exact|gpu_kiss_icp_gate'
ctest --test-dir build --label-regex 'cpu|python' --output-on-failure -j 2
python scripts/benchmark_kiss_icp_normals.py \
  --sequence build/cudanav_real_gpu_stack_release_724d05ca/sequence.bin \
  --out-dir build/kiss_icp_normal_comparison --repeats 2 --plot
```

On Windows add `--config Release` to build and `-C Release` to CTest; the
benchmark accepts `--dll-dir`. Dataset preparation is described in the
[timed MCD guide](cudanav_timed_dataset_mcd.md). Input data is external and not
redistributed. `--maximum-frames` selects a development prefix.

For the supplementary Windows measurement, use `--above-normal-priority` and
`--repeats 3` with a new output directory. This applies AboveNormal priority
to both modes; it does not change the product's default process priority.

The benchmark refuses to overwrite its output directory, freezes the executable
and source snapshots, and hashes sources with normalized LF, plus raw input and
executable bytes. It runs full-route validation first, then full/incremental
twice in alternating order, retaining all CSV, JSON, logs and failures. Paired
checks require frame p95 below 40 ms, improvement over full updates, equal ATE
and drift, and matching recorded XY poses, inliers, map point counts and mapping
counts at every frame. Whole-route validation checks normal bits independently
of CSV precision.

Occupied-cell differences are recorded separately. Existing clamped atomic
occupancy updates can vary even between two full-mode executions; native
mapping/controller quality gates still apply. This benchmark gates equality
of the odometry outputs, observed voxels, integrated rays and unknown cells.

The cache test compares exact poses/alignment statistics against full updates
over stationary and moving random clouds, new near/far points, equal-distance
grids and reset, with K=1,12,20. Sparse maps with fewer than K neighbours do not
reuse. Its invalidation test compares against exhaustive GPU distance checks
at positive/negative cell boundaries, inclusive radius boundaries, large radii
and partial blocks. The host-map test verifies predecessor metadata survives
swap eviction, insertion and reset.

Frame time covers odometry, rolling voxel mapping, occupancy projection, ESDF
and MPPI every tenth scan. Loading, output writes and construction are excluded.
These are native shadow evaluations; commands are not applied. FIFO metrics
are calculated serial lossless replay, not observed ROS scheduling or
sensor-to-command latency. Timing repeats on one route are evidence for
this workload, not a guarantee across datasets or hardware. A p95 target does
not bound every frame.
