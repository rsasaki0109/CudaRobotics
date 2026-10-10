# KISS-ICP pipeline latency improvements

The shared GPU odometry core reduces work on every incoming scan, without
changing point density, scan/map voxel resolutions, map radius, neighbour count,
correspondence gates or ICP iteration limits. The
[paired real-scan results](results/kiss_icp_pipeline_2026-10-10.md) retain every
method/repeat, accuracy gates and source/input/executable hashes.

## Changes

Normal-equation assembly previously added 30 terms per correspondence into the
same global addresses using CUDA atomics. The default now sums those same terms
within 256-thread blocks using CUB, writes one partial per block, and combines
partials in a fixed block order. The robust point-to-plane residual, weights,
6x6 solve and convergence criteria are unchanged. The sum is floating point;
different addition order can change the estimated trajectory. The reduction
test compares all terms with an independent double-precision CPU calculation,
including rejected correspondences, empty inputs and partial blocks.

The default host map stores points in a persistent contiguous vector and maps
voxel keys to vector slots. Radius eviction scans that vector, moving the last
point into an evicted slot; new voxels append one point. It preserves the
reference map's first retained point per voxel and exact radius/capacity policy
for the same world-point inputs. It exports integer slots in reference container
order and gathers coordinates on the GPU. The GPU sees the same point indices,
preserving tie choices and normal calculations; the CPU avoids packing three
coordinates per point from unordered-map nodes. Map tests compare exact point
membership, representatives and exported order across moving centres, eviction,
insertion, radius boundaries, capacity overflow and reset. The streaming test
checks bit-identical dense/reference poses and alignment values over 20 frames
with block reduction. GPU gathering is also checked byte-for-byte.

Scan centroid aggregation keeps the same unordered container, input-order sums
and output order. Adjacent points with the same voxel key reuse the previous
value pointer. CPU tests compare the complete output byte-for-byte, including
negative cells, boundaries, repeated points and 200,000-point inputs.

Normal queries default to cell order using the existing spatial index's point
permutation. Nearby queries run together without allocating or sorting another
buffer. Only thread scheduling changes: map point IDs, tie choices, neighbour
order, PCA arithmetic and output positions remain the same. Exact query tests
compare both schedules, including sparse fallback and equal-distance ties;
the streaming test also compares cell/dense against input/unordered execution.

The default reduction scratch buffer is 91.6 KiB at a 200,000-point scan capacity.
The dense host points/order reserve 3.05 MiB at a 200,000-point map capacity,
with an additional 3.05 MiB of GPU gathering scratch. Host hash nodes/buckets
remain. This is not a GPU-resident map: host scan
aggregation, host map management and full map upload remain. Earlier profiling
measured the upload at about 0.15 ms per frame, so it was not the first target.

## Measurement and reproduction

```bash
cmake --build build --target cudanav_real_gpu_stack_sequence \
  kiss_icp_reduction_exact kiss_icp_host_map_exact kiss_icp_downsample_exact \
  kiss_icp_gpu_streaming_smoke gpu_kiss_icp
python scripts/benchmark_kiss_icp_pipeline.py \
  --sequence build/cudanav_real_gpu_stack_release_724d05ca/sequence.bin \
  --out-dir build/kiss_icp_pipeline_comparison --repeats 2 --plot
```

Add `--config Release` to the build for a multi-config generator on Windows.
The benchmark accepts `--dll-dir` for runtime DLLs. Dataset preparation is in
[the timed MCD guide](cudanav_timed_dataset_mcd.md); input data is external and
not redistributed. `--maximum-frames` is a development prefix, not the full
quality workload.

All four modes use exact voxel normal/correspondence queries and full map-normal
recomputation, explicitly pinned by the benchmark. The shared core now
also supports [exact incremental normals](kiss_icp_incremental_normals.md).
For this pipeline comparison, `baseline` uses
per-point atomic reduction, the unordered host map and uncached centroids;
`reduction` changes only assembly; `map` also changes map storage; `optimized`
also caches adjacent scan voxels and schedules normal queries by cell. The
script freezes one executable and runs
all modes sequentially, reversing order on the second repeat. It preserves raw
CSV/JSON/logs and failures, refusing to overwrite the output directory. Paired
checks require optimized frame p95 below 75 ms, ATE within 2 cm of its baseline,
and drift within 0.02 percentage points.

Per-frame CSVs add wall-clock validation, deskew, scan aggregation, map upload,
ICP and map-update durations. Prune/insert/pack are subdivisions of map-update;
normal-equation GPU time is within ICP wall time. Dense pack timing measures
integer order export; GPU reorder is within map-upload wall time. Normal/index/
NN/reorder timers use GPU events. Do not add nested durations to their parents.
The cumulative public `KissIcpTiming` resets with the odometry object.

Frame timing covers odometry, rolling mapping, occupancy projection, ESDF and
MPPI every tenth scan. Input loading, CSV writes and construction are excluded.
These are native shadow evaluations, with commands not applied. FIFO metrics
are calculated lossless serial replay from sensor timestamps and compute
durations, not observed ROS scheduling or sensor-to-command latency. A p95
target does not bound every frame.

## Verification

```bash
ctest --test-dir build -R \
  'kiss_icp_reduction_exact|kiss_icp_host_map_exact|kiss_icp_downsample_exact|kiss_icp_spatial_exact|kiss_icp_gpu_streaming_smoke|gpu_kiss_icp_gate' \
  --output-on-failure -j 1
ctest --test-dir build --label-regex 'cpu|python' --output-on-failure -j 2
```

Reference backends remain selectable through `KissIcpConfig` and the native
stack runner: `--kiss-reduction-backend atomic`, `--kiss-map-backend unordered`
and `--kiss-downsample-backend unordered`, plus `--kiss-normal-query-order input`.
The older spatial-only benchmark pins
these references so it still isolates normal/correspondence search changes.
