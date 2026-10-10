# Exact spatial queries for streaming LiDAR odometry

The shared GPU KISS-ICP core now accelerates the work done on every incoming
LiDAR scan. It keeps the input points, map radius, map voxel resolution,
correspondence gate, normal neighbour count and ICP iteration limit unchanged.
The CudaNav ROS component uses this same core; the measurements below are
native shadow execution, not measured ROS scheduling or vehicle control.

The [2026-10-10 paired results](results/kiss_icp_spatial_2026-10-10.md)
retain all six full-route replays, quality gates, timings and provenance.

## Implementation

The previous normal estimator compared every map point with every other map
point each frame. The previous voxel correspondence index linked points into
lists and visited all cells in the correspondence gate's bounding cube.

The new index counts points per hashed cell, prefix-sums the counts with CUB,
and scatters point indices into contiguous buckets. Normal estimation searches
successive cell shells, pruning cells whose point-to-box distance exceeds the
current kth distance. It stops only when every point outside the searched cube
is farther than the kth neighbour. Sparse queries retry with cells eight times
larger before falling back to exhaustive search. A conservative floating-point
margin affects pruning only; point distances and neighbours remain unchanged.
This fallback means worst-case normal work can still be quadratic.

Correspondence queries use the same fine index, pruning against the current
nearest distance and retaining the original strict distance gate. Equal-distance
ties use the lower point index; the old linked backend had scheduling-dependent
ties. Neither trajectories nor normal-equation atomic reductions are claimed
to be bit-identical across runs.

The default cell size is `max(0.5 m, 3 * map_voxel_size)`. This is an index
parameter, not point-cloud downsampling. GPU indices persist in the odometry
object and are rebuilt per frame. Host downsampling, host map management, full
map upload, and host pose updates remain; this is not a GPU-resident pipeline.
The two new indices add roughly 23.1 MiB of capacity storage at the default
200,000 map-point / 524,288 hash-slot capacities, plus CUB scratch storage.

## Reproduce

```bash
cmake --build build --target cudanav_real_gpu_stack_sequence \
  kiss_icp_spatial_exact kiss_icp_gpu_streaming_smoke gpu_kiss_icp
python scripts/benchmark_kiss_icp_spatial.py \
  --sequence build/cudanav_real_gpu_stack_release_724d05ca/sequence.bin \
  --out-dir build/kiss_spatial_comparison --repeats 2 --plot
```

Use `--config Release` when building with a multi-config generator on Windows.
The sequence is the existing timed MCD NTU day-02 export: 1,190 scans over
118.902 seconds, with 128-channel point timestamps and deskew enabled.
See [dataset preparation](cudanav_timed_dataset_mcd.md) for how to obtain and
export the external dataset. Input data is not redistributed.

The script freezes a copy of the executable, refuses to overwrite an output
directory, and runs three modes sequentially, reversing their order on alternate
repeats. `legacy` uses brute-force normals and linked correspondences;
`normals` changes only normal estimation; `spatial` changes both. It retains
source/executable/input hashes, commands, all per-frame CSVs, failed quality
gates and logs. `--maximum-frames 200` is a development prefix, not the full
quality workload. Timing runs should have no concurrent tests or GPU workloads.

Frame timing includes odometry, rolling ray mapping, occupancy projection,
ESDF and the existing MPPI evaluation every tenth scan. Initial file loading,
CSV output and object construction are excluded. Stage timers are GPU event
durations; frame/odometry timers are wall-clock durations. The script also
calculates lossless serial FIFO response times from measured compute durations
and original scan timestamps. Those queue metrics are a replay model, not an
observed ROS queue, and do not include decoding, transport or sensor latency.

## Verification

```bash
ctest --test-dir build -R \
  'kiss_icp_spatial_exact|kiss_icp_gpu_streaming_smoke|gpu_kiss_icp_gate|check_kiss_icp_spatial_report' \
  --output-on-failure -j 1
```

Spatial queries are checked against exhaustive queries for negative cells,
cell boundaries, ties, duplicate points, sparse isolated points, fewer-than-k
clouds and partial blocks. Both coarse-index and exhaustive fallback paths
are exercised. The existing streaming and synthetic odometry gates remain.
Queue arithmetic is checked for accumulation and recovery after a slow frame.
