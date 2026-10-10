# Exact pooled KISS-ICP scan centroids

Scan voxel aggregation reuses backing memory for hash nodes, buckets and the
centroid output vector. The hash container is rebuilt each scan using the same
hasher, reserve request and insertion order. Input point density, voxel keys,
adjacent-voxel caching, input-order sums, centroid arithmetic and output order
match the cached reference. [Real-scan results](results/kiss_icp_downsample_2026-10-10.md)
include full-route quality gates, bytewise validation and all timing repeats.

## Allocation and lifetime

`kiss_downsample::Sampler` owns an arena of retained slabs and a reserved output
vector. Slabs are rounded to multiples of 1 MiB and align each allocation for its
rebound allocator type. The allocator routes unordered-map node/bucket requests
into those slabs. Container destruction still destroys its elements; individual
deallocations leave the slab memory in place. The next call resets the arena only
after the previous container has been destroyed, and constructs fresh map values
so no old sums survive. The sampler is not copyable.

Reusing the same container across frames could preserve a different bucket count
and change iteration order. Here each frame creates a new container and calls
the original `reserve(input_points)` before insertion. The allocator owns memory,
not a different hash algorithm. Order equivalence is relative to the same C++
standard-library implementation; it is not a cross-library ordering guarantee.

The output vector reserves three floats per configured maximum scan point.
At a capacity of 200,000 points this is 2.29 MiB. Arena storage grows on demand
and is retained for the odometry object's lifetime, including reset. Its size
depends on input cardinality and the standard library's node/bucket layout.
Allocation failure still throws. There is no additional GPU storage. The
reported arena size excludes the output vector and small slab metadata.

## Modes and counters

`KissIcpDownsampleBackend::Pooled` uses the arena and persistent output vector.
`Cached` retains the preceding cached lookup implementation with ordinary
allocation. `Unordered` additionally disables adjacent-voxel lookup caching.
`Validate` runs pooled and cached aggregation on every deskewed scan, checking
the entire output byte-for-byte and throwing if values, cardinality or order
differ. Validation timing includes both implementations and comparison work.

The native stack accepts `--kiss-downsample-backend` with
`pooled|cached|unordered|validate`. Pooled execution is the default after all
three paired full-route comparisons passed. All comparisons in that report
pin full map-normal updates. The subsequent
[default-policy comparison](results/kiss_icp_normal_default_2026-10-10.md)
combines pooled centroids with incremental normals, now the default.

`KissIcpTiming::downsample_upstream_allocations` counts new arena slabs since
reset. `downsample_arena_bytes` is the currently retained slab capacity reported
after sampling; reset clears timing counters while retaining storage. The CSV
adds per-frame new-slab counts and retained capacity, and JSON adds
`downsample_memory`. These counters count slabs, not every C++ allocation:
slab metadata/vector growth and initial output reservation also allocate.
Once slabs/output have sufficient capacity, sampling needs no new backing
allocations. Core scan validation, deskew buffers and map operations retain
their existing allocations.

## Measurement and reproduction

```bash
cmake --build build --target kiss_icp_downsample_benchmark \
  kiss_icp_downsample_exact cudanav_real_gpu_stack_sequence \
  kiss_icp_gpu_streaming_smoke kiss_icp_normal_cache_exact gpu_kiss_icp
ctest --test-dir build --output-on-failure -j 1 -R \
  'kiss_icp_downsample_exact|kiss_icp_gpu_streaming_smoke|kiss_icp_normal_cache_exact|kiss_icp_spatial_exact|kiss_icp_reduction_exact|kiss_icp_host_map_exact|gpu_kiss_icp_gate'
ctest --test-dir build --label-regex 'cpu|python' --output-on-failure -j 2
python scripts/benchmark_kiss_icp_downsample.py \
  --sequence build/cudanav_real_gpu_stack_release_724d05ca/sequence.bin \
  --out-dir build/kiss_icp_downsample_comparison --repeats 3 --plot
```

On Windows add `--config Release` to build and `-C Release` to CTest. The
benchmark accepts `--dll-dir`. External dataset preparation is in the
[timed MCD guide](cudanav_timed_dataset_mcd.md); input data is not redistributed.
`--maximum-frames` is a development prefix, not the complete workload.

The script refuses to overwrite outputs, freezes native and CPU executables
and source snapshots, and hashes sources with normalized LF plus raw input
and executable bytes. It profiles 24 evenly spaced raw scans before GPU timing,
then validates every deskewed scan over the full route, then measures cached
and pooled aggregation in three forward/reverse pairs. Normal voxel queries,
block reduction, dense map, cell-order normal scheduling and full normal updates
are fixed across methods. Raw logs, metrics, per-frame CSVs and failures remain.

Paired checks require pooled centroid mean below 5 ms, faster centroid p95,
faster whole-frame p95, identical ATE/drift and matching recorded odometry/map
fields at each frame. Those CSV fields include XY, XY error, inliers, map points,
observed voxels, integrated rays and unknown cells. Occupied-cell counts can
differ because the existing clamped atomic occupancy updates depend on update
order; native mapping/controller quality gates still apply unchanged.

The CPU executable can also be run directly:

```bash
bin/kiss_icp_downsample_benchmark \
  --sequence build/cudanav_real_gpu_stack_release_724d05ca/sequence.bin \
  --csv build/kiss_icp_centroid_cpu.csv --samples 24 --repeats 6
```

It samples raw coordinates without deskew, checks every centroid/output order
against the reference, warms the arena for each scan, and times cached/pooled
calls in alternating order. Diagnostic rows separately time voxel-key calculation,
container construction/reserve, lookup/accumulation with precomputed keys,
output generation and map destruction. A counting allocator records map node
and bucket allocations; output/key vectors are excluded from those counts.
Diagnostic stages use precomputed keys and different cache conditions from
the original interleaved loop, so their sum is not a replacement for measured
whole-call time. Raw CPU probes do not measure GPU deskew or stack latency.

Tests compare cached, unordered and pooled output bytes for empty/repeated
clouds, positive/negative voxel boundaries, several resolutions and 200,000-point
inputs. Repeated large/empty scans verify old sums do not survive and no new
slabs are allocated after warmup. The GPU streaming test validates centroids
and compares complete poses/alignment values against cached-reference execution,
including moving timed scans and reset.

Whole-frame timing covers odometry, rolling mapping, projection, ESDF and MPPI
every tenth scan. Loading, writes and construction are excluded. Commands are
not applied; FIFO metrics are calculated serial replay, not observed ROS
scheduling or sensor-to-command latency. Three repeats on one route do not
establish performance across datasets, hardware or standard libraries.
