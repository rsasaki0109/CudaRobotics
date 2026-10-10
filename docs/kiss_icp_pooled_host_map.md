# Exact pooled KISS-ICP host map

The rolling map retains one point per voxel in dense coordinate storage. Its
unordered lookup nodes now have a persistent storage pool: radius eviction
returns individual nodes to a free list and later insertions reuse them.
Contiguous slabs improve lookup locality. Two compact integer arrays mirror the
container's iteration links, so point-order export reads small dense arrays
instead of chasing host hash-node pointers every frame.
The standard unordered container still owns hashing, bucket layout, reserve,
insertion and iteration order. No map density, voxel key, point representative,
radius test, capacity rule or ICP arithmetic changes.
[Paired real-scan results](results/kiss_icp_host_map_2026-10-10.md) retain the
full-route validation, all timing/default replays and source/input/executable
hashes.

When a new node is inserted, its public iterator successor identifies the
insertion position in the mirror. Eviction unlinks the erased slot and repairs
the moved last slot's links. The constructor reserves the entire map capacity,
and insertion rejects overflow before modifying the table, preventing rehash
while the mirror is live. Full-route Validate compares this mirror's complete
output against the ordinary iterator walk every frame.

The compact mirror is enabled for MSVC STL and libstdc++, whose current
implementations link only the new node during insertion without rehash. It uses
public iterators rather than private node layout. This is an implementation
property, not a general promise of the [unordered-container standard](https://eel.is/c++draft/unord.req).
See [MSVC STL](https://github.com/microsoft/STL/blob/main/stl/inc/xhash),
[libstdc++](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/hashtable.h).
Other standard libraries keep the original iterator walk, with pooled nodes.

`KissIcpMapBackend::Pooled` is the shared API/native runner default. `Dense` uses the same dense
coordinates with standard node allocation; `Unordered` retains the original
coordinate-packing reference. `Validate` executes Pooled and Dense updates on
the same world points and compares every coordinate bit and exported integer
slot every frame. It can also run with `KissIcpNormalUpdate::Validate` to compare
every incremental normal against full recomputation. Validation timing includes
the extra reference work and is excluded from performance pairs.

## Storage lifetime and correctness

The allocator routes individual nodes to size classes with max-align-t aligned
slots in approximately 256 KiB slabs. Multi-element bucket arrays keep the
standard allocator. Rebound allocators share the pool, which outlives the map.
Erase and clear recycle nodes without disturbing live entries. Reset retains
slabs for later scans; destruction releases them. Storage therefore follows the
high-water mark of concurrent nodes rather than the number of historical
insertions. Slab slack, allocator metadata and retained peak capacity can use
more memory than a standard map after it shrinks.

`host_map_pool_bytes` reports retained node slabs only; it excludes standard
bucket arrays, dense points/order, normal-cache metadata and small pool metadata.
`host_map_order_bytes` separately reports the two reserved integer arrays:
1.53 MiB at a 200,000-point capacity on supported standard libraries, zero on
the iterator-walk fallback. This is extra host storage relative to Dense.
`host_map_upstream_allocations` counts new slabs during scan updates since reset.
These slabs replace individually allocated host hash nodes; they are not an
additional GPU cache. Dense reports zero for both pool counters. Validate also
holds a second standard host map and its coordinate/order buffers.

CPU checks compare point bits, dense slots, cache tags and exported order over
240 moving frames with duplicated points, radius boundaries, swap deletion and
reset. Repeated capacity-error/fill/evict cycles require stable slab counts and
coherent partial maps. GPU streaming checks compare complete poses, alignment
values and reuse counts against standard Dense, as well as the Unordered/Full
reference. K=1/12/20, tied clouds and sparse supports remain covered by the
normal-cache tests. Unordered iteration order remains standard-library specific;
the comparison uses that platform's own reference implementation.

## Reproduction

```bash
cmake --build build --target cudanav_real_gpu_stack_sequence \
  kiss_icp_host_map_exact kiss_icp_gpu_streaming_smoke kiss_icp_normal_cache_exact
ctest --test-dir build --output-on-failure \
  -R '^kiss_icp_(host_map_exact|gpu_streaming_smoke|normal_cache_exact)$'
python scripts/benchmark_kiss_icp_host_map.py \
  --sequence build/cudanav_real_gpu_stack_release_724d05ca/sequence.bin \
  --out-dir build/kiss_icp_host_map_comparison --repeats 3 --verify-default --plot
```

Windows multi-config builds need `--config Release`; CTest needs `-C Release`.
The script accepts `--dll-dir` for runtime DLLs. It refuses an existing output
directory and freezes one executable and computation/benchmark/test sources.
It retains normalized-LF source hashes, raw executable/input hashes, every
command, per-frame CSV, JSON, logs and artifact hashes, including failures.
The schedule validates the whole route first, then alternates Dense/Pooled,
Pooled/Dense and Dense/Pooled. Both timing modes use pooled scan centroids,
incremental normals, exact spatial queries and block reduction. Each pair gates
whole-frame p95, map-update mean, recorded trajectory/map/reuse equality,
ATE/drift equality and the native quality checks. The separate 2 ms p95 target
is recorded even if the measured benefit falls short.
The separate default replay omits the map-backend flag and checks the actual
pooled policy, allocated slabs, p95 and recorded output/reuse equality.

`--maximum-frames` is a development prefix. Native frame timing covers odometry,
rolling mapping and projection/ESDF/MPPI every tenth scan. Commands are not
applied; FIFO numbers calculate serial replay rather than observed ROS or
sensor-to-command latency. One route/hardware result is not a universal speedup
or a bound on the worst frame.
