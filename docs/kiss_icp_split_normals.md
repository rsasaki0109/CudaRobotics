# Exact split KISS-ICP normal updates

The incremental normal cache avoids KNN/PCA for unchanged supports. Its fused
kernel still mixes cheap cached copies with expensive searches in the same
thread blocks. Split scheduling separates those operations without changing
which supports are accepted or the normal arithmetic.
The [retained timing comparison](results/kiss_icp_split_normals_2026-10-10.md)
includes all repeats and unsuccessful latency targets.

## GPU execution

The first kernel runs the conservative reuse check in cell query order.
It copies accepted normals, remaps their ordered neighbour IDs and retains their
support radii. A block scan compacts rejected point IDs into a device queue,
with one reservation per block. A block integer atomic counts accepted points.
Partial blocks contribute only valid points.

The second kernel processes one queued point per thread using the existing
exact K+1 search, K-neighbour PCA and tie/radius recording. The common K=12
case supplies a constant 13-neighbour search bound so the compiler can
specialize the support search and size its local arrays; other K values retain
the generic exact search. Its launch covers the maximum current map size;
threads beyond the device queue count return without
searching. Empty queues perform no recomputation. The queue count stays on the
GPU between kernels, so scheduling adds no CPU round trip or host wait.

Each point writes its own normal and support slot. Queue block reservations
can arrive in a different order, but this does not change point IDs, neighbour
selection or the per-point summation order. Cached copies use the same
invalidation rule as Fused: surviving support, no tied distances and no new
representative at or inside the old radius. Density, voxel sizes, map radius,
ICP gates and iteration limits are unchanged.

A development variant that processed several queued points per thread failed
bitwise normal validation because compiler-generated floating-point rounding
differed. It was discarded. The implemented one-point-per-thread variant is
checked against the existing full recomputation and fused cache paths.
An initial split-only full-route comparison passed bitwise validation but saved
only about 0.11 ms of mean normal work. A tighter added-point cell box provided
little further improvement and was discarded. The common support size was
specialized after those trials; their
original frozen executables and results remain separate.

## Selection and storage

`KissIcpConfig::normal_schedule` selects `Fused` or `Split` for incremental and
validated updates. `normal_update=Full` and unsupported cache combinations use
their existing full path and allocate no queue. Fused remains available as the
comparison reference.
The API and native runner currently default to Fused.
Three full-route timing pairs reduced mean GPU normal work by 3.2-7.3%, but
whole-frame p95 improved in only one pair and regressed in two. The 1 ms
whole-frame saving target was not reached, so Split is an opt-in evaluation
path. Its extra queue and launch cost should be evaluated on the caller's
workload before adopting it.

To select Split through the shared API:

```cpp
cudarobotics::KissIcpConfig config;
config.normal_schedule = cudarobotics::KissIcpNormalSchedule::Split;
cudarobotics::KissIcpOdometry odometry(config);
```

The native runner accepts `--kiss-normal-schedule fused|split` and records the
requested schedule. `normal_cache.recompute_queue_bytes` in its JSON, and
`KissIcpTiming::normal_recompute_queue_bytes`, report actual reserved storage.
Split adds one integer per configured map point and one device counter:
800,004 bytes (0.763 MiB) at 200,000 points. Allocation happens at construction
and remains across reset; each update clears the count. No extra host storage
or per-frame allocation is needed.

`normal_ms` encloses both kernels, including compaction and the empty threads;
`normal_cache_prepare_ms` includes clearing the queue count. Both are contained
in odometry wall time. Whole-frame time is the acceptance measure.

## Reproduction

```bash
cmake --build build --target gpu_kiss_icp kiss_icp_normal_cache_exact \
  kiss_icp_gpu_streaming_smoke kiss_icp_spatial_exact kiss_icp_reduction_exact \
  kiss_icp_host_map_exact kiss_icp_downsample_exact cudanav_real_gpu_stack_sequence
ctest --test-dir build --output-on-failure -R \
  '^(gpu_kiss_icp_gate|kiss_icp_(gpu_streaming_smoke|spatial_exact|reduction_exact|host_map_exact|downsample_exact|normal_cache_exact))$'
ctest --test-dir build --output-on-failure --label-regex 'cpu|python' -j 4
compute-sanitizer --tool memcheck --error-exitcode 99 bin/kiss_icp_normal_cache_exact
python scripts/benchmark_kiss_icp_normal_schedule.py \
  --sequence build/cudanav_real_gpu_stack_release_724d05ca/sequence.bin \
  --out-dir build/kiss_icp_normal_schedule --repeats 3 --verify-default --plot
```

Windows builds add `--config Release`, CTest adds `-C Release`, executables
live in `bin/Release/` and the runner accepts `--dll-dir`. Dataset preparation
is in the [timed MCD guide](cudanav_timed_dataset_mcd.md).

The benchmark freezes computation sources, executable and input hashes. It
validates all map point/order/normal bits on the full route, then alternates
Fused/Split and Split/Fused timing pairs with pooled maps and centroids. Paired
checks require matching recorded trajectory/map/reuse fields, ATE and drift,
faster normal mean and whole-frame p95, p95 below 40 ms and native quality
gates. The 1 ms whole-frame saving target is recorded separately, including
misses. `--verify-default` adds a replay without a scheduling override.
`--expected-default fused|split` selects the expected policy for that check.

The GPU test compares Split, Fused and Full over moving/random/tied clouds,
K=1/12/20, both query orders, insertion and reset. It separately exercises an
empty queue, a fully populated partial block and sparse maps without reuse.
Boundary tests compare the presence query with exhaustive GPU distance checks,
including zero radius, inclusive support boundaries,
negative cells, large coordinates and the conservative large-radius fallback.
Full-route CSV equality covers recorded XY poses and map/reuse fields; the
streaming tests compare full SE3 poses and alignment statistics. Occupied-cell
differences from existing clamped atomic mapping updates remain separately
recorded with independent native mapping/controller gates.

## Scope

This is native shadow execution with commands unapplied and MPPI every tenth
scan. Frame timing excludes loading, construction and output. Calculated FIFO
latency is not observed ROS scheduling or sensor-to-command latency. One route
and hardware do not establish a general speedup or a worst-frame bound.
When little can be reused, compaction and the second launch may cost more than
fused execution; both schedules remain selectable.
