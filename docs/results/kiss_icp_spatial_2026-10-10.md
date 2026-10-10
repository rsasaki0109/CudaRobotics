# Exact KISS-ICP spatial queries (2026-10-10)

Native GPU KISS-ICP, rolling voxel mapping, ESDF and MPPI shadow execution on the timed MCD NTU day-02 sequence: 1,190 frames / 118.902 seconds. The same route is replayed twice per method, not treated as independent datasets. GPU: NVIDIA consumer GPU.

[Implementation and reproduction](../kiss_icp_spatial_performance.md), [all replay metrics](kiss_icp_spatial_2026-10-10.csv), [source/executable/input hashes](kiss_icp_spatial_2026-10-10.json).

| Method / repeat | Frame mean ms | Frame p95 ms | Frame p99 ms | Normal mean ms | NN mean ms | ATE m | Drift % | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| legacy / 0 | 229.376 | 522.226 | 554.191 | 125.885 | 66.056 | 0.8152 | 0.4735 | PASS |
| normals / 0 | 120.578 | 271.515 | 286.107 | 20.692 | 63.725 | 0.8070 | 0.4672 | PASS |
| spatial / 0 | 61.821 | 90.606 | 95.743 | 20.704 | 3.045 | 0.8139 | 0.4712 | PASS |
| spatial / 1 | 61.140 | 89.783 | 94.591 | 20.700 | 3.052 | 0.8189 | 0.4742 | PASS |
| normals / 1 | 123.881 | 271.996 | 282.928 | 20.716 | 63.816 | 0.8283 | 0.4807 | PASS |
| legacy / 1 | 239.657 | 526.738 | 559.027 | 130.043 | 68.638 | 0.8122 | 0.4709 | PASS |

Paired checks: each spatial replay has frame p95 below 100 ms, ATE no more than 2 cm above its paired legacy replay, and final drift no more than 0.02 percentage points above that baseline.

| Method / repeat | Calculated FIFO p95 response ms | Calculated final response ms | Calculated deadline miss % | Frames over 100 ms % |
|---|---:|---:|---:|---:|
| legacy / 0 | 134140.184 | 160326.432 | 88.319 | 88.151 |
| normals / 0 | 24124.588 | 34725.385 | 85.294 | 47.731 |
| spatial / 0 | 90.679 | 95.071 | 0.252 | 0.168 |
| spatial / 1 | 89.783 | 96.521 | 0.084 | 0.084 |
| normals / 1 | 27559.032 | 38068.761 | 85.546 | 49.664 |
| legacy / 1 | 145807.718 | 172162.307 | 88.403 | 88.319 |

FIFO metrics assume serial, lossless processing at original sensor timestamps. They are calculated from compute durations, not observed ROS scheduling. A small number of overruns remain; p95 below 100 ms is not a hard real-time bound.

Legacy uses exhaustive normals and linked correspondences. Normals changes only normal estimation. Spatial changes both. Point input, voxel resolutions, 40 m map radius, neighbour count, ICP gates and iteration limits are unchanged. Frame timing excludes initial file loading, CSV output and construction; MPPI runs every tenth scan. Commands are evaluated but not applied.

All six final replays are retained. An earlier development legacy replay failed the MPPI valid-rollout-ratio gate (0.00244 against 0.01); it is retained under `build/kiss_spatial_20261010/full_v1/`. Normal-equation atomic reductions and linked ties permit small trajectory differences across runs. These results do not establish controller robustness or real-vehicle performance.

Checks: exact spatial query test, streaming API smoke and synthetic odometry gate pass (3/3); CPU/Python CTest checks pass (80/80). Tests completed before final timings. Final runs execute sequentially in forward then reverse method order using one frozen executable. No concurrent tests or GPU workloads were intentionally run during final measurements.

Raw per-frame CSVs, commands, logs, `results.json` and `comparison.png`: `build/kiss_spatial_20261010/final/`. Measured sources are an uncommitted worktree; the recorded HEAD does not contain this implementation, so source hashes bind the measured changes.
