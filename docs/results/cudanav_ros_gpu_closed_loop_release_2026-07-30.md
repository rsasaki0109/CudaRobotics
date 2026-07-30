# CudaNav ROS 2 GPU Closed-Loop Release — 2026-07-30

Status: **PASS**

The clean `6eb0d0dca5ced88bf5ff788f61bbc76267a2a573` checkout completed 30 alternating command-driven S-course traversals in ROS 2 Jazzy. GPU KISS-ICP odometry fed rolling voxel mapping and ESDF, and the Nav2 CUDA MPPI controller drove the simulated robot state.

| Metric | Result | Release gate |
|---|---:|---:|
| Elapsed time | 1325.535 s | >= 600 s |
| Traversals | 30 / 30 | 30 / 30 |
| Collisions | 0 | 0 |
| Final goal distance | 0.153 m | <= 0.25 m |
| Odometry drift | 0.00283% | < 1% |
| Controller deadline misses | 32 / 13,268 (0.241%) | < 1% |
| Diagnostic errors / warnings | 0 / 0 | 0 / 0 |
| Maximum failure counter | 0 | 0 |

Hardware was an NVIDIA GeForce GTX 1660 Ti (`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`, driver 596.36). The retained MCAP contains 1,233,851,559 bytes and is bound by tree SHA-256 `935face2588f0930ba2bc0348d82cfc775a1ae655837bb5097a32894b1d32859`; its MCAP payload SHA-256 is `15f05316c66b8b45e9808238eb3196ace5d8ad0e59e965c1dce007fd4beae433`.

This is closed-loop simulation evidence, not real-data replay or multi-GPU reproduction. The large MCAP, trajectory, and GIF are retained locally and content-bound by the adjacent portable JSON.
