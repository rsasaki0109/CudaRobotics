# cudanav_kiss_icp_real_2026-07-29

GPU KISS-ICP odometry on a content-addressed real PointCloud2 sequence. This is not a controller or closed-loop result.

- Source commit: `455e013843a77de8b6c78073048c115143cb6edc`
- Dataset: `autoware_istanbul_localization_smoke`
- GPU: `NVIDIA GeForce GTX 1660 Ti` (`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`)
- Frames / duration: 300 / 29.900 s
- Declared profile / startup offset: `smoke` / 1.000 s
- Points per frame (min / mean / max): 932 / 1214.00 / 1281
- Reference pose age p95: 0.357888 ms
- Reference path: 72.285 m
- ATE RMSE: 0.778 m
- Final drift: 2.961%
- Yaw error p95: 0.003508 rad
- Mean frame time: 9.669 ms
- GPU NN p95: 5.582 ms
- Minimum inliers: 417
- Quality gate: PASS

## Scope

- Real PointCloud2 GPU odometry: yes
- GPU controller run: no
- Closed-loop evidence: no
