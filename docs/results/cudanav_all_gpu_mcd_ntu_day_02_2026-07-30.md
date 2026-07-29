# cudanav_all_gpu_mcd_ntu_day_02_2026-07-30

Real PointCloud2 shadow execution through GPU KISS-ICP, rolling voxel mapping, GPU ESDF inflation, and CUDA MPPI. Commands are evaluated but not applied, so this is not closed-loop evidence.

- Source commit: `724d05caae07896a9ebab1a71980f0993f279335`
- Dataset: `mcd_ntu_day_02_os1_128_ros2_timed_120s`
- GPU: `NVIDIA GeForce GTX 1660 Ti` (`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`)
- Profile / startup offset: `release` / 1.000 s
- Frames / duration: 1190 / 118.902 s
- ATE RMSE / final drift: 0.812 m / 0.471%
- Final observed voxels / peak occupied cells: 1791642 / 8120
- ESDF p95: 1.068 ms
- MPPI evaluations / solve p95: 120 / 0.715 ms
- Nonzero valid-rollout ratio minimum: 0.1362
- Safety-stop evaluations: 0 (maximum |v| 0.000 m/s)
- End-to-end mean / p95 frame time: 232.294 / 514.232 ms
- Quality gate: PASS

## Scope

- Real PointCloud2 all-GPU core shadow: yes
- ROS 2 runtime: no
- Commands applied to vehicle or simulator: no
- Closed-loop evidence: no
