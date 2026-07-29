# cudanav_all_gpu_mcd_ntu_day_02_2026-07-29

Real PointCloud2 shadow execution through GPU KISS-ICP, rolling voxel mapping, GPU ESDF inflation, and CUDA MPPI. Commands are evaluated but not applied, so this is not closed-loop evidence.

- Source commit: `541a53d36f9dd600646ffcf921d06fab404d67f3`
- Dataset: `mcd_ntu_day_02_os1_128_ros2_timed_120s`
- GPU: `NVIDIA GeForce GTX 1660 Ti` (`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`)
- Profile / startup offset: `release` / 1.000 s
- Frames / duration: 1190 / 118.902 s
- ATE RMSE / final drift: 0.819 m / 0.475%
- Final observed voxels / peak occupied cells: 1778523 / 8162
- ESDF p95: 1.147 ms
- MPPI evaluations / solve p95: 120 / 0.836 ms
- Nonzero valid-rollout ratio minimum: 0.1284
- Safety-stop evaluations: 0 (maximum |v| 0.000 m/s)
- End-to-end mean / p95 frame time: 249.434 / 545.781 ms
- Quality gate: PASS

## Scope

- Real PointCloud2 all-GPU core shadow: yes
- ROS 2 runtime: no
- Commands applied to vehicle or simulator: no
- Closed-loop evidence: no
