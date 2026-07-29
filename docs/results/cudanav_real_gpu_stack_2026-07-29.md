# cudanav_real_gpu_stack_2026-07-29

Real PointCloud2 shadow execution through GPU KISS-ICP, rolling voxel mapping, GPU ESDF inflation, and CUDA MPPI. Commands are evaluated but not applied, so this is not closed-loop evidence.

- Source commit: `614af5681fd757d298ea835c98988f2cd930de5b`
- Dataset: `autoware_istanbul_localization_smoke`
- GPU: `NVIDIA GeForce GTX 1660 Ti` (`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`)
- Profile / startup offset: `smoke` / 1.000 s
- Frames / duration: 300 / 29.900 s
- ATE RMSE / final drift: 0.779 m / 2.985%
- Final observed voxels / peak occupied cells: 107737 / 1789
- ESDF p95: 1.153 ms
- MPPI evaluations / solve p95: 31 / 0.532 ms
- Nonzero valid-rollout ratio minimum: 0.0444
- Safety-stop evaluations: 2 (maximum |v| 0.000 m/s)
- End-to-end mean / p95 frame time: 9.999 / 21.365 ms
- Quality gate: PASS

## Scope

- Real PointCloud2 all-GPU core shadow: yes
- ROS 2 runtime: no
- Commands applied to vehicle or simulator: no
- Closed-loop evidence: no
