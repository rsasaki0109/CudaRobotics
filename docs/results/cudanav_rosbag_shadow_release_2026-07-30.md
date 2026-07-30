# CudaNav ROS 2 real-rosbag shadow release

- Source commit: `c4b91452d079fbfa6b285d80355632a5d1c9c716` (clean)
- Dataset: `autoware_istanbul_localization_smoke`
- GPU: NVIDIA GeForce GTX 1660 Ti (`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`)
- Claim boundary: real sensor data with a derived recorded Path; commands do not alter recorded motion.

## Results

- Diagnostics: 793 samples over 79.025 s
- Solve latency: mean 3.166 ms, p95 4.801 ms
- Valid rollout ratio: 0.9907
- Pointcloud pairing: 790/790 (100.00%)
- Front clearance: minimum 3.535 m, mean 5.800 m
- All-colliding recovery: 5 cycles (0.631%), 5 retreat cycles

## Recorded CudaNav outputs

- `/cuda_nav/cmd_vel`: 790 messages
- `/cuda_nav/odom`: 804 messages
- `/cuda_nav/occupancy`: 796 messages
- `/cuda_nav/esdf`: 794 messages

Overall release gate: **PASS**
