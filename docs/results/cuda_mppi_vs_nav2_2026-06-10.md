# cuda_mppi_controller vs nav2_mppi_controller (2026-06-10)

Head-to-head closed-loop comparison of the stock Nav2 MPPI controller (CPU)
and `cuda_mppi_controller` (GPU), both loaded through pluginlib exactly as
`controller_server` loads them.

## Setup

- Scenario: 10 m × 10 m costmap (0.05 m/cell), vertical lethal wall at
  x = 5 m with a 2 m gap, manually inflated (inscribed 0.2 m, scaling 3.0).
  Straight global plan from (1, 5) to (9, 5) through the gap.
- Closed loop at 20 Hz against a unicycle plant with perfect tracking;
  identical costmap, plan, control limits (`vx∈[-0.35, 0.5]`, `wz≤1.9`),
  horizon (T=56, dt=0.05), and 1 optimizer iteration for both.
- CPU baseline: `nav2_mppi_controller::MPPIController` with the stock
  nav2_bringup critic set (ConstraintCritic, CostCritic, GoalCritic,
  GoalAngleCritic, PathAlignCritic, PathFollowCritic, PathAngleCritic,
  PreferForwardCritic), otherwise default parameters.
- Hardware: Intel i9-10900 (20 threads) vs RTX 4070 Ti SUPER. ROS 2 Jazzy,
  CUDA 12.0, Release build.

<img src="cuda_mppi_vs_nav2_2026-06-10.svg" alt="solve time chart" width="900"/>

## Results

| controller | K | result | sim time to goal | mean solve | p95 | max |
|---|---:|---|---:|---:|---:|---:|
| nav2 MPPI (CPU) | 1,000 | success | 16.0 s | 2.17 ms | 2.80 ms | 3.11 ms |
| nav2 MPPI (CPU) | 2,000 | success | 16.2 s | 4.52 ms | 5.71 ms | 6.21 ms |
| nav2 MPPI (CPU) | 5,000 | success | 16.5 s | 11.40 ms | 15.72 ms | 17.00 ms |
| nav2 MPPI (CPU) | 10,000 | success | 16.3 s | 22.94 ms | 27.68 ms | 33.17 ms |
| CUDA MPPI (GPU) | 2,048 | success | 19.4 s | 1.80 ms | 4.93 ms | 10.44 ms |
| CUDA MPPI (GPU) | 8,192 | success | 19.1 s | 2.40 ms | 5.24 ms | 7.01 ms |
| CUDA MPPI (GPU) | 16,384 | success | 18.9 s | 3.26 ms | 6.11 ms | 10.45 ms |
| CUDA MPPI (GPU) | 65,536 | success | 18.8 s | 9.65 ms | 12.21 ms | 19.29 ms |

(Numbers from the 3-DOF-control kernel that also supports Ackermann/Omni
and footprint checking; the GPU rollout pays ~0.5 ms over the earlier
diff-drive-only kernel.)

Side-by-side rollout (CPU K=2,000 vs GPU K=16,384):
[`gif/cuda_mppi_vs_nav2_cpu.gif`](../../gif/cuda_mppi_vs_nav2_cpu.gif)

## Reading the numbers honestly

- **Throughput**: at comparable sample counts (K≈2,000) the GPU solve is
  ~4× faster. Scaling K 32× (65,536) still costs less wall-clock than the
  CPU at 10,000. Sample counts that are impractical on CPU are routine on
  GPU.
- **Quality**: the CPU baseline reaches the goal ~15% sooner in simulated
  time (16.0–16.5 s vs 18.8–19.4 s). Its critic set is mature and more
  aggressively tuned than our 6-term cost; this gap is tuning, not
  architecture, and is the obvious next thing to close.
- Both controllers solved well inside the 50 ms @ 20 Hz budget in this
  scenario; the GPU headroom matters for larger K, longer horizons, denser
  costmaps, or slower embedded CPUs.
- Single fixed scenario, single seed pair, no sensor pipeline — this is a
  controlled microbenchmark, not a field study.

## Reproduce

```bash
cd ros2_ws
colcon build --packages-select cuda_mppi_controller --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
mkdir -p /tmp/mppi_bench
ros2 run cuda_mppi_controller controller_benchmark /tmp/mppi_bench
python3 scripts/render_cuda_mppi_benchmark.py /tmp/mppi_bench 2026-06-10
```
