# Optional CUDA MPPI fast-math build (2026-07-12)

CUDA `--use_fast_math` was evaluated as a low-effort, opt-in optimization for
the production MPPI core. It changes floating-point semantics, so it remains
disabled by default.

## Wall-gap A/B smoke

- GPU: NVIDIA GeForce GTX 1660 Ti, 6 GB
- CUDA compiler: 12.8
- `T=56`, one optimizer iteration
- Same source and closed-loop test; only `--use_fast_math` differs

| K | Build | Result | Sim time to goal | Mean solve | Max solve |
|---:|---|---|---:|---:|---:|
| 2,048 | precise (default) | pass | 18.5 s | 0.34 ms | 0.63 ms |
| 2,048 | fast math | pass | 19.0 s | 0.24 ms | 0.62 ms |
| 65,536 | precise (default) | pass | 16.1 s | 3.41 ms | 5.86 ms |
| 65,536 | fast math | pass | 16.2 s | 3.40 ms | 14.36 ms |

At K=2,048, mean solve time improved by about 29%, with a 0.5 simulated-second
increase in time to goal. At K=65,536 there was no meaningful mean improvement
and the observed maximum was worse. Both runs remained collision-free with a
100% minimum valid-rollout ratio.

## Enable explicitly

Python development/source build:

```bash
CMAKE_ARGS="-DCUDA_MPPI_FAST_MATH=ON" pip install -e python/
```

ROS2:

```bash
colcon build --packages-select cuda_mppi_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCUDA_MPPI_FAST_MATH=ON
```

Use this only after validating the target scenarios. The default remains the
precise build because reduction and transcendental differences can alter an
MPPI trajectory even when the final success outcome is unchanged.
