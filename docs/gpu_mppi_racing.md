# GPU MPPI Autonomous Racing

`src/gpu_mppi_racing.cu`

A kinematic-bicycle race car drives a closed circuit by **Model-Predictive Path
Integral (MPPI) control**. Every control step the GPU rolls out **thousands of
noisy candidate trajectories in parallel**, scores each by how far it advances
along the track (heavily penalising leaving the asphalt), and the
softmax-weighted average of the perturbations updates the nominal control
sequence. **One GPU thread = one sample trajectory** — the classic MPPI parallel
pattern, shown here as the green fan of candidates spreading ahead of the car.

![demo](https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_racing.gif)

## How it works

- **Model** — kinematic bicycle `(x, y, θ, v)` with controls `(accel, steer)`,
  integrated `T = 40` steps ahead at `dt = 0.06 s`.
- **Track cost** — the circuit is baked once on the host into two grid look-ups:
  a *progress* field (arc-length along the centreline) and a *distance* field
  (metres from the centreline). The rollout cost is then an **O(1)** lookup per
  predicted state — reward for progress (with wrap handling at the start line),
  a quadratic penalty for going beyond the track half-width, plus mild speed and
  steering shaping. So the per-step work is dominated by the `K × T` model
  integration, exactly what the GPU parallelises.
- **MPPI update** — `K = 2048` rollouts perturb the nominal controls with
  Gaussian noise; weights `w_k ∝ exp(-(cost_k - min)/λ)` are formed from the
  costs and a second kernel (one thread = one horizon step) computes the
  weighted control update. The first control is applied to the true car and the
  horizon recedes.

## Thesis — the win is the parallel rollout (measured)

Each step the **identical** `K × T` rollout+cost is also run single-threaded on
the CPU and timed, so the speed-up is measured, not assumed.

| | time / control step | note |
|---|---|---|
| CPU rollout | `~11.0 ms` | 2048 × 40 single-threaded |
| **GPU rollout** | **`~0.015 ms`** | one thread / trajectory |

**≈ 740× speed-up.** At that cost the controller plans in real time and the car
actually races: it completes **3/3 laps** of the 122.9 m circuit, **best lap
6.72 s**, **top speed 16 m/s**, staying on the asphalt throughout.

## Reproduce

```bash
cd build && cmake .. && make gpu_mppi_racing -j$(nproc)
cd .. && ./bin/gpu_mppi_racing
```

Prints the per-step GPU/CPU rollout times + speed-up and the lap times, and
writes `gif/gpu_mppi_racing.gif`.

## Notes

- One demo = one `.cu`; reuses `include/cuda_check.cuh` and
  `include/cuda_video.h`, and the per-thread cuRAND / softmax-MPPI idioms from
  the existing `mppi.cu`.
- The track background (asphalt band, kerbs, dashed centreline, start/finish
  line) is rasterised once from the distance field and reused each frame.
- `--expt-relaxed-constexpr`; GIF served from gh-pages (not committed to the
  repo).
