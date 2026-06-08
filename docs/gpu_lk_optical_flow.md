# GPU Pyramidal Lucas-Kanade Optical Flow

`src/gpu_lk_optical_flow.cu`

Sparse Lucas-Kanade optical flow — the workhorse behind KLT trackers,
monocular VO front-ends, and feature-based scene-flow pipelines. Maps onto the
repo's canonical idiom: **one thread = one feature**. Each LK feature is an
independent small-window Gauss-Newton problem, so a feature grid parallelises
perfectly across threads.

![demo](https://rsasaki0109.github.io/CudaRobotics/gpu_lk_optical_flow.gif)

## Pipeline

1. Build a synthetic reference image `I` (analytic textures: three blobs at
   different frequencies + a low-frequency background gradient).
2. Define a known smooth flow field (uniform translation + rotation) and warp
   `I` by it to produce the second image `J`.
3. Scatter `1024` features on a `32 × 32` grid.
4. Build a `3`-level image pyramid (`256 → 128 → 64`, box `2×2` downsample).
5. Run pyramidal LK coarse-to-fine on both CPU and GPU; compare per-feature
   flow.

Per feature, per pyramid level:

- Build the `2×2` spatial Hessian `G` and cache `I`, `Iₓ`, `I_y` over a
  `9×9` window on the **reference** image once (inverse-compositional form).
- Gate trackability with a Hessian determinant floor (`DET_FLOOR = 1e-3`):
  features in flat/aperture-limited regions are dropped.
- Run `8` Gauss-Newton iterations: resample `J` at `(feature + current flow)`,
  form residual `r = I − J`, solve `Δ = G⁻¹ b`, accumulate.
- Propagate the flow estimate up the pyramid (`×2` per level).

## Correctness — deterministic by construction

Per feature, LK is a fixed-iteration Gauss-Newton step with no data-dependent
branch that forks into a different answer — the only branch is the
determinant-floor skip, which is itself bit-identical between CPU and GPU under
`--fmad=false`. Both paths use the same bilinear sampler, the same gradient
kernel, and the same `2×2` inversion, so per-feature estimates agree to the
last bit.

| metric | value |
|---|---|
| CPU vs GPU flow `max\|diff\|` | `0.0` (bit-identical) |
| trackable features (CPU == GPU) | `290 / 1024` |
| valid-flag mismatch | `0` |
| ground-truth endpoint error (over trackable features) | `1.54` px |

The aperture problem is visible in the numbers: only `290 / 1024` features sit
in regions textured enough to pass the determinant floor. The signal here is
that CPU and GPU agree on **exactly which 290** are trackable — the gate is
deterministic, not just the arithmetic.

## Result (this machine)

| | scale | time | note |
|---|---|---|---|
| CPU serial LK | `1024` features × `3` levels × `8` iters | `54.4 ms` | reference |
| **GPU LK** | one thread / feature | **`0.284 ms`** | **~192×** |

## Reproduce

```bash
cd build && cmake .. && make gpu_lk_optical_flow -j$(nproc)
cd .. && ./bin/gpu_lk_optical_flow
```

Prints the timing + correctness table and writes
`gif/gpu_lk_optical_flow.gif` (side-by-side `I` and `J`-with-flow-arrows,
progressive feature reveal).

## Notes

- One demo = one `.cu`; reuses `include/cuda_check.cuh` and
  `include/cuda_video.h`.
- Warp convention: `J(p) = I(p + Δ)`, so the LK estimate is `d = −Δ`; the
  ground-truth comparison negates the synthetic flow accordingly.
- Compiled with `--fmad=false` so the fused-multiply-add fusion does not
  introduce a CPU/GPU divergence — this is what makes the `max|diff| = 0`
  claim hold exactly rather than approximately.
- GIF served from gh-pages, not committed to the repo.
