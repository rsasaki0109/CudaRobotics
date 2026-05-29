# GPU Jump Flooding Voronoi / EDT

`gpu_jfa_edt` is the textbook **Jump Flooding Algorithm** (Rong & Tan 2006) —
the canonical `O(log N)` GPU-native Voronoi diagram and Euclidean Distance
Transform (EDT) builder.

The map is the repo's canonical 2D idiom: **one thread = one pixel**.  Each
sweep, every pixel inspects 9 candidate sites (itself plus 8 neighbours at
the current step `s`) and keeps the closest seed it has seen so far.  Step
sizes halve each pass — `s = W/2, W/4, ..., 1` — so log₂(W) sweeps converge.

## Setup

- Grid: `512 × 512 = 262,144` pixels.
- `96` seed points placed at random (fixed seed for reproducibility).
- JFA runs `9` passes (`s = 256, 128, ..., 1`).
- CPU reference: brute-force EDT — for every pixel, scan every seed and pick
  the nearest.  This is the *exact* 2-norm Voronoi diagram.

## Correctness — honest approximation

In contrast to TSDF (#130), DBSCAN (#131) and Marching Cubes (#132), JFA is
**not** bit-identical to the brute-force reference.  Each sweep only peeks at
9 sites, so on rare configurations a pixel close to a Voronoi cell boundary
ends up assigned to a slightly farther seed than the true argmin.  The
disagreement is always confined to cell boundaries where two seeds are
nearly equidistant.

| metric | value |
|---|---|
| Voronoi label agreement (GPU vs CPU brute force) | `99.9191 %` (`261,932 / 262,144`) |
| EDT `max\|diff\|` | `0.0538` px (sub-pixel) |
| EDT `mean\|diff\|` | `2.05e-07` px |

So `0.0809 %` of pixels disagree, and every one of them is wrong by *less
than 0.05 px* — entirely within sub-pixel tolerance for downstream EDT use.
This is the textbook JFA result and matches the bounds in the original paper.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_jfa_edt -j$(nproc)
./bin/gpu_jfa_edt
```

Generated files:

- `tmp/gpu_jfa_edt.avi`
- `gif/gpu_jfa_edt.gif`

## Output

The GIF replays JFA pass by pass: the initial scatter of seeds, then the
Voronoi diagram crystallising at `s = 256, 128, ..., 1`.  A panel tracks the
pass index, the CPU vs GPU timing, and the agreement against brute force.

Latest local run:

- `512 × 512 × 96` seeds = `25.2 M` candidate distance pairs in brute force.
- CPU serial brute-force EDT `75 ms`, GPU JFA `0.077 ms` — about **983×**.
- 9 JFA passes.
- Voronoi label agreement `99.92 %`, EDT `max|diff| 0.054 px`.

The win scales as `log(N) / K`: the GPU does `9` sweeps of `262 144` pixels
instead of `262 144 × 96` brute-force comparisons.
