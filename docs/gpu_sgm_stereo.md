# GPU Semi-Global Matching Stereo Disparity

`src/gpu_sgm_stereo.cu`

Semi-Global Matching (Hirschmuller, 2008) - the workhorse dense-stereo
estimator behind depth cameras and stereo VO front-ends, and the repo's first
dense-stereo demo. SGM approximates an expensive global 2D smoothness energy by
aggregating a per-pixel matching cost along several 1D paths and summing them.
The 1D recurrence is sequential along a path, but every scanline is independent,
so the natural GPU map is **one thread = one scanline (per direction)**.

![demo](https://rsasaki0109.github.io/CudaRobotics/gpu_sgm_stereo.gif)

## Pipeline (CPU and GPU run the SAME integer logic)

1. **Census transform** - each pixel becomes a 5x5 (24-bit) census descriptor.
2. **Matching cost** - C(p,d) = Hamming(censusL[p], censusR[p-d]) via popcount.
3. **Path aggregation**, 4 directions (L-R, R-L, T-B, B-T):

   ```
   Lr(p,d) = C(p,d) + min( Lr(p-r,d),
                           Lr(p-r,d-1)+P1, Lr(p-r,d+1)+P1,
                           min_k Lr(p-r,k)+P2 ) - min_k Lr(p-r,k)
   ```
   accumulate S(p,d) += Lr(p,d) over all directions.
4. **Winner-take-all** - disparity(p) = argmin_d S(p,d).

The demo animates the disparity map sharpening as each of the 4 paths is added,
next to the left image and the synthetic ground truth.

## Correctness - bit-identical by construction

Every stage is integer arithmetic with deterministic tie-breaks (min over
values; argmin scans d ascending with strict less-than), so the CPU and GPU
disparity maps agree exactly.

| metric | value |
|---|---|
| CPU vs GPU disparity mismatches | 0 / 98304 |
| CPU vs GPU max abs diff | 0 |

Accuracy against the synthetic ground truth (an honest number for the algorithm
itself - not a CPU/GPU comparison):

| metric | value |
|---|---|
| mean abs disparity error | ~3.5 px |
| within 1 px | ~84 % |
| within 2 px | ~86 % |

The error concentrates at occlusion boundaries - the disocclusion halos to one
side of each foreground object, where a left-visible pixel has no correct match
in the right image. This is the expected, well-known failure mode of stereo
(and is exactly what a left-right consistency check would mask out); the planar
surfaces themselves are recovered cleanly.

## Result (this machine)

| | scale | time | note |
|---|---|---|---|
| CPU serial 4-path SGM | 384x256 x D=64, census + 4 paths + WTA | ~290 ms | reference |
| **GPU SGM** | one thread / scanline | **~6.3 ms** | **~46x** |

## Reproduce

```bash
cd build && cmake .. && make gpu_sgm_stereo -j$(nproc)
cd .. && ./bin/gpu_sgm_stereo
```

Prints the timing + bit-identity + accuracy table and writes
`gif/gpu_sgm_stereo.gif`.

## Notes

- One demo = one `.cu`; reuses `include/cuda_check.cuh` and `include/cuda_video.h`.
- This is the 4-path SGM variant (horizontal + vertical). The full 8-path
  variant adds the diagonals and reduces streaking; 4-path keeps the scanline
  indexing simple while still demonstrating the method and the GPU mapping.
- Path aggregation accumulates into S with atomicAdd; integer sums are
  order-independent, so the result is identical to the CPU's sequential sum.
- The right image is synthesised by forward-warping the scene with a z-buffer
  and filling disocclusions with the background layer (hole-free, realistic).
- `--fmad=false`; GIF served from gh-pages (not committed to the repo).
