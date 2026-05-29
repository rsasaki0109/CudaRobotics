# GPU Marching Cubes

`gpu_marching_cubes` is the textbook Lorensen & Cline iso-surface mesh
extractor running on the GPU.  It is the natural successor to the TSDF demo
(`gpu_tsdf_fusion`, #130): TSDF fusion *builds* a dense SDF volume, Marching
Cubes *reads* it and turns the zero-level-set into a triangle mesh.

The algorithm maps cleanly onto the repo's canonical mapping idiom:
**one thread = one cube cell**.  For every cell, the kernel

1. gathers the 8 corner SDF values,
2. forms an 8-bit configuration index (one bit per corner inside/outside),
3. looks the case up in a 256-entry edge / triangle table (Bourke's
   public-domain layout),
4. linearly interpolates the iso-vertex on each crossed edge,
5. writes up to 5 triangles into a *fixed* output slot (`cell_idx * 5 + t`).

The fixed-slot layout is what makes the CPU/GPU comparison clean: both
implementations write to identical indices, so the two vertex buffers can be
compared *byte-by-byte*.

## Setup

- Volume: `128^3 = 2,097,152` voxels, axis span `5 m`, grid spacing
  `≈ 0.039 m`, iso = `0`.
- Scene SDF: the same 3-sphere "snowman" used by the TSDF demo (ground plane
  union 3 spheres), evaluated analytically at every grid point.
- A single `__host__ __device__` `mc_cell(...)` is called by **both** the
  serial CPU triple loop and the batch CUDA kernel.

## Correctness — deterministic by construction

In contrast to the iLQR demo, Marching Cubes has no data-dependent branches
that fork into different answers — every cell looks up the same fixed table
and writes its triangles to the same slot.  With FMA contraction disabled
(`--fmad=false`) the CPU and GPU vertex buffers are **bit-identical**:

| metric | value |
|---|---|
| triangles (CPU vs GPU) | `81,890` vs `81,890` |
| cells with mismatched triangle count | `0 / 2,048,383` |
| vertex `max\|diff\|` over `737,010` floats | `0.0` |

The GPU is doing the *same arithmetic in parallel*; the win is throughput on
the volume.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_marching_cubes -j$(nproc)
./bin/gpu_marching_cubes
```

Generated files:

- `tmp/gpu_marching_cubes.avi`
- `gif/gpu_marching_cubes.gif`

## Output

The GIF rotates the extracted GPU mesh around the snowman, with a panel
tracking the total triangle count, the CPU vs GPU timing, and the bit-identity
of the vertex buffers.

Latest local run:

- `128^3` voxels → `81,890` triangles.
- CPU serial MC `68 ms`, GPU MC `4.08 ms` — about **17×**.
- Per cell: CPU `33 ns` vs GPU `2.0 ns`.
- CPU/GPU vertex `max|diff|` = `0.0` (bit-identical).

The modest speedup (compared to TSDF's ~1075×) reflects that MC is genuinely
compute-light per cell: 8 corner reads, a table lookup, and at most 12 edge
interpolations.  The real headline is the bit-identical mesh.
