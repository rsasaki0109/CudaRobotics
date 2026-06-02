# Stanford bunny scan (decimated)

`bun000.xyz` is a voxel-downsampled (voxel 0.0022) point set extracted from the
`bun000.ply` Cyberware range scan of the **Stanford bunny**, from the
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).

Only the vertex `x y z` coordinates are kept (one point per line), decimated from
~40k to ~6k points so the files are small enough to vendor for a self-contained
demo. The data is used here purely for an algorithm validation demo
(`src/gpu_real_bunny_reg.cu`). Please credit the Stanford Computer Graphics
Laboratory if you reuse it.
