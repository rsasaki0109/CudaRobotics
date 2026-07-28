# Exact GPU ESDF Core

`cudarobotics::Esdf2DGpu` computes a metric 2D Euclidean distance field from a
standard occupancy grid.

Unlike the earlier visualization JFA demo, this production core uses the exact
separable squared Euclidean distance transform. Independent row transforms and
column transforms run on the GPU, followed by metric scaling and truncation.

Unknown space is never implicit:

- `UnknownSpacePolicy::Occupied` makes unknown cells obstacle seeds;
- `UnknownSpacePolicy::Free` excludes unknown cells from the seed set.

Every output is finite and clamped to `[0, max_distance]`.

## Reference gate

```bash
cmake --build build --target esdf_2d_gpu_smoke -j"$(nproc)"
ctest --test-dir build -R esdf_2d_gpu_smoke --output-on-failure
```

The GPU result is compared cell-by-cell with an O(cells × occupied) CPU oracle
for both unknown policies and shapes including 1x1, one-row, one-column,
rectangular, and near-square grids. Separate cases cover no obstacles, all
occupied, truncation, malformed shapes, counts, and finite-range invariants.
