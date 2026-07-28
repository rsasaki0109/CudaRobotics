# CudaNav Multi-GPU Reproducibility Matrix

`scripts/run_cudanav_multi_gpu.py` runs the deterministic CudaNav closed-loop
smoke independently on selected physical NVIDIA devices. Each child process
receives one `CUDA_VISIBLE_DEVICES` index and produces the complete retained
run directory defined in [cudanav_closed_loop.md](cudanav_closed_loop.md).

```bash
python scripts/run_cudanav_multi_gpu.py \
  --output-dir build/cudanav_multi_gpu/matrix_001 \
  --devices all \
  --repetitions 3

python scripts/validate_cudanav_multi_gpu.py \
  build/cudanav_multi_gpu/matrix_001
```

The publication gate defaults to:

- at least two distinct physical GPU UUIDs;
- at least two distinct GPU model names;
- every child smoke gate passing;
- exactly one visible GPU recorded by each child;
- declared physical index/name/UUID matching the child manifest;
- identical full git commit and controller-config SHA-256 across all runs;
- the requested device × repetition cell count present.

The output must be outside the repository or under a git-ignored path. This
prevents matrix artifacts from making child worktrees dirty and invalidating
their provenance gate.

For local harness development on a single GPU, explicitly lower both coverage
requirements:

```bash
python scripts/run_cudanav_multi_gpu.py \
  --output-dir build/cudanav_multi_gpu/dev_smoke \
  --minimum-gpu-devices 1 \
  --minimum-gpu-models 1
```

That run remains labelled with one-device/one-model thresholds and is not
accepted as the multi-GPU publication result.
