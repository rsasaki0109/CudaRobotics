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

## Cross-machine collection

The two GPU models do not need to be installed in the same host. Run the
closed-loop smoke once on each machine at the identical commit and copy the
complete evidence directories to the aggregation machine. Then import them:

```bash
python scripts/run_cudanav_multi_gpu.py \
  --output-dir build/cudanav_multi_gpu/cross_machine_001 \
  --import-run imported/gtx_1660_ti/run_00 \
  --import-run imported/rtx_4070/run_00

python scripts/validate_cudanav_multi_gpu.py \
  build/cudanav_multi_gpu/cross_machine_001
```

Every imported directory is independently checked as CudaNav smoke evidence
before it is copied. The aggregate gate then revalidates all copied runs and
requires identical commit and controller-config SHA-256 values, distinct
physical GPU UUIDs, distinct model names, and a complete repetition matrix.
If multiple repetitions are imported, every physical GPU must contribute the
same count. `--import-run` cannot be combined with local `--devices`.

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
