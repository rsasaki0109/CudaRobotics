# Performance regression check

## What it does
`scripts/perf_check.py` runs a curated set of `comparison_*` GPU demos,
extracts the headline GPU timing from their stdout, and compares each
measurement against `scripts/perf_baseline.json`. The script exits
non-zero if any measurement exceeds `baseline * (1 + tolerance)`.

Default tolerance is **30%**.

## Local usage

```bash
# build the perf-critical targets
cmake --build build --target \
  comparison_esdf comparison_esdf_3d comparison_voxel_map \
  comparison_collision_check comparison_rrtstar_rewire esdf_mppi -j$(nproc)

# check current performance against baseline
python3 scripts/perf_check.py

# loosen tolerance for noisy hardware
python3 scripts/perf_check.py --tolerance 0.50

# refresh the baseline after an intentional speed change
python3 scripts/perf_check.py --update
```

## Benchmarks tracked
| Label | Binary | What we time |
|---|---|---|
| `esdf_2d_gpu_ms`        | `comparison_esdf`            | GPU JFA per 640K-cell ESDF |
| `esdf_3d_gpu_ms`        | `comparison_esdf_3d`         | GPU JFA-3D per 1.05M-voxel ESDF |
| `voxel_map_gpu_ms`      | `comparison_voxel_map`       | GPU 3D-DDA log-odds raycast per LiDAR scan |
| `collision_check_gpu_ms`| `comparison_collision_check` | GPU 2D-DDA per 1M candidate segments |
| `rrtstar_rewire_gpu_ms` | `comparison_rrtstar_rewire`  | GPU parallel rewire per 200K-node forest |
| `esdf_mppi_rollout_ms`  | `esdf_mppi`                  | GPU MPPI rollout iter (K=4096, T=30) with ESDF cost |

## CI integration
`.github/workflows/perf.yml` runs `perf_check.py` on a self-hosted
runner with the `gpu` label. GitHub-hosted Ubuntu runners do not have
NVIDIA GPUs, so the workflow is a no-op on them.

To enable PR-time perf checks:
1. Register a self-hosted runner labelled `gpu` with the repo.
2. Set the repository variable `PERF_RUNNER_AVAILABLE` to `true`.

The workflow also accepts manual triggering via `workflow_dispatch` so
you can run it on demand once the runner is online.

## Updating the baseline
The baseline file `scripts/perf_baseline.json` should be updated:
- when adding a new entry to the BENCHMARKS list,
- after intentional code changes that affect timing,
- when moving the perf runner to different hardware.

Always commit the baseline update in the same PR as the change that
caused it, with a note in the commit body explaining the delta.
