# Contact-Rich Diff-MPPI Paper Plan

Status: pilot signal supported; submission package not ready.

The machine-readable source of truth is
[`artifacts/contact_rich_diff_mppi.json`](artifacts/contact_rich_diff_mppi.json).
It currently proves only the narrow fixed-seed signal:

- `box_align_contact_loss`, `K=256`, four checked-in seeds:
  `diff_mppi_3=4/4`, `soppi_fast=4/4`, `mppi=0/4`;
- `box_align_detour` remains a negative result:
  `diff_mppi_3=1/4`, `mppi=0/4`.

The current CSV is content-addressed and those values are recomputed by the
artifact validator. They are not enough for a paper-level robustness claim.

## Required Experiment Blocks

1. Contact robustness
   - at least 30 independently generated seeds;
   - multiple rollout budgets;
   - multiple object shapes, friction values, contact mobility/damping, and
     observation/control perturbations;
   - fixed seed lists and all failed cells retained.
2. Exact matched compute
   - calibrate each baseline to the same per-control-step wall-clock target;
   - report achieved p50/p95 latency and rollout count, not only requested
     budget;
   - include MPPI, Diff-MPPI-1/3, SOPPI, SOPPI-fast, and a strong non-hybrid
     contact baseline.
3. Fidelity transfer
   - repeat the frozen protocol in a declared higher-fidelity simulator or on a
     real manipulation platform;
   - retain simulator/robot version, contact settings, GPU, commit, commands,
     raw trajectories, videos, and failure logs.
4. Statistical package
   - confidence intervals for success and paired quality metrics;
   - effect sizes and multiple-comparison policy;
   - no aggregation across non-equivalent contact tasks.

## Claim Discipline

SOPPI-fast contains a nominal gradient step, so it must not be called pure
SVGD. The detour failure must remain visible. Until the three pending evidence
blocks pass, the paper may describe a promising fixed-seed contact signal but
not broad superiority, exact-time dominance, or external contact fidelity.

## Frozen Robustness Command

The first experiment block is now implemented as a resumable, content-addressed
32,400-episode protocol:

```bash
python3 scripts/run_contact_robustness.py \
  --output-dir build/contact_robustness_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile release
```

See [`../docs/contact_diff_mppi_robustness.md`](../docs/contact_diff_mppi_robustness.md).
The runner and validator are complete; the clean-commit release GPU run is
still pending and the claim remains `planned`.
