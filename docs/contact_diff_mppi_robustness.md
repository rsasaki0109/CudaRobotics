# Contact-Rich Diff-MPPI Robustness Suite

`scripts/run_contact_robustness.py` is the preregistered robustness runner for
the contact-rich Diff-MPPI paper. Its release profile executes 32,400 episodes:

| Axis | Release matrix |
|---|---|
| Seeds | 30 deterministic paired seeds per cell |
| K | 128, 256, 512 |
| Scenarios | `box_swivel`, `box_align_strict`, `box_align_detour`, `box_align_contact_loss`, `box_align_contact_arc` |
| Planners | MPPI, Diff-MPPI-1/3, SOPPI, SOPPI-fast, hard-model MPPI |
| True-plant conditions | nominal; gain 0.8/1.2; size 0.9/1.1; wide/tall aspect ratios; hard contact at friction 0.2/0.6/1.0; hard-contact damping 0.75/1.25 |

The controller keeps its nominal model while the true plant changes. The
`mppi_hardmodel` arm is retained as a model-exact hard-contact reference.

## Run

```bash
cmake --build build --target benchmark_diff_mppi_pushing_box -j$(nproc)
python3 scripts/run_contact_robustness.py \
  --output-dir build/contact_robustness_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile release
```

Release evidence requires a clean commit, an identified NVIDIA GPU, a
git-ignored or external output directory, and the complete fixed matrix. The
runner stages the exact benchmark binary into the evidence directory.

Each condition is one independently logged benchmark process. State is written
atomically after every attempt. An interrupted multi-hour run can continue
without overwriting or silently accepting partial data:

```bash
python3 scripts/run_contact_robustness.py \
  --output-dir build/contact_robustness_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile release \
  --resume
```

Resume is refused if the commit, binary SHA-256, matrix, or statistical policy
changes. A retry gets a new attempt-numbered CSV/log; failed attempts remain
visible.

Validate the complete directory independently:

```bash
python3 scripts/validate_contact_robustness.py \
  build/contact_robustness_release --profile release
```

## Statistics

The report includes:

- success rate with Wilson 95% intervals for every matrix cell;
- paired bootstrap intervals for success and final-distance differences using
  identical seeds;
- exact paired McNemar tests for success;
- Holm correction across the complete preregistered primary family
  (`diff_mppi_3` and `soppi_fast` versus MPPI);
- mean and p95 control latency;
- every failed episode and every negative comparison.

Statistical outcome is deliberately not an integrity gate. A complete,
well-formed experiment passes evidence validation even if the hypothesis fails.
The manifest reports positive and negative Holm-significant cells separately.

## Smoke

For an end-to-end runner check:

```bash
python3 scripts/run_contact_robustness.py \
  --output-dir build/contact_robustness_smoke \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile smoke
```

Smoke uses 12 episodes and cannot satisfy the paper robustness claim. A
2026-07-29 local GTX 1660 Ti smoke validated the full artifact chain: 12/12
episodes were retained; `box_align_contact_loss` gave Diff-MPPI-3 and
SOPPI-fast 2/2 versus MPPI 0/2, while every method remained 0/2 on
`box_align_detour`. This is runner validation, not statistical paper evidence.
