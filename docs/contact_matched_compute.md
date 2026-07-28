# Deadline-Matched Contact Control

This experiment tests Diff-MPPI and SOPPI-fast against vanilla MPPI under an
enforced, identical wall-clock control period. It is separate from the fixed-K
robustness suite: fixed K measures algorithmic behavior at equal samples, while
this protocol measures held-out task success under a real-time deadline.

## Real-time contract

`benchmark_diff_mppi_pushing_box` accepts:

```text
--control-deadline-ms <positive milliseconds>
--seed-offset <non-negative index>
```

At every control step, computation starts at the slot boundary. A result that
finishes early waits until the shared deadline. An overrun is retained as a
deadline miss. The CSV reports raw task success and the stricter
`real_time_success`, which requires task success and zero deadline misses.
Compute latency excludes the enforced wait; `avg_control_slot_ms` includes it.

The default value is zero, so existing fixed-K experiments neither wait nor
change their task-success semantics.

## Preregistered release protocol

The release profile uses a 10 ms period and two disjoint seed sets:

| Stage | Seeds | Purpose |
|---|---:|---|
| Calibration | 5 per scenario/planner/K, offset 0 | Select the largest registered K with no deadline miss |
| Evaluation | 30 per scenario/planner, offset 100 | Held-out comparison at the frozen selected K |

Registered K candidates are 64, 128, 256, 512, and 1024. The scenarios are
`box_swivel`, `box_align_strict`, `box_align_detour`,
`box_align_contact_loss`, and `box_align_contact_arc`. MPPI, Diff-MPPI-3, and
SOPPI-fast are compared. If even K=64 misses the calibration deadline for a
planner, the release run fails instead of silently dropping that planner.

Run:

```bash
python3 scripts/run_contact_matched_compute.py \
  --output-dir build/contact_matched_compute_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile release
```

Interrupted runs preserve attempt-numbered CSV and log files:

```bash
python3 scripts/run_contact_matched_compute.py \
  --output-dir build/contact_matched_compute_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile release \
  --resume
```

Validate independently:

```bash
python3 scripts/validate_contact_matched_compute.py \
  build/contact_matched_compute_release --profile release
```

The validator regenerates the selected budgets, summary, comparisons, and
report from the bound episode tables. It also checks the clean commit, staged
binary, GPU identity, raw attempt hashes, complete matrices, disjoint seed
ranges, and experiment identity.

## Statistics

Every scenario/planner cell reports real-time success with a Wilson 95%
interval, deadline misses, latency, and final distance. Comparisons use
seed-index-paired bootstrap intervals and exact McNemar tests with Holm
correction over the complete declared family. Failures and negative results
remain in the output; hypothesis outcome is not an integrity gate.
