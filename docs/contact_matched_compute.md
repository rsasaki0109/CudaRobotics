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

Publish the compact checked-in evidence bundle only after validation:

```bash
python3 scripts/publish_contact_matched_compute.py \
  build/contact_matched_compute_release \
  --output-dir docs/results \
  --profile release
```

The publication contains calibration, summary, comparisons, report, and a
provenance document. The provenance binds the larger calibration/evaluation
episode tables, raw attempts, staged binary, commit, GPU, and statistical
outputs by SHA-256 without copying those large runtime artifacts into Git.

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

## Release result

The 2026-07-28 UTC release run completed 375 calibration and 450 held-out
evaluation episodes on a GTX 1660 Ti. All three planners selected K=1024 and
were evaluated with the same enforced 10 ms slot. Aggregate real-time success
was 0.800 for Diff-MPPI-3, 0.673 for MPPI, and 0.793 for SOPPI-fast.

On `box_align_contact_loss`, Diff-MPPI-3 reached 1.00 versus MPPI at 0.467
(paired delta +0.533, Holm p=0.000305); SOPPI-fast reached 0.967
(+0.500, Holm p=0.002472). The other scenarios were not Holm-significant, and
`box_align_detour` remained 0/30 for every planner. Evaluation-time overruns
remain visible: MPPI recorded four deadline misses, Diff-MPPI-3 one, and
SOPPI-fast 55. See
[`contact_matched_compute_2026-07-28_report.md`](results/contact_matched_compute_2026-07-28_report.md)
and the adjacent provenance JSON.
