# Contact-Rich Diff-MPPI Paper Plan

Status: release evidence complete; artifact ledger ready; manuscript freeze
remains.

The machine-readable source of truth is
[`artifacts/contact_rich_diff_mppi.json`](artifacts/contact_rich_diff_mppi.json).
The ledger is content-addressed, validates with `--require-ready`, and supports
all submission-required claims. The generated results chapter is
[`contact_rich_diff_mppi_results.md`](contact_rich_diff_mppi_results.md).

## Frozen Evidence

| Block | Release scope | Main result |
|---|---|---|
| Contact robustness | 32,400 episodes; 12 conditions; 5 tasks; 6 planners; K={128,256,512}; 30 paired seeds | 33 Holm-significant positive and 6 negative cells versus MPPI |
| Exact matched compute | 375 calibration + 450 held-out episodes; disjoint seeds; enforced 10 ms slot | Diff-MPPI-3 reaches 0.800 real-time success; `contact_loss` delta +0.533, Holm p=0.000305 |
| External fidelity | 3,150 closed-loop MuJoCo 3.11.0 episodes over friction, mass, and observation-noise variations | 3 positive and 0 negative Holm-significant cells; aggregate Diff-MPPI-3 success 0.457 |

The primary artifacts are:

- [`contact_robustness_2026-07-28_report.md`](../docs/results/contact_robustness_2026-07-28_report.md)
- [`contact_matched_compute_2026-07-28_report.md`](../docs/results/contact_matched_compute_2026-07-28_report.md)
- [`contact_external_fidelity_2026-07-28_report.md`](../docs/results/contact_external_fidelity_2026-07-28_report.md)

All failed cells, the six statistically significant regressions on the
tall-box condition, and the zero-success detour family remain visible.

## Claim Discipline

The supported claim is localized: differentiable refinement improves selected
contact-rich, compute-matched cells and transfers to a declared closed-loop
MuJoCo plant. The evidence does not establish universal planner dominance.

SOPPI-fast contains one nominal gradient step, so it is not a pure SVGD or
sampling-only baseline. The MuJoCo block is a custom planar sim-to-sim transfer,
not a standard manipulator benchmark or real-robot result. All frozen results
were collected on one GTX 1660 Ti; independent-hardware replication remains
desirable.

## Remaining Submission Work

1. Freeze the manuscript narrative around the generated results chapter.
2. Generate final plots and tables only from the published CSVs.
3. Add artifact URLs/DOIs and the final anonymized reproduction entry point.
4. Run the full paper validation gate on the exact submission commit.

No new broad-performance claim should be added without first extending the
artifact ledger and its statistical contract.

## Reproduction

```bash
python3 scripts/validate_paper_artifacts.py \
  paper/artifacts/contact_rich_diff_mppi.json --require-ready
python3 scripts/render_contact_paper_results.py --check
```

The experiment protocols and exact release commands are documented in
[`contact_diff_mppi_robustness.md`](../docs/contact_diff_mppi_robustness.md),
[`contact_matched_compute.md`](../docs/contact_matched_compute.md), and
[`contact_external_fidelity.md`](../docs/contact_external_fidelity.md).
