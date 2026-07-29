# Contact-Rich Diff-MPPI Paper Plan

Status: release evidence complete; artifact ledger ready; submission narrative
frozen and machine-checked.

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

## Remaining Submission Packaging

The authoritative Markdown submission draft is
[`diff_mppi_submission_draft.md`](diff_mppi_submission_draft.md). Its title,
claim/status table, frozen numerical boundaries, and local links are checked
against the ready ledger in CTest.

1. Convert the frozen Markdown narrative to the selected venue template.
2. Select the final anonymous artifact URL/DOI.
3. Assemble and validate the bundle on the exact submission commit.

The final robustness, matched-compute, and external-fidelity plots are now
generated only from the frozen published CSVs. The renderer writes PDF, SVG,
PNG, and a source-hash/semantic manifest:

```bash
python3 scripts/render_contact_submission_figures.py \
  --output-dir build/contact_submission_figures
```

After choosing the venue and anonymous artifact entry point, create the
portable package from a clean commit:

```bash
python3 scripts/assemble_contact_submission_bundle.py \
  --output-dir build/contact_submission_bundle \
  --venue VENUE \
  --artifact-url https://ANONYMOUS_ARTIFACT_URL
python3 scripts/validate_contact_submission_bundle.py \
  build/contact_submission_bundle/submission_manifest.json \
  --commit "$(git rev-parse HEAD)" --require-ready
```

The assembler copies the frozen manuscript, generated results, protocols,
published CSVs/reports, and all three figure formats. It rewrites absolute
machine paths in the two ledger provenance payloads, updates their hashes in
an anonymous ledger, and reruns every claim assertion inside the portable
bundle. The validator then reopens every file, checks its size and SHA-256,
revalidates the anonymous ledger, binds the figure sources, scans for identity
leaks, and refuses `ready: true` until the venue, clean commit, and HTTPS
artifact entry point are final.

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
