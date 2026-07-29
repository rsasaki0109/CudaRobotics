# Contact-Rich Diff-MPPI Release Results

This chapter is generated from the validated, content-addressed release artifacts. Statistical outcomes are reported independently of evidence integrity; negative and zero-success cells remain in the source tables.

## Evidence freeze

| Block | Episodes | Commit | GPU |
|---|---:|---|---|
| Robustness | 32,400 | `af4f0542d23e` | NVIDIA GeForce GTX 1660 Ti |
| Matched compute | 375 calibration + 450 held-out | `a6ca48ae09e4` | NVIDIA GeForce GTX 1660 Ti |
| MuJoCo transfer | 3,150 | `61b9f518ab09` | NVIDIA GeForce GTX 1660 Ti |

## Broad robustness

The fixed release matrix spans 12 plant conditions, five contact tasks, six planners, K={128,256,512}, and 30 paired seeds. It contains 33 Holm-significant positive and 6 Holm-significant negative success cells versus MPPI.

| Planner | Episodes | Success | Mean control ms |
|---|---:|---:|---:|
| diff_mppi_3 | 5,400 | 0.614 | 2.655 |
| diff_mppi_1 | 5,400 | 0.595 | 0.959 |
| soppi_fast | 5,400 | 0.557 | 1.464 |
| mppi | 5,400 | 0.453 | 0.120 |
| soppi | 5,400 | 0.446 | 0.426 |
| mppi_hardmodel | 5,400 | 0.331 | 0.159 |

The aggregate ordering favors Diff-MPPI-3, but the effect is not universal. All six Holm-significant negative cells are Diff-MPPI-3 on the tall-box condition. The detour task remains a visible negative control rather than being removed from the family.

## Exact 10 ms matched compute

Calibration and evaluation seeds are disjoint. Each planner selected K=1024 and received the same enforced 10 ms control slot. `real_time_success` requires task success and zero deadline misses.

| Planner | Held-out episodes | Real-time success | Deadline misses |
|---|---:|---:|---:|
| diff_mppi_3 | 150 | 0.800 | 1 |
| soppi_fast | 150 | 0.793 | 55 |
| mppi | 150 | 0.673 | 4 |

On `box_align_contact_loss`, Diff-MPPI-3 improves real-time success by +0.533 (95% bootstrap CI [+0.367, +0.700], Holm p=0.000305). SOPPI-fast improves it by +0.500 (Holm p=0.002472). The other four scenario families are not Holm-significant, and every planner remains at 0/30 on `box_align_detour`.

## Closed-loop MuJoCo transfer

The CUDA planners retain the nominal smooth rollout model while MuJoCo 3.11.0 executes every selected command and returns the next state. The matrix declares friction, mass, and observation-noise variations.

| Planner | Episodes | Success | Mean control ms |
|---|---:|---:|---:|
| diff_mppi_3 | 1050 | 0.457 | 2.614 |
| soppi_fast | 1050 | 0.340 | 1.432 |
| mppi | 1050 | 0.289 | 0.117 |

The full 70-cell family contains 3 Holm-significant positive and 0 negative cells. No individual observation-noise cell survives full-family Holm correction, so sensing-noise effects remain descriptive.

## Claim boundary

- These experiments support a contact-rich, compute-quality result; they do not establish universal planner dominance.
- SOPPI-fast contains one nominal gradient step and is not a pure sampling-only or pure-SVGD baseline.
- The MuJoCo task is a custom planar closed-loop sim-to-sim transfer, not a standard manipulator benchmark or real-robot result.
- All results are from one GTX 1660 Ti. Independent hardware replication is desirable but is not silently implied.

## Reproduction

```bash
python3 scripts/validate_paper_artifacts.py paper/artifacts/contact_rich_diff_mppi.json --require-ready
python3 scripts/render_contact_paper_results.py --check
```

Source artifacts:

- `docs/results/contact_robustness_2026-07-28_provenance.json`
- `docs/results/contact_matched_compute_2026-07-28_provenance.json`
- `docs/results/contact_external_fidelity_2026-07-28_provenance.json`
