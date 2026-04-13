# Diff-MPPI Submission Checklist

Date: 2026-04-14

This note is the final pre-submit checklist for the current `Diff-MPPI` package.
It is intentionally narrow: the goal is to freeze one reviewer-safe paper, not to keep expanding the experiment scope.

## 1. Main Claim Freeze

The paper should make only this claim:

> A minimal, training-free post-MPPI gradient refinement improves the compute-quality tradeoff on hard dynamic-obstacle tasks, and the effect survives exact matched-time comparison against strong non-hybrid in-repo baselines.

The paper should not claim:
- a fully literature-faithful reproduction of the complete Feedback-MPPI controller stack
- universal exact-time dominance across all manipulation settings
- a decisive MuJoCo win
- novelty of "sampling + gradient" in general

## 2. Headline Results To Keep In Main Paper

### Dynamic navigation

Use these as the submission-critical points:
- fixed-budget `dynamic_slalom @ K=1024`: `diff_mppi_3 = success 1.00, dist 1.91, 0.29 ms`
- exact-time `dynamic_slalom @ 1.0 ms` fixed-controller table:
  - `mppi K=7167 @ 0.99 ms, success 0.00, dist 14.15`
  - `step_mppi K=7422 @ 0.99 ms, success 0.00, dist 14.10`
  - `feedback_mppi_ref K=949 @ 1.01 ms, success 0.00, dist 11.86`
  - `feedback_mppi_paper K=129 @ 1.02 ms, success 0.00, dist 11.61`
  - `feedback_mppi_fused K=128 @ 1.88 ms, success 0.00, dist 10.33`
  - `diff_mppi_3 K=5966 @ 0.99 ms, success 1.00, dist 1.90`

### 7-DOF manipulation

Keep only the strongest fixed-budget result in main text:
- `7dof_dynamic_avoid @ K=512`
  - `mppi = success 0.25, dist 0.635, 0.39 ms`
  - `feedback_mppi_ref = success 0.75, dist 0.283, 4.01 ms`
  - `diff_mppi_3 = success 1.00, dist 0.090, 0.84 ms`

### Mechanism

Keep the narrow interpretation:
- the gradient correction is front-loaded on dynamic tasks
- the method is not just "more compute in general"
- `step_mppi` and `feedback_mppi_faithful` both fail to reproduce the `dynamic_slalom` win

## 3. What Belongs In Appendix / Follow-Up

Do not overload the main paper with these:
- family-level `--multi-param` matched-time robustness sweeps
- full 7-DOF exact-time `3.0 / 5.0 ms` tables
- MuJoCo `InvertedPendulum-v4` exact-time tables
- MuJoCo `Reacher` terminal-heavy sweep details
- dynamic-bicycle, cartpole, uncertainty, and planar-manipulator follow-ups

These are still valuable because they answer reviewer concerns, but they are support material, not the headline story.

## 4. Final Text Checks

- Abstract says `fixed-budget sweeps and exact matched-time comparisons`, not only one of them.
- Introduction does not oversell novelty over CEM-GD, MPPI-IPDDP, or Feedback-MPPI.
- Experimental setup distinguishes `main exact-time table` from `family robustness sweep`.
- Limitations explicitly say the feedback baselines are strong in-repo proxies, not full-stack reproductions.
- Limitations explicitly say MuJoCo is a transfer / standardization check, not a decisive hybrid-only win.
- 7-DOF text says the exact-time sweep is mixed and the main manipulation point is the fixed-budget `K=512` result.

## 5. Numerical Sanity Checks

Before submission, verify these files still agree:
- `paper/diff_mppi_paper.md`
- `paper/latex/diff_mppi.tex`
- `paper/diff_mppi_submission_draft.md`
- `build/exact_time_1ms.csv`
- `build/benchmark_diff_mppi_exact_time_summary.md`
- `build/benchmark_diff_mppi_7dof_exact_time_summary.md`

If a number differs between docs:
- prefer the fixed-controller numbers already frozen in `build/exact_time_1ms.csv` for the main table
- prefer the later `--multi-param` summaries only for robustness / appendix wording

## 6. Build / Packaging Checks

Run:

```bash
python3 -m py_compile scripts/tune_diff_mppi_time_targets.py
python3 scripts/generate_paper_figures.py --csv build/benchmark_full_final.csv --pareto-csv build/paper_pareto_exact_time.csv
cd paper/latex && make
```

Expected:
- `paper/latex/diff_mppi.pdf` builds successfully
- page count remains 5
- only non-fatal `Underfull \hbox` warnings remain

## 7. Reviewer-Risk Checklist

Be ready for these objections:
- "Why is MuJoCo not a main result?"
- "Why separate the fixed-controller table from the multi-parameter matched-time sweep?"
- "Why is the 7-DOF exact-time result mixed?"
- "How close is `feedback_mppi_paper` to the external method?"
- "Is this just more compute?"

Prepared answers belong in `paper/diff_mppi_rebuttal_note.md`.

## 8. Final Recommendation

If time is limited, stop adding experiments.
Freeze the current package, submit the narrow paper, and keep MuJoCo / 7-DOF exact-time / follow-up material for rebuttal and artifact support.
