# Diff-MPPI Rebuttal Note

Date: 2026-04-14

This note is an internal reviewer-response prep document for the current `Diff-MPPI` submission package.
It is not meant to expand the claim.
It is meant to keep the response narrow, defensible, and consistent with the paper.

## Core Response In Two Sentences

The paper's main claim is not that Diff-MPPI dominates every controller on every domain.
The claim is that a minimal post-MPPI gradient refinement crosses the success boundary on a hard dynamic-obstacle task under shared wall-clock budgets where strong non-hybrid in-repo baselines still fail, while also transferring to a nontrivial 7-DOF manipulation setting.

## 1. "Is this just more compute?"

Short answer:

No. The central comparison is wall-clock matched, not just fixed-`K`.

Backing evidence:
- the main exact-time table is built around a shared `1.0 ms` controller budget
- at that budget on `dynamic_slalom`, `mppi`, `step_mppi`, `feedback_mppi_ref`, `feedback_mppi_paper`, and `feedback_mppi_fused` all fail, while `diff_mppi_3` succeeds
- the parallelized refinement still preserves most of the sampling budget: `diff_mppi_3 K=5966` vs `mppi K=7167`

Safe interpretation:

> The gain is not coming from escaping the time budget. It comes from spending a similar per-step budget differently: MPPI update first, then a short local gradient correction.

## 2. "Why separate the fixed-controller exact-time table from the multi-parameter sweep?"

Short answer:

Because they answer different questions.

Backing evidence:
- the fixed-controller table is the clean headline comparison for one concrete controller instance
- the later `--multi-param` sweep is a family-level robustness check over `K`, feedback gain scale, and Diff-MPPI gradient hyperparameters
- in that sweep, the best Diff family point can shift between `diff_mppi_3` and `diff_mppi_1` depending on the target

Safe interpretation:

> The fixed-controller table is the main paper result; the family-level sweep is robustness evidence that the qualitative non-hybrid failure pattern survives broader tuning.

## 3. "Why is MuJoCo not a main result?"

Short answer:

Because the current MuJoCo tasks are transfer checks, not decisive hybrid-only wins.

Backing evidence:
- `InvertedPendulum-v4` and the wider-reset variant can be solved by `mppi`, `feedback_mppi_ref`, and `diff_mppi_3` under matched `1.0-1.5 ms` tuning
- the `Reacher` terminal-heavy variant gives a hybrid-over-plain-MPPI split, but tuned feedback catches up

Safe interpretation:

> The MuJoCo results matter because they weaken the "custom benchmark only" criticism, not because they replace the main dynamic-obstacle claim.

## 4. "The 7-DOF exact-time result is mixed. What is the actual manipulation claim?"

Short answer:

The main manipulation claim comes from the medium-budget fixed-`K` result, not from universal exact-time dominance.

Backing evidence:
- on `7dof_dynamic_avoid @ K=512`, `diff_mppi_3` reaches `success=1.00` at `0.84 ms`
- at the same fixed budget, `feedback_mppi_ref` reaches `0.75` at `4.01 ms`
- in the later exact-time `3.0 / 5.0 ms` sweep, feedback catches up on `7dof_dynamic_avoid`
- `7dof_shelf_reach` remains weak for all methods

Safe interpretation:

> The 7-DOF benchmark shows that the method does not collapse in a higher-dimensional manipulation domain, and that there is at least one genuine compute-quality win there. It does not show uniform superiority.

## 5. "How close is `feedback_mppi_paper` to Feedback-MPPI?"

Short answer:

It is a closer in-repo proxy, not a full reproduction of the external controller stack.

Backing evidence:
- the repo now includes `feedback_mppi_ref`, `feedback_mppi_paper`, `feedback_mppi_faithful`, `feedback_mppi_cov`, `feedback_mppi_hf`, and `feedback_mppi_fused`
- the main paper explicitly calls these strong in-repo baselines rather than claiming a full literature-faithful reproduction
- even the closer variants still fail on `dynamic_slalom`

Safe interpretation:

> The point is not "we exactly reproduced Feedback-MPPI and beat it everywhere." The point is that several stronger feedback-style non-hybrid baselines were tested, and none closed the hard dynamic-task gap.

## 6. "What is actually new beyond CEM-GD or prior sampling-plus-gradient hybrids?"

Short answer:

The novelty claim is narrow: minimality, matched-time competitiveness, and the specific hard-task evidence.

Backing evidence:
- no claim that sampling + gradient is new in general
- no claim that differentiable MPPI is new in general
- the paper positions itself as a minimal training-free post-MPPI refinement
- the contribution is compute-competitive matched-time evidence on hard dynamic-obstacle tasks, plus a lightweight mechanism analysis

Safe interpretation:

> The paper is about a particularly small and practical hybrid design, not about inventing the whole hybrid category.

## 7. "Why should a reviewer care if the claim is this narrow?"

Short answer:

Because the narrow claim is still surprising and useful.

Backing evidence:
- the hard `dynamic_slalom` split survives both fixed-budget and matched-time views
- `step_mppi` shows that improving the sampling distribution alone is not enough
- `feedback_mppi_faithful` shows that a two-rate feedback architecture with current-action-only gains is also not enough
- the gradient freshness analysis gives a concrete mechanism story

Safe interpretation:

> The value is not broad domination. The value is a precise empirical finding: a very small post-sampling gradient stage can rescue a hard class of dynamic tasks that strong non-hybrid alternatives still fail under the same compute regime.

## 8. Fallback Narrowing If Reviews Push Hard

If the paper needs to be narrowed further, use this fallback framing:

> We present Diff-MPPI as a compute-quality tradeoff study of a minimal post-MPPI gradient refinement. The strongest evidence is a matched-time dynamic-obstacle result where non-hybrid baselines still fail, plus one fixed-budget 7-DOF manipulation win. We do not claim universal dominance or a complete replacement for MPPI.

## 9. Pointers

Use these files when drafting a response:
- `paper/diff_mppi_paper.md`
- `paper/diff_mppi_submission_draft.md`
- `paper/icra_iros_gap_list.md`
- `paper/diff_mppi_7dof_followup.md`
- `paper/diff_mppi_mujoco_followup.md`
- `build/benchmark_diff_mppi_exact_time_summary.md`
- `build/benchmark_diff_mppi_7dof_exact_time_summary.md`
