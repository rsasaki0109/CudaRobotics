# Diff-MPPI Main-Paper Draft

Date: 2026-04-03

This document is a submission-oriented draft for the strongest paper the current repository can support.
It is intentionally narrower than the repository itself.

The goal is not to describe every benchmark or every baseline variant.
The goal is to produce one paper argument that a reviewer can follow in a short read.

## Working Title

Candidate title:

> Diff-MPPI: A Lightweight Hybrid MPPI Controller with Local Autodiff Refinement Under Matched Compute Budgets

Safer alternative:

> A Lightweight Hybrid MPPI Controller with Local Autodiff Refinement Under Matched Compute Budgets

The safer title is better if we want to reduce novelty pushback.

## One-Sentence Claim

The entire paper should revolve around this sentence:

> A minimal hybrid controller that augments vanilla MPPI with a short autodiff refinement stage improves trajectory quality under matched per-step compute budgets beyond strong non-hybrid MPPI feedback baselines, especially on hard dynamic-obstacle tasks.

Everything that does not help this claim should move to appendix or be removed.

## What Goes In Main Text

Main text should contain only four empirical blocks:

1. Base dynamic suite
   - `dynamic_crossing`
   - `dynamic_slalom`

2. Exact-time evaluation
   - same two tasks
   - same matched-time protocol

3. Mechanism analysis
   - one figure from the trace workflow
   - show front-loaded correction and why gradient-only is insufficient

4. One stronger outside-domain benchmark
   - use the planar manipulator pilot
   - focus on `arm_static_shelf`
   - mention `arm_dynamic_sweep` as a harder quality-only follow-up

Everything else should move to appendix:
- old static 2D four-task suite
- CartPole pilot
- dynamic-bicycle pilot
- uncertainty follow-up
- all secondary baseline variants that do not make the closest comparison sharper

## What Should Stay In The Baseline Table

Do not show every in-repo proxy in the main paper.
That reads like exploration, not a final claim.

Main table should keep only:
- `mppi`
- `feedback_mppi_ref`
- `feedback_mppi_cov`
- `diff_mppi_3`

Optional fifth row if space permits:
- `feedback_mppi_fused`

Do not put these in the main table:
- `feedback_mppi_release`
- `feedback_mppi_hf`
- `feedback_mppi_sens`
- `grad_only_3`

Those are useful rebuttal or appendix baselines, but they dilute the paper story.

## Why These Four Rows

`mppi` is the default baseline.

`feedback_mppi_ref` is the closest released-gain proxy currently in the repo.
It is the right answer to "what if a reviewer asks for a Feedback-MPPI-style comparison?"

`feedback_mppi_cov` is the strongest lighter non-hybrid feedback baseline in the outside-domain manipulator pilot and still competitive in dynamic navigation.

`diff_mppi_3` is the clearest final hybrid method.
`diff_mppi_1` is useful for appendix ablation, not the main story.

## Abstract Draft

Sampling-based controllers such as MPPI remain attractive for nonlinear obstacle-avoidance control, but their solution quality can degrade under tight rollout budgets. We study a lightweight hybrid controller that first performs a standard MPPI sampling update and then applies a short local autodiff refinement to the control sequence. The resulting controller is intentionally minimal: it does not differentiate through the full sampling process and it preserves the original MPPI update as the dominant planning step. We evaluate the method against vanilla MPPI and stronger non-hybrid feedback MPPI baselines under fixed rollout budgets and exact matched-time controller budgets. On two dynamic-obstacle navigation tasks, the hybrid controller remains successful under matched per-step compute budgets where vanilla MPPI remains unsuccessful, while strong non-hybrid feedback baselines reduce but do not close the gap on the harder task. We further evaluate the method on a planar manipulator obstacle-avoidance pilot, where a current-action feedback baseline and a covariance-feedback baseline both outperform vanilla MPPI on a static shelf-reaching task. These results support a narrow empirical claim: a short autodiff refinement stage can improve the compute-quality tradeoff of MPPI beyond strong non-hybrid feedback variants, particularly on hard dynamic-obstacle tasks.

## Introduction Draft

### Problem framing

MPPI is widely used because it handles nonlinear dynamics and nonconvex costs with a simple rollout-based update.
However, under limited rollout budgets, vanilla MPPI can produce trajectories that are qualitatively close to useful behavior without reaching a successful control sequence.

This is the regime we target.
We do not claim to replace MPPI.
We study whether a very short local refinement stage can improve the control sequence after the MPPI update without discarding the basic MPPI controller structure.

### Positioning

The paper should not claim:
- "differentiable MPPI is new"
- "gradient information in sampling-based control is new"
- "dynamic-obstacle MPPI is new"

The paper should claim:

> a minimal CUDA hybrid controller, built on top of a plain MPPI update and a short local autodiff refinement, improves trajectory quality under matched compute budgets beyond strong non-hybrid feedback MPPI baselines on hard dynamic-obstacle tasks.

### Contributions

Use only three contributions in the paper:

1. A minimal hybrid MPPI controller with a short local autodiff refinement stage that preserves the standard MPPI sampling update.
2. A matched-time evaluation protocol that compares the hybrid controller against vanilla MPPI and strong non-hybrid feedback MPPI baselines under shared per-step controller budgets.
3. Evidence across dynamic-obstacle navigation and a planar manipulator obstacle-avoidance pilot that the hybrid controller improves the compute-quality tradeoff, while clarifying the limits of closer non-hybrid feedback baselines.

## Method Draft

### Controller structure

Describe the method in exactly three steps:

1. Sample rollouts around the current nominal control sequence and perform the standard MPPI weighted update.
2. Roll out the updated nominal sequence and differentiate the trajectory cost with respect to the nominal controls using a lightweight backward pass.
3. Apply a small number of local gradient steps to the control sequence before execution.

Important wording:
- say "local refinement"
- say "short refinement stage"
- say "post-MPPI refinement"
- do not say "end-to-end differentiable MPPI layer"

### What makes it lightweight

Be explicit:
- the MPPI update is unchanged
- the refinement uses only a few gradient steps
- the backward pass is local to the post-update nominal sequence
- the method is compared under exact matched-time budgets

### Baseline language

State clearly that the repo includes multiple in-house feedback proxies, but the main paper keeps only the closest and strongest non-hybrid ones.
This reads as deliberate curation instead of uncontrolled benchmark growth.

## Experimental Section Draft

### Main suite

Main suite should be:
- `dynamic_crossing`
- `dynamic_slalom`

Main planners:
- `mppi`
- `feedback_mppi_ref`
- `feedback_mppi_cov`
- `diff_mppi_3`

Optional appendix planners:
- `feedback_mppi_fused`
- `feedback_mppi_hf`
- `feedback_mppi_release`
- `feedback_mppi_sens`
- `grad_only_3`
- `diff_mppi_1`

### Main result table

The main table should have two blocks:

Block A: fixed-budget, `K in {256, 512, 1024}`
- success
- final distance
- average control ms

Block B: exact-time, targets `{1.0, 1.5, 2.0} ms`
- success
- final distance
- matched `K`
- measured control ms

### Main narrative

The main narrative should be:

1. Vanilla MPPI fails on both dynamic tasks at low and medium budgets.
2. `feedback_mppi_ref` recovers the easier `dynamic_crossing` task under both fixed-budget and exact-time tuning.
3. Stronger non-hybrid feedback baselines reduce terminal error further, but still do not solve `dynamic_slalom`.
4. `diff_mppi_3` remains the only controller family that consistently succeeds on both tasks.

That sequence is easy for reviewers to follow.

## Key Numbers To Use

These are the cleanest current numbers for the paper story.

### Dynamic navigation, fixed budget

From the current dynamic follow-up and gap-closure runs:
- `dynamic_crossing`, `mppi K=256`: success `0.00`, final distance about `3.04`
- `dynamic_crossing`, `feedback_mppi_ref K=256`: success `1.00`, final distance about `1.90`
- `dynamic_crossing`, `feedback_mppi_release K=256`: success `1.00`, final distance about `1.86`
- `dynamic_crossing`, `feedback_mppi_fused K=256`: success `1.00`, final distance about `1.87`
- `dynamic_crossing`, `diff_mppi_3 K=256`: success `1.00`, final distance about `1.91`

- `dynamic_slalom`, `mppi K=256`: success `0.00`, final distance about `14.33`
- `dynamic_slalom`, `feedback_mppi_ref K=256`: success `0.00`, final distance about `11.87`
- `dynamic_slalom`, `feedback_mppi_cov K=256`: success `0.00`, final distance about `11.49`
- `dynamic_slalom`, `feedback_mppi_fused K=256`: success `0.00`, final distance about `10.28`
- `dynamic_slalom`, `diff_mppi_3 K=256`: success `1.00`, final distance about `1.89`

The main sentence should be:

> stronger non-hybrid feedback closes much of the easy-task gap, but the hard dynamic-slalom success split remains unique to the hybrid controller.

### Dynamic navigation, exact time

From the latest exact-time tuning (including `feedback_mppi_faithful`):

- `1.00 ms`, `dynamic_crossing`
  - `mppi`: `K=7404 @ 0.979 ms`, success `0.00`, dist `3.02`
  - `feedback_mppi_ref`: `K=1218 @ 0.998 ms`, success `1.00`, dist `1.93`
  - `feedback_mppi_faithful`: `K=3584 @ 0.994 ms`, success `0.00`, dist `2.90`
  - `diff_mppi_3`: `K=1279 @ 0.989 ms`, success `1.00`, dist `1.85`

- `1.00 ms`, `dynamic_slalom`
  - `mppi`: `K=7513 @ 0.987 ms`, success `0.00`, dist `14.16`
  - `feedback_mppi_ref`: `K=1180 @ 1.020 ms`, success `0.00`, dist `11.87`
  - `feedback_mppi_faithful`: `K=3403 @ 0.976 ms`, success `0.00`, dist `14.09`
  - `diff_mppi_3`: `K=455 @ 1.012 ms`, success `1.00`, dist `1.91`

- `2.00 ms`, `dynamic_slalom`
  - `mppi`: `K=14777 @ 1.993 ms`, success `0.00`, dist `14.15`
  - `feedback_mppi_ref`: `K=3424 @ 1.998 ms`, success `0.00`, dist `11.90`
  - `feedback_mppi_faithful`: `K=7930 @ 1.994 ms`, success `0.00`, dist `14.03`
  - `diff_mppi_3`: `K=8361 @ 1.993 ms`, success `1.00`, dist `1.92`

This is the cleanest matched-time claim in the repository. Key observations:
- `diff_mppi_3` is the only planner family that solves `dynamic_slalom` at any matched-time budget
- `feedback_mppi_faithful` performs at MPPI level on `dynamic_slalom`, confirming the two-rate architecture failure
- On the easier `dynamic_crossing`, both `feedback_mppi_ref` and `diff_mppi_3` succeed, with `diff_mppi_3` slightly better on terminal distance

### Outside-domain manipulator

Use the static shelf task in the main text and dynamic sweep as supporting text.

Fixed-budget shelf result:
- `arm_static_shelf`, `mppi K=256`: success `0.00`, final distance `0.23`
- `arm_static_shelf`, `feedback_mppi_cov K=256`: success `1.00`, final distance `0.15`, avg ms `2.65`
- `arm_static_shelf`, `feedback_mppi_ref K=256`: success `1.00`, final distance `0.15`, avg ms `1.90`
- `arm_static_shelf`, `diff_mppi_1 K=256`: success `0.75`, final distance `0.16`

Dynamic sweep quality result:
- `arm_dynamic_sweep`, `mppi K=256`: final distance `0.33`
- `arm_dynamic_sweep`, `feedback_mppi_cov K=256`: final distance `0.29`
- `arm_dynamic_sweep`, `feedback_mppi_ref K=256`: final distance `0.30`
- `arm_dynamic_sweep`, `diff_mppi_1 K=256`: final distance `0.30`

This should not be oversold as a decisive manipulation win.
It should be presented as a stronger outside-domain pilot that confirms the method does not collapse immediately outside the navigation suite.

## 7-DOF Manipulator Results (New)

A Panda-like 7-DOF serial-arm benchmark with 14D state, 7D control, 3D workspace obstacles, and analytical dynamics Jacobians.

### Two scenarios

- `7dof_shelf_reach`: reach a target while avoiding a static workspace obstacle
- `7dof_dynamic_avoid`: reach a target while avoiding a moving 3D obstacle

### Fixed-budget key numbers

`7dof_dynamic_avoid`:
- `mppi K=256`: success `0.75`, final distance `0.273`, avg ms `0.38`
- `feedback_mppi_ref K=256`: success `1.00`, final distance `0.090`, avg ms `3.47`
- `diff_mppi_3 K=256`: success `0.50`, final distance `0.465`, avg ms `5.71`

`7dof_shelf_reach`:
- `mppi K=256`: success `0.25`, final distance `0.340`, avg ms `0.33`
- `diff_mppi_1 K=256`: success `0.50`, final distance `0.331`, avg ms `1.44`
- `diff_mppi_3 K=256`: success `0.50`, final distance `0.259`, avg ms `3.63`

### Exact-time key numbers (new)

`7dof_shelf_reach` at `3.0 ms` target:
- `mppi` (K=4096 @ 0.97 ms): success `0.00`, final distance `0.41`
- `diff_mppi_3` (K=32 @ 3.53 ms): success `1.00`, final distance `0.14`

This is a strong result: the hybrid controller finds the obstacle-free path at very low K (32 samples) where high-K pure sampling fails. The gradient refinement compensates for the small sample budget.

### Narrative for main text

Use `7dof_dynamic_avoid` as the main outside-domain result showing that feedback-MPPI extends to high-DOF manipulation. Use `7dof_shelf_reach` exact-time result as a supporting finding where the hybrid controller outperforms at matched compute.

### What not to say

Do not claim Diff-MPPI dominates on all 7-DOF tasks. `feedback_mppi_ref` wins on `7dof_dynamic_avoid` and that should be acknowledged honestly.

## feedback_mppi_faithful Finding (New)

A `feedback_mppi_faithful` variant was tested on the base dynamic navigation suite. It combines the released current-action gain computation with a two-rate controller architecture (replan every 2 steps, local feedback between replans).

Result: fails on both `dynamic_crossing` and `dynamic_slalom` even at K=8192 (2.1 ms/step).

Comparison:
- `feedback_mppi_ref` (every-step replan): success `1.00` on `dynamic_crossing` at K=256 (0.60 ms)
- `feedback_mppi_faithful` (stride=2 replan): success `0.00` on `dynamic_crossing` at K=8192 (2.06 ms)
- `diff_mppi_3` (hybrid): success `1.00` on both tasks at K=256 (0.91 ms)

This finding belongs in the paper as evidence that the two-rate feedback architecture with current-action-only gains is insufficient for dynamic-obstacle tasks. The autodiff refinement provides complementary value that pure feedback cannot replicate.

## Mechanism Figure Draft

Use one figure only.

Recommended figure:
- `dynamic_slalom` correction-vs-horizon
- `dynamic_slalom` success-vs-K

Main explanation:
- correction magnitude is front-loaded in the horizon
- the refinement sharpens near-term controls rather than rewriting the whole plan
- gradient-only is not enough
- non-hybrid feedback reduces terminal error but does not cross the hard-task success boundary

That is enough for a mechanism section.

## What To Put In Appendix

Appendix A:
- full baseline zoo
- `feedback_mppi_release`
- `feedback_mppi_hf`
- `feedback_mppi_sens`
- `feedback_mppi_fused`
- `grad_only_3`

Appendix B:
- 7-DOF manipulator full results (fixed-budget and exact-time)
- feedback_mppi_faithful two-rate architecture analysis
- uncertainty follow-up
- dynamic-bicycle pilot
- CartPole pilot

Appendix C:
- additional exact-time sweeps
- search traces

Appendix D:
- implementation details and CUDA kernels

## Limitations Draft

Keep limitations short and direct.

Recommended limitations paragraph:

The current contribution is empirical and intentionally narrow. The closest feedback baselines in the repository are strong in-repo proxies, including a two-rate variant tested under the released gain computation, but they are not a full paper-faithful reproduction of the complete sensitivity-aware MPPI controller stack. The outside-domain evaluation now spans a 7-DOF manipulator with 3D workspace obstacles, but it still relies on custom benchmark domains rather than standardized suites like MuJoCo manipulation tasks or Isaac Gym environments. Accordingly, we position the paper as a compute-quality tradeoff study of a lightweight hybrid controller, not as a definitive replacement for MPPI.

## Reviewer-Facing Framing

If the paper is written from the current evidence, the intended reviewer reaction should be:

- the claim is narrow
- the evidence is careful
- the matched-time comparison is real
- the 7-DOF evaluation is non-trivial (14D state, 3D obstacles)
- the two-rate feedback analysis is informative
- the paper knows its limits
- the hard dynamic task split is interesting

That is the path to `accept`.
The path to `strong accept` still needs one stronger standardized benchmark or one truly literature-faithful full-stack baseline reproduction.

## Immediate Next Writing Step

Turn this draft into four files:

1. `paper/diff_mppi_abstract.md`
2. `paper/diff_mppi_intro.md`
3. `paper/diff_mppi_method.md`
4. `paper/diff_mppi_experiments.md`

If time is limited, do not split yet.
Submit from this single draft first and only modularize later.
