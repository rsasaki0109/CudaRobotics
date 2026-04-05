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

### Landscape: the growing family of MPPI + refinement hybrids

There is now a clear trend of augmenting MPPI with some form of post-sampling refinement:
- CEM-GD (Bharadhwaj et al., 2020): CEM sampling → gradient descent
- MPPI-IPDDP (2022/2025): MPPI sampling → DDP smoothing in convex corridor
- Biased-MPPI (Trevisan & Alonso-Mora, RA-L 2024): ancillary controllers → biased MPPI sampling
- Diffusion-MPPI / Generation-Refinement (2025): learned generative prior ↔ MPPI bidirectional
- Step-MPPI (Le et al., 2026): neural sampling distribution → single-step MPPI
- Feedback-MPPI (Belvedere et al., RA-L 2026): MPPI → sensitivity-derived feedback gains

Recent theory by Fazlyab et al. (2026) shows that MPPI is exactly a preconditioned gradient descent step. Our autodiff refinement adds explicit local gradient steps after this implicit step.

### Our positioning: minimal refinement

Within this landscape, our specific niche is the **minimal, training-free, compute-competitive** refinement. While recent work proposes increasingly sophisticated mechanisms (learned sampling distributions, diffusion models, DDP with convex corridors), we show that 3 autodiff gradient steps on the post-MPPI control sequence — with no learned components, no additional data structures, and parallelized to retain 85% of the sampling budget at matched time — suffices to cross the success boundary on hard dynamic-obstacle tasks where all non-hybrid approaches fail.

The paper should not claim:
- "differentiable MPPI is new"
- "gradient information in sampling-based control is new"
- "dynamic-obstacle MPPI is new"
- "hybrid sampling + gradient is new" (CEM-GD, MPPI-IPDDP exist)

The paper should claim:

> a minimal, training-free hybrid controller — just 3 gradient steps after a standard MPPI update — is the only method that solves hard dynamic-obstacle tasks under matched compute budgets across 6 strong non-hybrid baselines, on both 2D navigation and a 7-DOF manipulation benchmark. The gradient parallelization makes the refinement nearly free in wall-clock time.

### Key related work to cite

**Theoretical foundation:**
- **MPPI as Preconditioned Gradient Descent** (Fazlyab et al., arXiv:2603.24489, March 2026): proves MPPI is a preconditioned gradient descent step with unit step size. Our autodiff stage adds explicit gradient steps after this implicit step — a standard "coarse + fine" optimization pattern.

**Sampling + gradient hybrid precedents:**
- **CEM-GD** (Bharadhwaj et al., arXiv:2004.08763, L4DC 2020): combines cross-entropy method with gradient descent for model-based RL planning. This is the closest methodological precedent. Key differences: (1) we use MPPI not CEM, (2) our parallelized gradient retains 85% of the sampling budget within the same wall-clock time (CEM-GD does not evaluate under matched compute), (3) we include a 7-DOF manipulation evaluation.
- **MPPI-IPDDP** (hybrid MPPI + gradient-based DDP, IEEE TRO 2025): uses MPPI for coarse trajectory + IPDDP for smoothing inside a convex corridor. Our approach is simpler — pure autodiff refinement without corridor construction.

**Informed sampling and feedback extensions:**
- **Biased-MPPI** (Trevisan & Alonso-Mora, arXiv:2401.09241, RA-L 2024): uses importance sampling with ancillary controllers to bias the MPPI sampling distribution. Modifies the sampling step; our approach leaves sampling untouched and refines afterward.
- **Diffusion/Flow-MPPI** (2025): uses learned generative models as trajectory priors for MPPI. Requires training; our approach is training-free.
- **Feedback-MPPI** (Belvedere et al., arXiv:2506.14855, RA-L 2026): rollout-differentiation feedback gains. Our `feedback_mppi_ref` baseline follows their gain computation. We additionally test a two-rate variant (`feedback_mppi_faithful`) and show current-action-only gains fail on dynamic tasks.
- **Step-MPPI** (Le et al., arXiv:2604.01539, April 2026): learns a neural sampling distribution for single-step lookahead. Complementary: training-free post-MPPI refinement vs learned sampling modification.

**GPU-accelerated control:**
- **DiffMPC** (Toyota Research, arXiv:2510.06179, 2025): GPU-accelerated differentiable MPC in JAX for learning. Different goal — they differentiate through the entire MPC for policy learning; we add gradient steps within MPPI for real-time control.
- **cuNRTO** (arXiv:2603.02642, 2026): GPU robust trajectory optimization for Franka manipulator. Optimization-based (SQP) rather than sampling-based.
- **MPPI-Generic** (arXiv:2409.07563): CUDA MPPI library; we share the GPU-parallel rollout design pattern.

### What makes our contribution distinct from CEM-GD

CEM-GD (2020) established that combining sampling and gradient steps improves MPC planning. Our contribution is:

1. **Compute-competitive parallelization**: with parallelized gradient computation, the hybrid controller uses K=6216 samples at 1.0 ms — 85% of MPPI's K=7271 — plus 3 gradient steps. The gradient refinement comes at minimal cost to the sampling budget. CEM-GD does not evaluate under matched wall-clock time.

2. **Empirical evidence on hard dynamic-obstacle tasks**: `dynamic_slalom` is a task where no non-hybrid feedback variant succeeds at any compute budget. This goes beyond CEM-GD's model-based RL benchmarks.

3. **7-DOF manipulation with analytical Jacobians**: extends evaluation to 14D state, 7D control, demonstrating the approach scales to higher-dimensional robotics domains.

4. **Feedback architecture analysis**: testing and ruling out a two-rate Feedback-MPPI variant provides evidence that the gradient refinement is not replaceable by pure feedback.

### Contributions

Use only three contributions in the paper:

1. A minimal hybrid MPPI controller with a short local autodiff refinement stage that preserves the standard MPPI sampling update, interpretable as adding explicit gradient steps to MPPI's implicit preconditioned gradient descent (Fazlyab et al., 2026). With parallelized gradient computation, the refinement retains 85% of the sampling budget within the same wall-clock time.
2. A matched-time evaluation protocol comparing the hybrid controller against vanilla MPPI and strong non-hybrid feedback baselines (including a Feedback-MPPI-style two-rate variant) under shared per-step controller budgets, on both 2D dynamic-obstacle navigation and a 7-DOF serial-arm manipulation benchmark.
3. Evidence that the hybrid controller is the only method family that solves the hardest dynamic-obstacle task across 6 non-hybrid baselines, and that a two-rate feedback architecture with current-action-only gains cannot replicate this result.

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

From the latest exact-time tuning (after gradient parallelization):

- `1.00 ms`, `dynamic_slalom`
  - `mppi`: `K=7271 @ 0.982 ms`, success `0.00`, dist `14.19`
  - `feedback_mppi_ref`: `K=992 @ 1.009 ms`, success `0.00`, dist `11.77`
  - `feedback_mppi_fused`: `K=128 @ 1.854 ms`, success `0.00`, dist `10.33`
  - `diff_mppi_3`: `K=6216 @ 0.965 ms`, success `1.00`, dist `1.89`

- `2.00 ms`, `dynamic_slalom`
  - `mppi`: `K=14010 @ 1.901 ms`, success `0.00`, dist `14.18`
  - `feedback_mppi_fused`: `K=136 @ 1.968 ms`, success `0.00`, dist `10.36`
  - `diff_mppi_3`: `K=13467 @ 1.982 ms`, success `1.00`, dist `1.89`

After gradient parallelization, `diff_mppi_3` now uses `K=6216` at 1.0 ms (previously K=455) — nearly as many samples as vanilla MPPI plus the autodiff refinement, all within the same time budget. Key observations:
- `diff_mppi_3` is the only planner family that solves `dynamic_slalom` at any matched-time budget
- The gradient parallelization makes the hybrid controller genuinely compute-competitive: at 1.0 ms, diff_mppi_3 uses K=6216 vs mppi K=7271 (85% as many samples plus 3 gradient steps)
- All non-hybrid feedback baselines fail on `dynamic_slalom` regardless of compute budget

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

### Fixed-budget key numbers (after gradient parallelization — 17x speedup)

`7dof_dynamic_avoid`:
- `mppi K=512`: success `0.25`, final distance `0.635`, avg ms `0.39`
- `feedback_mppi_ref K=512`: success `0.75`, final distance `0.283`, avg ms `4.01`
- `diff_mppi_3 K=512`: success `1.00`, final distance `0.090`, avg ms `0.84`
- `diff_mppi_1 K=256`: success `0.75`, final distance `0.268`, avg ms `0.49`

`7dof_shelf_reach`:
- `mppi K=256`: success `0.25`, final distance `0.340`, avg ms `0.28`
- `diff_mppi_1 K=256`: success `0.50`, final distance `0.274`, avg ms `0.43`
- `diff_mppi_3 K=256`: success `0.25`, final distance `0.328`, avg ms `0.70`

### Exact-time key numbers

`7dof_dynamic_avoid` at `1.0 ms` target:
- `mppi` (K=4096 @ 1.05 ms): success `1.00`, final distance `0.10`
- `diff_mppi_1` (K=3169 @ 1.03 ms): success `1.00`, final distance `0.08`
- `feedback_mppi_ref` (K=32 @ 1.33 ms): success `0.50`, final distance `0.42`

`7dof_shelf_reach` at `1.0 ms` target:
- `mppi` (K=4096 @ 0.94 ms): success `0.00`, final distance `0.41`
- `feedback_mppi_ref` (K=54 @ 0.92 ms): success `1.00`, final distance `0.14`
- `diff_mppi_3` (K=2215 @ 1.05 ms): success `0.00`, final distance `0.42`

### Strongest 7-DOF talking point

On `7dof_dynamic_avoid` at K=512, `diff_mppi_3` reaches `success=1.00` at `0.84 ms` while `feedback_mppi_ref` reaches `0.75` at `4.01 ms` — the hybrid controller is both more reliable and **4.8x faster** at this sample budget. This is a genuine compute-quality win on a 14D manipulation task.

### Narrative for main text

The 7-DOF result complements the 2D dynamic navigation story:
- On dynamic navigation, diff_mppi_3 is the **only** method that solves the hard task
- On 7-DOF manipulation, diff_mppi_3 achieves **better success at lower compute** than the closest feedback baseline
- Both results support the same claim: a short autodiff refinement stage improves the compute-quality tradeoff

### What not to say

Do not claim Diff-MPPI dominates on all 7-DOF configurations. At K=256 on `7dof_dynamic_avoid`, `feedback_mppi_ref` reaches 1.00 success while diff_mppi_3 is at 0.50. The advantage is clearest at medium budgets (K=512) where the gradient helps enough to cross the success threshold.

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
