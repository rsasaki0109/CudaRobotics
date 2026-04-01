# Diff-MPPI ICRA/IROS Gap List

Date: 2026-04-02

This note is a submission-oriented gap analysis for the current `Diff-MPPI` line.
It is written for an `ICRA/IROS full paper` decision, not for a workshop, demo, or repository release.

Important scope note:
- The venue judgment below is an inference from recent `ICRA/IROS`-adjacent MPPI papers and current flagship-conference expectations.
- It is not an official decision rule from the conference organizers.
- The closest official signal I found is that `ICRA 2026` had `5,088` submissions and reviewers are not required to inspect material beyond the submitted paper and optional video, which raises the bar for what must be obvious inside the main paper itself.

Primary references:
- `ICRA 2026 Calls for Papers and Posters`: https://2026.ieee-icra.org/contribute/
- `ICRA 2026 Papers - Submission closed`: https://2026.ieee-icra.org/contribute/call-for-icra-2026-papers-now-accepting-submissions/
- `ICRA 2026 Record Number of Submissions`: https://2026.ieee-icra.org/announcements/record-of-submissions/
- `PI-Net (2017)`: https://arxiv.org/abs/1706.09597
- `Differentiable MPC (2018)`: https://arxiv.org/abs/1810.13400
- `Safety in Augmented Importance Sampling / Robust MPPI (2022)`: https://arxiv.org/abs/2204.05963
- `Path Integral Control with Rollout Clustering and Dynamic Obstacles (2024)`: https://arxiv.org/abs/2403.18066
- `Chance-Constrained Sampling-Based MPC for Collision Avoidance in Uncertain Dynamic Environments / C2U-MPPI (2025)`: https://arxiv.org/abs/2501.08520
- `DRPA-MPPI (2025)`: https://arxiv.org/abs/2503.20134
- `Feedback-MPPI (2025)`: https://arxiv.org/abs/2506.14855
- `One-Step CDF-MPPI (2025)`: https://arxiv.org/abs/2509.00836

## Bottom Line

Current status:
- `Workshop / spotlight demo / open-source systems contribution`: strong
- `ICRA/IROS full paper, submitted today`: weak
- `ICRA/IROS full paper after one more literature-faithful baseline and one higher-fidelity experiment`: plausible

Short version:

> The project now has a defensible narrow empirical claim, but not yet a strong enough paper contribution for a flagship robotics main track.

The main reason is not that the results are bad.
The main reason is that the current paper would still look like:
- an incremental MPPI variant
- tested in 2D kinematic toy scenarios
- without a literature-faithful direct baseline

That combination usually struggles at `ICRA/IROS` unless the empirical evidence is unusually strong or the systems story is unusually concrete.

## What Is Already Good

The current line now has real positives:
- fixed-budget, cap-based, and equal-time comparisons
- two dynamic scenarios, not just one
- a simplified `feedback_mppi` baseline inside the same harness
- a `grad_only_3` ablation that removes one weak alternative explanation
- a narrow claim that is honest and empirically supported

The strongest current claim is:

> A minimal CUDA hybrid controller that augments vanilla MPPI with a short autodiff-based refinement stage reaches successful trajectories on two dynamic-obstacle tasks under matched per-step compute budgets where vanilla MPPI remains unsuccessful.

That is substantially better than the earlier single-scenario story.

## Why It Is Still Weak For ICRA/IROS Full Paper

### 1. The method contribution still looks incremental against nearby literature

Existing papers already cover much of the surrounding idea space:
- `PI-Net (2017)` already made path-integral control differentiable.
- `Differentiable MPC (2018)` already made optimization-based control layers differentiable.
- `Feedback-MPPI (2025)` adds sensitivity-derived local feedback to MPPI and evaluates on a quadruped and quadrotor.
- `Rollout Clustering + Dynamic Obstacles (2024)` and `DRPA-MPPI (2025)` already claim dynamic-obstacle improvements for MPPI.
- `C2U-MPPI (2025)` already pushes MPPI-style dynamic-obstacle avoidance into uncertain and real-world human-shared settings.
- `One-Step CDF-MPPI (2025)` combines distance-field gradients with MPPI and evaluates on high-dimensional manipulation.

Because of that, the current paper cannot rely on any of these claims alone:
- "MPPI but differentiable"
- "MPPI but better on dynamic obstacles"
- "MPPI plus gradient information"
- "GPU MPPI variant"

Those claims are too broad relative to the literature above.

### 2. The direct baseline story is better, but still incomplete

Right now you compare against:
- vanilla `mppi`
- simplified `feedback_mppi`
- `grad_only_3`

What is still missing is a literature-faithful sensitivity-aware MPPI baseline, for example:
- a stronger `Feedback-MPPI`-style local feedback baseline
- another rollout-differentiation or local linearization baseline

This is the most dangerous missing experiment because a reviewer can reasonably say:

> The paper shows that hybrid search plus local sensitivity helps over vanilla MPPI and over a simplified in-repo feedback controller, but does not yet show whether the proposed implementation is actually better than existing sensitivity-aware MPPI variants.

That is a direct novelty threat, not just a "future work" point.

### 3. The experiment tier is still below flagship-conference expectations

Current evaluation is still:
- 2D
- kinematic
- hand-designed environments
- no hardware
- no standard public robotics benchmark

That does not make the work invalid.
It does make it harder to justify a main-track acceptance when nearby papers evaluate on:
- real robots
- high-fidelity locomotion
- 7-DOF manipulators
- uncertain dynamic environments with perception noise

The current dynamic tasks are good for internal iteration, but they are not yet strong enough as the final evaluation layer.

### 4. The matched-time story is much better, but still not airtight

The project now has:
- cap-based comparisons
- equal-time nearest-match comparisons

That is already much better than most quick research repos.

But a skeptical reviewer can still say:
- equal-time matching is discrete, not fully optimized
- the controller families are not tuned under a common search budget
- the time-matched claim may depend on the sampled `K` grid

That is fixable, but right now it remains a vulnerability.

### 5. The paper contribution is still empirical-only

At the moment, the paper story is:
- we implemented a hybrid controller
- it works better on our tasks

What is still missing is a cleaner explanation of mechanism, for example:
- why the gradient stage helps specifically after sampling
- when the hybrid controller should beat pure MPPI
- when it should fail
- whether the gain comes from better sample efficiency, better local stabilization, or better obstacle timing

Even a lightweight analysis section would help.

## What Would Make This ICRA/IROS-Plausible

These are ordered by importance, not by ease.

### Tier 1: Must-Have

1. Strengthen the current direct sensitivity-aware baseline

Minimum acceptable version:
- upgrade the current simplified `feedback_mppi` baseline into a closer `Feedback-MPPI`-style comparison inside the same benchmark harness
- compare under fixed-budget and exact matched-time settings

Why this is critical:
- it is the cleanest answer to "is this actually new enough relative to nearby MPPI literature?"

2. Add one higher-fidelity evaluation domain

Best options:
- `7-DOF manipulator` with obstacle avoidance
- `Isaac`/high-fidelity mobile robot navigation with dynamics
- a small real robot demo if available

Why this is critical:
- it moves the paper out of "2D toy benchmark only"

3. Tighten time matching

Needed change:
- replace nearest-grid equal-time matching with a tuning procedure that chooses controller parameters to hit the same compute target more exactly

Why this matters:
- it makes the compute-quality claim much harder to dismiss

### Tier 2: Strongly Recommended

4. Add uncertainty, not only moving obstacles

Examples:
- obstacle state noise
- delayed obstacle observations
- randomized obstacle speeds
- mild model mismatch

Why this helps:
- `C2U-MPPI` and related work already push toward uncertain dynamic settings
- it raises the realism of the benchmark substantially

5. Add one mechanism analysis

Examples:
- show that refinement reduces rollout count needed for a fixed success target
- show where the gradient step changes control relative to nominal MPPI
- plot success as a function of both `K` and grad-step count

Why this helps:
- it makes the contribution more than "we tried a variant and it looked better"

### Tier 3: Nice-To-Have

6. Add a hardware or real-time deployment angle

Examples:
- onboard timing
- actual control frequency achieved on a robot computer
- deployment on a mobile base or manipulator

7. Add one standardized benchmark or dataset-style protocol

This does not need to be huge.
Even a small, reusable public benchmark protocol helps the paper look less bespoke.

## Minimum Submission Bar

If the goal is a serious `ICRA/IROS full paper`, the minimum package I would trust is:

1. Current static benchmark
2. Current two dynamic tasks
3. `grad_only_3` ablation
4. literature-faithful `Feedback-MPPI`-style baseline beyond the current simplified `feedback_mppi`
5. exact matched-time tuning
6. one higher-fidelity experiment outside 2D kinematic navigation

Without items `4-6`, the paper is still too easy to down-score on originality and significance.

## Fastest Acceptable Paper Framing

If you do complete the missing items, the safest framing is not:

> We introduce a new differentiable MPPI algorithm.

That framing invites direct novelty attacks.

The safer framing is:

> We study a minimal hybrid MPPI controller that combines stochastic rollouts with a short autodiff refinement stage, and show that this combination improves compute-quality tradeoffs under matched real-time budgets.

That framing is narrower, but more defensible.

## Recommended Next Steps

Immediate next work:
1. Strengthen the simplified `feedback_mppi` baseline in `benchmark_diff_mppi` into a more literature-faithful comparison.
2. Replace nearest-match equal-time analysis with exact matched-time tuning.
3. Port the benchmark to one higher-fidelity domain.

If time is limited:
1. aim for `workshop / late-breaking results / open-source systems demo`
2. keep collecting stronger evidence before attempting `ICRA/IROS` full paper

## My Current Judgment

As of `2026-04-02`, my judgment is:
- `today, as-is`: not strong enough for `ICRA/IROS` full paper
- `after one more literature-faithful baseline + one stronger experiment`: plausible
- `for workshop/demo right now`: yes

That is a conservative judgment, but it is the one most likely to survive actual review pressure.
