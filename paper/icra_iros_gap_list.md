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
- fixed-budget, cap-based, equal-time, and exact-time-tuned comparisons
- two dynamic scenarios, not just one
- an uncertainty follow-up with nominal-vs-actual obstacle mismatch on the dynamic pair
- a strengthened in-repo `feedback_mppi` baseline inside the same harness
- a closer rollout-sensitivity `feedback_mppi_sens` baseline inside the same harness
- a `grad_only_3` ablation that removes one weak alternative explanation
- two outside-domain pilots: nonlinear `CartPole` and dynamic-bicycle mobile navigation
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
- strengthened in-repo `feedback_mppi`
- rollout-sensitivity `feedback_mppi_sens`
- `grad_only_3`

What is still missing is a literature-faithful sensitivity-aware MPPI baseline, for example:
- a stronger `Feedback-MPPI`-style local feedback baseline
- another rollout-differentiation or local linearization baseline

The new `feedback_mppi_sens` comparison is a meaningful improvement over the earlier state of the repo:
- it derives feedback gains from rollout initial-state sensitivities instead of only from a nominal local linearization
- on `dynamic_crossing`, it reaches `0.75` success across `K={256,512,1024}` while vanilla MPPI remains at `0.00`
- on `dynamic_slalom`, it still fails, and it is still weaker than the simpler `feedback_mppi` baseline

So the baseline gap is narrower than before, but not closed.

This is the most dangerous missing experiment because a reviewer can reasonably say:

> The paper shows that hybrid search plus local sensitivity helps over vanilla MPPI and over stronger in-repo feedback controllers, but does not yet show whether the proposed implementation is actually better than existing sensitivity-aware MPPI variants.

That is a direct novelty threat, not just a "future work" point.

### 3. The experiment tier is still below flagship-conference expectations

Current evaluation is still:
- mostly 2D
- mostly kinematic
- hand-designed environments
- no hardware
- no standard public robotics benchmark

That does not make the work invalid.
It does make it harder to justify a main-track acceptance when nearby papers evaluate on:
- real robots
- high-fidelity locomotion
- 7-DOF manipulators
- uncertain dynamic environments with perception noise

There are now two partial exceptions:
- a pilot nonlinear CartPole benchmark outside the 2D navigation suite
- a dynamic-bicycle mobile-navigation pilot with steering lag and drag

That helps because the project is no longer purely a 2D kinematic story.
The dynamic-bicycle result is the more useful of the two for reviewer defense, because it stays in obstacle-avoidance planning while adding richer vehicle dynamics.
That pilot is now better than the earlier version because it also includes the closer `feedback_mppi_sens` baseline and a `1.80 ms` exact-time spot check, so the outside-domain story is no longer just `mppi` versus `Diff-MPPI`.
But neither pilot yet counts as the kind of stronger robotics-domain evaluation that fully closes this gap.

### 4. The matched-time story is much better, and now has direct tuning, but is still not complete

The project now has:
- cap-based comparisons
- equal-time nearest-match comparisons
- exact matched-time tuning on the current dynamic two-task suite
- exact matched-time tuning on the dynamic-bicycle follow-up pilot

That is already much better than most quick research repos.

But a skeptical reviewer can still say:
- the exact-time tuning currently searches `K` only, not the full controller design space
- the exact-time result is currently concentrated on the dynamic two-task suite, not the full benchmark portfolio
- outside the base suite, the time-matched claim currently reaches only a custom medium-fidelity dynamic-bicycle pilot

The dynamic-bicycle exact-time result is still useful, but it currently reads as a conservative compute-matched spot check:
- `mppi`, `feedback_mppi_sens`, and `diff_mppi_1` are all competitive on terminal distance at `1.80 ms`
- the stronger signal there is rollout-efficiency, not a decisive matched-time terminal-distance win

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

A lightweight analysis section now exists in the repo-level follow-up:
- `benchmark_diff_mppi` can emit per-step trace CSVs with sampled controls, refined controls, and local gradients
- `scripts/plot_diff_mppi_mechanism.py` produces correction-vs-episode, correction-vs-horizon, and success-vs-`K` figures
- on `dynamic_slalom @ K=1024`, the correction is strongly front-loaded, with early-horizon correction `0.018 -> 0.025` for Diff-MPPI versus late-horizon correction `0.001`

That partially addresses the empirical-only criticism, because it shows where the extra compute is going. But it is still a lightweight empirical mechanism check, not a deeper theoretical account.

## What Would Make This ICRA/IROS-Plausible

These are ordered by importance, not by ease.

### Tier 1: Must-Have

1. Strengthen the current direct sensitivity-aware baseline

Minimum acceptable version:
- keep the current nominal-linearization `feedback_mppi` and rollout-sensitivity `feedback_mppi_sens` baselines, but tighten the latter into a closer `Feedback-MPPI`-style comparison inside the same benchmark harness
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

Current status:
- partially addressed by the new `benchmark_diff_mppi_cartpole` pilot
- partially addressed again by the new `benchmark_diff_mppi_dynamic_bicycle` pilot
- not closed, because CartPole is still an underactuated control toy domain and the dynamic-bicycle benchmark is still a custom medium-fidelity pilot rather than a standardized high-fidelity robotics task

3. Extend the direct time-tuning protocol

Needed change:
- keep the new exact matched-time search in the final experimental package and extend it beyond `K`-only tuning on the current dynamic suite

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

Current status:
- partially addressed by the new `uncertain_crossing` and `uncertain_slalom` follow-up
- that follow-up uses seed-dependent obstacle time-offset, speed-scale, and lateral-offset mismatch while the planner still optimizes against the nominal obstacle motion
- it is still model mismatch, not yet observation noise, delayed sensing, or probabilistic prediction

5. Add one mechanism analysis

Examples:
- show that refinement reduces rollout count needed for a fixed success target
- show where the gradient step changes control relative to nominal MPPI
- plot success as a function of both `K` and grad-step count

Why this helps:
- it makes the contribution more than "we tried a variant and it looked better"

Current status:
- partially done via the new trace-based `dynamic_slalom` analysis
- still missing broader multi-task / multi-seed mechanism plots and a tighter causal account

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
4. literature-faithful `Feedback-MPPI`-style baseline beyond the current nominal-linearization `feedback_mppi` and rollout-sensitivity `feedback_mppi_sens`
5. exact matched-time tuning on the final evaluation suite
6. one higher-fidelity experiment outside 2D kinematic navigation

The new CartPole pilot means item `6` is now partially addressed, but not at the level that would make me remove it from the minimum list. Without item `4`, and without a stronger version of item `6`, and without carrying item `5` through the final evaluation suite, the paper is still too easy to down-score on originality and significance.

## Fastest Acceptable Paper Framing

If you do complete the missing items, the safest framing is not:

> We introduce a new differentiable MPPI algorithm.

That framing invites direct novelty attacks.

The safer framing is:

> We study a minimal hybrid MPPI controller that combines stochastic rollouts with a short autodiff refinement stage, and show that this combination improves compute-quality tradeoffs under matched real-time budgets.

That framing is narrower, but more defensible.

## Recommended Next Steps

Immediate next work:
1. Strengthen the current rollout-sensitivity `feedback_mppi_sens` baseline in `benchmark_diff_mppi` into a more literature-faithful comparison.
2. Port the benchmark to one higher-fidelity domain.
3. Carry the new exact matched-time tuning workflow into that stronger evaluation domain.

If time is limited:
1. aim for `workshop / late-breaking results / open-source systems demo`
2. keep collecting stronger evidence before attempting `ICRA/IROS` full paper

## My Current Judgment

As of `2026-04-02`, my judgment is:
- `today, as-is`: not strong enough for `ICRA/IROS` full paper
- `after one more literature-faithful baseline + one stronger experiment`: plausible
- `for workshop/demo right now`: yes

That is a conservative judgment, but it is the one most likely to survive actual review pressure.
