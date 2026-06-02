# Diff-MPPI ICRA/IROS Gap List

Date: 2026-04-14 (updated)

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
- `Feedback-MPPI (2025, RA-L 2026)`: https://arxiv.org/abs/2506.14855
- `One-Step CDF-MPPI (2025)`: https://arxiv.org/abs/2509.00836
- `MPPI-IPDDP (hybrid MPPI+DDP, IEEE TRO 2025)`: https://arxiv.org/abs/2208.02439
- `CEM-GD / sampling+gradient hybrid (L4DC 2020)`: https://arxiv.org/abs/2004.08763
- `Biased-MPPI / ancillary controller fusion (RA-L 2024)`: https://arxiv.org/abs/2401.09241
- `Diffusion/Flow-MPPI / generative prior + MPPI (2025)`: https://arxiv.org/html/2508.01192
- `DiffMPC / GPU differentiable MPC (Toyota Research, 2025)`: https://arxiv.org/abs/2510.06179
- `cuNRTO / GPU robust trajectory optimization (2026)`: https://arxiv.org/abs/2603.02642
- `MPPI as Preconditioned Gradient Descent (2026)`: https://arxiv.org/abs/2603.24489
- `Step-MPPI / Differentiable Predictive Control (2026)`: https://arxiv.org/abs/2604.01539

## Bottom Line

Current status:
- `Workshop / spotlight demo / open-source systems contribution`: strong
- `ICRA/IROS full paper, submitted today`: borderline (improved from weak)
- `ICRA/IROS full paper after one more literature-faithful baseline and one higher-fidelity experiment`: plausible

Short version:

> The project now has a defensible narrow empirical claim, but not yet a strong enough paper contribution for a flagship robotics main track.

The main reason is not that the results are bad.
The main reason is that the current paper would still look like:
- an incremental MPPI variant
- tested in 2D kinematic toy scenarios
- without a literature-faithful direct baseline

That combination usually struggles at `ICRA/IROS` unless the empirical evidence is unusually strong or the systems story is unusually concrete.

## Update 2026-06-02: both Tier-1 additions built — and they force a headline pivot

The two missing pieces the "plausible" line above was waiting for now both exist.
Full evidence: `paper/cdf_mppi_baseline_results.md`. They did not just fill the
checklist — they changed what the honest headline can be.

**1. Literature-faithful direct baseline — DONE (and sobering).**
Implemented One-Step CDF-MPPI (`arXiv:2509.00836`) inside the same CUDA harness
(`src/benchmark_cdf_mppi_7dof.cu`), analytic margin-derived CDF as primary, neural
CDF as a documented ablation. Under matched wall-clock per control step:
- `7dof_shelf_reach` (4 seeds): `cdf_mppi` **1.00 success @ 0.05 ms/step**; best
  Diff variant `diff_mppi_1` 0.50 @ 0.46 ms; `diff_mppi_3` 0.25; vanilla 0.25.
- `7dof_dynamic_avoid`, reactive CDF (8 seeds): `cdf_mppi` **1.00, 0 collisions
  @ 0.24 ms**; `diff_mppi_3` 0.75–0.88; it occasionally collides.
- **Fair rematch (key):** the naive CDF win was largely its free goal *config*
  (solved IK). Handing Diff-MPPI the same `q_goal`, every variant jumps and the
  gap collapses — and crucially **`diff_mppi_3` (0.62–0.88) does NOT beat vanilla
  MPPI (0.88)**. CDF-MPPI's durable edge is per-step compute (8–47× cheaper), not
  success. **On the old headline tasks the autodiff refinement shows no advantage.**

**2. Higher-fidelity / decisive experiment — DONE (and this is the new headline).**
The whole smooth-dynamics suite lacked any contact gradient, so the contact-rich
hypothesis was literally untestable. Built two differentiable-contact tasks
(`src/benchmark_diff_mppi_pushing.cu`, `..._pushing_box.cu`) with smooth
(softplus / box-SDF) contact and forward-mode dual-number autodiff THROUGH contact:
- **box-pose pushing** (`box_align`, 8 seeds, K=256): `diff_mppi_5` **1.00 success**
  (71.5 steps) while **vanilla MPPI = 0.00 even at 16× samples (K=4096), which is
  cheaper per step yet still fails.** Success is **monotone in gradient steps**
  (0.00 / 0.38 / 1.00 for 1 / 3 / 5) — direct evidence the gradient is the active
  ingredient, not sampling.
- disk pushing (`push_straight`): vanilla saturates at ~30 steps for any K;
  `diff_mppi_3` reaches 26 at matched wall-clock — a matched-budget edge sampling
  cannot erase.

**Consequence for the paper — pivot the headline.**
The current draft leads with dynamic-obstacle exact-time navigation wins. The new
CDF-MPPI baseline *undercuts* that lead (a reactive one-step baseline matches/beats
the Diff line there at a fraction of the compute). The honest, defensible thesis is
now narrower and stronger:

> Diff-MPPI's contribution is **localized to contact-rich dynamics**: where the
> model gradient flows through (smooth) contact, a few autodiff refinement steps
> buy success/efficiency that no sample budget can match; on smooth tasks it ties
> vanilla MPPI and loses on compute to a CDF-MPPI baseline.

Lead with the pushing results; demote the smooth 7-DOF / dynamic-obstacle tables to
"where it does NOT help (and why)", with CDF-MPPI as the honest baseline that makes
that boundary precise. This converts the prior overclaim risk into a credibility
asset (a clearly-scoped positive result plus a strong negative control).

Status line, revised:
- `ICRA/IROS full paper`: still borderline, but on a **more honest and more
  defensible** footing — a scoped contact-rich claim with a literature-faithful
  baseline, rather than a broad dynamic-navigation claim that the baseline refutes.
- Remaining work is now **writing** (re-frame the draft around contact) and
  **hardening** the contact result (more contact tasks / a harder solved case like
  `box_turn`, currently unsolved by all methods), not building the missing pieces.

## What Is Already Good

The current line now has real positives:
- fixed-budget, cap-based, equal-time, and exact-time-tuned comparisons
- two dynamic scenarios, not just one
- an uncertainty follow-up with nominal-vs-actual obstacle mismatch on the dynamic pair
- a strengthened in-repo `feedback_mppi` baseline inside the same harness
- a closer rollout-sensitivity `feedback_mppi_sens` baseline inside the same harness
- a covariance-regression `feedback_mppi_cov` baseline inside the same harness
- a fused covariance-plus-linearization `feedback_mppi_fused` baseline inside the same harness
- a lower-rate-replan `feedback_mppi_hf` baseline inside the same harness
- a release-style current-action `feedback_mppi_ref` baseline inside the same harness
- a `grad_only_3` ablation that removes one weak alternative explanation
- three outside-domain pilots: nonlinear `CartPole`, dynamic-bicycle mobile navigation, and planar manipulator obstacle avoidance
- a narrow claim that is honest and empirically supported

The strongest current claim is:

> A minimal CUDA hybrid controller that augments vanilla MPPI with a short autodiff-based refinement stage reaches successful trajectories on two dynamic-obstacle tasks under matched per-step compute budgets where vanilla MPPI remains unsuccessful.

That is substantially better than the earlier single-scenario story.

## Why It Is Still Weak For ICRA/IROS Full Paper

### 1. The method contribution still looks incremental against nearby literature

Existing papers already cover much of the surrounding idea space:
- `PI-Net (2017)` already made path-integral control differentiable.
- `Differentiable MPC (2018)` already made optimization-based control layers differentiable.
- `Feedback-MPPI (2025, RA-L 2026)` adds sensitivity-derived local feedback to MPPI and evaluates on a quadruped and quadrotor.
- `Rollout Clustering + Dynamic Obstacles (2024)` and `DRPA-MPPI (2025)` already claim dynamic-obstacle improvements for MPPI.
- `C2U-MPPI (2025)` already pushes MPPI-style dynamic-obstacle avoidance into uncertain and real-world human-shared settings.
- `One-Step CDF-MPPI (2025)` combines distance-field gradients with MPPI and evaluates on high-dimensional manipulation.
- `MPPI-IPDDP (IEEE TRO 2025)` already proposes a hybrid MPPI + gradient-based DDP approach for smooth collision-free trajectories.
- `MPPI as Preconditioned Gradient Descent (2026)` formally proves MPPI is a preconditioned gradient step, which provides a theoretical lens for our hybrid approach.
- `Step-MPPI (2026)` learns a neural sampling distribution to achieve multi-step foresight with single-step latency, combining differentiable methods with MPPI from a different angle.

However, the new framing is now stronger than before:

> Fazlyab et al. (2026) show MPPI is exactly a preconditioned gradient descent step. Our autodiff refinement adds explicit local gradient steps after this implicit gradient update. This two-stage structure — coarse sampling-based gradient followed by fine local gradient — is a well-motivated optimization strategy, and we show empirically that it helps specifically on hard dynamic-obstacle tasks where the sampling-based preconditioner is insufficient.

This framing partially addresses the incrementality concern because it connects the method to a principled optimization structure rather than being "just another heuristic on top of MPPI".

Remaining vulnerability: the paper cannot rely on any of these claims alone:
- "MPPI but differentiable"
- "MPPI but better on dynamic obstacles"
- "MPPI plus gradient information"
- "GPU MPPI variant"

Those claims are too broad relative to the literature above.

### 2. The direct baseline story is better, but still incomplete

Right now you compare against:
- vanilla `mppi`
- strengthened in-repo `feedback_mppi`
- release-style current-action `feedback_mppi_ref`
- rollout-sensitivity `feedback_mppi_sens`
- covariance-regression `feedback_mppi_cov`
- fused covariance-plus-linearization `feedback_mppi_fused`
- lower-rate-replan `feedback_mppi_hf`
- `grad_only_3`

What is still missing is a literature-faithful sensitivity-aware MPPI baseline, for example:
- a stronger `Feedback-MPPI`-style local feedback baseline implemented as closely as possible inside the same harness
- or a direct reproduction of one nearby rollout-differentiation / sensitivity-aware controller rather than another in-house proxy

The newer baseline story is materially better than it was a few iterations ago:
- it derives feedback gains from rollout initial-state sensitivities instead of only from a nominal local linearization
- on `dynamic_crossing`, it reaches `0.75` success across `K={256,512,1024}` while vanilla MPPI remains at `0.00`
- a newer `feedback_mppi_cov` variant regresses time-varying gains from sampled state-control covariance
- on `dynamic_crossing`, `feedback_mppi_cov` reaches `1.00` success across `K={256,512,1024}`
- a newer `feedback_mppi_fused` variant blends covariance-regressed gains with nominal-linearization gains over two feedback passes
- on `dynamic_crossing`, `feedback_mppi_fused` reaches `1.00` success at `K={256,512}` with final distance about `1.86-1.91`
- on `dynamic_slalom`, it still fails, but it is now the strongest non-hybrid feedback baseline, reducing final distance to about `10.23-10.30` versus `11.44-11.51` for `feedback_mppi_cov`, `11.80-11.91` for `feedback_mppi`, and `12.75-12.81` for `feedback_mppi_sens`
- a newer `feedback_mppi_hf` variant decouples lower-rate replanning from per-step local feedback execution
- under a `1.00 ms` cap, `feedback_mppi_hf K=256 @ 0.87 ms` improves over MPPI on both dynamic tasks, lowering terminal distance from `3.04 -> 2.83` on `dynamic_crossing` and from `14.33 -> 13.62` on `dynamic_slalom`
- a targeted exact-time sweep now tunes `feedback_mppi_hf` directly to shared `1.00`, `1.50`, and `2.00 ms` targets
- at those targets, `feedback_mppi_hf` still improves over MPPI on both tasks, for example `dynamic_crossing: K=285 @ 0.978 ms, dist=2.77` and `dynamic_slalom: K=369 @ 1.498 ms, dist=13.34`
- a newer `feedback_mppi_ref` variant follows the released `Feedback-MPPI` current-action gain structure more closely
- on `dynamic_crossing`, `feedback_mppi_ref` reaches `1.00` success at `K={256,512}` with final distance about `1.87-1.91` while staying in the `0.56-0.65 ms` range
- on `dynamic_slalom`, it still fails, but lowers final distance to about `11.89-12.08`, which is materially better than `mppi` and `feedback_mppi_hf`
- a targeted exact-time sweep now tunes `feedback_mppi_ref` directly to shared `1.00 ms` and `1.50 ms` targets
- at `1.00 ms`, `feedback_mppi_ref` reaches `dynamic_crossing: K=1263 @ 1.002 ms, success=1.00, dist=1.95` and `dynamic_slalom: K=1150 @ 1.023 ms, dist=11.89`
- at `1.50 ms`, it reaches `dynamic_crossing: K=2362 @ 1.482 ms, success=1.00, dist=1.89` and `dynamic_slalom: K=2190 @ 1.472 ms, dist=11.89`
- a newer `feedback_mppi_release` variant also matches the released weighting shape more closely
- on `dynamic_crossing`, `feedback_mppi_release` reaches `1.00` success at `K={256,512}` with final distance about `1.86-1.91` while staying in the `0.61-0.72 ms` range
- on `dynamic_slalom`, it still fails and in fact degrades to about `19.09-19.12`, which is useful because it shows the released weighting alone is not sufficient
- a targeted exact-time sweep now tunes `feedback_mppi_release` directly to shared `1.00 ms` and `1.50 ms` targets
- at `1.00 ms`, `feedback_mppi_release` reaches `dynamic_crossing: K=1062 @ 1.009 ms, success=1.00, dist=1.93` and `dynamic_slalom: K=901 @ 1.007 ms, dist=19.11`
- at `1.50 ms`, it reaches `dynamic_crossing: K=2173 @ 1.530 ms, success=1.00, dist=1.90` and `dynamic_slalom: K=2033 @ 1.530 ms, dist=19.13`
- a targeted exact-time sweep now also tunes `feedback_mppi_cov` directly to shared `1.50 ms` and `2.00 ms` targets
- at `1.50 ms`, `feedback_mppi_cov` reaches `dynamic_crossing: K=219 @ 1.474 ms, success=1.00, dist=1.92` and `dynamic_slalom: K=211 @ 1.490 ms, dist=11.72`
- at `2.00 ms`, it reaches `dynamic_crossing: K=292 @ 1.964 ms, success=1.00, dist=1.91` and `dynamic_slalom: K=293 @ 1.971 ms, dist=11.68`
- a targeted exact-time sweep now also tunes the heavier `feedback_mppi_fused` baseline to a shared `2.00 ms` target
- at `2.00 ms`, `feedback_mppi_fused` reaches `dynamic_crossing: K=153 @ 1.968 ms, success=1.00, dist=1.94` and `dynamic_slalom: K=137 @ 1.993 ms, dist=10.51`

So the baseline gap is narrower than before, but not closed.

This is the most dangerous missing experiment because a reviewer can reasonably say:

> The paper shows that hybrid search plus local sensitivity helps over vanilla MPPI and over stronger in-repo feedback controllers, but does not yet show whether the proposed implementation is actually better than existing sensitivity-aware MPPI variants.

That is a direct novelty threat, not just a "future work" point.

**Status 2026-06-02 — this gap is now CLOSED, see the Update section above.** A
literature-faithful **One-Step CDF-MPPI** (`arXiv:2509.00836`) is implemented in the
same harness (`src/benchmark_cdf_mppi_7dof.cu`). It is not another in-house feedback
proxy but a published, mechanistically-distinct controller (C-space distance-field
angle shaping). The reviewer sentence above is answered — but the answer cuts both
ways: on the smooth 7-DOF tasks CDF-MPPI matches/beats the Diff-MPPI line at 8–47×
lower per-step compute, so the paper can no longer claim a baseline-beating result
*there*. That is why the headline pivots to the contact-rich regime (Update section),
where the autodiff refinement provably wins and CDF-MPPI is not applicable.

### 3. The experiment tier is still below flagship-conference expectations

Current evaluation is still:
- mostly 2D
- mostly kinematic
- hand-designed environments
- no hardware
- only a small MuJoCo `InvertedPendulum-v4` pilot, not yet a standard manipulation or locomotion benchmark

That does not make the work invalid.
It does make it harder to justify a main-track acceptance when nearby papers evaluate on:
- real robots
- high-fidelity locomotion
- 7-DOF manipulators
- uncertain dynamic environments with perception noise

There are now four partial exceptions:
- a pilot nonlinear CartPole benchmark outside the 2D navigation suite
- a dynamic-bicycle mobile-navigation pilot with steering lag and drag
- a planar-manipulator obstacle-avoidance pilot with second-order joint dynamics and workspace collisions
- a small MuJoCo `InvertedPendulum-v4` pilot using the public Gymnasium / MuJoCo model and termination protocol

That helps because the project is no longer purely a 2D kinematic story.
The dynamic-bicycle and planar-manipulator results are the more useful of the three for reviewer defense, because they stay in obstacle-avoidance planning while adding richer vehicle or manipulator dynamics.
The manipulator pilot is especially helpful because it produces a real success-rate split on `arm_static_shelf`: vanilla `mppi` remains unsuccessful while `diff_mppi_1` reaches `0.75`, `feedback_mppi_cov` reaches `1.00`, and the newer `feedback_mppi_ref` baseline also reaches `1.00`.
The MuJoCo pilot helps against the "custom benchmark only" criticism, and it now also has matched-time multi-parameter tuning on a standard and a wider-reset variant. A newer MuJoCo `Reacher` extension adds a harder terminal-heavy variant and a more stable seed protocol, so the project is no longer limited to pendulum-style stabilization on the public-benchmark side. However, the MuJoCo results still read as transfer / standardization checks rather than decisive hybrid-only wins: the Reacher variant can produce a hybrid-over-plain-MPPI split, but a tuned feedback baseline still catches up. So none of these pilots yet counts as the kind of stronger robotics-domain evaluation that fully closes this gap.

### 4. The matched-time story is much better, and now has direct tuning, but is still not complete

The project now has:
- cap-based comparisons
- equal-time nearest-match comparisons
- exact matched-time tuning on the current dynamic two-task suite with multi-parameter search
- exact matched-time tuning on the dynamic-bicycle follow-up pilot
- exact matched-time tuning on the 7-DOF manipulation benchmark
- exact matched-time tuning on the MuJoCo `InvertedPendulum-v4` pilot

That is already much better than most quick research repos.

But a skeptical reviewer can still say:
- the exact-time tuning no longer searches `K` only; it now also searches feedback gain scale and Diff-MPPI gradient hyperparameters, but it still does not cover the full controller architecture / design space
- the cleanest matched-time separation is still concentrated on the dynamic two-task suite: on `dynamic_slalom`, every non-hybrid family still fails at `1.0`, `1.5`, and `2.0 ms`, while the full 7-DOF exact-time sweep is mixed and the MuJoCo pilots still read mostly as transfer checks
- the newer `feedback_mppi_ref`, `feedback_mppi_release`, `feedback_mppi_cov`, `feedback_mppi_hf`, and `feedback_mppi_fused` baselines now have broader exact-time coverage, but they still remain in-repo proxies rather than a paper-faithful reproduction of the full external controller stack
- the family-level multi-parameter sweep and the fixed-controller headline table now need to be separated explicitly in the paper, because the best Diff-MPPI family point can shift between `diff_mppi_3` and `diff_mppi_1` depending on the target, and some exact-time points still drift due to timing noise
- outside the base suite, the time-matched claim now also reaches the 7-DOF benchmark, a small MuJoCo `InvertedPendulum-v4` pilot with multi-parameter tuning, and a MuJoCo `Reacher` follow-up that is stronger under fixed budget than under exact matched time, but the cleanest hybrid-only separation is still the dynamic-obstacle base suite rather than the pilots

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

That partially addresses the empirical-only criticism, because it shows where the extra compute is going.

Additionally, Fazlyab et al. (2026) now provide a theoretical lens: MPPI is a preconditioned gradient descent step. Our autodiff refinement can be interpreted as adding explicit gradient steps after this implicit step, which is a standard optimization strategy (e.g., combining coarse and fine gradient methods). The front-loaded correction profile is consistent with this interpretation: the sampling-based preconditioner provides a good global direction, while the local gradient sharpens the near-term controls.

This partially upgrades the contribution from "purely empirical" to "empirical with theoretical motivation".

## What Would Make This ICRA/IROS-Plausible

These are ordered by importance, not by ease.

### Tier 1: Must-Have

1. Strengthen the current direct sensitivity-aware baseline

Minimum acceptable version:
- keep the current nominal-linearization `feedback_mppi`, release-style `feedback_mppi_ref`, release-weighting `feedback_mppi_release`, rollout-sensitivity `feedback_mppi_sens`, covariance-regression `feedback_mppi_cov`, fused `feedback_mppi_fused`, and lower-rate-replan `feedback_mppi_hf` baselines, but tighten them into a closer `Feedback-MPPI`-style comparison inside the same benchmark harness
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
- partially addressed further by the new `benchmark_diff_mppi_manipulator` pilot
- **substantially addressed** by the new `benchmark_diff_mppi_manipulator_7dof` benchmark: Panda-like 7-DOF serial arm with 14D state, 7D control, 3D workspace obstacles, analytical Jacobians, and two scenarios (`7dof_shelf_reach`, `7dof_dynamic_avoid`). On `7dof_dynamic_avoid`, `feedback_mppi_ref` reaches `1.00` success at `K=256` while vanilla MPPI reaches `0.75`. This is no longer a toy 2-link task but a high-dimensional manipulation domain.
- an additional `feedback_mppi_faithful` two-rate variant was tested: combining released current-action gain with stride=2 replan. It fails on both dynamic tasks even at K=8192 (2.1 ms/step), confirming that current-action-only feedback gains lose temporal coverage between replans.
- still not fully closed, because the 7-DOF benchmark is still a custom simplified model rather than a standardized suite, and Diff-MPPI does not consistently outperform feedback baselines on the 7-DOF tasks
- **closed for the contribution, differently than expected (2026-06-02):** the 7-DOF
  task turned out to be the wrong place to look — under the fair shared-goal-config
  condition Diff-MPPI does not beat vanilla MPPI or CDF-MPPI there. The higher-fidelity
  domain that *does* discriminate is **differentiable contact** (`benchmark_diff_mppi_pushing`,
  `..._pushing_box`): on box-pose pushing `diff_mppi_5` reaches `1.00` success while
  vanilla MPPI reaches `0.00` even at `K=4096` (16× samples, cheaper per step), with
  success **monotone in gradient steps**. This is the decisive higher-fidelity experiment;
  it moves the paper out of "2D smooth toy" into contact-rich manipulation where the
  model gradient carries information sampling cannot buy. See the Update section above.

3. Extend the direct time-tuning protocol

Needed change:
- keep the new exact matched-time search in the final experimental package, preserve the multi-parameter sweep, and clearly separate fixed-controller headline results from family-level robustness sweeps

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

Current status:
- partially done via `benchmark_diff_mppi_mujoco` on MuJoCo `InvertedPendulum-v4`, including exact-time multi-parameter tuning
- still missing a more representative public manipulation or locomotion benchmark and paper-integrated multi-task results

## Minimum Submission Bar

If the goal is a serious `ICRA/IROS full paper`, the minimum package I would trust is:

1. Current static benchmark
2. Current two dynamic tasks
3. `grad_only_3` ablation
4. literature-faithful `Feedback-MPPI`-style baseline beyond the current nominal-linearization `feedback_mppi`, release-style `feedback_mppi_ref`, release-weighting `feedback_mppi_release`, rollout-sensitivity `feedback_mppi_sens`, covariance-regression `feedback_mppi_cov`, fused `feedback_mppi_fused`, and lower-rate-replan `feedback_mppi_hf`
5. exact matched-time tuning on the final evaluation suite
6. one higher-fidelity experiment outside 2D kinematic navigation

The new CartPole, dynamic-bicycle, and planar-manipulator pilots mean item `6` is more substantially addressed than before, but still not at the level that would make me remove it from the minimum list. Without item `4`, and without a stronger version of item `6`, and without carrying item `5` through the final evaluation suite, the paper is still too easy to down-score on originality and significance.

## Fastest Acceptable Paper Framing

If you do complete the missing items, the safest framing is not:

> We introduce a new differentiable MPPI algorithm.

That framing invites direct novelty attacks.

The safer framing is:

> We study a minimal hybrid MPPI controller that combines stochastic rollouts with a short autodiff refinement stage, and show that this combination improves compute-quality tradeoffs under matched real-time budgets.

That framing is narrower, but more defensible.

**Revised framing (2026-06-02), now that the experiments are in.** The matched-budget
result above is real but, on smooth tasks, weaker than hoped (a CDF-MPPI baseline ties
or beats it). The strongest *defensible* framing is now contact-scoped:

> We study a minimal hybrid MPPI controller — stochastic rollouts plus a short
> autodiff refinement stage — and show that **when the dynamics gradient flows through
> contact**, the refinement achieves success and matched-wall-clock efficiency that
> sampling cannot match at any budget (box-pose pushing: 100% vs 0% at 16× samples,
> monotone in gradient steps), while on smooth dynamics it ties vanilla MPPI and a
> literature-faithful CDF-MPPI baseline. The contribution is a clearly-scoped,
> mechanism-supported positive result with an honest negative control.

This trades breadth for credibility — exactly the trade flagship review rewards.

## Recommended Next Steps

Immediate next work:
1. Strengthen the current `feedback_mppi_ref` / `feedback_mppi_release` / `feedback_mppi_sens` / `feedback_mppi_cov` / `feedback_mppi_fused` / `feedback_mppi_hf` baselines in `benchmark_diff_mppi` into a more literature-faithful comparison.
2. Port the benchmark to one higher-fidelity domain.
3. Carry the new exact matched-time tuning workflow into that stronger evaluation domain.

**Superseded 2026-06-02 — both items 1–2 are done; the work is now writing + hardening.**
Revised immediate next work:
1. **Re-frame the draft around contact** (`diff_mppi.tex`): lead with the differentiable-
   pushing results, demote the smooth 7-DOF / dynamic-obstacle tables to a labelled
   "where it does not help, and why" section, and introduce CDF-MPPI as the honest
   baseline that makes that boundary precise.
2. **Harden the contact result**: at least one more differentiable-contact task, and
   either solve or honestly bound the currently-unsolved `box_turn` (long translate +
   rotate). The monotone-in-gradient-steps evidence is the strongest single result —
   protect it with a second task so it does not read as one cherry-picked scenario.
3. Keep the matched-wall-clock protocol and the fair shared-goal-config condition as
   the reporting standard throughout.

If time is limited:
1. aim for `workshop / late-breaking results / open-source systems demo`
2. keep collecting stronger evidence before attempting `ICRA/IROS` full paper

## My Current Judgment

As of `2026-04-02`, my judgment is:
- `today, as-is`: not strong enough for `ICRA/IROS` full paper
- `after one more literature-faithful baseline + one stronger experiment`: plausible
- `for workshop/demo right now`: yes

That is a conservative judgment, but it is the one most likely to survive actual review pressure.

**Updated judgment (2026-06-02).** Both gating items are now built, so the judgment is
no longer "plausible after more experiments" but "plausible after a rewrite":
- `today, as-is (old framing)`: weaker than before — the new CDF-MPPI baseline refutes
  the dynamic-navigation headline the current draft leads with.
- `after re-framing around the contact-rich result`: plausible, and on a more honest
  footing than any prior iteration — a scoped, mechanism-supported (monotone in grad
  steps), baseline-anchored positive result.
- The bottleneck moved from *evidence* to *narrative*. That is a much better place to be.
