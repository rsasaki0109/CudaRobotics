# Diff-MPPI Baseline & Differentiation Literature Review (2026-06-02)

Deep-research output (fan-out web search → source fetch → 3-vote adversarial
verification → synthesis). 6 search angles, 17 primary sources fetched, 82
candidate claims extracted, top 25 adversarially verified (22 confirmed, 3
killed). All surviving claims are backed by arXiv/published primaries.

This note answers the `icra_iros_gap_list.md` gap #1 ("one literature-faithful
direct baseline") and gap #3 (honest differentiation) by literature, BEFORE any
implementation. It is the scoping input for the next implementation task.

## Implementation status (2026-06-02)

Baseline IMPLEMENTED — see `paper/cdf_mppi_baseline_results.md` and
`src/benchmark_cdf_mppi_7dof.cu`. The angle-based one-step CDF-MPPI controller
works (analytic-CDF variant: 100% success, 0 collisions, ~0.05 ms/step on
`7dof_shelf_reach`, beating the Diff-MPPI family at ~9× lower per-step compute).
Key finding: the **neural** CDF needs eikonal/Sobolev gradient supervision (the
repo's value-MSE MLP cannot learn the 7-D contact gradient) → shipped analytic
CDF as primary, neural as a documented ablation. Open: goal-config fairness
asymmetry (CDF-MPPI gets a precomputed goal config); gap #2 (high-fidelity expt).

## Bottom line (recommendation)

> Re-implement **One-Step CDF-MPPI (arXiv:2509.00836)** as the literature-faithful
> direct baseline inside the CudaRobotics CUDA harness, and frame Diff-MPPI's
> novelty strictly as **the minimal autodiff gradient-descent refinement of the
> MPPI control mean under a matched per-step compute budget** — NOT as parallelism
> and NOT as the bare "beats vanilla MPPI on dynamic obstacles" outcome.

## 1. Recommended baseline: One-Step CDF-MPPI (arXiv:2509.00836)

Strongest direct baseline because it maximizes overlap with our claim while
staying reproducible:

- gradient-augmented MPPI variant (fuses Configuration Space Distance Field
  gradients into MPPI, joint-space cost, horizon reduced to one step)
- evaluated on the **exact 7-DOF Franka obstacle-avoidance task** Diff-MPPI claims
- explicit comparison against both optimization-based and **standard MPPI** baselines
- beats vanilla MPPI on success in the 2-DOF case (99.6% vs 71.2%)
- >750 Hz (776 Hz vs 61-222 Hz baselines)

**Key mechanistic distinction (this is our differentiation, not a weakness):**
CDF-MPPI injects *domain-specific distance-field gradients via cost shaping*,
whereas Diff-MPPI applies an *explicit autodiff gradient-descent step on the
control sequence/mean*. Re-implementing CDF-MPPI head-to-head is recommended
precisely because reviewers may otherwise see it as "close enough" and demand the
comparison.

## 2. Differentiation — STILL DEFENSIBLE (niche empirically vacant)

No surveyed system occupies "short autodiff gradient-descent refinement applied
to the MPPI control mean":

- **Biased-MPPI (RA-L 2024, 2401.09241)**: injects classical/learned feedback
  controllers (LQR, LQI, energy swing-up, H-inf, hand-crafted) as
  importance-sampled rollouts — samples, not gradient-stepping the mean.
- **Feedback-MPPI (RA-L 2026, 2506.14855)**: Riccati-based local linear feedback
  gains via rollout differentiation for closed-loop tracking — not control-seq refinement.
- **CDF-MPPI / CSC-MPPI**: inject domain-specific cost/constraint gradients (cost
  shaping / primal-dual feasibility step), not autodiff descent on the mean.
- **DiffMPC (Toyota Research 2025, 2510.06179)**: gradient-QP differentiable MPC,
  not sampling-based.
- **Hydrax (JAX MJX toolkit)** and **MPPI-Generic (C++ CUDA: MPPI/Tube/RMPPI)**:
  sampling only, no autodiff/gradient controller.

→ The matched-budget + short-autodiff-refinement-of-the-mean hybrid is genuinely open.

## 3. OCCUPIED territory — do NOT claim

- **Parallelism-first framing is taken.** MPPI is natively GPU-parallel (Williams
  et al.); DiffMPC explicitly claims GPU-parallel differentiable MPC (SQP + custom
  preconditioned CG, tridiagonal preconditioning). Diff-MPPI builds on an
  already-parallel MPPI base → cannot claim "first GPU-parallel differentiable control".
- **"Beats vanilla MPPI on dynamic obstacles" outcome is crowded.** Biased-MPPI
  (0 collisions vs 4-10 for vanilla IA-MPPI), CSC-MPPI (0% vs 30% collision,
  2501.x), C2U-MPPI (uncertain human-shared, real-world). The *result pattern*
  alone is not novel — the *mechanism* (autodiff refinement) + matched per-step
  budget must carry the contribution.

## 4. High-fidelity experiment — most persuasive direction

Current flagship bar = **GPU physics-simulator MPPI with sim-to-real on
high-DOF / contact-rich tasks**:

- **IsaacGym-MPPI (RA-L 2024, 2307.09105)**: GPU physics simulator as the dynamics
  model directly; sim-to-real on mobile nav, non-prehensile manipulation,
  whole-body high-DOF control.
- **MuJoCo-MPPI (2025, 2511.21264)**: GPU MuJoCo world model, real-time bimanual
  manipulation, sim-to-real on two UR5e arms (10-16 Hz replan, <100 ms compute).

→ Most credible addition for Diff-MPPI: a **7-DOF manipulator or contact-rich
MuJoCo task with sim-to-real framing**. Reusable harness: MPPI-Generic (C++ CUDA)
is the closest same-language harness, but re-implementing the chosen baseline
inside CudaRobotics' own harness preserves comparison fairness.

## 5. CAVEATS / open risks (must resolve before claiming novelty)

- **[RESOLVED 2026-06-02] Suspect papers verified — niche survives.** The
  "future-dated" IDs were real (2603 = Mar 2026, 2604 = Apr 2026; we are in
  Jun 2026). All three threats checked against primaries; none occupies the
  *online explicit autodiff gradient-descent refinement of the MPPI control
  mean at inference under matched budget* niche:
  - **MPPI-IPDDP (2208.02439)** — sequential 3-stage pipeline: MPPI warm-start →
    convex collision-free corridor → IPDDP smoothing. Corridor-constrained DDP
    smoothing of a coarse trajectory, NOT autodiff descent on the mean. Different
    mechanism. Niche safe.
  - **MPPI-as-Preconditioned-GD (2603.24489)** — purely THEORETICAL. Shows the
    *existing* MPPI update IS a unit-step preconditioned gradient update;
    gradients derived analytically from the KL/variational free-energy objective,
    NOT via autodiff; no multi-step extension running extra steps on the mean; no
    matched-budget robotics experiment (numerical illustrations only). Does not
    add a refinement stage → does not occupy our system niche. **BUT forces a
    framing tightening (see below) — must be cited and engaged, not ignored.**
  - **Step-MPPI (2604.01539, "Toward Single-Step MPPI via Differentiable
    Predictive Control")** — trains an NN proposal distribution OFFLINE via a
    differentiable predictive-control loss, then samples online. Differentiability
    is in the offline training loss, not an online per-step gradient step on the
    controls. Adjacent (must cite) but does not occupy the niche.
  - DRPA-MPPI (2503.20134) — switching repulsive-potential cost, sampling-based,
    no gradient refinement. Not a threat.

- **[NEW, CRITICAL] Framing constraint from Preconditioned-GD (2603.24489).**
  Because that paper proves *vanilla MPPI already is a (zeroth-order)
  preconditioned gradient step*, we MUST NOT frame Diff-MPPI loosely as "we add
  gradient descent to MPPI" — a reviewer will call that theoretically confused.
  Correct framing: Diff-MPPI augments MPPI's **zeroth-order, sample-estimated
  preconditioned step** with an **explicit first-order autodiff refinement of the
  control mean** (true cost/dynamics derivatives), under a matched per-step
  budget. Position relative to 2603.24489 explicitly: they characterize the
  zeroth-order step; we add and budget-match a first-order step. This paper is an
  asset for precise positioning, not a killer.
- **Budget-matching protocol.** Define a defensible per-step accounting standard
  (wall-clock vs FLOPs vs rollout-count equivalence) reviewers will accept.
- Several anchor papers are very recent preprints (CDF-MPPI Sep 2025, DiffMPC Oct
  2025, MuJoCo-MPPI Nov 2025), not yet peer-reviewed; re-check at submission time.

## Primary sources

- One-Step CDF-MPPI: https://arxiv.org/abs/2509.00836
- Biased-MPPI: https://arxiv.org/abs/2401.09241
- Feedback-MPPI: https://arxiv.org/abs/2506.14855 / https://feedback-mppi.github.io/
- DiffMPC (GPU diff MPC): https://arxiv.org/abs/2510.06179
- C2U-MPPI: https://arxiv.org/html/2501.08520v2
- CSC-MPPI: https://arxiv.org/html/2506.16386
- IsaacGym-MPPI: https://arxiv.org/abs/2307.09105
- MuJoCo-MPPI (bimanual, sim-to-real): https://arxiv.org/pdf/2511.21264
- MPPI-Generic (CUDA harness): https://arxiv.org/html/2409.07563v4
- Hydrax (JAX MJX toolkit): https://github.com/vincekurtz/hydrax
- MPPI-IPDDP (verified 2026-06-02): https://arxiv.org/abs/2208.02439
- MPPI as Preconditioned Gradient Descent (verified 2026-06-02): https://arxiv.org/abs/2603.24489
- Step-MPPI / Differentiable Predictive Control (verified 2026-06-02): https://arxiv.org/abs/2604.01539
- DRPA-MPPI (verified 2026-06-02): https://arxiv.org/abs/2503.20134
