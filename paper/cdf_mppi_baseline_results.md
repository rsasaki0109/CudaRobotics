# One-Step CDF-MPPI Baseline — Implementation & Results (2026-06-02)

Literature-faithful direct baseline for the Diff-MPPI research line, closing
gap #1 of `paper/icra_iros_gap_list.md` ("one literature-faithful direct
baseline"). Method: **One-Step CDF-MPPI** (arXiv:2509.00836). Scoping rationale
in `paper/diff_mppi_baseline_literature_2026-06.md`.

- Binary: `bin/benchmark_cdf_mppi_7dof` (`src/benchmark_cdf_mppi_7dof.cu`)
- Neural CDF header: `include/neural_cdf.cuh`
- Opponent: `bin/benchmark_diff_mppi_manipulator_7dof` (unchanged)

## Method as implemented

Angle-based **one-step (H=1)** MPPI on joint velocities, faithful to the paper:
per control step, sample N=200 velocity inputs `u_i ~ N(mu, sigma)`; cost
`c_i = a1*theta1*gate + a2*theta2` where `theta1` = angle(CDF ascent, motion),
`theta2` = angle(goal direction, motion); deactivate `theta1` when motion
already recedes (`theta1 < pi/2`) or far from obstacles (`fc >= d_act`) or goal
closer than obstacle (`fc >= ||q_goal - q||`). Importance-weighted mean update
`mu <- (1-a_mu)mu + a_mu * sum w_i u_i / sum w_i`, `w_i = exp(-c_i/beta)`.
Params: N=200, beta=1.0, a1=20, a2=10, d_act=1.0, dt=0.01, vel bounds = arm
limits. Plant: velocity-kinematic `q <- clamp(q + dt*u, joint_limits)`.

**Same task scaffold as the Diff-MPPI 7-DOF benchmark** (copied verbatim): FK
`fk_joint_positions`, workspace obstacles, `host_min_margin` collision oracle,
EE-goal success test, identical CSV schema. Only the controller's action space
+ integrator differ (CDF-MPPI velocity/H=1 vs Diff-MPPI torque/T=30), which is
intrinsic to the two published methods.

### CDF source: analytic (primary) vs neural (ablation)

The CDF is built from the workspace signed margin `m(q)`:
`f_c(q) = clamp(max(m,0)/||grad_q m||, 0, 1.5)`, ascent direction `grad_q m`
(away from the obstacle) — a first-order (Newton-step) joint-space
distance-to-contact estimate. Both variants share this transform; they differ
only in where `m, grad_q m` come from:

- **`cdf_mppi` (PRIMARY)** — analytic: `m` = `host_min_margin`, `grad_q m` by 8
  FK finite differences. Works.
- **`cdf_mppi_neural` (ABLATION)** — `m` from an MLP (7→64→64→64→1, tanh)
  trained on `host_min_margin` samples (neural-SDF style). Does NOT work — see
  finding below.

## Headline result — `7dof_shelf_reach`, 4 seeds, matched task

Budget axis (agreed): **matched wall-clock per control step** (`avg_control_ms`,
identical chrono meter in both binaries); sample-count and replan dt disclosed.

| controller | success | steps | final_dist | min_dist | avg_ms/step | samples/step |
|---|---|---|---|---|---|---|
| **cdf_mppi (analytic)** | **1.00** | 124 | 0.144 | 0.144 | **0.05** | 200 |
| diff_mppi_1 | 0.50 | 186 | 0.274 | 0.188 | 0.46 | 7,680 (256×30) |
| diff_mppi_3 | 0.25 | 238 | 0.328 | 0.286 | 0.75 | 7,680 |
| mppi (vanilla) | 0.25 | 240 | 0.340 | 0.269 | 0.38 | 7,680 |
| feedback_mppi_ref | 0.00 | 300 | 0.398 | 0.295 | 1.66 | 7,680 |

CDF-MPPI also succeeds 100% (0 collisions) on a second static scenario
`7dof_static_reach2` (two obstacles), avg 46 steps, 0.05 ms/step.

On this task CDF-MPPI reaches the goal collision-free at ~0.05 ms/step where the
best Diff-MPPI variant manages 50% at ~9× the per-step compute. **But this
workspace-goal comparison is unfair to Diff-MPPI** — CDF-MPPI is handed the goal
configuration. See "Fair rematch" below, which closes most of the gap and is the
condition to report.

## Fair rematch — give Diff-MPPI the SAME goal configuration (KEY)

The headline gap was suspected to be CDF-MPPI's free IK (goal config). We tested
it directly: added a **gated joint-space goal cost** to the Diff-MPPI binary
(`CostParams7::qgoal_weight` + `goal_q[]`, default 0 → published numbers
reproduce exactly; verified byte-identical) and fed it the SAME exported
`q_goal`. Run: `--goal-config-dir build --qgoal-weight 15`. 8 seeds, K=256.

| controller | success (workspace goal) | success (SAME goal config) | avg_ms/step |
|---|---|---|---|
| **cdf_mppi (analytic)** | **1.00** | — (config-native) | **0.05** |
| feedback_mppi_ref | 0.00 | **1.00** | 2.33 |
| mppi (vanilla) | 0.38 | 0.88 | 0.39 |
| diff_mppi_1 | 0.50 | 0.62 | 0.54 |
| diff_mppi_3 | 0.25 | 0.62 | 0.85 |

**Conclusion — the naive CDF-MPPI win was largely the free-IK advantage.** Once
Diff-MPPI is handed the same goal configuration, every variant jumps (e.g.
feedback_mppi_ref 0.00→1.00, vanilla mppi 0.38→0.88) and the success gap to
CDF-MPPI collapses. CDF-MPPI's durable edge is **per-step compute** (0.05 ms vs
0.39–2.33 ms, 8–47×), not success rate.

Two consequences for the Diff-MPPI paper:
1. Report the comparison under the FAIR (shared goal-config) condition; the
   workspace-goal version overstates CDF-MPPI and is not a controller-only test.
2. With the goal config shared, **diff_mppi_3's gradient refinement does NOT
   beat vanilla MPPI** at matched K (0.62 vs 0.88). Diff-MPPI's contribution
   therefore cannot rest on out-succeeding these baselines on this static task —
   it must rest on the autodiff-refinement mechanism under a matched *per-step
   budget* (where CDF-MPPI's 0.05 ms vs Diff-MPPI's ~0.8 ms is the real axis),
   and likely needs the high-fidelity / dynamic settings of gap #2 to show value.

(qgoal_weight sweep, 4 seeds: w=5 → mppi 0.75 / diff_mppi_3 0.50; w=15 → 1.00 /
0.75; w=40 → 0.75 / 0.75. w=15 used for the 8-seed table.)

## Dynamic obstacle — does Diff-MPPI's multi-step lookahead finally win? (no)

Hypothesis: CDF-MPPI is a static-obstacle, one-step (H=1) method, so a MOVING
obstacle should expose it — the multi-step rollout + autodiff refinement of
Diff-MPPI should anticipate the motion and win. We made the analytic CDF-MPPI
REACTIVE (margin evaluated at the obstacle's CURRENT step position) and ran
`7dof_dynamic_avoid` (one obstacle crossing at 0.15 m/s), 8 seeds.

| controller | success (workspace) | success (config goal w=15) | collisions | avg_ms/step |
|---|---|---|---|---|
| **cdf_mppi (analytic, reactive)** | **1.00** | — | **0** | 0.24 |
| diff_mppi_3 | 0.75 | 0.88 | some (diff_mppi_1 hit 2) | 0.91 |
| mppi | 0.12 | 0.88 | — | 0.42 |
| diff_mppi_1 | 0.38 | 0.50 | some | 0.58 |
| feedback_mppi_ref | 0.75 | 0.62 | — | 2.76 |

**Hypothesis NOT supported.** Reactive CDF-MPPI is the MOST reliable here (1.00,
zero collisions) and the cheapest; the Diff-MPPI family tops out at 0.88 and
occasionally collides. And again `diff_mppi_3 (0.88) ≈ mppi (0.88)` under the
fair condition — the gradient refinement does not clearly beat vanilla MPPI.

Caveat: this obstacle is slow (0.15 m/s; it moves ~0.22 m over the ~1.5 s the
arm takes). A genuinely anticipation-demanding obstacle (fast, on a direct
collision course at the arrival moment) is UNTESTED and is the one regime that
could favour multi-step lookahead. We did not hand-tune such a scenario — doing
so to make Diff-MPPI win would be cherry-picking; if pursued it must be
principled and pre-registered.

## Consolidated conclusion (for the paper)

Across static reach (×2), and this dynamic-obstacle task, a literature-faithful
CDF-MPPI **matches or beats** the entire Diff-MPPI line on success and collisions
at **8–47× lower per-step compute**, and crucially `diff_mppi_3` (the autodiff
refinement) **does not outperform vanilla MPPI** once the goal configuration is
shared. The Diff-MPPI line, as currently evaluated, has **no demonstrated
advantage** over this strong simple baseline. Honest implications:

1. The autodiff-refinement contribution needs a regime where it provably pays
   off — high-fidelity / contact-rich dynamics (gap #2, where gradients through
   accurate physics matter) or a principled anticipation-demanding task — or
2. the Diff-MPPI claim must be reframed (e.g. purely the matched-wall-clock-
   budget framing) and the systems-paper headline tempered.

This is the pre-submission reality check the baseline was built to provide.

## gap #2 probe: where could the autodiff refinement pay off? (contact is missing)

We checked whether existing underactuated / higher-fidelity tasks reveal a
Diff-MPPI advantage, and surveyed the suite for the contact-rich regime.

- **Existing cartpole swing-up** (`benchmark_diff_mppi_cartpole`, 8 seeds): no
  clean win. `cartpole_large_angle`: all methods fail (success 0, incl. K=2048).
  `cartpole_recover`: best is diff_mppi_1@K256 = 0.50 vs mppi 0.0–0.25 (a faint
  hint), but diff_mppi_3 = 0.125 (extra steps hurt); low + noisy overall.
- **Existing dynamic-bicycle** CSV (already in repo): diff_mppi ties vanilla.
- **Critical infra gap (survey):** EVERY benchmark uses smooth dynamics, and the
  dual-number autodiff + the `if (d<margin)` penalty mean **no gradient flows
  through contact anywhere in the repo**. The contact-rich hypothesis — that a
  model gradient through accurate contact dynamics beats pure sampling — is
  therefore literally untestable with current code. Real MuJoCo is linked only
  for ground-truth eval (gradients go through a hand-coded smooth approx), and
  the MuJoCo models in-repo (pendulum, reacher) have no contact anyway.

**Implication:** demonstrating Diff-MPPI's value needs a NEW differentiable
contact task — e.g. planar non-prehensile pushing with a smooth (softplus/
barrier) friction-contact model so autodiff gradients flow through contact.
That is the decisive experiment; until it exists, the evidence says the autodiff
refinement has no advantage over vanilla MPPI / a CDF-MPPI baseline.

### Fairness caveats (MUST disclose in the paper)

1. **Goal asymmetry (significant) — now quantified, see "Fair rematch" above.**
   CDF-MPPI is handed a precomputed collision-free goal *configuration* `q_goal`
   (offline EE-distance descent, EE residual 0.001) — effectively a solved IK.
   The Diff-MPPI family default sees only the *workspace* EE goal. We resolved
   this by giving Diff-MPPI the same `q_goal`; the success gap then collapses.
   Always report the fair (shared-config) condition.
2. **Action space / replan rate differ** (CDF dt=0.01, Diff dt=0.04) — disclosed
   as a secondary column, not equalized (each method uses its native rate).
3. Single simplified-Panda FK + sphere obstacles; not a high-fidelity simulator
   (that is gap #2, separate).

## Finding: neural CDF needs gradient supervision (ablation, documented)

The neural variant fits the margin VALUE well (held-out RMSE ≈ 0.11 rad) but its
**gradient is unusable**: cosine similarity to the true margin gradient near the
contact manifold stays ≈ 0.05–0.07 regardless of FD secant width (0.001→0.2),
target weighting (near-contact oversampling), far-field cap (1.5/3.0), or
training the smooth `m(q)` field directly instead of the composite CDF. Result:
`cdf_mppi_neural` plows through the obstacle (≈8 collisions, 0% success).

Root cause is fundamental, not a tuning miss: a smooth value-MSE MLP averages
out the high-frequency 7-D contact gradient; value error (~0.1) exceeds the
gradient signal `eps*||grad m||` (~0.03). This empirically reproduces exactly
why the CDF / neural-distance-field literature uses **eikonal (`||grad f||=1`)
and/or Sobolev (direct gradient) supervision** plus larger networks — which the
repo's value-MSE `GpuMLP` does not provide. We therefore ship the analytic CDF
as the primary baseline and keep the neural variant as this documented negative
result. (Decision: analytic primary, neural ablation — 2026-06-02.)

## Reproduce

```bash
cd build && cmake .. && make -j$(nproc) benchmark_cdf_mppi_7dof benchmark_diff_mppi_manipulator_7dof
# primary CDF-MPPI (analytic), both static scenarios
bin/benchmark_cdf_mppi_7dof --planners cdf_mppi --seed-count 4 --csv build/cdf.csv
# neural ablation + validation (RMSE, grad cos-sim, 2D slice PNG)
bin/benchmark_cdf_mppi_7dof --validate --planners cdf_mppi_neural --scenarios 7dof_shelf_reach
# opponent family, workspace goal (default; published numbers)
bin/benchmark_diff_mppi_manipulator_7dof --scenarios 7dof_shelf_reach --k-values 256 --seed-count 8 --csv build/diff_ws.csv
# opponent family, FAIR rematch with the SAME goal config CDF-MPPI used
bin/benchmark_diff_mppi_manipulator_7dof --scenarios 7dof_shelf_reach --k-values 256 --seed-count 8 \
    --goal-config-dir build --qgoal-weight 15 --csv build/diff_fair.csv
```

The `--goal-config-dir`/`--qgoal-weight` flags default OFF; with them unset the
binary reproduces the original workspace-goal numbers byte-for-byte.

Both CSVs share the same header → concatenate for analysis. Apples-to-apples
checklist: identical scenario/obstacles (copied `make_7dof_shelf_reach`),
identical `goal_tol`=0.15 and collision threshold 0.02, shared `host_min_margin`
success oracle; the only uncontrolled variable is the controller (plus the
disclosed goal-config asymmetry).

## Open follow-ups

- [DONE] Equalize the goal-config asymmetry — fair rematch implemented; gap collapses.
- Diff-MPPI's value over vanilla MPPI is NOT shown on this static fair task
  (diff_mppi_3 0.62 < mppi 0.88) → the paper needs a setting where the autodiff
  refinement pays off: matched per-step *wall-clock* budget and/or gap #2.
- If a faithful neural CDF is wanted: add eikonal/Sobolev supervision to GpuMLP.
- gap #2 (high-fidelity experiment) remains: 7-DOF/contact-rich MuJoCo + sim-to-real.
