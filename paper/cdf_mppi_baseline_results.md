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

**UPDATE (gap #2 confirmed below):** option 1 is now realized — on differentiable-
contact pushing, the autodiff refinement DOES provably pay off (box-pose: 100% vs
0%, monotone in gradient steps, unbeatable by samples). The honest paper thesis:
**Diff-MPPI's contribution is real but LOCALIZED to contact-rich dynamics**;
on smooth tasks it ties vanilla MPPI and a CDF-MPPI baseline. Scope the claim
there and lead with the pushing results, not the smooth reaching tasks.

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

### gap #2 RESULT: differentiable planar pushing (the regime where it pays off)

Built `src/benchmark_diff_mppi_pushing.cu`: quasi-static point-pusher → disk,
SMOOTH contact (softplus penetration → normal force), forward-mode dual-number
autodiff for the exact gradient THROUGH contact; Diff-MPPI refines the control
mean with it. Matched-budget vs vanilla MPPI, 8 seeds. All methods reach the
goal (task is solvable), so the discriminator is **efficiency (steps / cost)**.

At matched sample count K, Diff-MPPI reaches goal in ~20–26% fewer steps. The
sharper test is matched WALL-CLOCK (diff_mppi_3 is ~13× the per-step cost):

| scenario | diff_mppi_3 (K=16) | mppi (K=1024, matched ms) | mppi saturation |
|---|---|---|---|
| **push_straight** | **26.1 steps** @0.38 ms | 29.8 steps @0.39 ms | K 512→2048 all ~30 |
| push_diagonal | 30.4 steps @0.40 ms | 30.9 steps @0.16 ms | ~31 (tie) |

**On push_straight, vanilla MPPI saturates at ~30 steps no matter how many
samples — it cannot buy the gradient's directness — while diff_mppi_3 reaches 26
at matched wall-clock.** This is the FIRST regime in the whole study where the
autodiff refinement holds a matched-budget edge that sampling cannot erase. It
is modest and task-dependent (a tie on push_diagonal), but it LOCALIZES the
Diff-MPPI contribution: contact-rich dynamics where the model gradient carries
information that is expensive to sample. The smooth 7-DOF / cartpole / dynamic
tasks lack this property, which is why the refinement showed no advantage there.

→ Honest framing for the paper: Diff-MPPI's value is specific to differentiable-
contact / contact-rich settings, shown as a matched-wall-clock efficiency gain
that extra samples cannot match — NOT a general success-rate win over MPPI or
the CDF-MPPI baseline.

### gap #2 stronger task: box pushing to a target POSE (orientation)

Built `src/benchmark_diff_mppi_pushing_box.cu`: a point pusher pushes a
RECTANGULAR box to a target (x, y, theta). Reaching the orientation needs
off-centre contact (torque); the smooth box-SDF contact model makes the wrench
(force + torque) differentiable, so autodiff gives the contact-point gradient.
`box_align` (rotate + small translate), 8 seeds, K=256:

| controller | success | steps | pos_err | ang_err | ms/step |
|---|---|---|---|---|---|
| **diff_mppi_5** | **1.00** | 71.5 | 0.217 | 0.044 | 2.73 |
| diff_mppi_3 | 0.38 | 168 | 0.230 | 0.036 | 1.50 |
| diff_mppi_1 | 0.00 | 240 | 0.242 | 0.029 | 0.58 |
| mppi (K=256) | 0.00 | 240 | 0.274 | 0.028 | 0.08 |
| mppi (K=2048) | 0.00 | 240 | 0.273 | 0.033 | 0.28 |
| mppi (K=4096) | 0.00 | 240 | 0.272 | 0.029 | 0.52 |

**Here the contact gradient delivers a categorical success-rate win.**
diff_mppi_5 reaches the target pose 100% of the time (and ~3× faster, 71 vs 240
steps) while vanilla MPPI reaches it 0% of the time **even at 16× the samples
(K=4096), which is CHEAPER per step than diff_mppi_5 yet still fails**. Success
scales **monotonically with the number of gradient steps** (0.00 / 0.38 / 1.00
for 1 / 3 / 5 steps) — direct evidence that the autodiff refinement through
contact is the active ingredient, not sampling. The bottleneck is fine pose
control near the goal: pure sampling plateaus at pos_err ~0.27 regardless of K;
the gradient closes the last few cm. (A coarse static-friction deadzone was tried
and REMOVED: it equally blocks every method's fine corrections, helping no one.)

### gap #2 corroboration: second contact task `box_pivot` (different mechanics)

To guard against `box_align` being a single lucky scenario, added a structurally
distinct orientation task `box_pivot`: **opposite handedness** (+0.7 rad vs
box_align's −0.7), the pusher engages the **left face** (push +x) instead of the
bottom, and a **tighter angular tolerance** (`ang_tol=0.11`). Tight-tolerance
rotation is exactly the regime where sampling plateaus and the contact gradient
closes the last bit. 8 seeds:

| controller | success (K=1024) | ang_err (K=1024) | ang_err (K=256) |
|---|---|---|---|
| **diff_mppi_5** | **0.50** | **0.112** | 0.117 |
| diff_mppi_3 | 0.00 | 0.124 | 0.121 |
| diff_mppi_1 | 0.00 | 0.139 | 0.142 |
| mppi (K=1024 / 256) | 0.00 | 0.193 | 0.193 |

Two things reproduce from `box_align`: (1) **only the deepest-gradient controller
ever crosses the tolerance; vanilla MPPI succeeds 0% at every K**, and (2) the
**continuous angular residual is strictly monotone in gradient steps**
(`0.193 → 0.139 → 0.124 → 0.112`), and K-independent — the gradient, not sampling,
drives the orientation correction. The success rate is *deliberately modest* (0.50,
not 1.00): `box_pivot` sits near the contact-mechanics rotation ceiling of a single
fixed-face push (~0.65 rad before the contact normal rotates away and further
rotation needs contact repositioning), so this is reported primarily on the
**continuous monotone metric**, with the binary win as corroboration rather than a
threshold-tuned headline. It confirms the box_align mechanism on different contact
geometry without duplicating it.

### Honest boundary: `box_turn` (long translate + rotate) is unsolved by all

`box_turn` (translate ~0.9 + rotate 0.9 rad) is solved by **no method** (all 0%,
pos_err floors at ~0.39 > tol 0.20). This is a genuine limit, reported as such:
- The blocker is **translation, not orientation** (ang_err is already ~0.10). A
  longer planning horizon monotonically reduces the residual (diff_mppi_5 pos_err
  `0.378 → 0.304 → 0.274 → 0.220` for `H = 16/24/32/48` via `--horizon`), but
  **vanilla MPPI keeps pace** (both reach 0.17 success at H=48, ~23 ms/step):
  translation is sample-friendly, so this task does *not* isolate the gradient's
  contribution the way the orientation-dominant tasks do.
- Single-point non-prehensile pushing fundamentally needs **contact-point
  switching** for a long translate-then-rotate maneuver; receding-horizon mean
  refinement does not plan that re-grasp. We do **not** hand-tune a win here — that
  would be cherry-picking the wrong axis. `box_turn` stands as the stated boundary
  of the method and a clean direction for future work (contact-mode planning).

This is the strongest result of the study: a literature-faithful sampling method
(vanilla MPPI) provably cannot match Diff-MPPI here at any sample budget, and the
effect is monotone in gradient steps. Combined with the disk-pushing efficiency
result, TWO independent differentiable-contact tasks localize the Diff-MPPI
contribution to contact-rich dynamics — exactly the regime the smooth 7-DOF /
cartpole / dynamic-obstacle tasks lacked, where it showed no advantage.

### gap #2 robustness: does the contact win survive contact-model MISMATCH? (yes, asymmetrically)

The single strongest attack on the new contact-rich headline is: *"the gradient
win is an artifact of a convenient smooth contact model whose parameters exactly
match the plant — give the controller the wrong model and the gradient becomes
garbage."* We tested it directly. `benchmark_diff_mppi_pushing_box` gained a
default-noop flag `--plant-gain-scale G`: the **true plant**'s contact mobility
(`push_gain`, `rot_gain`) is multiplied by `G`, while the controller's internal
rollout **and** its autodiff gradient keep the NOMINAL gains — i.e. the
controller's contact model is deliberately *wrong* by factor `G`. `G=1.0`
reproduces the published numbers byte-identically. `box_align`, 8 seeds,
`diff_mppi_5 @K=1024` vs the strongest sampler `mppi @K=4096` (16× samples,
cheaper per step):

| plant/model gain `G` | mppi (K=4096) | diff_mppi_5 (K=1024) | regime |
|---|---|---|---|
| 0.6 | **1.00** | 1.00 | task easy for all — discriminator vanishes |
| 0.7 | 0.00 | **1.00** | gradient dominates |
| 0.85 | 0.00 | **1.00** | gradient dominates |
| 1.0 (matched) | 0.00 | 0.75 | gradient dominates |
| 1.2 | 0.00 | 0.50 | gradient edge |
| 1.4 | 0.00 | 0.25 | gradient edge, degrading |
| 1.6 | 0.00 | 0.12 | gradient nearly broken (still > mppi) |

Three findings, all honest:

1. **The win is NOT a matched-model knife's edge.** Across the whole band
   `G ∈ [0.7, 1.4]` (±30–40% contact-mobility error) `diff_mppi_5` keeps a
   *categorical* success advantage over vanilla MPPI, which stays flat at `0.00`
   for every `G` in that band regardless of 16× samples. A model wrong by a third
   does not erase the gradient's contribution.
2. **Robustness is asymmetric, and interpretably so.** Under-modelled mobility
   (`G<1`, plant *less* mobile than the controller believes) is actually *better*
   than matched — `1.00` at `G=0.7–0.85` vs `0.75` at `G=1.0`, and faster
   (~57 vs ~100 steps): the controller over-predicts motion, so it commits to
   sustained contact and converges conservatively. Over-modelled mobility (`G>1`,
   plant *more* mobile) degrades monotonically (`0.75→0.50→0.25→0.12`): the
   controller under-predicts motion and overshoots the fine orientation. This is
   a clean, mechanistic characterization rather than a single robustness number.
3. **Stated boundaries (both ends).** At extreme over-mobility (`G=1.6`) the
   gradient advantage nearly collapses (`0.12` vs `0.00`) — the model is simply
   too wrong; at extreme under-mobility (`G=0.6`) the dynamics become so gentle
   that *vanilla MPPI also reaches `1.00`*, so the task no longer discriminates.
   The gradient advantage lives precisely in the stiff-contact regime where
   sampling overshoots the orientation — and it is robust across a wide band of
   model error within it.

This converts the "convenient matched model" objection into a strength: the
contact-gradient benefit survives substantial contact-model mismatch, with an
honestly-bounded, mechanistically-explained asymmetric degradation. Reproduce:

```bash
for G in 0.6 0.7 0.85 1.0 1.2 1.4 1.6; do
  bin/benchmark_diff_mppi_pushing_box --scenarios box_align --planners diff_mppi_5 \
      --k-values 1024 --seed-count 8 --plant-gain-scale $G --csv build/mm_$G.csv
  bin/benchmark_diff_mppi_pushing_box --scenarios box_align --planners mppi \
      --k-values 4096 --seed-count 8 --plant-gain-scale $G --csv build/mm_mppi_$G.csv
done
```

### gap #2 robustness, second axis: OBJECT-SIZE (geometry) mismatch

The gain-scale study above varies contact *mobility*. A structurally different
sim-to-real error is getting the object's *dimensions* wrong: the controller
rarely knows the exact box size. `--plant-size-scale G` scales the TRUE plant's
box half-extents (`hx, hy`) by `G` while the controller's rollout + gradient keep
the nominal size. `box_align`, 8 seeds, same matchup:

| plant/model box size `G` | mppi (K=4096) | diff_mppi_5 (K=1024) |
|---|---|---|
| 0.7 (plant box smaller) | 0.00 | 0.00 |
| 0.85 | 0.00 | 0.25 |
| 1.0 (matched) | 0.00 | 0.75 |
| 1.15 | 0.00 | 0.62 |
| 1.3 (plant box bigger) | 0.00 | 0.00 |

- **mppi is flat `0.00` across the whole range** — sampling never solves the
  orientation regardless of object-size error.
- **diff_mppi_5 keeps its advantage within `±15%` size error** (`G ∈ [0.85, 1.15]`)
  but degrades to `0.00` at `±30%`. The tolerance band is **tighter than for the
  gain axis** (`±15%` vs `±30–40%`), and mechanistically so: the box size sets the
  contact point and torque lever arm, so a wrong size mis-locates *where* torque
  is generated — corrupting the orientation gradient more directly than a uniform
  mobility scale. At `G=1.3` the controller mis-places the contact badly enough to
  shove the box away (`pos_err 0.90`) — a clean, stated failure boundary.

Together the two axes bound the contact-gradient win honestly: robust to
contact-mobility error up to `±30–40%` and to object-size error up to `±15%`;
outside those bands it degrades to the sampling baseline — which itself *never*
solves the task at any budget. Reproduce: as above with `--plant-size-scale`
`{0.7,0.85,1.0,1.15,1.3}`.

### gap #2 robustness, task-generality: both axes replicate on `box_pivot`

Both mismatch sweeps above were run on `box_align`, leaving one reviewer attack
open: *"is the robustness an artifact of that one scenario?"* We re-ran **both
axes on the structurally distinct `box_pivot`** (opposite handedness, left-face
contact, **tight `ang_tol=0.11`**). On `box_pivot` the binary success latch is
near-degenerate for every method (the task sits right at the single-contact
rotation ceiling), so a binary table would be uninformative — and trusting a
near-threshold latch is exactly the trap the noise non-result below documents.
The discriminating quantity is therefore the **continuous final angular residual
`ang_err`** (settling proxy), which exposes the mechanism directly: pure
sampling plateaus on the orientation; the contact gradient drives the residual
down. `box_pivot`, 8 seeds, same matchup, `ang_err` (rad, lower is better):

| `G` | **gain axis** mppi K4096 | diff_mppi_5 K1024 | **size axis** mppi K4096 | diff_mppi_5 K1024 |
|---|---|---|---|---|
| 0.7  | 0.260 | 0.171 | 0.700 | 0.700 |
| 0.85 | 0.222 | 0.137 | 0.248 | 0.162 |
| 1.0  | 0.193 | 0.115 | 0.193 | 0.115 |
| 1.15/1.2 | 0.163 | 0.106 | 0.192 | 0.111 |
| 1.3/1.4 | 0.138 | 0.101 | 0.196 | 0.106 |

- **The gradient's angular residual stays strictly below the sampling floor at
  every point of both bands.** On gain, diff leads by `0.04–0.09` rad across
  `G ∈ [0.7, 1.4]`; on size, by a similar margin across `G ∈ [0.85, 1.3]`. mppi's
  `ang_err` floor never drops below `0.138` (gain) / `0.192` (size) — it never
  reaches the `0.11` latch at any budget, so its binary success is `0.00`
  throughout. The `box_align` signature ("gradient closes the contact-driven
  rotation residual that sampling cannot") **replicates on a different task and
  different contact face** — it is not a `box_align` artifact.
- **Asymmetry flips, consistently with the mechanics.** Where `box_align`'s win
  *degraded* with larger plant gain/size, `box_pivot`'s *improves*: a higher gain
  or a bigger box lengthens the torque lever arm, so the gradient-directed push
  rotates more per step and crosses the tight latch (`diff_mppi_5` success climbs
  `0.12 → 1.00` by `G=1.2` gain / `G=1.3` size). The failure boundary moves to the
  *small* side: at size `G=0.7` the box shrinks enough that both methods collapse
  identically (`ang_err 0.700`) — an honest, shared boundary, not a tuned one.
- Reproduce: as the two sweeps above with `--scenarios box_pivot` and, for the
  negative control, `--planners mppi --k-values 4096`.

This makes the contact-gradient robustness **two axes × two tasks**, measured on a
binary latch where it is clean (`box_align`) and on a continuous residual where the
latch is tight (`box_pivot`) — consistent across all four cells.

### gap #2 mechanism: WHY 16× samples cannot rescue vanilla MPPI (quantified)

The headline observation is that vanilla MPPI fails the box-pose tasks *even at
$16\times$ the samples*. We instrumented exactly why, via a new `--diag-mechanism`
mode on `benchmark_diff_mppi_pushing_box` that runs vanilla MPPI ($K=4096$) and
logs, per control step, the K-sample statistics (replaying the same `d_perturbed`
controls the rollout drew, so cost and rotation pair 1:1 per sample). Two numbers
fall out, both on `box_pivot` where the box stalls at angular residual $\approx
0.19$ for $\sim$200 steps (the published `mppi` plateau):

1. **The cost-reducing controls are a starved minority.** At the plateau decision
   state, the per-sample (net box rotation, cost) cloud is sharply U-shaped in
   rotation: cost is minimized in a thin **positive-rotation band $\approx[0.05,
   0.15]$ rad** (mean cost $\approx 1.9$ vs the stall cost $2.65$), but **64% of
   the 4096 samples have net rotation in $[-0.05, 0)$** — they barely turn the box
   at all. Isotropic velocity noise rarely produces the *sustained off-centre*
   contact needed to rotate; the box only turns when the push happens to stay
   off-centre across the horizon. The softmax-weighted MPPI mean, dominated by the
   inactive 64% pile sitting at the stall cost, cannot lock onto the useful tail.

2. **The starvation is structural in $K$, not a budget shortfall.** Define
   `escape_frac` = fraction of samples whose net rotation is toward the goal by
   enough to break the angular-tolerance latch this step. At the plateau it is
   $\approx 0.04$–$0.09$ and **does not grow with $K$**: `0.043 / 0.089 / 0.070`
   at $K = 256 / 1024 / 4096$ ($1\times / 4\times / 16\times$). So $16\times$ the
   samples buys $16\times$ the *raw* count but the *useful fraction* is fixed by
   the noise-vs-contact geometry — the weighted mean's per-step probability of
   escaping the latch is essentially constant. This is precisely why adding
   **gradient steps** (which point deterministically into the low-cost rotation
   band every step) breaks the plateau while adding **samples** does not — the
   mechanism behind the monotone-in-$N_g$ signature.

Figures: `scripts/plot_contact_figures.py` adds `fig_mechanism_sampling` (the
cost-vs-rotation scatter + the `escape_frac`-vs-$K$ bar) and
`fig_robustness_pivot` (the box_pivot two-axis continuous-residual separation).
Reproduce: `bin/benchmark_diff_mppi_pushing_box --diag-mechanism build/diag
--k-values 4096`. The diagnostic is gated behind the flag; the published sweep
path is byte-unchanged.

### A noted non-result: stochastic pose noise games the success latch

We also tried unmodelled Gaussian *process noise* on the box pose as a third
axis, but dropped it as a clean robustness test: the episode's success is latched
the first step the (noisy) pose touches the tolerance, so injecting pose noise
lets a stuck vanilla MPPI "succeed" on a transient excursion (e.g. at noise std
`0.04`, mppi reports `success 0.50` with final `pos_err 0.63 ≫ tol`). The noise
games the metric rather than testing control quality. A faithful noise study would
need a settling-based success criterion (within-tolerance for several consecutive
steps); we leave that to future work rather than report a metric-confounded number.

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
