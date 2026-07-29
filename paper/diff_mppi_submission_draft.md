# Contact-Rich Diff-MPPI: Hybrid Sampling and Gradient Refinement for GPU Manipulation

Draft date: 2026-07-29

## Evidence status

This is the authoritative submission-oriented Markdown draft. Its claim
boundary is synchronized with
[`artifacts/contact_rich_diff_mppi.json`](artifacts/contact_rich_diff_mppi.json),
whose submission-required evidence is `ready: true`. Final venue formatting,
artifact URLs, and anonymization remain editorial release tasks; they do not
change the frozen experimental claims.

## Abstract

Sampling-based model predictive control handles discontinuous costs and broad
regions of the action space, but finite rollout budgets can leave the selected
control sequence close to, yet outside, a narrow contact-success basin. We
study Contact-Rich Diff-MPPI, a training-free hybrid controller that performs
the standard model predictive path integral update and then applies three
local autodiff refinement steps to the nominal controls. We evaluate where
this refinement helps and where it fails using three independently frozen
blocks. First, a 32,400-episode GPU matrix spans five box-contact tasks, twelve
plant conditions, six planners, three rollout budgets, and thirty paired
seeds. Diff-MPPI-3 attains 0.614 aggregate success versus 0.453 for MPPI, but
the family includes 33 Holm-significant positive and 6 Holm-significant
negative cells. Second, with disjoint calibration/evaluation seeds and an
enforced 10 ms control slot, Diff-MPPI-3 reaches 1.00 real-time success on
contact loss versus 0.467 for MPPI, a paired delta of +0.533 with Holm
\(p=0.000305\). Third, when MuJoCo executes every command under friction,
mass, and observation variations, Diff-MPPI-3 reaches 0.457 aggregate success
versus 0.289 for MPPI, with three positive and zero negative
Holm-significant cells. Detour and tall-box failures remain visible. These
results support a localized compute-quality claim for contact-rich control,
not universal planner dominance or real-robot transfer.

## 1. Introduction

Model predictive path integral control (MPPI) is attractive for robotics
because it transforms a nonlinear receding-horizon problem into massively
parallel trajectory sampling. Its update can explore nonconvex objectives
without differentiating through every rollout. That robustness comes with a
finite-sample limitation: under a fixed rollout budget, the weighted update
may identify the correct interaction mode while still missing the narrow
control sequence that maintains or recovers contact.

Gradient trajectory optimization has the complementary behavior. Once a
nominal sequence enters an informative basin, local derivatives can make a
sharp correction with little additional search. The same local method can
also fail when the basin is wrong, the model is mismatched, or the task
requires a topological change. Contact-rich manipulation exposes both sides:
alignment and contact-loss tasks provide useful local geometry, while detours
and tall objects create failure modes that refinement cannot erase.

We investigate a deliberately small hybrid. The MPPI update remains intact.
Afterward, a forward-mode autodiff pass computes the nominal trajectory cost
gradient and applies one or three clipped steps. There is no learned sampler,
offline dataset, or differentiation through the sampling distribution.

The paper makes three contributions:

1. A minimal GPU hybrid of standard MPPI sampling and short local autodiff
   refinement for planar contact-rich manipulation.
2. A preregistered robustness protocol with paired seeds, full-family
   correction, smooth and structurally different hard-contact plants, and
   retained negative cells.
3. Deadline-matched and closed-loop MuJoCo transfer evaluations that separate
   same-\(K\), equal-slot, and external-plant claims.

## 2. Related work

MPPI uses information-theoretic importance weighting to update a nominal
control sequence from parallel stochastic rollouts
[[Williams et al., 2017](https://doi.org/10.2514/1.G001921)]. Its GPU-friendly
structure makes it a strong baseline when costs are nonsmooth or dynamics are
expensive to differentiate.

Sampling-plus-gradient hybrids are not new. CEM-GD combines cross-entropy
sampling with gradient descent for planning, while other methods combine MPPI
with DDP, sensitivity feedback, learned sampling distributions, or ancillary
controllers. Our contribution is narrower: a minimal post-MPPI refinement,
evaluated under a declared contact-family statistical protocol and an
enforced wall-clock control slot.

MuJoCo provides an independent rigid-body/contact engine for the external
fidelity block. We use it as closed-loop sim-to-sim transfer: CUDA planners
retain their nominal smooth model, MuJoCo advances the true plant after each
selected command, and the resulting state is returned to the controller. This
is stronger than open-loop replay but is neither a standard robot benchmark
nor real-robot evidence.

## 3. Method

### 3.1 MPPI sampling update

Let \(U=(u_0,\ldots,u_{T-1})\) be the nominal control sequence. For sampled
perturbations \(\epsilon_k\), the controller rolls out the nominal dynamics
and obtains trajectory costs \(S_k\). With temperature \(\lambda\) and
\(\rho=\min_k S_k\), normalized weights are

\[
w_k =
\frac{\exp(-(S_k-\rho)/\lambda)}
{\sum_j \exp(-(S_j-\rho)/\lambda)}.
\]

The standard update is retained:

\[
U^+ = U + \sum_k w_k \epsilon_k.
\]

All rollouts, stage costs, and weighted reductions execute on the GPU.

### 3.2 Local autodiff refinement

Starting from \(U^{(0)}=U^+\), the hybrid rolls out the nominal smooth contact
model and computes \(\nabla_U J(U^{(j)})\) with forward-mode autodiff and an
adjoint accumulation. It then applies

\[
U^{(j+1)} =
\Pi_{\mathcal U}
\left[
U^{(j)}-\alpha\,
\operatorname{clip}(\nabla_U J(U^{(j)}),g_{\max})
\right],
\]

where \(\Pi_{\mathcal U}\) enforces control limits. `diff_mppi_1` takes one
step and `diff_mppi_3` takes three. The controller executes the first refined
command, shifts the sequence, and repeats at the next state.

The gradient is local to the nominal post-MPPI sequence. We do not
differentiate through sampling, importance weights, or random-number
generation. The sampling stage remains responsible for broad search; the
gradient stage only sharpens the selected mode.

### 3.3 Contact models

The nominal planner model uses a smooth planar box-contact approximation. The
robustness matrix varies gain, object size and aspect ratio, and includes a
structurally different momentum-carrying hard-contact plant with friction and
damping sweeps. `mppi_hardmodel` is retained as a model-exact hard-contact
reference rather than being omitted when it underperforms.

The external block replaces the executed plant with MuJoCo 3.11.0. The
planners still optimize their nominal smooth dynamics; MuJoCo advances each
chosen command and returns the next state. This tests closed-loop model
mismatch, not zero-shot transfer to a physical robot.

### 3.4 Compared planners

The broad matrix includes:

- `mppi`: vanilla sampling baseline;
- `diff_mppi_1`: MPPI plus one local gradient step;
- `diff_mppi_3`: MPPI plus three local gradient steps;
- `soppi`: particle/SVGD-inspired control-sequence update;
- `soppi_fast`: accelerated variant containing one nominal gradient step;
- `mppi_hardmodel`: MPPI using the exact hard-contact model where applicable.

SOPPI-fast is not described as a pure SVGD or sampling-only baseline because
its nominal gradient step would make that label false.

## 4. Experimental protocol

### 4.1 Tasks

Five planar box-manipulation tasks cover distinct interaction structures:

- `box_swivel`: rotate and translate through sustained contact;
- `box_align_strict`: align pose under tight final tolerances;
- `box_align_detour`: move around an axis-aligned obstruction;
- `box_align_contact_loss`: recover after the nominal contact mode breaks;
- `box_align_contact_arc`: maintain a curved contact maneuver.

Success is computed from the preregistered task-specific terminal thresholds.
Episode failures and solver deadline misses remain in the released tables.

### 4.2 Broad robustness matrix

The release matrix contains:

| Dimension | Values |
|---|---|
| Conditions | 12 nominal, gain, size/aspect, hard-friction, and hard-damping conditions |
| Tasks | 5 |
| Planners | 6 |
| Rollout budgets | \(K\in\{128,256,512\}\) |
| Seeds | 30 paired seeds per cell |
| Total | 32,400 episodes; 1,080 summary cells |

Planner comparisons use paired success differences, paired bootstrap 95%
intervals, exact McNemar tests, and Holm correction over the declared family.
Outcome significance is not an artifact-integrity gate: the release remains
valid when a method loses.

### 4.3 Exact 10 ms control slots

The compute-matched block separates calibration from evaluation. For each of
three planners and five tasks, 25 calibration seeds select the largest \(K\)
with zero calibration deadline misses, producing 375 calibration episodes.
Thirty disjoint held-out seeds per cell produce 450 evaluation episodes.

Every planner receives an enforced 10 ms control slot. `real_time_success`
requires both task success and zero deadline misses. The calibration selected
\(K=1024\) for MPPI, Diff-MPPI-3, and SOPPI-fast on the reference GPU; fairness
therefore comes from the enforced slot rather than unequal sample counts.

### 4.4 MuJoCo transfer

The external-fidelity matrix contains 3,150 closed-loop episodes:

| Dimension | Values |
|---|---|
| Plant | MuJoCo 3.11.0 custom planar box MJCF |
| Conditions | 7 nominal, friction, mass, and observation-noise settings |
| Tasks | 5 |
| Planners | MPPI, Diff-MPPI-3, SOPPI-fast |
| Budget | \(K=256\) |
| Seeds | 30 paired seeds per cell |

The same paired bootstrap, McNemar, and full-family Holm procedure is applied.
Observation-noise cells are not promoted when their individual effects do not
survive correction.

### 4.5 Hardware and evidence freeze

All frozen results were generated on an NVIDIA GeForce GTX 1660 Ti. Each block
records its source commit, clean/dirty state, GPU identity, commands, matrix
shape, raw CSV hashes, report hashes, and validator result. Independent
hardware replication is desirable but is not implied by the current ledger.

## 5. Results

### 5.1 Fixed-seed contact signal

The earlier four-seed \(K=256\) suite establishes the motivating signal. On
`box_align_contact_loss`, Diff-MPPI-3 and SOPPI-fast solve 4/4 runs while MPPI
solves 0/4. The same suite is not universally positive:
`box_align_detour` remains 0/4 for MPPI and only 1/4 for Diff-MPPI-3. We use
this block as a fixed-seed signal, not the statistical headline.

### 5.2 Broad robustness

| Planner | Episodes | Success | Mean control ms |
|---|---:|---:|---:|
| Diff-MPPI-3 | 5,400 | 0.614 | 2.655 |
| Diff-MPPI-1 | 5,400 | 0.595 | 0.959 |
| SOPPI-fast | 5,400 | 0.557 | 1.464 |
| MPPI | 5,400 | 0.453 | 0.120 |
| SOPPI | 5,400 | 0.446 | 0.426 |
| MPPI-hardmodel | 5,400 | 0.331 | 0.159 |

Across the 360 paired comparisons versus MPPI, there are 33 Holm-significant
positive success cells and 6 Holm-significant negative success cells. All six
negative cells are Diff-MPPI-3 under the tall-box condition. Thus, the
aggregate ordering favors the three-step hybrid, while object geometry exposes
a repeatable regression that forbids universal-dominance wording.

The hard-contact friction/damping conditions reproduce positive cells under a
momentum-carrying plant that is structurally different from the nominal smooth
model. This supports `hard_contact_transfer` as sim-to-sim evidence; it does
not replace the MuJoCo or real-robot experiments.

### 5.3 Deadline-matched control

| Planner | Held-out episodes | Real-time success | Deadline misses |
|---|---:|---:|---:|
| Diff-MPPI-3 | 150 | 0.800 | 1 |
| SOPPI-fast | 150 | 0.793 | 55 |
| MPPI | 150 | 0.673 | 4 |

The aggregate row mixes tasks with different difficulty, so the registered
paired family is the primary interpretation. On `box_align_contact_loss`,
Diff-MPPI-3 reaches 1.00 real-time success versus 0.467 for MPPI. The paired
difference is +0.533 with bootstrap 95% CI \([+0.367,+0.700]\) and Holm
\(p=0.000305\). SOPPI-fast reaches 0.967 and improves by +0.500 with Holm
\(p=0.002472\).

The other four task cells are not Holm-significant. In particular, every
planner remains 0/30 on `box_align_detour`. The equal-slot experiment
therefore identifies one strong compute-matched contact-loss result, not broad
matched-time superiority.

### 5.4 Closed-loop MuJoCo transfer

| Planner | Episodes | Success | Mean control ms |
|---|---:|---:|---:|
| Diff-MPPI-3 | 1,050 | 0.457 | 2.614 |
| SOPPI-fast | 1,050 | 0.340 | 1.432 |
| MPPI | 1,050 | 0.289 | 0.117 |

The 70-cell paired family contains three Holm-significant positive
Diff-MPPI-3 cells and zero negative cells. The positive cells are
`box_align_strict` at friction 0.3 and mass scale 0.75, each with success delta
+0.50, and `box_align_contact_arc` at mass scale 1.25 with delta +0.40.

No individual observation-noise cell survives full-family Holm correction.
Noise effects are therefore descriptive. Detour remains zero-success across
planners and conditions, which indicates that local refinement does not solve
the missing topological search problem.

## 6. Claim ledger

| Claim ID | Status | Supported interpretation |
|---|---|---|
| `fixed_seed_contact_signal` | Supported | 4/4 hybrid versus 0/4 MPPI signal on contact loss |
| `contact_suite_robustness` | Supported | 32,400-episode family with positive and negative corrected cells |
| `matched_compute_contact` | Supported | One Holm-significant contact-loss win under the enforced 10 ms slot |
| `contact_model_fidelity` | Supported | Closed-loop MuJoCo aggregate and three corrected positive cells |
| `hard_contact_transfer` | Supported | Structurally different hard-contact friction/damping family |
| `negative_result_detour` | Supported | Retained 0/4 versus 1/4 fixed-seed negative control |

The ledger being `ready: true` means all declared submission-required claims
have valid content-bound evidence. It does not mean the method wins every cell,
that the paper demonstrates real-robot transfer, or that final venue packaging
is complete.

## 7. Discussion

The experiments suggest a specific division of labor. MPPI provides mode-level
search. When that search enters a contact-success basin, a few local gradient
steps can correct alignment or contact loss without replacing the sampler.
The contact-loss result survives an enforced control slot and the aggregate
advantage transfers to an independently executed MuJoCo plant.

The failure structure is equally informative. Tall objects produce six
corrected negative cells for Diff-MPPI-3, implying sensitivity to geometry or
gradient scaling. Detour stays unsolved because its obstacle-induced topology
requires a different mode rather than a sharper local update. SOPPI-fast is
competitive, but its nominal gradient step means the comparison does not
isolate pure SVGD from differentiation.

These findings motivate a localized claim: post-MPPI autodiff refinement can
improve selected contact-rich control cells under fixed samples, enforced
deadlines, and sim-to-sim model mismatch. They do not support replacing MPPI,
claiming universal hybrid superiority, or asserting real-world manipulation.

## 8. Limitations

- All frozen experiments use one GTX 1660 Ti.
- The nominal and hard-contact robustness plants are custom GPU simulations.
- The external plant is a custom planar MuJoCo MJCF, not a standard
  manipulator suite.
- MuJoCo transfer is closed-loop sim-to-sim transfer, not real-robot evidence.
- No observation-noise cell survives full-family correction.
- `box_align_detour` remains unsolved in the deadline-matched and MuJoCo
  families.
- The method has not been evaluated for deformable contact, 3D grasping, or
  hardware safety constraints.

## 9. Reproduction

Validate the frozen ledger and regenerate the results chapter:

```bash
python3 scripts/validate_paper_artifacts.py \
  paper/artifacts/contact_rich_diff_mppi.json --require-ready
python3 scripts/render_contact_paper_results.py --check
```

The three release blocks are reproduced with:

```bash
python3 scripts/run_contact_robustness.py \
  --output-dir build/contact_robustness_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile release
python3 scripts/run_contact_matched_compute.py \
  --output-dir build/contact_matched_compute_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --profile release
python3 scripts/run_contact_external_fidelity.py \
  --output-dir build/contact_external_fidelity_release \
  --binary bin/benchmark_diff_mppi_pushing_box \
  --model mujoco_models/contact_box_push.xml \
  --profile release
```

Protocols and retained artifacts:

- [`contact_diff_mppi_robustness.md`](../docs/contact_diff_mppi_robustness.md)
- [`contact_matched_compute.md`](../docs/contact_matched_compute.md)
- [`contact_external_fidelity.md`](../docs/contact_external_fidelity.md)
- [`contact_rich_diff_mppi_results.md`](contact_rich_diff_mppi_results.md)

## 10. Conclusion

Contact-Rich Diff-MPPI adds a short, training-free local refinement to the
standard MPPI update. A 32,400-episode robustness family, an enforced 10 ms
held-out evaluation, and closed-loop MuJoCo transfer show significant positive
contact cells alongside explicit tall-box and detour failures. The evidence
supports a narrow compute-quality advantage for selected contact-rich tasks.
It does not establish universal planner dominance or physical-robot transfer.
