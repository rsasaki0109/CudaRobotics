# Differentiable Particle Filter Observation-Model Scenarios

Last updated: 2026-05-21 JST

This note consolidates the `diff_pf_mlp` observation-model experiments into a
paper-facing table. All rows use the same 8-landmark 40 m x 30 m localization
setup, 1,024 particles for evaluation, soft resampling, and `alpha=3.14`
learned by `src/diff_pf.cu`. The MLP observation model is 2 -> 16 -> 1 and
receives scaled `(distance, residual)` inputs.

## Scenario Matrix

| Scenario | Observation mismatch | Gaussian likelihood RMSE | Supervised MLP RMSE | Tracking-tuned MLP RMSE | Calibrated-surrogate MLP RMSE | Best learned ratio vs Gaussian |
|---|---|---:|---:|---:|---:|---:|
| Clean Gaussian | None; simulator noise matches analytic Gaussian | 6.97 m | 7.19 m | **6.16 m** | not run | **0.88x** |
| Range outliers | 18% of visible ranges get uniform +/-9 m outliers | 9.80 m | 7.18 m | 7.22 m | **7.04 m** | **0.72x** |
| Distance-dependent bias | `z = d + N(0, 1.0) + 0.35 * max(0, d - 10.0)` | 8.41 m | 7.11 m | 7.03 m | **6.03 m** | **0.72x** |
| Occlusion + hidden kidnap | 30% landmark dropout + 25% short returns from 1.0-16.0 s; hidden pose jump (-4 m, +3 m) at 3.0 s | 8.25 m | 7.40 m | 8.08 m | **6.93 m** | **0.84x** |
| Occlusion only | 30% landmark dropout + 25% short returns from 1.0-16.0 s; no pose jump | 7.98 m | 7.27 m | **6.69 m** | 6.99 m | **0.84x** |
| Kidnap only | Clean Gaussian observations; hidden pose jump (-4 m, +3 m) at 3.0 s | 7.66 m | 8.24 m | 12.27 m | **6.90 m** | **0.90x** |

## Interpretation

The clean Gaussian scenario is intentionally conservative: the handcrafted
likelihood is correctly specified, so the MLP is not expected to win by
model-class advantage. It still validates the DPF plumbing: the learned
observation model can be inserted into the particle-filter update and tuned
through the full tracking loss.

The range-outlier scenario is the clearest robust-tail calibration result. A
trace of 8,192 known-distance samples estimates a 96-bin residual likelihood
over the non-Gaussian tail. Training the MLP against that calibrated density
with ordinary GPU backprop reaches 7.04 m, ahead of the Gaussian baseline at
9.80 m and the rollout finite-difference MLP at 7.22 m in this run.

The distance-dependent bias scenario isolates systematic observation-model
mismatch without particle depletion from a hidden state jump. Two-seed
finite-difference tracking remains noisy in this run, reaching 7.03 m. The
calibrated surrogate estimates a 24-bin bias curve and residual sigma from
8,192 known-distance calibration traces, then trains the observation MLP with
ordinary GPU backprop. That trace-learned path reaches 6.03 m after about
2.8 s of surrogate training, versus 8.41 m for the Gaussian baseline.

The occlusion+kidnap scenario mixes two failure modes: corrupted/missing
measurements and a hidden state jump. The MLP still improves over the Gaussian
likelihood, but the calibrated result is more constrained than the outlier-only
and bias-only scenes. The trace-learned surrogate observes known-distance
measurements during the occlusion window and skips dropouts, matching the PF
update where invalid landmarks contribute no likelihood. It therefore learns
the valid short-return residual tail, reaching 6.93 m versus 8.25 m for the
Gaussian baseline. Dropout and kidnap recovery still depend on particle support
and resampling mechanics, not only on the observation likelihood.

The follow-up ablations split those two effects. In occlusion-only, the
tracking-tuned MLP reaches 6.69 m versus 7.98 m for the Gaussian baseline,
while the calibrated valid-residual surrogate reaches 6.99 m. In kidnap-only,
clean Gaussian observations make the learned likelihoods sensitive to particle
support after the hidden jump; the calibration-learned Gaussian residual
surrogate lands at 6.90 m versus 7.66 m for the handcrafted Gaussian, while
the rollout-tuned MLP is worse in this seeded run. That isolates the remaining
hard part as recovery after an unmodeled state jump, not just
observation-model mismatch.

## Particle-Support Recovery

The kidnap-only recovery follow-up adds range-reset particle injection after
the hidden jump. At each step after `t=3.0s`, 12% of the resampled particles
are replaced by samples drawn on measured landmark-range circles, then the
usual likelihood update selects particles consistent with the full observation
set.

| Kidnap recovery variant | RMSE | Ratio vs no-injection Gaussian |
|---|---:|---:|
| Gaussian likelihood | 7.30 m | 1.00x |
| Gaussian likelihood + range-reset injection | **1.01 m** | **0.14x** |
| Calibrated-surrogate MLP likelihood | 5.87 m | 0.80x |
| Calibrated-surrogate MLP + range-reset injection | 1.08 m | 0.15x |

This closes most of the hidden-jump gap without changing the observation
likelihood, confirming that kidnap recovery is primarily a particle-support
problem once the measurements are clean.

## Current Research Claim

`diff_pf_mlp` now demonstrates a progression:

1. A supervised MLP can replace the analytic Gaussian likelihood.
2. The MLP weights can be fine-tuned end-to-end through a soft-resampling DPF
   rollout using finite-difference tracking gradients.
3. When calibration traces are available, a differentiable observation
   surrogate can be estimated and used to train the same MLP much faster than
   rollout finite differences.
4. Under misspecified observation models, the best learned likelihoods reduce
   localization RMSE by 10-28% versus the handcrafted Gaussian baseline across
   the current stress and ablation matrix.
5. In the kidnap-only case, range-reset particle injection reduces RMSE by 86%
   versus the no-injection Gaussian baseline, showing that explicit support
   recovery is the right tool for hidden state jumps.

## Next Useful Ablation

The next clean experiment is an injection trigger/rate ablation: compare fixed
12% range-reset injection against an effective-sample-size trigger, lower
steady-state rates, expansion reset, and the existing AMCL-style recovery
baseline.
