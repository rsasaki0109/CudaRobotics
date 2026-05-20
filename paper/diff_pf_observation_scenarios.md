# Differentiable Particle Filter Observation-Model Scenarios

Last updated: 2026-05-20 JST

This note consolidates the `diff_pf_mlp` observation-model experiments into a
paper-facing table. All rows use the same 8-landmark 40 m x 30 m localization
setup, 1,024 particles for evaluation, soft resampling, and `alpha=3.14`
learned by `src/diff_pf.cu`. The MLP observation model is 2 -> 16 -> 1 and
receives scaled `(distance, residual)` inputs.

## Scenario Matrix

| Scenario | Observation mismatch | Gaussian likelihood RMSE | Supervised MLP RMSE | Tracking-tuned MLP RMSE | Best learned ratio vs Gaussian |
|---|---|---:|---:|---:|---:|
| Clean Gaussian | None; simulator noise matches analytic Gaussian | 6.97 m | 7.19 m | **6.16 m** | **0.88x** |
| Range outliers | 18% of visible ranges get uniform +/-9 m outliers | 10.27 m | 7.45 m | **6.91 m** | **0.67x** |
| Distance-dependent bias | `z = d + N(0, 1.0) + 0.35 * max(0, d - 10.0)` | 8.33 m | **6.13 m** | 7.20 m | **0.74x** |
| Occlusion + hidden kidnap | 30% landmark dropout + 25% short returns from 1.0-16.0 s; hidden pose jump (-4 m, +3 m) at 3.0 s | 10.38 m | 7.66 m | **7.56 m** | **0.73x** |

## Interpretation

The clean Gaussian scenario is intentionally conservative: the handcrafted
likelihood is correctly specified, so the MLP is not expected to win by
model-class advantage. It still validates the DPF plumbing: the learned
observation model can be inserted into the particle-filter update and tuned
through the full tracking loss.

The range-outlier scenario is the clearest learned-observation result. The
analytic Gaussian over-penalizes large residuals caused by injected outliers,
while tracking-loss tuning pushes the MLP toward a heavier-tailed effective
likelihood. The result improves from 10.27 m to 6.91 m.

The distance-dependent bias scenario isolates systematic observation-model
mismatch without particle depletion from a hidden state jump. The learned MLP
likelihood improves over the Gaussian baseline, but the supervised MLP is the
best row entry: 6.13 m versus 8.33 m. Tracking-loss fine-tuning still beats the
Gaussian at 7.20 m, but this scene exposes the current finite-difference
tuning noise more clearly than the outlier-only case.

The occlusion+kidnap scenario mixes two failure modes: corrupted/missing
measurements and a hidden state jump. The MLP still improves over the Gaussian
likelihood, but the gap is smaller than in the outlier-only scene because
kidnap recovery also depends on particle support and resampling mechanics, not
only on the observation likelihood.

## Current Research Claim

`diff_pf_mlp` now demonstrates a progression:

1. A supervised MLP can replace the analytic Gaussian likelihood.
2. The MLP weights can be fine-tuned end-to-end through a soft-resampling DPF
   rollout using finite-difference tracking gradients.
3. Under misspecified observation models, learned likelihoods reduce
   localization RMSE by 14-33% versus the handcrafted Gaussian baseline.
   Tracking-loss tuning wins on range outliers and occlusion+kidnap; the
   supervised likelihood wins on the systematic biased-range scene.

## Next Useful Ablation

The next clean experiment is to reduce the finite-difference tuning variance
on the biased-range scene: average gradients over multiple rollout seeds,
increase the training horizon, or replace the current hard central-difference
loop with a smoother resampling relaxation. The target is to see whether
tracking-loss tuning can recover the supervised MLP's 6.13 m biased-range
result without overfitting to rollout noise.
