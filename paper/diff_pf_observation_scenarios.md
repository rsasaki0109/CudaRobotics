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
| Distance-dependent bias | `z = d + N(0, 1.0) + 0.35 * max(0, d - 10.0)` | 8.33 m | 6.76 m | **6.64 m** | **0.80x** |
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
mismatch without particle depletion from a hidden state jump. The first
single-seed finite-difference run improved over the Gaussian but trailed the
supervised MLP. Averaging the finite-difference gradient over two rollout seeds
and restoring the best held-out validation checkpoint reduces the
tracking-tuned result to 6.64 m, narrowly ahead of the supervised MLP at
6.76 m and well ahead of the Gaussian at 8.33 m.

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
   localization RMSE by 20-33% versus the handcrafted Gaussian baseline.
   Tracking-loss tuning now wins on range outliers, distance-dependent bias,
   and occlusion+kidnap.

## Next Useful Ablation

The next clean experiment is to replace the finite-difference MLP update with a
smoother resampling relaxation or direct differentiable surrogate. The two-seed
central-difference update is enough to recover the biased-range scene, but it
still takes roughly one minute for a tiny 65-parameter MLP because each epoch
reruns many full PF rollouts.
