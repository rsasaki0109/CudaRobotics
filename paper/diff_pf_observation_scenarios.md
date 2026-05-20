# Differentiable Particle Filter Observation-Model Scenarios

Last updated: 2026-05-20 JST

This note consolidates the `diff_pf_mlp` observation-model experiments into a
paper-facing table. All rows use the same 8-landmark 40 m x 30 m localization
setup, 1,024 particles for evaluation, soft resampling, and `alpha=3.14`
learned by `src/diff_pf.cu`. The MLP observation model is 2 -> 16 -> 1 and
receives scaled `(distance, residual)` inputs.

## Scenario Matrix

| Scenario | Observation mismatch | Gaussian likelihood RMSE | Supervised MLP RMSE | Tracking-tuned MLP RMSE | Direct-surrogate MLP RMSE | Best learned ratio vs Gaussian |
|---|---|---:|---:|---:|---:|---:|
| Clean Gaussian | None; simulator noise matches analytic Gaussian | 6.97 m | 7.19 m | **6.16 m** | not run | **0.88x** |
| Range outliers | 18% of visible ranges get uniform +/-9 m outliers | 10.27 m | 7.45 m | **6.91 m** | not run | **0.67x** |
| Distance-dependent bias | `z = d + N(0, 1.0) + 0.35 * max(0, d - 10.0)` | 8.33 m | 6.21 m | 6.94 m | **5.81 m** | **0.70x** |
| Occlusion + hidden kidnap | 30% landmark dropout + 25% short returns from 1.0-16.0 s; hidden pose jump (-4 m, +3 m) at 3.0 s | 10.38 m | 7.66 m | **7.56 m** | not run | **0.73x** |

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
mismatch without particle depletion from a hidden state jump. Two-seed
finite-difference tracking remains noisy in this run, reaching 6.94 m. A direct
biased-observation surrogate trained with ordinary GPU backprop is both faster
and better: 5.81 m after about 2.9 s of surrogate training, versus 8.33 m for
the Gaussian baseline.

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
3. When a differentiable observation surrogate is available, ordinary GPU
   backprop can train the same MLP much faster than rollout finite differences.
4. Under misspecified observation models, the best learned likelihoods reduce
   localization RMSE by 27-33% versus the handcrafted Gaussian baseline.

## Next Useful Ablation

The next clean experiment is to remove the hand-coded biased sensor from the
direct surrogate and learn it from calibration traces. That would keep the
fast direct-gradient path while making the observation model useful for unknown
range bias, outliers, or occlusion mixtures.
