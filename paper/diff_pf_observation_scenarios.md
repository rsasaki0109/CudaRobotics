# Differentiable Particle Filter Observation-Model Scenarios

Last updated: 2026-05-20 JST

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
| Occlusion + hidden kidnap | 30% landmark dropout + 25% short returns from 1.0-16.0 s; hidden pose jump (-4 m, +3 m) at 3.0 s | 10.38 m | 7.66 m | **7.56 m** | not run | **0.73x** |

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
likelihood, but the gap is smaller than in the outlier-only scene because
kidnap recovery also depends on particle support and resampling mechanics, not
only on the observation likelihood.

## Current Research Claim

`diff_pf_mlp` now demonstrates a progression:

1. A supervised MLP can replace the analytic Gaussian likelihood.
2. The MLP weights can be fine-tuned end-to-end through a soft-resampling DPF
   rollout using finite-difference tracking gradients.
3. When calibration traces are available, a differentiable observation
   surrogate can be estimated and used to train the same MLP much faster than
   rollout finite differences.
4. Under misspecified observation models, the best learned likelihoods reduce
   localization RMSE by roughly 27-28% versus the handcrafted Gaussian
   baseline across the current misspecified scenario matrix.

## Next Useful Ablation

The next clean experiment is to feed calibration traces from occlusion mixtures
into the same surrogate path. That tests whether a trace-learned likelihood can
handle missing/short returns, not only smooth bias or range outlier tails.
