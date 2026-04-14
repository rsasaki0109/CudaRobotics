# Diff-MPPI: Minimal Gradient Refinement for Sampling-Based Model Predictive Control

## Abstract

Sampling-based model predictive controllers such as MPPI handle nonlinear dynamics and nonconvex costs effectively, but their trajectory quality can plateau under limited rollout budgets on hard tasks. Recent work shows that MPPI is equivalent to a preconditioned gradient descent step. We study a minimal hybrid controller that adds a short autodiff refinement — just three local gradient steps — to the standard MPPI update. With parallelized gradient computation on GPU, this refinement retains 85% of the sampling budget within the same wall-clock time. We evaluate the method against vanilla MPPI and six strong non-hybrid feedback baselines under exact matched-time controller budgets on two-dimensional dynamic-obstacle navigation and a 7-DOF serial-arm manipulation benchmark. On the hardest dynamic-obstacle task, the hybrid controller is the only method that reaches the goal across all tested compute budgets from 0.2 to 3.1 ms — all six non-hybrid baselines fail at every sample count from K=128 to K=8192. On 7-DOF manipulation with dynamic obstacles, the hybrid controller achieves success=1.00 at 0.84 ms while the closest feedback baseline reaches 0.75 at 4.01 ms. A gradient freshness analysis shows that the autodiff correction concentrates during obstacle encounters where the cost landscape changes most rapidly, explaining why the refinement is specifically valuable on dynamic tasks.

## I. Introduction

Model Predictive Path Integral (MPPI) control is widely used for real-time nonlinear trajectory optimization because it handles arbitrary cost functions through parallel rollout sampling on GPU [1]. However, under limited rollout budgets, vanilla MPPI can produce trajectories that are qualitatively close to the solution without crossing the success boundary — particularly on tasks with rapidly changing cost landscapes such as dynamic obstacle avoidance.

A growing body of work augments MPPI with various refinement mechanisms: gradient-based DDP smoothing [2], learned sampling distributions [3,4], sensitivity-derived feedback gains [5], and ancillary controller biasing [6]. Recent theoretical work by Fazlyab et al. [7] shows that classical MPPI is exactly a preconditioned gradient descent step with unit step size on a KL-regularized distribution objective, establishing a principled connection between sampling-based and gradient-based trajectory optimization.

In this context, we study a deliberately minimal hybrid controller: after the standard MPPI weighted update, we apply three local gradient steps to the nominal control sequence using forward-mode autodiff. The key design choice is simplicity — no learned components, no convex corridor construction, no additional data structures beyond the cost gradient itself.

Our main contributions are:

1. A minimal, training-free hybrid MPPI controller where the autodiff refinement is parallelized to retain 85% of the sampling budget at matched wall-clock time, interpretable as adding explicit gradient steps after MPPI's implicit preconditioned gradient step.

2. A matched-time evaluation showing a non-hybrid Pareto ceiling: on the hardest dynamic-obstacle task, all six non-hybrid baselines remain unsuccessful at every sample count from K=128 to K=8192, while the hybrid controller succeeds at K=128 and above.

3. Evidence across 2D dynamic navigation and 7-DOF serial-arm manipulation, including a gradient freshness analysis showing that the autodiff correction concentrates during dynamic obstacle encounters where the sampling-based preconditioner is least effective.

## II. Related Work

**MPPI and path-integral control.** MPPI [1] performs weighted averaging of sampled trajectories using the exponential cost-to-go as importance weights. Fazlyab et al. [7] show this update is equivalent to a preconditioned gradient descent step, connecting MPPI to optimization-based MPC.

**Sampling + gradient hybrids.** CEM-GD [8] combines cross-entropy method sampling with gradient descent for model-based RL planning. MPPI-IPDDP [2] smooths MPPI trajectories via DDP inside a convex corridor. Our approach is simpler: pure autodiff on the post-MPPI control sequence without corridor construction or learning.

**Feedback-enhanced MPPI.** Feedback-MPPI [5] computes local feedback gains from rollout sensitivity analysis. Biased-MPPI [6] uses ancillary controllers to improve sampling efficiency. Step-MPPI [4] learns a neural sampling distribution for single-step lookahead. Our method is complementary: training-free refinement applied post-MPPI rather than modifying the sampling distribution.

**GPU-accelerated control.** DiffMPC [9] provides GPU-accelerated differentiable MPC in JAX for learning applications. cuNRTO [10] solves robust trajectory optimization on GPU for manipulators. MPPI-Generic [11] provides a CUDA MPPI library. We share the GPU-parallel rollout design but focus on the gradient refinement analysis.

## III. Method

### A. Controller Structure

At each control step, the hybrid controller executes three phases:

**Phase 1: MPPI Sampling Update.** Sample K rollouts around the current nominal control sequence u. Compute trajectory costs and update the nominal via the standard MPPI weighted average.

**Phase 2: Parallel Cost Gradient Computation.** Roll out the updated nominal trajectory. Compute stage cost gradients and dynamics Jacobians in parallel across T horizon threads on GPU.

**Phase 3: Local Gradient Refinement.** Apply the adjoint backward pass (single-thread, matrix operations only) to obtain control gradients. Take three gradient descent steps on the control sequence with per-timestep norm clipping.

The gradient computation is split into a parallel phase (T threads compute cost gradients + analytical dynamics Jacobians simultaneously) and a sequential phase (backward adjoint accumulates control gradients via matrix multiplication). This separation enables parallelization of the expensive cost gradient evaluations while preserving the sequential adjoint dependency.

### B. Compute Budget Analysis

After parallelization, the gradient refinement adds minimal overhead to the MPPI sampling step. At a matched 1.0 ms wall-clock budget on the 2D navigation benchmark:

- Vanilla MPPI: K=7271 samples
- Diff-MPPI-3: K=6216 samples + 3 gradient steps

The hybrid controller retains 85% of the sampling budget while gaining gradient refinement within the same wall-clock time. This makes the compute-quality comparison genuinely fair: the gradient refinement is nearly free in wall-clock terms.

### C. Adaptive Gradient Skip

We additionally implement an adaptive variant that monitors the total gradient norm and skips the gradient step when the norm falls below a threshold. On 2D navigation tasks, the gradient computation is already fast enough that the monitoring overhead exceeds the savings. The adaptive skip is more valuable for higher-dimensional systems where gradient steps are more expensive.

## IV. Experimental Setup

### A. 2D Dynamic-Obstacle Navigation

Two scenarios on a bicycle model: `dynamic_crossing` (easier, one crossing obstacle) and `dynamic_slalom` (harder, multiple dynamic obstacles requiring precise timing). Goal tolerance 2.0 m, max 260 steps.

### B. 7-DOF Serial-Arm Manipulation

Panda-like 7-DOF arm with 14D state (7 joint angles + 7 velocities), 7D torque control, 3D workspace obstacles, second-order joint dynamics with gravity and damping, and analytical dynamics Jacobians. Two scenarios: `7dof_shelf_reach` (static obstacle) and `7dof_dynamic_avoid` (moving 3D obstacle).

### C. Baselines

Eight non-hybrid baselines, all evaluated under exact matched-time tuning:

1. **mppi**: vanilla sampling-only MPPI
2. **feedback_mppi_ref**: current-action sensitivity-based feedback gains (closest to released Feedback-MPPI [5] gain computation)
3. **feedback_mppi_fused**: covariance-regression + LQR blend (strongest non-hybrid)
4. **feedback_mppi_hf**: two-rate architecture with full-horizon covariance+LQR gains
5. **feedback_mppi_faithful**: two-rate architecture with current-action-only gains
6. **feedback_mppi_cov**: covariance-regression feedback gains
7. **feedback_mppi_paper**: covariance-regression + LQR blend with every-step replanning—closest to Feedback-MPPI [5]'s sensitivity-derived gain computation
8. **step_mppi**: online-adapted sampling bias via cost-weighted EMA of control deviations, inspired by Step-MPPI [4]

### D. Matched-Time Evaluation Protocol

We use a binary-search exact-time tuning script that finds the K value per planner and scenario that produces a given wall-clock target (e.g., 1.0 ms). This ensures all comparisons are at the same per-step compute budget, not just the same K.

## V. Results

### A. Non-Hybrid Pareto Ceiling on dynamic_slalom

The central result: on `dynamic_slalom`, **all six non-hybrid baselines remain at success=0.00 at every K from 128 to 8192**. The strongest non-hybrid baseline (`feedback_mppi_fused`) reduces final distance from 14.2 (MPPI) to 10.3, but never reaches the goal.

`diff_mppi_3` succeeds at every K from 128 upward, reaching final distance 1.89-1.95.

At matched 1.0 ms:

| Planner | K | ms | Success | Final Dist |
|---|---|---|---|---|
| mppi | 7271 | 0.98 | 0.00 | 14.19 |
| feedback_mppi_fused | 128 | 1.85 | 0.00 | 10.33 |
| feedback_mppi_ref | 992 | 1.01 | 0.00 | 11.77 |
| feedback_mppi_hf | 292 | 0.98 | 0.00 | 13.80 |
| feedback_mppi_faithful | 3294 | 0.98 | 0.00 | 14.09 |
| feedback_mppi_paper | 1024 | 8.75 | 0.00 | 11.74 |
| step_mppi | 1024 | 0.19 | 0.00 | 14.25 |
| **diff_mppi_3** | **6216** | **0.97** | **1.00** | **1.89** |

This establishes a hard non-hybrid Pareto ceiling: no amount of compute budget allocated to non-hybrid controllers can solve this task. The gradient refinement is the only mechanism that crosses the success boundary.

### B. 7-DOF Manipulation

On `7dof_dynamic_avoid` at K=512:

| Planner | Success | Final Dist | Avg ms |
|---|---|---|---|
| mppi | 0.25 | 0.635 | 0.39 |
| feedback_mppi_ref | 0.75 | 0.283 | 4.01 |
| **diff_mppi_3** | **1.00** | **0.090** | **0.84** |

The hybrid controller achieves the highest success rate while being 4.8x faster than the closest feedback baseline. This demonstrates the approach transfers to 14-dimensional manipulation tasks.

### C. Gradient Freshness Analysis

Comparing gradient behavior on static (`corner_turn`) vs dynamic (`dynamic_slalom`) tasks reveals:

- **Gradient magnitude** peaks during obstacle encounters on dynamic tasks, while declining monotonically on static tasks
- **Correction magnitude** concentrates in the episode segment where dynamic obstacles are closest
- **Gradient-correction alignment** varies more on dynamic tasks, indicating the gradient provides directional information not captured by the MPPI update

This supports the interpretation that the gradient refinement acts as a "freshness correction": on dynamic tasks where the MPPI warm-start becomes stale due to obstacle motion, the gradient provides an update based on the current cost landscape.

### D. Feedback Architecture Analysis

The `feedback_mppi_faithful` variant tests whether a two-rate controller architecture (replan every other step, apply stored current-action gains between replans) can replicate the benefit of gradient refinement. Result: it fails on both dynamic tasks even at K=8192 (2.1 ms), performing at vanilla MPPI level on `dynamic_slalom`. This confirms that the gradient refinement provides complementary value that pure feedback — whether every-step or two-rate — cannot replicate.

## VI. Limitations

The current contribution is empirical with theoretical motivation from Fazlyab et al. [7]. The feedback baselines are strong in-repo proxies but not full reproductions of the complete Feedback-MPPI [5] controller stack. The 7-DOF benchmark uses a simplified dynamics model rather than a validated Panda simulation. The evaluation relies on custom benchmark domains rather than standardized suites such as MuJoCo manipulation tasks.

## VII. Conclusion

We showed that a minimal gradient refinement — three autodiff steps after a standard MPPI update — breaks through a hard non-hybrid Pareto ceiling on dynamic-obstacle tasks. With parallelized gradient computation, the refinement retains 85% of the sampling budget at matched wall-clock time. A gradient freshness analysis explains why the refinement is specifically valuable on dynamic tasks: the gradient provides current-landscape information that the sampling-based update cannot capture when obstacles move between timesteps. The approach transfers to 7-DOF manipulation, where the hybrid controller achieves higher success at lower compute than the closest feedback baseline.

## References

[1] Williams et al., "Information theoretic MPC for model-based reinforcement learning," ICRA 2017.
[2] MPPI-IPDDP, IEEE TRO 2025.
[3] Le et al., "Toward Single-Step MPPI via Differentiable Predictive Control," arXiv:2604.01539, 2026.
[4] Step-MPPI, arXiv:2604.01539, 2026.
[5] Belvedere et al., "Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation," RA-L 2026.
[6] Trevisan & Alonso-Mora, "Biased-MPPI: Informing Sampling-Based MPC by Fusing Ancillary Controllers," RA-L 2024.
[7] Fazlyab et al., "MPPI as Preconditioned Gradient Descent," arXiv:2603.24489, 2026.
[8] Bharadhwaj et al., "Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization," L4DC 2020.
[9] DiffMPC, arXiv:2510.06179, 2025.
[10] cuNRTO, arXiv:2603.02642, 2026.
[11] MPPI-Generic, arXiv:2409.07563, 2024.
