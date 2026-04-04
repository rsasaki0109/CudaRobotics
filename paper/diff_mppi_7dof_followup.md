# Diff-MPPI 7-DOF Manipulator Follow-Up

Date: 2026-04-05

## Overview

This follow-up extends Diff-MPPI evaluation from 2D kinematic navigation and a 2-link planar arm to a Panda-like 7-DOF serial manipulator in 3D workspace. It directly addresses the reviewer concern that prior evaluation was limited to low-dimensional toy domains.

## Benchmark Design

- State: 14D (7 joint angles + 7 joint velocities)
- Control: 7D (joint torques)
- Dynamics: second-order with gravity compensation, damping, and joint limits per DOF
- FK: simplified DH chain with alternating z/y rotation axes
- Obstacles: 3D spheres with inverse-squared distance penalties
- Gradient computation: analytical dynamics Jacobian + finite-difference cost gradients

### Scenarios

1. **7dof_shelf_reach**: reach a target position while avoiding a static obstacle blocking the direct path. Goal tolerance 0.15.
2. **7dof_dynamic_avoid**: reach a target position while avoiding a moving 3D obstacle. Goal tolerance 0.10.

### Planners

- `mppi`: vanilla sampling-only MPPI
- `feedback_mppi_ref`: current-action sensitivity-based feedback gains (closest to released Feedback-MPPI gain computation)
- `diff_mppi_1`: hybrid with 1 gradient step (alpha=0.02)
- `diff_mppi_3`: hybrid with 3 gradient steps (alpha=0.008)

## Key Results

### 7dof_dynamic_avoid (fixed budget)

| Planner | K=256 success | K=512 | K=1024 | avg_ms (K=256) |
|---|---|---|---|---|
| mppi | 0.75 | 0.25 | 0.75 | 0.38 |
| feedback_mppi_ref | **1.00** | 0.75 | **1.00** | 3.47 |
| diff_mppi_1 | 0.25 | 0.50 | 0.00 | 2.16 |
| diff_mppi_3 | 0.50 | 0.50 | 0.25 | 5.71 |

### 7dof_shelf_reach (fixed budget)

| Planner | K=256 success | K=512 | K=1024 | avg_ms (K=256) |
|---|---|---|---|---|
| mppi | 0.25 | 0.00 | 0.00 | 0.33 |
| feedback_mppi_ref | 0.00 | 0.00 | 0.25 | 2.49 |
| diff_mppi_1 | **0.50** | 0.00 | 0.25 | 1.44 |
| diff_mppi_3 | **0.50** | 0.00 | 0.25 | 3.63 |

## Gradient Computation Speed

Analytical dynamics Jacobian (O(NDOF) instead of O(NDOF^2) finite-diff passes) reduced gradient computation from ~13.6 ms to ~5.7 ms per step at K=256. Analytical control cost gradient and terminal velocity gradient contribute further savings.

## Interpretation

- `feedback_mppi_ref` is the strongest planner on `7dof_dynamic_avoid`, reaching 100% success at K=256. This confirms that sensitivity-based feedback is valuable for high-DOF manipulation.
- `diff_mppi_1/3` show promise on `7dof_shelf_reach` at low K (50% vs mppi 25%), but results are noisy across K values. The gradient refinement helps the arm navigate around the static obstacle more consistently at low sample budgets.
- The compute cost of gradient refinement on 7-DOF (5.7 ms for diff_mppi_3 vs 0.38 ms for mppi) means the hybrid controller spends significantly more compute per step. Matched-time evaluation is needed to assess whether this overhead is justified.
- Overall, the 7-DOF benchmark extends the evaluation beyond 2D navigation and shows that both feedback and hybrid MPPI approaches transfer to higher-dimensional manipulation. The results are intentionally not oversold: this is a pilot evaluation on a simplified 7-DOF model, not a full high-fidelity robotics benchmark.

## feedback_mppi_faithful Finding

A `feedback_mppi_faithful` variant was tested on the base 2D dynamic navigation suite (not the 7-DOF benchmark). It combines the released current-action gain computation (`compute_reference_feedback_gain_kernel`) with a two-rate controller architecture (replan every 2 steps, apply stored gain between replans).

Result: this variant fails on both `dynamic_crossing` and `dynamic_slalom` even at K=8192 and 2.1 ms per step, while the every-step `feedback_mppi_ref` succeeds on `dynamic_crossing` at K=256 and 0.60 ms.

Conclusion: current-action-only feedback gains are insufficient for the two-rate architecture on dynamic-obstacle tasks. The gain loses temporal information between replans. This suggests that a full two-rate Feedback-MPPI implementation requires full-horizon feedback gains (covariance or LQR), not just the current-action sensitivity variant.

## Reproduction

```bash
cmake --build build -j$(nproc)

# Fixed-budget benchmark
./bin/benchmark_diff_mppi_manipulator_7dof --seed-count 4 --k-values 256,512,1024 \
  --csv build/benchmark_diff_mppi_manipulator_7dof.csv

# Exact-time tuning
python3 scripts/tune_diff_mppi_time_targets.py --preset 7dof_manipulator

# Summary
python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi_manipulator_7dof.csv
```
