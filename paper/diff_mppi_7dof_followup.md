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

## Key Results (after gradient parallelization — cumulative 17x speedup)

### 7dof_dynamic_avoid (fixed budget)

| Planner | K=256 success | K=512 | K=1024 | avg_ms (K=256) |
|---|---|---|---|---|
| mppi | 0.75 | 0.25 | 0.75 | 0.33 |
| feedback_mppi_ref | **1.00** | 0.75 | **1.00** | 2.06 |
| diff_mppi_1 | 0.75 | 0.25 | 0.75 | 0.49 |
| **diff_mppi_3** | 0.50 | **1.00** | 0.75 | **0.79** |

Key finding: at K=512, `diff_mppi_3` reaches `success=1.00` at `0.84 ms` while `feedback_mppi_ref` reaches `0.75` at `4.01 ms`. The hybrid controller is both more reliable and **4.8x faster**.

### 7dof_shelf_reach (fixed budget)

| Planner | K=256 success | K=512 | K=1024 | avg_ms (K=256) |
|---|---|---|---|---|
| mppi | 0.25 | 0.00 | 0.00 | 0.28 |
| feedback_mppi_ref | 0.00 | 0.00 | 0.25 | 1.55 |
| **diff_mppi_1** | **0.50** | 0.00 | 0.25 | 0.43 |
| diff_mppi_3 | 0.25 | 0.00 | 0.00 | 0.70 |

### Exact-time key numbers

`7dof_dynamic_avoid` at `1.0 ms` target:
- `diff_mppi_1` (K=3169 @ 1.03 ms): success `1.00`, dist `0.08`
- `mppi` (K=4096 @ 1.05 ms): success `1.00`, dist `0.10`
- `feedback_mppi_ref` (K=32 @ 1.33 ms): success `0.50`, dist `0.42`

## Gradient Computation Speed

Three rounds of optimization brought diff_mppi_3 K=256 from 13.6 ms to 0.79 ms (17x cumulative):
1. Analytical dynamics Jacobian + analytical control/velocity gradients: 13.6 → 5.7 ms
2. Merged FK calls in stage_cost_device + analytical dq gradient: 5.7 → 2.75 ms
3. Parallel stage cost gradient computation (T threads) + gradient norm clipping: 2.75 → 0.79 ms

## Interpretation

- At K=512, `diff_mppi_3` is now the strongest planner on `7dof_dynamic_avoid`: success=1.00 at 0.84 ms, while feedback_mppi_ref needs 4.01 ms for 0.75 success. The gradient parallelization made the hybrid controller compute-competitive.
- At K=256 on `7dof_dynamic_avoid`, `feedback_mppi_ref` still reaches 1.00 success while diff_mppi_3 is at 0.50. The advantage is budget-dependent.
- On `7dof_shelf_reach`, diff_mppi_1 shows modest improvement over mppi at K=256 (0.50 vs 0.25 success). Results remain noisy across K values.
- At matched time (1.0 ms), diff_mppi_1 matches mppi success on dynamic_avoid while achieving slightly better terminal distance. feedback_mppi_ref cannot reach the 1.0 ms target without dropping to very low K.
- Overall, the gradient parallelization shifted the story from "diff_mppi is too slow for 7-DOF" to "diff_mppi is compute-competitive and sometimes superior on high-dimensional tasks".

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
