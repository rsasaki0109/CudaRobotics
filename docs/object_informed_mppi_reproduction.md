# Object-Informed MPPI Reproduction Notes

Date: 2026-06-04

## Target

Paper: **Object-Informed Model Predictive Path Integral Control for Model-Based Dynamic Non-Prehensile Manipulation**.

- arXiv: https://arxiv.org/abs/2605.30778
- Web search checked on 2026-06-04 for a dedicated public Object-Informed MPPI implementation; no clear standalone repository was found in the top results.
- Core idea: plan first in an object-level space, then use that object-level trajectory to guide the full robot-object MPPI rollout. The goal is to avoid wasting samples on robot motions that do not make useful object progress.

## Implementation

Implemented a lightweight object-informed MPPI reproduction in the pushing benchmarks:

- `src/benchmark_diff_mppi_pushing.cu`
  - Added `Variant::use_object_informed` and object-reference parameters.
  - Added `object_ref_disk_f`: a direct-actuated object-only straight-line reference from current object position to goal.
  - Added `push_object_informed_rollout_kernel`: standard MPPI sampling plus an object-reference tracking cost.
  - Added `seed_object_informed_nominal`: a host-side nominal-control seed that places the pusher behind the object along the object-level reference.
  - Added planners:
    - `oi_mppi`
    - `oi_lp_mppi`

- `src/benchmark_diff_mppi_pushing_box.cu`
  - Added a pose reference for box position and orientation.
  - Added `rollout_object_informed_kernel`: standard MPPI sampling plus object pose reference cost.
  - Added `seed_object_informed_nominal`: a two-mode seed:
    - translation mode places the pusher behind the box along the object-level position reference,
    - rotation mode switches near the position tolerance and places the pusher at an off-center contact point to induce torque.
  - Added planners:
    - `oi_mppi`
    - `oi_lp_mppi`

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- The paper's object-level planner is not reproduced. This implementation uses a simple direct-actuated line/pose reference.
- There is no obstacle-aware object planner or learned/contact-rich object-level dynamics.
- The robot level is the repo's existing 2D point pusher / box pusher, not the paper's full manipulator setting.
- `oi_mppi` and `oi_lp_mppi` are intentionally minimal so they can be compared against MPPI, LP-MPPI, Diff-MPPI, and SOPPI on the existing benchmark surface.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi_pushing -j$(nproc)
cmake --build build-docker-smoke --target benchmark_diff_mppi_pushing_box -j$(nproc)
```

Disk pushing comparison:

```bash
./bin/benchmark_diff_mppi_pushing \
  --quick \
  --scenarios push_straight,push_diagonal \
  --planners mppi,lp_mppi,lp_mppi_smooth,oi_mppi,oi_lp_mppi,diff_mppi_1,diff_mppi_3,soppi_fast \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/oi_mppi_pushing_compare_tuned.csv
```

Box pushing comparison:

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --scenarios box_align,box_swivel \
  --planners mppi,lp_mppi_smooth,oi_mppi,oi_lp_mppi,diff_mppi_3,soppi_fast \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/oi_mppi_box_compare_tuned.csv
```

Box-align confirmation:

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --scenarios box_align \
  --planners mppi,lp_mppi_smooth,oi_lp_mppi,diff_mppi_3,soppi_fast \
  --k-values 128,256 \
  --seed-count 6 \
  --csv build-docker-smoke/oi_mppi_box_align_confirm.csv
```

## Results

Artifacts:

- `build-docker-smoke/oi_mppi_pushing_compare_tuned.csv`
- `build-docker-smoke/oi_mppi_pushing_compare_tuned_summary.md`
- `build-docker-smoke/oi_mppi_box_compare_tuned.csv`
- `build-docker-smoke/oi_mppi_box_align_confirm.csv`
- `build-docker-smoke/oi_mppi_box_align_confirm_summary.md`

Disk pushing, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| push_diagonal | 128 | mppi | 1.00 | 34.0 | 0.194 | 3.7 | 0.097 |
| push_diagonal | 128 | lp_mppi_smooth | 1.00 | 27.0 | 0.185 | 3.3 | 0.098 |
| push_diagonal | 128 | oi_lp_mppi | 1.00 | 33.7 | 0.180 | 3.5 | 0.102 |
| push_diagonal | 128 | diff_mppi_3 | 1.00 | 28.3 | 0.186 | 3.3 | 0.679 |
| push_straight | 128 | mppi | 1.00 | 31.0 | 0.186 | 2.2 | 0.077 |
| push_straight | 128 | lp_mppi_smooth | 1.00 | 25.3 | 0.179 | 2.0 | 0.075 |
| push_straight | 128 | oi_lp_mppi | 1.00 | 30.3 | 0.166 | 2.2 | 0.092 |
| push_straight | 128 | diff_mppi_3 | 1.00 | 25.0 | 0.186 | 1.8 | 0.481 |

Box-align confirmation, seed-count 6:

| Scenario | K | Planner | Success | Steps | Pos Err | Ang Err | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| box_align | 128 | mppi | 0.00 | 240.0 | 0.279 | 0.030 | 4.0 | 0.108 |
| box_align | 128 | lp_mppi_smooth | 0.00 | 240.0 | 0.289 | 0.045 | 3.0 | 0.122 |
| box_align | 128 | soppi_fast | 0.00 | 240.0 | 0.281 | 0.038 | 3.6 | 0.893 |
| box_align | 128 | diff_mppi_3 | 0.17 | 208.2 | 0.231 | 0.046 | 3.3 | 1.959 |
| box_align | 128 | oi_lp_mppi | 1.00 | 74.2 | 0.181 | 0.232 | 2.2 | 0.114 |
| box_align | 256 | mppi | 0.00 | 240.0 | 0.277 | 0.029 | 3.9 | 0.130 |
| box_align | 256 | lp_mppi_smooth | 0.00 | 240.0 | 0.295 | 0.041 | 2.9 | 0.132 |
| box_align | 256 | soppi_fast | 0.00 | 240.0 | 0.272 | 0.027 | 3.6 | 0.996 |
| box_align | 256 | diff_mppi_3 | 0.33 | 177.3 | 0.227 | 0.033 | 3.1 | 1.953 |
| box_align | 256 | oi_lp_mppi | 1.00 | 66.3 | 0.192 | 0.238 | 2.1 | 0.140 |

Box-swivel, seed-count 3:

| Scenario | K | Planner | Success | Steps | Pos Err | Ang Err | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| box_swivel | 256 | mppi | 1.00 | 67.0 | 0.216 | 0.006 | 1.1 | 0.117 |
| box_swivel | 256 | lp_mppi_smooth | 1.00 | 22.0 | 0.217 | 0.011 | 0.3 | 0.118 |
| box_swivel | 256 | soppi_fast | 1.00 | 58.0 | 0.215 | 0.015 | 0.9 | 1.013 |
| box_swivel | 256 | oi_lp_mppi | 0.00 | 240.0 | 0.054 | 0.403 | 1.6 | 0.130 |

Observed pattern:

- On the simple disk pushing tasks, `oi_lp_mppi` is only a weak positive signal. It improves final distance slightly against vanilla MPPI in some rows, but LP-MPPI and Diff-MPPI remain better on steps/cost.
- On `box_align`, `oi_lp_mppi` is a strong positive result. It solves all 6 seeds at both `K=128` and `K=256`, while MPPI, LP-MPPI, and SOPPI remain at zero success. It is also much cheaper than Diff-MPPI because it avoids per-step autodiff gradients.
- The `box_align` success is near the angular tolerance: `oi_lp_mppi` ends with `ang_err` around `0.23-0.24`. This is valid under the benchmark tolerance, but less precise than the failed baselines' final orientation; the win comes from satisfying both position and angle jointly instead of missing position.
- On `box_swivel`, `oi_lp_mppi` is a negative result. The lightweight object-level pose reference over-focuses position and leaves a large angular error. The existing LP-MPPI remains the preferred planner there.
- The best next step toward the paper is replacing the hand-authored line/pose reference with an actual object-level MPPI/CEM planner, then passing that object trajectory into the full pusher MPPI. That should make the method less scenario-specific and avoid the `box_swivel` failure mode.
