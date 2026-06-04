# PR/EMPPI Reproduction Notes

Date: 2026-06-04

## Target

Primary recent target: **Parameter-Robust MPPI for Safe Online Learning of Unknown Parameters**.

- arXiv: https://arxiv.org/abs/2601.02948
- Submitted on 2026-01-06.
- Core idea: maintain a particle-based belief over unknown physical parameters, update that belief online, and optimize nominal and safety-focused trajectories so MPPI remains safe while the model is uncertain.

Precursor target: **Model-Based Generalization Under Parameter Uncertainty Using Path Integral Control**.

- arXiv: https://arxiv.org/abs/2006.03106
- Published as RA-L / ICRA 2020.
- Core idea: expand the path-integral sample space to include uncertainty, so the controller optimizes over action and model-uncertainty effects rather than assuming one fixed dynamics model.

Related robustness paper:

- RMPPI: https://arxiv.org/abs/2102.09027
- This uses an augmented nominal/actual state-space architecture, tracking controllers, safety logic, and an importance-sampling scheme for robust off-road MPPI. It is related, but the lightweight implementation below focuses on parameter particles rather than RMPPI's full tracking-controller stack.

## Implementation

Implemented a lightweight parameter-robust MPPI reproduction in `src/benchmark_diff_mppi.cu`:

- Added model-mismatch evaluation fields to `Scenario`:
  - `use_model_mismatch`
  - `eval_wheelbase_scale`
  - `eval_max_speed_scale`
  - `eval_max_steer_scale`
- Added `PlannerVariant::use_parameter_robust_sampling` and PR tuning fields:
  - `pr_param_particles`
  - `pr_wheelbase_span`
  - `pr_max_speed_span`
  - `pr_max_steer_span`
  - `pr_worst_blend`
- Added `rollout_parameter_robust_kernel`.
- Added `robust_param_particle` to create a small deterministic set of bicycle parameter particles.
- Added `nav_stage_cost_float` for repeated stage-cost evaluation inside the robust rollout.
- Added planners:
  - `pr_mppi`: robust parameter particles with standard MPPI noise.
  - `pr_mppi_smooth`: robust parameter particles plus low-pass sampled control noise.
  - `pr_mppi_cautious`: more particles and stronger worst-case cost blend.
- Added scenarios:
  - `model_mismatch_slalom`: slalom evaluation uses longer wheelbase, lower max speed, and lower steering authority than the planning model.
  - `model_mismatch_crossing`: dynamic crossing with the same evaluation-side bicycle model mismatch.

## Scope Caveats

This is a reproduction scaffold, not a paper-faithful implementation:

- No online parameter learning.
- No SVGD update over parameter belief particles.
- No conformal prediction safety constraints.
- No separate safety-focused backup trajectory.
- No augmented nominal/actual RMPPI state or downstream tracking controller.
- The parameter set is a fixed deterministic grid over wheelbase, max speed, and max steering authority.
- The robust objective is a simple average/worst-case cost blend:
  - low `pr_worst_blend` behaves like expected-cost uncertainty sampling;
  - high `pr_worst_blend` is more conservative and can stall.

## Commands

Build:

```bash
cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)
```

Main comparison:

```bash
./bin/benchmark_diff_mppi \
  --quick \
  --scenarios model_mismatch_slalom,model_mismatch_crossing,slalom,dynamic_crossing \
  --planners mppi,lp_mppi_smooth,pr_mppi,pr_mppi_smooth,pr_mppi_cautious,step_mppi_smooth,pi_mppi,ds_mppi \
  --k-values 128,256 \
  --seed-count 3 \
  --csv build-docker-smoke/pr_mppi_compare.csv
```

Summary:

```bash
python3 scripts/summarize_diff_mppi.py \
  --csv build-docker-smoke/pr_mppi_compare.csv \
  --markdown-out build-docker-smoke/pr_mppi_compare_summary.md \
  --time-caps 0.5,1.0,2.0 \
  --time-targets 0.5,1.0
```

## Results

Artifacts:

- `build-docker-smoke/pr_mppi_compare.csv`
- `build-docker-smoke/pr_mppi_compare_summary.md`
- `build-docker-smoke/pr_mppi_compare_summary.tex`

Dynamic crossing, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_crossing | 128 | mppi | 0.00 | 260.0 | 2.75 | 45557.9 | 0.13 |
| dynamic_crossing | 128 | lp_mppi_smooth | 1.00 | 252.0 | 1.89 | 41520.3 | 0.14 |
| dynamic_crossing | 128 | pr_mppi | 0.00 | 260.0 | 2.43 | 45012.8 | 0.18 |
| dynamic_crossing | 128 | pr_mppi_smooth | 1.00 | 251.0 | 1.92 | 41385.2 | 0.18 |
| dynamic_crossing | 128 | pr_mppi_cautious | 1.00 | 252.3 | 1.92 | 41757.0 | 0.22 |
| dynamic_crossing | 128 | step_mppi_smooth | 1.00 | 251.7 | 1.88 | 41278.4 | 0.14 |
| dynamic_crossing | 256 | mppi | 0.00 | 260.0 | 3.15 | 45870.7 | 0.16 |
| dynamic_crossing | 256 | lp_mppi_smooth | 1.00 | 252.0 | 1.94 | 41575.2 | 0.16 |
| dynamic_crossing | 256 | pr_mppi_smooth | 1.00 | 252.0 | 1.92 | 41501.1 | 0.22 |
| dynamic_crossing | 256 | step_mppi_smooth | 1.00 | 251.0 | 1.89 | 41200.1 | 0.17 |

Model-mismatch crossing, seed-count 3:

| Scenario | K | Planner | Success | Steps | Final Dist | Cost | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| model_mismatch_crossing | 128 | mppi | 0.00 | 300.0 | 4.96 | 53209.7 | 0.13 |
| model_mismatch_crossing | 128 | lp_mppi_smooth | 1.00 | 299.7 | 1.95 | 48815.0 | 0.13 |
| model_mismatch_crossing | 128 | pi_mppi | 0.00 | 300.0 | 3.10 | 50716.3 | 0.22 |
| model_mismatch_crossing | 128 | pr_mppi | 0.00 | 300.0 | 4.03 | 51955.6 | 0.18 |
| model_mismatch_crossing | 128 | pr_mppi_smooth | 0.67 | 300.0 | 1.97 | 48584.4 | 0.18 |
| model_mismatch_crossing | 128 | pr_mppi_cautious | 0.00 | 300.0 | 2.15 | 48894.8 | 0.22 |
| model_mismatch_crossing | 128 | step_mppi_smooth | 0.67 | 299.7 | 1.97 | 48514.3 | 0.14 |
| model_mismatch_crossing | 256 | mppi | 0.00 | 300.0 | 4.88 | 53107.1 | 0.16 |
| model_mismatch_crossing | 256 | lp_mppi_smooth | 0.33 | 300.0 | 2.04 | 48948.1 | 0.16 |
| model_mismatch_crossing | 256 | pi_mppi | 0.00 | 300.0 | 3.09 | 50661.2 | 0.31 |
| model_mismatch_crossing | 256 | pr_mppi_smooth | 0.67 | 300.0 | 2.00 | 48655.7 | 0.22 |
| model_mismatch_crossing | 256 | step_mppi_smooth | 1.00 | 300.0 | 1.93 | 48630.5 | 0.17 |

Model-mismatch slalom, seed-count 3:

| Scenario | K | Planner | Success | Final Dist | Cost | Collisions | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| model_mismatch_slalom | 128 | mppi | 0.00 | 22.97 | 58020.4 | 0.00 | 0.13 |
| model_mismatch_slalom | 128 | lp_mppi_smooth | 0.00 | 37.31 | 60294.7 | 0.00 | 0.14 |
| model_mismatch_slalom | 128 | pi_mppi | 0.00 | 18.25 | 55758.3 | 0.00 | 0.22 |
| model_mismatch_slalom | 128 | pr_mppi_smooth | 0.00 | 37.24 | 60189.0 | 0.00 | 0.19 |
| model_mismatch_slalom | 128 | step_mppi_smooth | 0.00 | 18.58 | 61207.8 | 2.33 | 0.15 |
| model_mismatch_slalom | 256 | mppi | 0.00 | 29.42 | 59848.3 | 0.00 | 0.16 |
| model_mismatch_slalom | 256 | pi_mppi | 0.00 | 18.26 | 55805.6 | 0.00 | 0.31 |
| model_mismatch_slalom | 256 | pr_mppi_smooth | 0.00 | 37.30 | 60192.0 | 0.00 | 0.22 |
| model_mismatch_slalom | 256 | step_mppi_smooth | 0.00 | 12.67 | 61463.3 | 2.33 | 0.17 |

Observed pattern:

- `pr_mppi_smooth` is positive on `dynamic_crossing`: vanilla MPPI fails, while robust low-pass sampling reaches `1.00` success.
- On `model_mismatch_crossing`, `pr_mppi_smooth` improves strongly over vanilla MPPI and pi-MPPI. It is also more stable than `lp_mppi_smooth` at `K=256` in this small seed run, but does not beat `step_mppi_smooth`.
- `pr_mppi` without low-pass sampling is not enough. The robust cost alone still produces rough controls and misses the threshold.
- `pr_mppi_cautious` is too conservative: it lowers final distance versus vanilla MPPI on mismatch crossing, but often stalls outside the goal tolerance.
- `model_mismatch_slalom` is a negative result. The fixed parameter-particle robust objective over-penalizes aggressive slalom trajectories and stalls; pi-MPPI and Step-MPPI are better directions there.
- The most useful next faithful step would be an online parameter-belief update and a separate safety backup trajectory. Without those, the fixed robust cost can become either helpful in open dynamic scenes or overly cautious in tight slalom geometry.
