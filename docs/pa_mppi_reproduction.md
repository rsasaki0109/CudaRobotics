# PA-MPPI Lightweight Reproduction

## Target

- Paper: "PA-MPPI: Perception-Aware Model Predictive Path Integral Control
  for Quadrotor Navigation in Unknown Environments"
- arXiv: https://arxiv.org/abs/2509.14978
- Accepted RAL preprint PDF: https://rpg.ifi.uzh.ch/docs/RAL26_Zhai.pdf
- Public reference implementation: no dedicated PA-MPPI implementation was
  found in the web/GitHub search performed for this pass.

PA-MPPI adds a perception objective to MPPI. The paper's key setting is
unknown-environment quadrotor navigation: when the goal is occluded, the
controller biases sampled trajectories toward viewpoints that can perceive
unknown regions in the goal direction, expanding the mapped traversable space
and improving the chance of finding an alternate route.

## Implemented Scope

This repository now has a lightweight 2D CUDA reproduction in
`src/benchmark_diff_mppi.cu`:

- `use_pa_perception_cost` plus perception-cost tuning fields in
  `PlannerVariant`.
- `pa_segment_occlusion_device`, a line-of-sight obstruction surrogate against
  static and dynamic circular obstacles.
- `compute_pa_perception_scores_kernel`, which modifies rollout scores before
  the MPPI weight update.
- Planner variants:
  - `pa_mppi`: perception-aware weighting without low-pass sampling.
  - `pa_mppi_soft`: low-weight tie-breaker setting with a small score cap.
  - `pa_mppi_smooth`: perception-aware weighting plus low-pass sampling.
  - `pa_mppi_frontier`: stronger frontier reward / viewpoint-seeking setting.

The implemented score term is:

```text
occ_t = line_of_sight_occlusion(x_t, goal)
occ_0 = line_of_sight_occlusion(x_0, goal)
active_t = gate(max(occ_0, occ_t))
poi_t = 1 - cos(theta_t - bearing_to_goal_t)
forward_occ_t = line_of_sight_occlusion(x_t, forward_ray(theta_t))
exposed_t = max(0, occ_0 - occ_t)

score = nominal_cost + mean_t active_t * (
          occlusion_weight * occ_t^2
        + poi_weight * poi_t^2
        + forward_occ_weight * forward_occ_t^2
        - frontier_reward * exposed_t
)
score_delta is clipped by pa_score_cap.
```

This is not a full paper-faithful reproduction. It does not use a 3D occupancy
grid, depth-image updates, ROG-Map, quadrotor dynamics, camera frustum
geometry, unknown/free/occupied map states, or a true frontier detector. The
current benchmark uses fully known 2D obstacle fields, so the reproduced hook
is only the online MPPI-side perception score: line-of-sight, viewpoint
alignment, and a frontier-like reward when a sampled trajectory reduces goal
occlusion.

## Build And Benchmark

Build:

```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc 'cmake --build build-docker-smoke --target benchmark_diff_mppi -j$(nproc)'
```

Static/occlusion benchmark:

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace map4_engine_v2:validation \
  bash -lc './bin/benchmark_diff_mppi --quick \
    --scenarios static_u_trap,static_s_corridor,cluttered,narrow_passage,dynamic_bottleneck \
    --planners mppi,lp_mppi_smooth,pa_mppi_soft,pa_mppi,pa_mppi_smooth,pa_mppi_frontier,cdf_lp_mppi,step_mppi_smooth,sc_mppi_smooth,tsallis_mppi_smooth \
    --k-values 64,128 \
    --seed-count 3 \
    --csv build-docker-smoke/pa_mppi_tuned_compare.csv && \
    python3 scripts/summarize_diff_mppi.py \
      --csv build-docker-smoke/pa_mppi_tuned_compare.csv \
      --markdown-out build-docker-smoke/pa_mppi_tuned_compare_summary.md \
      --time-caps 0.25,0.5,1.0 \
      --time-targets 0.25,0.5'
```

Artifacts:

- `build-docker-smoke/pa_mppi_static_compare.csv`
- `build-docker-smoke/pa_mppi_static_compare_summary.md`
- `build-docker-smoke/pa_mppi_tuned_compare.csv`
- `build-docker-smoke/pa_mppi_tuned_compare_summary.md`

All table values below are means over 3 seeds.

## Positive Result: Narrow Passage

The perception-aware cost is useful as a viewpoint/tie-breaker in
`narrow_passage`. It keeps 100% success and cuts the vanilla MPPI step count by
roughly 20 steps. `pa_mppi_frontier` and `pa_mppi_smooth` are close to the
Step/SC-style smooth samplers here, but are slower because of the extra
line-of-sight scoring.

| Planner | K | Success | Steps | Final Dist | Avg ms |
|---|---:|---:|---:|---:|---:|
| mppi | 64 | 1.00 | 252.0 | 1.84 | 0.13 |
| lp_mppi_smooth | 64 | 1.00 | 232.7 | 1.90 | 0.13 |
| pa_mppi | 64 | 1.00 | 244.7 | 1.96 | 0.17 |
| pa_mppi_frontier | 64 | 1.00 | 230.7 | 1.94 | 0.16 |
| pa_mppi_smooth | 64 | 1.00 | 231.0 | 1.93 | 0.16 |
| pa_mppi_soft | 64 | 1.00 | 231.3 | 1.93 | 0.17 |
| step_mppi_smooth | 64 | 1.00 | 230.3 | 1.91 | 0.14 |
| tsallis_mppi_smooth | 64 | 1.00 | 228.7 | 1.93 | 0.13 |
| mppi | 128 | 1.00 | 251.3 | 1.86 | 0.14 |
| pa_mppi_frontier | 128 | 1.00 | 231.0 | 1.89 | 0.17 |
| pa_mppi_soft | 128 | 1.00 | 231.3 | 1.93 | 0.17 |

## Partial Result: Cluttered Progress

On `cluttered`, no tested planner reaches the success threshold in this quick
run. PA-MPPI does improve final distance over vanilla MPPI, LP-MPPI, Step, and
SC, but it remains far behind CDF-MPPI and Tsallis at higher K.

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| mppi | 64 | 0.00 | 38.50 | 0.17 |
| lp_mppi_smooth | 64 | 0.00 | 39.47 | 0.16 |
| pa_mppi | 64 | 0.00 | 36.79 | 0.24 |
| pa_mppi_soft | 64 | 0.00 | 37.26 | 0.21 |
| cdf_lp_mppi | 64 | 0.00 | 17.77 | 0.11 |
| tsallis_mppi_smooth | 128 | 0.00 | 22.72 | 0.16 |
| pa_mppi | 128 | 0.00 | 36.81 | 0.21 |
| pa_mppi_smooth | 128 | 0.00 | 37.13 | 0.23 |

## Negative Result: Full-Known Maps Are A Bad Fit

The paper's strongest claim depends on partial maps and unknown frontiers. This
repo's benchmark exposes a useful failure mode: in fully known corridor/trap
maps, a naive goal-visibility cost can pull the robot toward bad viewpoints
instead of through the feasible passage.

`static_s_corridor` is the clearest negative case:

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| mppi | 128 | 0.00 | 13.51 | 0.14 |
| lp_mppi_smooth | 128 | 0.00 | 14.80 | 0.14 |
| pa_mppi | 128 | 0.00 | 40.29 | 0.22 |
| pa_mppi_frontier | 128 | 0.00 | 41.00 | 0.22 |
| pa_mppi_soft | 128 | 0.00 | 39.96 | 0.23 |
| sc_mppi_smooth | 128 | 1.00 | 1.92 | 0.17 |
| step_mppi_smooth | 128 | 1.00 | 1.96 | 0.15 |

`static_u_trap` also shows that strong PA weights are unsafe in a fully known
trap. The soft score cap avoids the worst regression, but still does not solve
the scenario.

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| cdf_lp_mppi | 128 | 0.00 | 8.51 | 0.13 |
| mppi | 128 | 0.00 | 11.47 | 0.15 |
| pa_mppi | 128 | 0.00 | 30.03 | 0.23 |
| pa_mppi_frontier | 128 | 0.00 | 29.67 | 0.23 |
| pa_mppi_soft | 128 | 0.00 | 11.99 | 0.23 |

On `dynamic_bottleneck`, PA-MPPI does not discover the timing policy needed to
wait for the obstacle, while CDF and Tsallis can solve two or three seeds:

| Planner | K | Success | Final Dist | Avg ms |
|---|---:|---:|---:|---:|
| cdf_lp_mppi | 128 | 1.00 | 1.89 | 0.13 |
| tsallis_mppi_smooth | 128 | 0.67 | 9.35 | 0.16 |
| mppi | 64 | 0.67 | 9.46 | 0.13 |
| pa_mppi_frontier | 128 | 0.00 | 24.71 | 0.20 |
| pa_mppi_soft | 128 | 0.00 | 25.25 | 0.20 |

## Takeaways

- The PA hook is useful to keep as a benchmarkable perception-aware MPPI
  mechanism, especially as a viewpoint tie-breaker in narrow/open cluttered
  scenes.
- In a fully known 2D map, line-of-sight-to-goal is not equivalent to useful
  unknown-space exploration. Without an occupancy state that distinguishes
  unknown from free/occupied, the PA objective can be actively harmful.
- `pa_mppi_soft` is the safest default from this pass because its score cap
  prevents U-trap collapse, but `pa_mppi_frontier` has the strongest
  narrow-passage step reduction.
- A more faithful next version needs an explicit partial-map simulator:
  unknown cells, camera field-of-view ray casting, frontier extraction, and a
  reward for newly observed cells in the goal direction.
