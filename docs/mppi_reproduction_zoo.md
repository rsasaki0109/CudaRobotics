# MPPI Reproduction Zoo

This page indexes the CUDA MPPI-family reproduction work in CudaRobotics. The
goal is not to claim full paper-faithful reproductions for every method. The
goal is to make each idea runnable, benchmarkable, and honest about the gap
between the paper and the current lightweight implementation.

## Quick Smoke

From a machine with NVIDIA Container Toolkit and a CUDA-capable GPU:

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_smoke.py --bin ./bin/benchmark_diff_mppi --out-dir build/mppi_zoo'
```

The script writes CSV and a compact Markdown report under `build/mppi_zoo/`
through the Docker volume mount.

## Checked-In Smoke Result

The latest checked-in fixed-seed smoke artifact is
[`results/mppi_zoo_smoke_2026-06-05.md`](results/mppi_zoo_smoke_2026-06-05.md)
with its source CSV at
[`results/mppi_zoo_smoke_2026-06-05.csv`](results/mppi_zoo_smoke_2026-06-05.csv).

Scope: `dynamic_crossing,narrow_passage`, `K=64,128`, 3 seeds per
scenario/planner/K cell, and the nine planners listed in the report. The useful
negative control is `dynamic_crossing`: vanilla `mppi` fails at both K values,
while the paper-inspired zoo variants solve the same cells in this smoke run.

## Checked-In Suite Result

The expanded fixed-seed suite artifact is
[`results/mppi_zoo_suite_2026-06-09.md`](results/mppi_zoo_suite_2026-06-09.md)
with its source CSV at
[`results/mppi_zoo_suite_2026-06-09.csv`](results/mppi_zoo_suite_2026-06-09.csv)
and chart at
[`results/mppi_zoo_suite_2026-06-09.svg`](results/mppi_zoo_suite_2026-06-09.svg).

Scope: five navigation scenarios
(`dynamic_crossing`, `narrow_passage`, `model_mismatch_crossing`,
`dynamic_pincer`, `uncertain_crossing`), eight curated planners, `K=64,128`,
and 3 seeds per scenario/planner/K cell.

Reproduce:

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_suite.py --bin ./bin/benchmark_diff_mppi && python3 scripts/render_mppi_zoo_suite_chart.py'
```

## Comparison GIF

The checked-in side-by-side rollout is
[`gpu_mppi_zoo_dynamic_crossing.gif`](https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_zoo_dynamic_crossing.gif)
(local copy: [`../gif/gpu_mppi_zoo_dynamic_crossing.gif`](../gif/gpu_mppi_zoo_dynamic_crossing.gif)).
It contrasts vanilla `mppi` and `step_mppi_smooth` on `dynamic_crossing` at
`K=128`, matching the suite's strongest negative control.

Reproduce:

```bash
cmake --build build --target benchmark_diff_mppi -j$(nproc)
python3 scripts/render_mppi_zoo_gif.py --bin bin/benchmark_diff_mppi
```

Useful signals from the checked-in run:

- Vanilla `mppi` solves only 2/10 scenario-K cells in this suite.
- `tsallis_mppi_smooth` is the only curated planner with 10/10 solved cells;
  `step_mppi_smooth` and `sc_mppi_smooth` follow at 9/10.
- `model_mismatch_crossing` replaces `dynamic_slalom` in the suite because
  the slalom geometry needs gradient/hybrid guidance at low K; mismatch crossing
  still discriminates vanilla `mppi` from Step/Tsallis.
- `dynamic_pincer` and `uncertain_crossing` add stress beyond the smoke pair.

## Index

| Family | Paper / idea | Implementation | Best signal | Limit | Doc |
|---|---|---|---|---|---|
| LP-MPPI | Low-pass filtered control noise | `lp_mppi`, `lp_mppi_smooth` | Strong dynamic-crossing success where vanilla MPPI fails | Reproduction scaffold, not the full paper system | [`lp_mppi_reproduction.md`](lp_mppi_reproduction.md) |
| Step-MPPI | State-conditioned or step-wise proposal shaping | `step_mppi_fast`, `step_mppi_smooth`, adaptive variants | Preferred lightweight default in dynamic crossing; smooth variant keeps success with lower roughness | Uses EMA/table-like proposal logic instead of a trained proposal network | [`step_mppi_reproduction.md`](step_mppi_reproduction.md) |
| Tsallis-MPPI | q-exponential / Tsallis weighting | `tsallis_mppi_q07`, `tsallis_mppi_smooth`, `tsallis_mppi_q13` | Strong cheap fix for dynamic bottlenecks and open crossings | Sensitive to q shape; harder scenes still need more structure | [`tsallis_mppi_reproduction.md`](tsallis_mppi_reproduction.md) |
| SOPPI | SVGD-style sample optimization | `soppi`, `soppi_fast` across navigation, CartPole, pushing, box pushing | Box pushing exposes useful final-error and success improvements | Navigation gains are modest; SVGD score is simplified | [`soppi_reproduction.md`](soppi_reproduction.md) |
| SVG-MPPI | Stein-mode guidance | `svg_mppi` variants | Adds Stein-style mode guidance in the shared benchmark | Lightweight scaffold rather than full differentiable-through-time reproduction | [`svg_mppi_reproduction.md`](svg_mppi_reproduction.md) |
| pi-MPPI | Projection-informed controls | `pi_mppi` variants | Tests projection-filtered controls in the same navigation scenarios | Helpful as a constraint layer, not a standalone default | [`pi_mppi_reproduction.md`](pi_mppi_reproduction.md) |
| CDF-MPPI | Configuration-space distance field guidance | `cdf_*` variants | Useful C-space guidance experiment in 2D navigation | Paper target is manipulator motion planning; current benchmark is 2D nav | [`cdf_mppi_reproduction.md`](cdf_mppi_reproduction.md) |
| SC-MPPI | Safety-controlled rollouts | `sc_mppi_smooth` | Good safety-controlled sampling baseline for open scenes | Less effective as a universal bottleneck solver | [`sc_mppi_reproduction.md`](sc_mppi_reproduction.md) |
| Shield-MPPI | CBF-style shield and repair | `shield_*` variants | Makes safety repair behavior measurable in the MPPI suite | Current implementation is a scaffold, not a full shield stack | [`shield_mppi_reproduction.md`](shield_mppi_reproduction.md) |
| PR / EMPPI | Parameter-robust particles | `pr_*` variants | Tests robustness to model parameter spread | Reproduction scaffold; still needs richer uncertainty experiments | [`pr_mppi_reproduction.md`](pr_mppi_reproduction.md) |
| CC-MPPI | Covariance-controlled weighting | `cc_mppi_smooth` | Sometimes lowers final distance versus baseline variants | Not promoted as default because wins are scenario dependent | [`cc_mppi_reproduction.md`](cc_mppi_reproduction.md) |
| TD-CD-MPPI | Terminal value / constraint discounting | `td_cd_*` variants | Shows the value-function and discounting tradeoff explicitly | Needs a learned value function for a faithful result | [`td_cd_mppi_reproduction.md`](td_cd_mppi_reproduction.md) |
| CSC-MPPI | Clustering / representative samples | `csc_mppi_smooth` | More principled robust planner for hard bottlenecks | Higher complexity than the cheap smooth baselines | [`csc_mppi_reproduction.md`](csc_mppi_reproduction.md) |
| DM-MPPI | Data-model surrogate and influence pruning | `dm_mppi_smooth` | Documents surrogate/pruning direction inside the same benchmark | Current reproduction still evaluates more samples than the paper idea wants | [`dm_mppi_reproduction.md`](dm_mppi_reproduction.md) |
| BC-MPPI | Bayesian or feasibility layer | `bc_mppi_smooth` | Makes feasibility-layer behavior comparable to SC/C2U/CC variants | Computationally inefficient in the current lightweight form | [`bc_mppi_reproduction.md`](bc_mppi_reproduction.md) |
| Object-Informed MPPI | Object-level pushing guidance | Pushing and box-pushing benchmarks | Object-aware cost helps expose contact-planning behavior | Navigation benchmark is not the target domain | [`object_informed_mppi_reproduction.md`](object_informed_mppi_reproduction.md) |
| C2U-MPPI | Chance-constrained unscented layer | `c2u_mppi_smooth`, `c2u_mppi_strict` | Cheap open-scene chance layer; 0.92 success in one capped aggregate | Not a default planner; slalom and mismatch cases remain negative | [`c2u_mppi_reproduction.md`](c2u_mppi_reproduction.md) |
| DUCCT-MPPI | Dual uncertainty / conservative risk | `ducct_mppi_smooth`, cautious and diluted variants | Useful negative-control family for uncertainty inflation | Conservative variants can freeze or underperform | [`ducct_mppi_reproduction.md`](ducct_mppi_reproduction.md) |
| DRA-MPPI | Dynamic risk-aware collision probability | `dra_mppi_soft`, `dra_mppi_hard`, `dra_mppi_multimodal` | Strong signal on dynamic crossing and pincer scenes | Threshold sensitive; not a full shared-random-sample paper reproduction | [`dra_mppi_reproduction.md`](dra_mppi_reproduction.md) |
| DBaS-Log-MPPI | Log-normal / DBaS-style sampling | `log_mppi`, `dbas_log_mppi_*` | Agile variant improves crossing and narrow-passage behavior over Log-MPPI | Behind Step, SC, DRA, and Tsallis on several stress scenes | [`dbas_log_mppi_reproduction.md`](dbas_log_mppi_reproduction.md) |
| PA-MPPI | Perception-aware line-of-sight scoring | `pa_mppi_*` variants | Positive narrow-passage result with lower step count than vanilla MPPI | Full-known maps are a bad fit; needs a real 3D perception stack | [`pa_mppi_reproduction.md`](pa_mppi_reproduction.md) |
| dsMPPI | Deterministic sampling / adaptive distribution | `ds_mppi` and adaptive variants | Useful scaffold for sampler experiments | Naive adaptive sigma is a documented negative result | [`ds_mppi_reproduction.md`](ds_mppi_reproduction.md) |

## How To Read These Results

- Treat each document as a lightweight CUDA reproduction note, not as a final
  paper replication claim.
- Prefer methods with both a positive result and a clearly documented negative
  result. That usually means the benchmark is discriminating instead of
  saturated.
- Check the command blocks in each document before comparing results. Some
  entries target navigation, while others target CartPole or pushing.

## Public-Facing Gaps

- Add `soppi` / `soppi_fast` to the fixed-seed suite now that the navigation
  kernel is faster; navigation gains are still modest, so treat it as coverage
  rather than a headline win.
