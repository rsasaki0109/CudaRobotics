# Performance regression check

## What it does
`scripts/perf_check.py` runs a curated set of `comparison_*` GPU demos,
extracts the headline GPU timing from their stdout, and compares each
measurement against `scripts/perf_baseline.json`. The workflow also runs
the planner showdown and adversarial falsifier target gates through CTest.
It exits non-zero if any measurement exceeds `baseline * (1 + tolerance)` or
either planner gate misses.

Default tolerance is **30%**.

## Local usage

```bash
# build the perf-critical targets
cmake --build build --target \
  comparison_esdf comparison_esdf_3d comparison_voxel_map \
  comparison_collision_check comparison_rrtstar_rewire esdf_mppi \
  gpu_planner_showdown_benchmark gpu_planner_falsifier_benchmark -j$(nproc)

# check current performance against baseline
python3 scripts/perf_check.py

# check the planner showdown and falsifier gates
cd build && ctest --output-on-failure --label-regex 'showdown|falsifier' -j1

# render a compact Markdown report from the showdown JSON
python3 scripts/summarize_planner_showdown.py \
  --json build/gpu_planner_showdown_benchmark.json \
  --markdown-out build/gpu_planner_showdown_benchmark.md \
  --strict

# render a compact Markdown report from the falsifier JSON
python3 scripts/summarize_planner_falsifier.py \
  --json build/gpu_planner_falsifier_benchmark.json \
  --markdown-out build/gpu_planner_falsifier_benchmark.md \
  --strict

# render a scenario matrix after manual stress-probe runs
python3 scripts/summarize_planner_showdown.py \
  --json build/gpu_planner_showdown_benchmark.json \
  --json build/gpu_planner_showdown_tight.json \
  --json build/gpu_planner_showdown_priority_flip.json \
  --json build/gpu_planner_showdown_adversarial_density.json \
  --markdown-out build/gpu_planner_showdown_matrix.md \
  --strict

# render a pressure-controller ablation after manual ablation runs
python3 scripts/summarize_planner_showdown.py \
  --json build/gpu_planner_showdown_benchmark.json \
  --json build/gpu_planner_showdown_pressure_teacher.json \
  --json build/gpu_planner_showdown_pressure_none.json \
  --json build/gpu_planner_showdown_adversarial_density.json \
  --json build/gpu_planner_showdown_pressure_none_adversarial_density.json \
  --markdown-out build/gpu_planner_showdown_pressure_ablation.md

# loosen tolerance for noisy hardware
python3 scripts/perf_check.py --tolerance 0.50

# refresh the baseline after an intentional speed change
python3 scripts/perf_check.py --update
```

## Benchmarks tracked
| Label | Binary | What we time |
|---|---|---|
| `esdf_2d_gpu_ms`        | `comparison_esdf`            | GPU JFA per 640K-cell ESDF |
| `esdf_3d_gpu_ms`        | `comparison_esdf_3d`         | GPU JFA-3D per 1.05M-voxel ESDF |
| `voxel_map_gpu_ms`      | `comparison_voxel_map`       | GPU 3D-DDA log-odds raycast per LiDAR scan |
| `collision_check_gpu_ms`| `comparison_collision_check` | GPU 2D-DDA per 1M candidate segments |
| `rrtstar_rewire_gpu_ms` | `comparison_rrtstar_rewire`  | GPU parallel rewire per 200K-node forest |
| `esdf_mppi_rollout_ms`  | `esdf_mppi`                  | GPU MPPI rollout iter (K=4096, T=30) with ESDF cost |

## Target gates tracked
| Label | Binary | Gate |
|---|---|---|
| `showdown` | `gpu_planner_showdown_benchmark --check --no-video --scenario baseline` | trainable safety-dual MPPI with learned pressure and adaptive budget must keep reach 48/48, deadlocks 0, collisions <= 8, collision CVaR <= 26.5, residual <= 12.0%, and runtime <= 15.0 ms |
| `falsifier` | `gpu_planner_falsifier_benchmark --check` | worst-K adversarial scenario search must make no-pressure and no-regret fail, keep the learned target inside hard gates, and accept at least one adaptive repair |

`scripts/summarize_planner_showdown.py` turns the emitted JSON into a
Markdown summary that highlights the hard-gate status and the remaining
enemy planner, currently no-regret MPPI.
When multiple `--json` inputs are supplied, the same script emits a
scenario matrix across baseline and stress probes.
The planner uses a scenario-conditioned learned safety-pressure controller
distilled from observed CVaR, collisions, minimum separation, lane tightness,
graph conflict density, cross-shift load, and priority flips instead of
scenario-specific convergence boosts.
Use `--pressure-mode learned|teacher|none` to compare the distilled
controller with its teacher formula and a no-pressure bypass. In the tracked
ablation, no-pressure still passes baseline with narrow margins but misses the
adversarial-density target gate.
Use `--adaptive-budget learned|off` to compare the default budget policy with
fixed pass scheduling. The learned policy scores pass-2 CVaR, residual
pressure, and scenario difficulty; in the tracked matrix it flags the
adversarial-density probe, reports whether a refinement candidate was accepted,
and keeps all final runtimes below the 15 ms gate.
`gpu_planner_falsifier_benchmark` scans 719,712 scenario variants over lane
tightness, jitter, cross-shift, spawn phase, goal offset, and priority flips.
Its gate requires the worst 12 discovered cases to break no-pressure and
no-regret, keep learned safety-pressure inside the showdown hard gates, and
accept at least one adaptive repair. The tracked run has 12/12 no-pressure
failures, 12/12 learned passes, and 12/12 accepted repairs.
Only the `baseline` scenario is gated in CI; `--scenario tight`,
`--scenario priority_flip`, and `--scenario adversarial_density` are manual
stress probes for narrower crossings, flipped priority ordering, and dense
centerline conflicts.

## CI integration
`.github/workflows/perf.yml` runs `perf_check.py` plus the showdown and
falsifier CTest gates on a self-hosted runner with the `gpu` label.
GitHub-hosted Ubuntu runners do not have NVIDIA GPUs, so the workflow is a
no-op on them.

To enable PR-time perf checks:
1. Register a self-hosted runner labelled `gpu` with the repo.
2. Set the repository variable `PERF_RUNNER_AVAILABLE` to `true`.

The workflow also accepts manual triggering via `workflow_dispatch` and
runs nightly on a schedule once the runner is online.

## Updating the baseline
The baseline file `scripts/perf_baseline.json` should be updated:
- when adding a new entry to the BENCHMARKS list,
- after intentional code changes that affect timing,
- when moving the perf runner to different hardware.

Always commit the baseline update in the same PR as the change that
caused it, with a note in the commit body explaining the delta.
