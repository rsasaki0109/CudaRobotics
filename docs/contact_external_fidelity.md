# MuJoCo Contact External-Fidelity Suite

This suite tests the CUDA MPPI family against a structurally separate MuJoCo
contact plant. The controller retains its nominal smooth box-pushing model.
MuJoCo executes every selected command, advances contact dynamics, and returns
the next observed state before the following controller call. It is therefore
closed-loop sim-to-sim transfer, not open-loop replay and not real-robot
evidence.

The release profile contains 3,150 episodes:

| Axis | Release matrix |
|---|---|
| Seeds | 30 deterministic paired seeds per cell |
| K | 256 |
| Scenarios | `box_swivel`, `box_align_strict`, `box_align_detour`, `box_align_contact_loss`, `box_align_contact_arc` |
| Planners | MPPI, Diff-MPPI-3, SOPPI-fast |
| MuJoCo conditions | nominal; friction 0.3/0.9; box mass 0.75/1.25; position/yaw observation noise 0.01 m/0.02 rad and 0.02 m/0.04 rad |

## Build and run

MuJoCo is optional. Once CMake finds its headers and library:

```bash
cmake -S . -B build
cmake --build build --target benchmark_diff_mppi_pushing_box_mujoco -j
python3 scripts/run_contact_external_fidelity.py \
  --output-dir build/contact_external_fidelity_release \
  --binary bin/benchmark_diff_mppi_pushing_box_mujoco \
  --model mujoco_models/contact_box_push.xml \
  --profile release
```

Release evidence requires a clean commit, an identified NVIDIA GPU, the exact
MuJoCo engine/header version, a staged model, executable, and MuJoCo runtime
library, and the complete fixed matrix. The staged runtime directory is
prepended to the child process library search path. Each condition is an
independent attempt-numbered process. State is written atomically after each
attempt, so interrupted runs resume without discarding failures:

```bash
python3 scripts/run_contact_external_fidelity.py \
  --output-dir build/contact_external_fidelity_release \
  --profile release --resume
```

Resume is refused when the commit, executable, MuJoCo XML, matrix, or
statistical policy changes.

## Validate and publish

```bash
python3 scripts/validate_contact_external_fidelity.py \
  build/contact_external_fidelity_release --profile release
python3 scripts/publish_contact_external_fidelity.py \
  build/contact_external_fidelity_release \
  --output-dir docs/results --profile release
```

Validation checks complete matrix coverage, raw attempt provenance, engine/GPU
identity, clean release state, and SHA-256 identities for every artifact.
Publication revalidates the source and writes compact summary, comparison,
report, and provenance files. Outcomes—including negative results—are never
used as evidence-integrity gates.

## Smoke

Use `--profile smoke` for a four-episode pipeline check. Smoke evidence cannot
satisfy the paper's external-fidelity claim.

## Release result

The 2026-07-28 UTC release run completed all 3,150 episodes on a GTX 1660 Ti
with MuJoCo 3.11.0. Diff-MPPI-3 reached 0.46 aggregate success, SOPPI-fast
0.34, and MPPI 0.29 across the fixed matrix. The full 70-cell paired comparison
family contained three Holm-significant positive Diff-MPPI-3 cells and no
Holm-significant negative cells.

The significant cells were `box_align_strict` at friction 0.3 and mass scale
0.75 (success delta +0.50 in each), plus `box_align_contact_arc` at mass scale
1.25 (+0.40). No individual sensing-noise cell survived full-family Holm
correction, and difficult zero/near-zero-success scenarios remain in the
tables. See
[`contact_external_fidelity_2026-07-28_report.md`](results/contact_external_fidelity_2026-07-28_report.md)
and the adjacent provenance JSON.
