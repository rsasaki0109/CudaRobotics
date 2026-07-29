# Reproducibility

CudaRobotics benchmark results are meant to be regenerated from source, not only read from checked-in figures.
The primary entry point is:

```bash
python3 scripts/run_repro_suite.py --suite smoke
```

The runner writes per-benchmark CSV files, Markdown summaries, logs, a human-readable report, and a
machine-readable manifest under `build/repro_suite/` by default. It does not hide the underlying commands:
every command line is recorded in `manifest.json` and `report.md`.

## Suites

| Suite | Scope | Use |
|---|---|---|
| `smoke` | One tiny Diff-MPPI dynamic-navigation run | Fast GPU sanity check and CI dry-run validation |
| `diff-mppi` | Dynamic navigation, CartPole, dynamic bicycle, 2-link arm, and 7-DOF arm quick runs | Main research reproduction pass without optional MuJoCo |
| `standard` | Optional MuJoCo pendulum and reacher quick runs | Standard-environment check when MuJoCo is installed |
| `all` | `diff-mppi` plus optional MuJoCo tasks | Broadest quick reproduction suite |

List the exact suite contents with:

```bash
python3 scripts/run_repro_suite.py --list
```

## Common Commands

Plan a run without requiring CUDA, MuJoCo, or benchmark binaries:

```bash
python3 scripts/run_repro_suite.py --dry-run --suite smoke
```

Build selected targets before running:

```bash
python3 scripts/run_repro_suite.py --build --suite diff-mppi
```

Run one explicit task:

```bash
python3 scripts/run_repro_suite.py --only dynamic_bicycle_quick --build
```

Generate plots in addition to CSV and Markdown summaries:

```bash
python3 scripts/run_repro_suite.py --suite diff-mppi --plots
```

Use a separate output directory for a paper or release snapshot:

```bash
python3 scripts/run_repro_suite.py --suite all --output-dir build/repro_suite_$(date +%Y%m%d)
```

## Output Layout

Each task writes:

- `<task>.csv`: raw per-episode benchmark rows
- `<task>_summary.md`: summary tables from `scripts/summarize_diff_mppi.py`
- `logs/<task>.log`: benchmark stdout/stderr
- `logs/<task>_summary.log`: summarizer stdout/stderr
- `plots/<task>/`: optional figures when `--plots` is set
- `report.md`: human-readable run overview with task statuses, links, logs, and commands
- `manifest.json`: commands, outputs, statuses, timings, and git commit

Regenerate a report from an existing manifest with:

```bash
python3 scripts/render_repro_report.py --manifest build/repro_suite/manifest.json
```

Optional MuJoCo tasks are skipped when their binaries are missing unless `--strict-optional` is set.
This keeps the main suite usable on CUDA-only machines while still documenting how to reproduce the
standard-environment checks.

## CI Contract

The CTest target `check_repro_suite_dry_run` validates the runner and manifest path without requiring a GPU:

```bash
ctest --test-dir build -R check_repro_suite_dry_run --output-on-failure
```

The actual benchmark suites remain GPU/runtime checks and should be run on a machine with a working NVIDIA CUDA stack.

## End-to-End CudaNav Evidence

The autonomy-stack evidence has separate gates because closed-loop simulation,
real recorded data, and multi-GPU reproducibility prove different claims:

| Evidence | Command | Claim |
|---|---|---|
| Deterministic closed loop | `scripts/run_cudanav_closed_loop.py` | Commands affect subsequent simulated state |
| Native all-GPU core closed loop | `scripts/run_cudanav_gpu_closed_loop.py` | GPU-estimated state drives MPPI commands that affect later LiDAR scans |
| Real rosbag shadow replay | `scripts/run_cudanav_rosbag_replay.py` | Real sensor/motion data passes the GPU controller quality gate |
| Real dataset pipeline | `scripts/run_cudanav_real_dataset_pipeline.py` | Acquisition inspection, derived Path, materialization, and replay use one content-bound plan |
| Real-bag GPU KISS-ICP | `scripts/run_cudanav_kiss_icp_real.py` | Recorded PointCloud2 GPU odometry passes reference and artifact-integrity gates |
| Real-data all-GPU core shadow | `scripts/run_cudanav_real_gpu_stack.py` | KISS-ICP, voxel mapping, ESDF inflation, and CUDA MPPI share one content-bound real sequence |
| GPU matrix | `scripts/run_cudanav_multi_gpu.py` | ROS smoke or native 30-traversal release reproduces from the same commit/config across physical GPU UUIDs and models |
| Full autonomy suite | `scripts/run_autonomy_suite.py` | Closed-loop, recorded/shadow, and multi-GPU gates from one content-bound release entry point |

All three runners write self-describing manifests and refuse dirty release
evidence. The real-rosbag validator re-hashes the external input dataset and
binds the selected database, diagnostics, controller configuration, evaluation,
and exact commands. See
[`cuda_mppi_bag_eval.md`](cuda_mppi_bag_eval.md),
[`cudanav_closed_loop.md`](cudanav_closed_loop.md), and
[`cudanav_multi_gpu.md`](cudanav_multi_gpu.md).
The standalone real-sensor odometry gate is documented in
[`cudanav_kiss_icp_real.md`](cudanav_kiss_icp_real.md); it is not a controller
or closed-loop claim.
The four-stage native shadow gate is documented in
[`cudanav_real_gpu_stack.md`](cudanav_real_gpu_stack.md). Its commands are not
applied, so it remains distinct from ROS 2 and closed-loop evidence.
The native S-course gate is documented in
[`cudanav_gpu_closed_loop.md`](cudanav_gpu_closed_loop.md), with a checked-in
[30-traversal GPU release result](results/cudanav_gpu_closed_loop_release_2026-07-29.md).
Its commands are applied continuously to the simulated plant for 1059.4
simulated seconds, but it remains distinct from the ROS 2 release-profile gate
with retained MCAP/video and from real-data evidence.
The aggregate release workflow is documented in
[`cudanav_autonomy_suite.md`](cudanav_autonomy_suite.md).

## Paper Claim Gates

Paper prose is not treated as evidence. The manifests under
`paper/artifacts/` bind supported claims to content-addressed files and numeric
CSV/JSON assertions:

```bash
python3 scripts/validate_paper_artifacts.py
python3 scripts/validate_paper_artifacts.py --require-ready
```

The first command validates the ledger and is suitable for CI while experiments
are still pending. The second is the submission gate and fails until every
submission-required systems or contact-rich Diff-MPPI claim is supported.

The contact paper's full robustness matrix has its own resumable GPU runner and
artifact validator:

```bash
python3 scripts/run_contact_robustness.py \
  --output-dir build/contact_robustness_release \
  --profile release
python3 scripts/validate_contact_robustness.py \
  build/contact_robustness_release --profile release
```

See [`contact_diff_mppi_robustness.md`](contact_diff_mppi_robustness.md).

The paper's equal-wall-clock claim uses a separate calibrated deadline
protocol. It selects the largest zero-miss K for each planner on calibration
seeds, freezes those budgets, then evaluates 30 held-out seeds:

```bash
python3 scripts/run_contact_matched_compute.py \
  --output-dir build/contact_matched_compute_release \
  --profile release
python3 scripts/validate_contact_matched_compute.py \
  build/contact_matched_compute_release --profile release
```

See [`contact_matched_compute.md`](contact_matched_compute.md).

The external-fidelity gate keeps the CUDA controller's nominal internal model
but advances the selected command through a MuJoCo contact plant at every
control step:

```bash
python3 scripts/run_contact_external_fidelity.py \
  --output-dir build/contact_external_fidelity_release \
  --profile release
python3 scripts/validate_contact_external_fidelity.py \
  build/contact_external_fidelity_release --profile release
```

See [`contact_external_fidelity.md`](contact_external_fidelity.md). This is
closed-loop sim-to-sim transfer; it is not real-robot evidence.

## Release Preflight

The benchmark suites above regenerate research results. Release candidates use
a separate preflight so package, registration, artifact, and repository gates
are recorded together without treating local checks as proof of remote CI:

```bash
python3 scripts/run_release_preflight.py \
  --profile gpu \
  --build-dir build \
  --dist-dir build/release_v0.2.0/dist \
  --output-dir build/release_v0.2.0/preflight \
  --require-clean
python3 scripts/validate_release_preflight.py \
  build/release_v0.2.0/preflight \
  --profile gpu --commit "$(git rev-parse HEAD)"
```

The output directory contains `manifest.json`, `report.md`, per-gate logs,
registration CSV/Markdown, and the Python artifact SHA256 manifest. The
manifest content-addresses every per-gate log and generated evidence file; the
independent validator rejects missing, replaced, or post-run edited content,
profile downgrades, dirty runs, and commit mismatches. `report.md` is a
human-readable rendering and is not a substitute for the manifest gate.
GitHub Build, manylinux wheel, ROS 2, and final evidence checks remain explicit
external gates and must be attached to the same release-candidate commit.

The Build and Python package workflows support manual execution on a
release-candidate branch or tag. Successful runs upload JSON attestations
bound to the exact SHA checked out from that ref:

```bash
python3 scripts/validate_release_ci.py github_build_ci_evidence.json \
  --gate github_build --commit "$RC_COMMIT"
python3 scripts/validate_release_ci.py python_package_ci_evidence.json \
  --gate python_manylinux_wheels --commit "$RC_COMMIT"
```

Pull-request runs are intentionally rejected as final release evidence. The
Python attestation also requires both versioned build/test jobs, CPython
3.10/3.12 manylinux completion, both declared artifact groups, and the
SHA-256 of an artifact manifest regenerated after downloading those groups
inside the same workflow run.

`scripts/validate_v0_2_release.py` is the final v0.2 decision point. It
independently reruns both local preflight gates, all three remote CI gates,
Python archive structure/hash checks, exact-commit checks, and the explicit
real-rosbag negative-result contract. A successful child report by itself does
not authorize tagging; the aggregate must emit `status: ready`.
