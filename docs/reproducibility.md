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
