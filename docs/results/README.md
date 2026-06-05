# Checked-In Benchmark Results

This directory stores benchmark artifacts that are meant to be inspected before
running a GPU locally. They are not final paper-faithful claims; they are
fixed-seed smoke results with enough detail to make wins and failures visible.

## MPPI Zoo Smoke, 2026-06-05

- Report: [`mppi_zoo_smoke_2026-06-05.md`](mppi_zoo_smoke_2026-06-05.md)
- CSV: [`mppi_zoo_smoke_2026-06-05.csv`](mppi_zoo_smoke_2026-06-05.csv)
- Chart: [`mppi_zoo_smoke_2026-06-05.svg`](mppi_zoo_smoke_2026-06-05.svg)
- Scope: `dynamic_crossing,narrow_passage`
- Planners: `mppi`, `lp_mppi_smooth`, `step_mppi_smooth`,
  `tsallis_mppi_smooth`, `dra_mppi_soft`, `c2u_mppi_smooth`,
  `ducct_mppi_smooth`, `dbas_log_mppi_agile`, `pa_mppi_smooth`
- Sample counts: `K=64,128`
- Seeds: 3 per scenario/planner/K cell

Reproduce from the repository root with a CUDA-capable Docker setup:

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_smoke.py --bin ./bin/benchmark_diff_mppi --out-dir docs/results --csv docs/results/mppi_zoo_smoke_2026-06-05.csv --markdown-out docs/results/mppi_zoo_smoke_2026-06-05.md --seed-count 3 --k-values 64,128'
python3 scripts/render_mppi_zoo_chart.py --csv docs/results/mppi_zoo_smoke_2026-06-05.csv --svg-out docs/results/mppi_zoo_smoke_2026-06-05.svg
```

Key signals:

- `dynamic_crossing` is the negative control: vanilla `mppi` has success 0.00
  at `K=64` and `K=128`, while the zoo variants solve the same cells.
- `narrow_passage` is an efficiency check: vanilla `mppi` succeeds, but the
  smooth variants reduce the average number of control steps in this smoke run.
