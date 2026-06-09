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

## MPPI Zoo Suite, 2026-06-10

- Report: [`mppi_zoo_suite_2026-06-10.md`](mppi_zoo_suite_2026-06-10.md)
- CSV: [`mppi_zoo_suite_2026-06-10.csv`](mppi_zoo_suite_2026-06-10.csv)
- Chart: [`mppi_zoo_suite_2026-06-10.svg`](mppi_zoo_suite_2026-06-10.svg)
- GIF: [gh-pages](https://rsasaki0109.github.io/CudaRobotics/gpu_mppi_zoo_dynamic_crossing.gif)
  (local: [`../gif/gpu_mppi_zoo_dynamic_crossing.gif`](../gif/gpu_mppi_zoo_dynamic_crossing.gif))
- Scope: `dynamic_crossing,narrow_passage,model_mismatch_crossing,dynamic_pincer,uncertain_crossing`
- Planners: `mppi`, `step_mppi_smooth`, `tsallis_mppi_smooth`, `ducct_mppi_smooth`,
  `dra_mppi_soft`, `lp_mppi_smooth`, `c2u_mppi_smooth`, `sc_mppi_smooth`,
  `soppi`, `soppi_fast`
- Sample counts: `K=64,128`
- Seeds: 3 per scenario/planner/K cell

Reproduce from the repository root:

```bash
cmake --build build --target benchmark_diff_mppi -j$(nproc)
python3 scripts/run_mppi_zoo_suite.py --bin bin/benchmark_diff_mppi
python3 scripts/render_mppi_zoo_suite_chart.py
python3 scripts/render_mppi_zoo_gif.py --bin bin/benchmark_diff_mppi
```

Docker equivalent:

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_suite.py --bin ./bin/benchmark_diff_mppi && python3 scripts/render_mppi_zoo_suite_chart.py'
```

Key signals:

- `dynamic_crossing` and `uncertain_crossing` remain strong negative controls:
  vanilla `mppi` fails while the curated zoo variants solve the same cells.
- `model_mismatch_crossing` separates Step/Tsallis from vanilla `mppi`; DRA and
  DUCCT are only partially solved on this cell at `K=64,128`.
- `dynamic_pincer` separates risk-aware planners from vanilla `mppi` (success
  0.00 vs 1.00 for the zoo set in this run).
- `dynamic_slalom` stays outside this suite: it needs gradient/hybrid guidance
  at low K and remains a separate benchmark cell in `benchmark_diff_mppi`.
- `narrow_passage` stays an efficiency check: all curated planners succeed and
  finish in fewer steps than vanilla `mppi`.
- Aggregate: `step_mppi_smooth`, `tsallis_mppi_smooth`, and `sc_mppi_smooth`
  9/10 solved cells; `soppi` and `soppi_fast` 2/10 (navigation coverage only);
  vanilla `mppi` 2/10.

## MPPI Zoo Suite, 2026-06-09

Eight-planner predecessor run, kept for comparison:

- Report: [`mppi_zoo_suite_2026-06-09.md`](mppi_zoo_suite_2026-06-09.md)
- CSV: [`mppi_zoo_suite_2026-06-09.csv`](mppi_zoo_suite_2026-06-09.csv)
- Chart: [`mppi_zoo_suite_2026-06-09.svg`](mppi_zoo_suite_2026-06-09.svg)
