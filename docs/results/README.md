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

## SOPPI Box Pushing, 2026-06-14

- Report: [`soppi_box_pushing_2026-06-14.md`](soppi_box_pushing_2026-06-14.md)
- CSV: [`soppi_box_pushing_2026-06-14.csv`](soppi_box_pushing_2026-06-14.csv)
- Scope: `box_turn,box_align,box_pivot,box_swivel,box_align_strict,box_align_detour,box_align_contact_loss`
- Planners: `mppi`, `diff_mppi_1`, `diff_mppi_3`, `soppi`, `soppi_fast`
- Sample count: `K=256`
- Seeds: 4 per scenario/planner cell

Reproduce from the repository root:

```bash
cmake --build build --target benchmark_diff_mppi_pushing_box -j$(nproc)
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-14.csv
```

Key signals:

- `box_align_contact_loss`: pure `soppi` `0.25` vs `mppi` `0.00` (contact-loss cell,
  no nominal grad steps).
- `box_swivel` all-pairs `soppi` remains `1.00` vs MPPI `0.75`.

## SOPPI Box Pushing, 2026-06-13

- Report: [`soppi_box_pushing_2026-06-13.md`](soppi_box_pushing_2026-06-13.md)
- CSV: [`soppi_box_pushing_2026-06-13.csv`](soppi_box_pushing_2026-06-13.csv)
- Scope: `box_turn,box_align,box_pivot,box_swivel,box_align_strict,box_align_detour`
- Planners: `mppi`, `diff_mppi_1`, `diff_mppi_3`, `soppi`, `soppi_fast`, `soppi_fast_g3`
- Key signal: `soppi_fast_g3` matches `diff_mppi_3` on `box_align_detour` (`0.25`).

## SOPPI Box Pushing, 2026-06-12

- Report: [`soppi_box_pushing_2026-06-12.md`](soppi_box_pushing_2026-06-12.md)
- CSV: [`soppi_box_pushing_2026-06-12.csv`](soppi_box_pushing_2026-06-12.csv)
- Scope: `box_turn,box_align,box_pivot,box_swivel,box_align_strict,box_align_detour`
- Planners: `mppi`, `diff_mppi_1`, `diff_mppi_3`, `soppi`, `soppi_fast`
- Sample count: `K=256`
- Seeds: 4 per scenario/planner cell

Reproduce from the repository root:

```bash
cmake --build build --target benchmark_diff_mppi_pushing_box -j$(nproc)
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-12.csv
```

Key signals:

- `box_align_detour` adds an axis-aligned wall; only `diff_mppi_3` reaches partial
  success (`0.25`) while MPPI and SOPPI stay at `0.00`.
- `box_swivel` still shows all-pairs `soppi` at `1.00` vs MPPI `0.75`.

## SOPPI Box Pushing, 2026-06-11

- Report: [`soppi_box_pushing_2026-06-11.md`](soppi_box_pushing_2026-06-11.md)
- CSV: [`soppi_box_pushing_2026-06-11.csv`](soppi_box_pushing_2026-06-11.csv)
- Scope: `box_turn,box_align,box_pivot,box_swivel,box_align_strict`
- Planners: `mppi`, `diff_mppi_1`, `diff_mppi_3`, `soppi`, `soppi_fast`
- Sample count: `K=256`
- Seeds: 4 per scenario/planner cell

Reproduce from the repository root:

```bash
cmake --build build --target benchmark_diff_mppi_pushing_box -j$(nproc)
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-11.csv
```

Key signals:

- `box_align_strict` is the new orientation-binding cell: Diff-MPPI `1.00`,
  `soppi_fast` ties MPPI at `0.75` with lower cost.
- `box_swivel` still shows all-pairs `soppi` at `1.00` vs MPPI `0.75`.

## SOPPI Box Pushing, 2026-06-10

- Report: [`soppi_box_pushing_2026-06-10.md`](soppi_box_pushing_2026-06-10.md)
- CSV: [`soppi_box_pushing_2026-06-10.csv`](soppi_box_pushing_2026-06-10.csv)
- Scope: `box_turn,box_align,box_pivot,box_swivel`
- Planners: `mppi`, `diff_mppi_1`, `diff_mppi_3`, `soppi`, `soppi_fast`
- Sample count: `K=256`
- Seeds: 4 per scenario/planner cell

Reproduce from the repository root:

```bash
cmake --build build --target benchmark_diff_mppi_pushing_box -j$(nproc)
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-10.csv
```

Key signals:

- `box_swivel` is the discriminating cell: all-pairs `soppi` reaches `1.00`
  success where vanilla `mppi` and Diff-MPPI stop at `0.75`.
- `box_align` shows a large final-distance/cost gap for SOPPI even though the
  strict success threshold is not crossed.
- Post-kernel `soppi_fast` is about 3.4x faster than the pre-optimization note
  on `box_swivel` and about 1.8x slower than MPPI, down from roughly 9x slower.

## MPPI Zoo Suite, 2026-06-09

Eight-planner predecessor run, kept for comparison:

- Report: [`mppi_zoo_suite_2026-06-09.md`](mppi_zoo_suite_2026-06-09.md)
- CSV: [`mppi_zoo_suite_2026-06-09.csv`](mppi_zoo_suite_2026-06-09.csv)
- Chart: [`mppi_zoo_suite_2026-06-09.svg`](mppi_zoo_suite_2026-06-09.svg)
