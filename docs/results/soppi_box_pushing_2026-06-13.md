# SOPPI Box Pushing Report

_Adds `soppi_g3` / `soppi_fast_g3` hybrid planners (SVGD + nominal Diff-MPPI grad steps)._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast,soppi_fast_g3 \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-13.csv
```

## Inputs

| field | value |
| --- | --- |
| csv | docs/results/soppi_box_pushing_2026-06-13.csv |
| scenarios | box_turn, box_align, box_pivot, box_swivel, box_align_strict, box_align_detour |
| planners | mppi, diff_mppi_1, diff_mppi_3, soppi, soppi_fast, soppi_fast_g3 |
| k_values | 256 |
| seed_count | 4 |

## `box_align_detour`

`box_align` geometry with pusher start `(1.10, 1.20)` and a narrow wall on the
direct push lane (`x=[1.48,1.72]`, `y=[1.98,2.14]`). Success requires reaching
the goal pose without obstacle penetration.

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 280.0 | 0.294 | 2.9 | 0.423 |
| diff_mppi_1 | 0.00 | 280.0 | 0.269 | 2.5 | 1.602 |
| diff_mppi_3 | 0.25 | 219.5 | 0.256 | 2.1 | 3.058 |
| soppi | 0.00 | 280.0 | 0.294 | 2.9 | 0.850 |
| soppi_fast | 0.00 | 280.0 | 0.296 | 2.9 | 0.651 |
| soppi_fast_g3 | 0.25 | 219.2 | 0.249 | 2.0 | 3.167 |

## Key Signals

- Pure SVGD (`soppi`, `soppi_fast`) still fails on `box_align_detour`; adding three
  nominal grad steps (`soppi_fast_g3`, same `alpha=0.010` as `diff_mppi_3`) matches
  `diff_mppi_3` at `0.25` success (`37` steps on seed 2).
- `soppi_fast_g3` also lifts `box_align_strict` to `1.00` and keeps `box_swivel` at
  `1.00`; cost is about `4.7x` slower than `soppi_fast` on detour.
- `box_swivel` all-pairs `soppi` remains `1.00` vs MPPI `0.75`.
