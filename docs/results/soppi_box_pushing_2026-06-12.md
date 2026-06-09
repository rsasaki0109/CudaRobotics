# SOPPI Box Pushing Report

_Includes axis-aligned obstacle support and the `box_align_detour` scenario._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-12.csv
```

## Inputs

| field | value |
| --- | --- |
| csv | docs/results/soppi_box_pushing_2026-06-12.csv |
| scenarios | box_turn, box_align, box_pivot, box_swivel, box_align_strict, box_align_detour |
| planners | mppi, diff_mppi_1, diff_mppi_3, soppi, soppi_fast |
| k_values | 256 |
| seed_count | 4 |

## `box_align_detour`

`box_align` geometry with pusher start `(1.10, 1.20)` and a narrow wall on the
direct push lane (`x=[1.48,1.72]`, `y=[1.98,2.14]`). Success requires reaching
the goal pose without obstacle penetration.

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 280.0 | 0.294 | 2.9 | 0.419 |
| diff_mppi_1 | 0.00 | 280.0 | 0.269 | 2.5 | 1.644 |
| diff_mppi_3 | 0.25 | 219.5 | 0.256 | 2.1 | 3.059 |
| soppi | 0.00 | 280.0 | 0.294 | 2.9 | 0.851 |
| soppi_fast | 0.00 | 280.0 | 0.296 | 2.9 | 0.738 |

## Key Signals

- `box_align_detour` is the first obstacle cell: only `diff_mppi_3` clears a seed
  (`0.25` success, `38` steps on seed 2); vanilla MPPI and SOPPI stay at `0.00`.
- `box_swivel` and `box_align_strict` cells are unchanged from the 2026-06-11 row.
- Obstacle penetration is tracked in the CSV (`collision_free`, `collisions`) and
  folded into success on obstacle scenarios.
