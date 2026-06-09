# SOPPI Box Pushing Report

_Includes the new `box_align_strict` orientation-binding scenario._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-11.csv
```

## Inputs

| field | value |
| --- | --- |
| csv | docs/results/soppi_box_pushing_2026-06-11.csv |
| scenarios | box_turn, box_align, box_pivot, box_swivel, box_align_strict |
| planners | mppi, diff_mppi_1, diff_mppi_3, soppi, soppi_fast |
| k_values | 256 |
| seed_count | 4 |

## `box_align_strict`

Same geometry as `box_align` with `pos_tol=0.28 m` and `ang_tol=0.08 rad`
(parent: `0.22 m` / `0.25 rad`). The combined gate turns the parent task's
near-misses into partial or full success for gradient/SVGD planners.

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.75 | 121.0 | 0.283 | 4.5 | 0.091 |
| diff_mppi_1 | 1.00 | 73.0 | 0.274 | 4.1 | 0.689 |
| diff_mppi_3 | 1.00 | 71.0 | 0.269 | 4.0 | 1.745 |
| soppi | 0.50 | 159.0 | 0.283 | 4.4 | 0.309 |
| soppi_fast | 0.75 | 119.0 | 0.279 | 4.1 | 0.191 |

## Other Scenarios (unchanged cells)

| scenario | best SOPPI signal |
| --- | --- |
| box_align | final_d `0.28` vs MPPI `0.43`; success still `0.00` for all |
| box_swivel | `soppi` `1.00` vs MPPI `0.75` |
| box_pivot / box_turn | insensitive at this budget |

## Key Signals

- `box_align_strict` is the new discriminating cell: Diff-MPPI reaches `1.00`
  success while the parent `box_align` row stays at `0.00` for vanilla MPPI and SOPPI.
- `soppi_fast` ties MPPI on success (`0.75`) but finishes with lower cost
  (`4.1` vs `4.5`) and fewer steps (`119` vs `121`).
- `box_swivel` remains the rotation escape cell where all-pairs `soppi` is the
  only planner at `1.00` success.
