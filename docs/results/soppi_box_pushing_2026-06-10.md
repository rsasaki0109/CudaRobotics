# SOPPI Box Pushing Report

_Post-kernel-optimization checked-in run (PR #168)._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-10.csv
```

## Inputs

| field | value |
| --- | --- |
| csv | docs/results/soppi_box_pushing_2026-06-10.csv |
| scenarios | box_turn, box_align, box_pivot, box_swivel |
| planners | mppi, diff_mppi_1, diff_mppi_3, soppi, soppi_fast |
| k_values | 256 |
| seed_count | 4 |

## Planner Aggregate

| planner | scenarios | success | final_d | steps | cost | avg_ms |
| --- | --- | --- | --- | --- | --- | --- |
| diff_mppi_1 | 4 | 0.25 | 0.295 | 200.4 | 3.6 | 0.944 |
| diff_mppi_3 | 4 | 0.31 | 0.293 | 187.5 | 3.5 | 2.339 |
| mppi | 4 | 0.19 | 0.307 | 220.8 | 3.9 | 0.142 |
| soppi | 4 | 0.25 | 0.249 | 245.0 | 2.9 | 0.489 |
| soppi_fast | 4 | 0.19 | 0.255 | 245.2 | 2.9 | 0.289 |

## Per Scenario

### `box_align`

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 240.0 | 0.433 | 7.8 | 0.122 |
| diff_mppi_1 | 0.25 | 201.0 | 0.400 | 7.3 | 0.951 |
| diff_mppi_3 | 0.50 | 149.8 | 0.397 | 7.1 | 2.356 |
| soppi | 0.00 | 240.0 | 0.275 | 4.2 | 0.501 |
| soppi_fast | 0.00 | 240.0 | 0.276 | 4.2 | 0.276 |

### `box_pivot`

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 240.0 | 0.108 | 1.2 | 0.132 |
| diff_mppi_1 | 0.00 | 240.0 | 0.113 | 1.1 | 0.962 |
| diff_mppi_3 | 0.00 | 240.0 | 0.115 | 1.0 | 2.282 |
| soppi | 0.00 | 240.0 | 0.107 | 1.1 | 0.473 |
| soppi_fast | 0.00 | 240.0 | 0.107 | 1.1 | 0.307 |

### `box_swivel`

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.75 | 103.0 | 0.275 | 1.8 | 0.160 |
| diff_mppi_1 | 0.75 | 100.5 | 0.273 | 1.8 | 0.908 |
| diff_mppi_3 | 0.75 | 100.0 | 0.271 | 1.8 | 2.353 |
| soppi | 1.00 | 98.0 | 0.215 | 1.7 | 0.549 |
| soppi_fast | 0.75 | 100.8 | 0.236 | 1.8 | 0.291 |

### `box_turn`

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 260.0 | 0.411 | 4.6 | 0.152 |
| diff_mppi_1 | 0.00 | 260.0 | 0.393 | 4.2 | 0.957 |
| diff_mppi_3 | 0.00 | 260.0 | 0.388 | 4.1 | 2.364 |
| soppi | 0.00 | 260.0 | 0.400 | 4.5 | 0.434 |
| soppi_fast | 0.00 | 260.0 | 0.402 | 4.5 | 0.280 |

## Key Signals

- `box_swivel` is the discriminating cell: all-pairs `soppi` reaches `1.00` success
  where vanilla `mppi` and Diff-MPPI stop at `0.75`.
- `box_align` still shows a large final-distance/cost gap for SOPPI (`0.43 -> 0.28`,
  `7.8 -> 4.2`) even though the strict success threshold is not crossed.
- Post-kernel `soppi_fast` is about **3.4x faster** than the pre-optimization note
  on `box_swivel` (`1.00 ms -> 0.29 ms`) and about **1.8x slower than MPPI**,
  down from roughly 9x slower before the kernel pass.
- `box_turn` and `box_pivot` remain mostly insensitive under this quick budget.
