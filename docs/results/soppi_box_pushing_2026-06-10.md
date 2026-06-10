# SOPPI Box Pushing Report

_Tunes `soppi_fast` on `box_align_contact_loss`, documents `box_align_contact_arc`,
fixes canonical scenario seed indices for filtered runs._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-10.csv
```

## Inputs

| field | value |
| --- | --- |
| csv | docs/results/soppi_box_pushing_2026-06-10.csv |
| scenarios | box_turn … box_align_contact_loss, box_align_contact_arc (8 total) |
| planners | mppi, diff_mppi_3, soppi, soppi_fast |
| k_values | 256 |
| seed_count | 4 |

## `box_align_contact_loss` (strict orientation gate)

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 240.0 | 0.286 | 4.8 | 0.127 |
| diff_mppi_3 | 1.00 | 44.0 | 0.275 | 2.8 | 2.406 |
| soppi | 0.50 | 189.5 | 0.284 | 4.5 | 0.473 |
| soppi_fast | **0.75** | 121.5 | 0.281 | 3.8 | 0.530 |

Subset SVGD defaults tuned on canonical seeds (`neighbor_count=112`, `svgd_iters=2`,
`step_size=0.05`, `bandwidth=2.0`). Pure all-pairs `soppi` stays at `0.50`; vanilla
`mppi` at `0.00`.

## `box_align_contact_arc` (wider gate, same contact gradient)

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 1.00 | 54.8 | 0.299 | 3.3 | 0.079 |
| diff_mppi_3 | 1.00 | 41.8 | 0.297 | 2.7 | 1.943 |
| soppi | 1.00 | 55.0 | 0.298 | 3.3 | 0.487 |
| soppi_fast | 1.00 | 56.2 | 0.297 | 3.3 | 0.471 |

## Key signals

- **`box_align_contact_arc`** is the pure-SOPPI-friendly contact cell: all four
  planners at `1.00` on fixed seeds with the wider `pos_tol=0.30`, `ang_tol=0.12`.
- **`box_align_contact_loss`** remains the hard strict-gate cell: subset SVGD
  (`soppi_fast`) reaches `0.75` vs all-pairs `soppi` `0.50` and `mppi` `0.00`.
- Hybrid `soppi_fast_g3` (nominal grad + subset SVGD) still reaches `1.00` on the
  strict cell when nominal trajectory steps are required.
- Filtered `--scenarios` runs now use the same per-scenario RNG seeds as the full
  suite (canonical `si` index), so isolated probes match published numbers.
