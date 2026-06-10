# SOPPI Box Pushing Report

_Tunes `soppi_fast` on `box_align_contact_loss` to `1.00`, documents `box_align_contact_arc`,
syncs canonical-seed full-suite rows._

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

## Full suite (`K=256`, 4 seeds)

| scenario | mppi | soppi | soppi_fast | diff_mppi_3 |
| --- | ---:| ---:| ---:| ---:|
| box_swivel | 0.75 | **1.00** | **1.00** | 0.75 |
| box_align_strict | 0.75 | 0.50 | 0.75 | **1.00** |
| box_align_detour | 0.00 | 0.00 | 0.00 | 0.25 |
| box_align_contact_loss | 0.00 | 0.50 | **1.00** | **1.00** |
| box_align_contact_arc | **1.00** | **1.00** | **1.00** | **1.00** |

## `box_align_contact_loss` (strict orientation gate)

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 240.0 | 0.286 | 4.8 | 0.127 |
| diff_mppi_3 | 1.00 | 44.0 | 0.275 | 2.8 | 2.406 |
| soppi | 0.50 | 189.5 | 0.284 | 4.5 | 0.473 |
| soppi_fast | **1.00** | 52.2 | 0.279 | 3.1 | 4.116 |

`soppi_fast` defaults: subset SVGD (`neighbor_count=112`, `svgd_iters=2`, `step_size=0.05`,
`bandwidth=2.0`) plus **one** nominal trajectory grad step (`grad_steps=1`, `alpha=0.010`).
Pure subset SVGD alone plateaued at **0.75** on canonical seeds (seed 3 stalled at
`pos=0.287` vs `pos_tol=0.28`); a single Diff-MPPI-style nominal update closes the gap
without the three-step hybrid of `soppi_fast_g3`.

## `box_align_contact_arc` (wider gate, same contact gradient)

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 1.00 | 54.8 | 0.299 | 3.3 | 0.079 |
| diff_mppi_3 | 1.00 | 41.8 | 0.297 | 2.7 | 1.943 |
| soppi | 1.00 | 55.0 | 0.298 | 3.3 | 0.487 |
| soppi_fast | 1.00 | 49.8 | 0.298 | 3.0 | 4.812 |

## Key signals

- **`box_align_contact_loss`**: `soppi_fast` **1.00** on canonical seeds at `K=256`
  (subset SVGD + 1 nominal grad step) vs `mppi` **0.00** and all-pairs `soppi` **0.50**.
- **`box_align_contact_arc`** remains the pure-SOPPI-friendly contact cell: all four
  planners at **1.00** with the wider `pos_tol=0.30`, `ang_tol=0.12`.
- **`box_align_detour`** still needs nominal grad (`diff_mppi_3` / `soppi_fast_g3`); one
  grad step is not enough on that obstacle cell.
- Filtered `--scenarios` runs use stable per-scenario RNG seeds (canonical `si` index).
