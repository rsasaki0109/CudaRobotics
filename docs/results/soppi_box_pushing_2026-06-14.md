# SOPPI Box Pushing Report

_Seven-scenario suite (no `box_align_contact_arc`); `soppi_fast` contact-loss row
updated after subset-SVGD + one nominal-grad tuning — see also
[`soppi_box_pushing_2026-06-10.md`](soppi_box_pushing_2026-06-10.md) for the
eight-scenario canonical row._

## Command

```bash
./bin/benchmark_diff_mppi_pushing_box \
  --quick \
  --planners mppi,diff_mppi_1,diff_mppi_3,soppi,soppi_fast \
  --k-values 256 \
  --seed-count 4 \
  --csv docs/results/soppi_box_pushing_2026-06-14.csv
```

## Inputs

| field | value |
| --- | --- |
| csv | docs/results/soppi_box_pushing_2026-06-14.csv |
| scenarios | box_turn, box_align, box_pivot, box_swivel, box_align_strict, box_align_detour, box_align_contact_loss |
| planners | mppi, diff_mppi_1, diff_mppi_3, soppi, soppi_fast |
| k_values | 256 |
| seed_count | 4 |

## Signal cells (`K=256`, 4 seeds)

| scenario | mppi | soppi | soppi_fast | diff_mppi_3 |
| --- | ---:| ---:| ---:| ---:|
| box_swivel | 0.75 | **1.00** | **1.00** | 0.75 |
| box_align_strict | 0.75 | 0.50 | 0.75 | **1.00** |
| box_align_detour | 0.00 | 0.00 | 0.00 | 0.25 |
| box_align_contact_loss | 0.00 | 0.50 | **1.00** | **1.00** |

## `box_align_contact_loss`

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 240.0 | 0.286 | 4.8 | 0.654 |
| diff_mppi_1 | 1.00 | 45.2 | 0.277 | 2.9 | 2.771 |
| diff_mppi_3 | 1.00 | 44.0 | 0.275 | 2.8 | 3.542 |
| soppi | 0.50 | 189.5 | 0.284 | 4.5 | 1.520 |
| soppi_fast | **1.00** | 52.2 | 0.279 | 3.1 | 3.361 |

`soppi_fast` uses subset SVGD (`neighbor_count=112`, `svgd_iters=2`) plus one nominal
trajectory grad step (`grad_steps=1`, `alpha=0.010`). Pure subset SVGD plateaued at
`0.75` on canonical seeds; the single nominal update closes the strict gate.

## Key signals

- **`box_align_contact_loss`**: `soppi_fast` **1.00** vs `mppi` **0.00**; all-pairs
  `soppi` **0.50** without nominal grad.
- **`box_align_detour`**: still gradient-positive / sampling-negative — only
  `diff_mppi_3` at **0.25**; one nominal grad step is not enough on that cell.
- **`box_swivel`**: all-pairs `soppi` **1.00** vs MPPI **0.75**.
