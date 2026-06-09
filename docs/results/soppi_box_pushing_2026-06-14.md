# SOPPI Box Pushing Report

_Adds `box_align_contact_loss` contact-loss cell and stage gap penalty (`w_contact_loss`)._

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

## `box_align_contact_loss`

`box_align_strict` geometry with `w_near=0`, `w_contact_loss=47`, and a small
contact deadzone (`pen_thresh=0.009`). Stage cost penalizes squared pusher-box gap
when the pusher leaves smooth contact during the rotation arc.

| planner | success | steps | final_d | cost | avg_ms |
| --- | --- | --- | --- | --- | --- |
| mppi | 0.00 | 240.0 | 0.287 | 4.9 | 0.289 |
| diff_mppi_1 | 1.00 | 48.0 | 0.279 | 3.0 | 1.525 |
| diff_mppi_3 | 1.00 | 44.0 | 0.277 | 2.8 | 2.770 |
| soppi | 0.25 | 216.2 | 0.286 | 4.7 | 0.866 |
| soppi_fast | 0.00 | 240.0 | 0.289 | 4.9 | 0.553 |

## Key Signals

- Pure all-pairs `soppi` reaches `0.25` success on `box_align_contact_loss` while
  vanilla `mppi` stays at `0.00` — a contact-loss cell where SVGD helps without
  nominal Diff-MPPI grad steps (`seed 3`, `145` steps).
- `diff_mppi_3` remains strongest at `1.00`; the gap penalty is in both rollout
  cost and the SOPPI autodiff score kernel.
- Prior scenarios are unchanged (contact-loss appended last; seeds `si=0..5` match
  the 2026-06-13 run). `box_swivel` all-pairs `soppi` stays `1.00` vs MPPI `0.75`.
