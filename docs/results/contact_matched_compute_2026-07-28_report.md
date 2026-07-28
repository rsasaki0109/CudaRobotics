# Deadline-Matched Contact Control

Every planner receives the same enforced wall-clock control slot. `real_time_success` requires both task success and zero deadline misses.

## Calibrated budgets

| Planner | Selected K |
|---|---:|
| mppi | 1024 |
| diff_mppi_3 | 1024 |
| soppi_fast | 1024 |

## Evaluation

| Scenario | Planner | K | RT success | Deadline misses | Max ms |
|---|---|---:|---:|---:|---:|
| box_swivel | mppi | 1024 | 0.967 | 0 | 4.221 |
| box_swivel | diff_mppi_3 | 1024 | 1.000 | 0 | 6.880 |
| box_swivel | soppi_fast | 1024 | 1.000 | 0 | 6.444 |
| box_align_strict | mppi | 1024 | 0.933 | 1 | 10.781 |
| box_align_strict | diff_mppi_3 | 1024 | 1.000 | 0 | 6.866 |
| box_align_strict | soppi_fast | 1024 | 1.000 | 0 | 6.264 |
| box_align_detour | mppi | 1024 | 0.000 | 0 | 4.645 |
| box_align_detour | diff_mppi_3 | 1024 | 0.000 | 1 | 11.000 |
| box_align_detour | soppi_fast | 1024 | 0.000 | 54 | 65.903 |
| box_align_contact_loss | mppi | 1024 | 0.467 | 3 | 12.243 |
| box_align_contact_loss | diff_mppi_3 | 1024 | 1.000 | 0 | 6.710 |
| box_align_contact_loss | soppi_fast | 1024 | 0.967 | 1 | 10.791 |
| box_align_contact_arc | mppi | 1024 | 1.000 | 0 | 4.302 |
| box_align_contact_arc | diff_mppi_3 | 1024 | 1.000 | 0 | 6.392 |
| box_align_contact_arc | soppi_fast | 1024 | 1.000 | 0 | 6.466 |

## Paired comparisons versus MPPI

| Scenario | Planner | N | RT success delta [95% CI] | Holm p |
|---|---|---:|---:|---:|
| box_swivel | diff_mppi_3 | 30 | +0.033 [+0.000, +0.100] | 1 |
| box_swivel | soppi_fast | 30 | +0.033 [+0.000, +0.100] | 1 |
| box_align_strict | diff_mppi_3 | 30 | +0.067 [+0.000, +0.167] | 1 |
| box_align_strict | soppi_fast | 30 | +0.067 [+0.000, +0.167] | 1 |
| box_align_detour | diff_mppi_3 | 30 | +0.000 [+0.000, +0.000] | 1 |
| box_align_detour | soppi_fast | 30 | +0.000 [+0.000, +0.000] | 1 |
| box_align_contact_loss | diff_mppi_3 | 30 | +0.533 [+0.367, +0.700] | 0.0003052 |
| box_align_contact_loss | soppi_fast | 30 | +0.500 [+0.300, +0.700] | 0.002472 |
| box_align_contact_arc | diff_mppi_3 | 30 | +0.000 [+0.000, +0.000] | 1 |
| box_align_contact_arc | soppi_fast | 30 | +0.000 [+0.000, +0.000] | 1 |

Calibration seeds and evaluation seeds are disjoint. The largest registered K with zero calibration deadline misses is selected independently for each planner before held-out evaluation.
