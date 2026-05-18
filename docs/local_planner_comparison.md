# Local planner cross-comparison

DWA, STOMP and Diff-MPPI variants share the same scenario, bicycle dynamics, cost components and obstacle representation in benchmark_diff_mppi. Each cell is a (scenario, dyn_speed_scale, dyn_radius_scale) tuple; success is averaged across seeds.

## Per-planner summary across all cells

| planner | family | cells | cells solved | mean succ | mean final_d | mean ms |
|---|---|---|---|---|---|---|
| dwa_fast | DWA | 30 | 28 | 0.93 | 1.94 | 0.06 |
| dwa_fine | DWA | 30 | 30 | 1.00 | 1.92 | 0.06 |
| dwa_med | DWA | 30 | 30 | 1.00 | 1.91 | 0.06 |
| stomp_1 | STOMP | 30 | 17 | 0.57 | 5.98 | 0.48 |
| stomp_2 | STOMP | 30 | 17 | 0.58 | 2.89 | 0.94 |
| stomp_3_smooth | STOMP | 30 | 18 | 0.60 | 2.85 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 30 | 21 | 0.70 | 2.64 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 30 | 23 | 0.77 | 2.88 | 0.68 |

## Hard-cell focus (speed >= 1.5)

Filter cells with dyn_speed_scale >= 1.5 to capture the regime where the obstacle moves fast enough to force genuine replanning. Lower bound on success differentiates planners.

| planner | family | hard cells | hard cells solved | mean succ | mean final_d |
|---|---|---|---|---|---|
| dwa_fast | DWA | 12 | 10 | 0.83 | 2.00 |
| dwa_fine | DWA | 12 | 12 | 1.00 | 1.94 |
| dwa_med | DWA | 12 | 12 | 1.00 | 1.91 |
| stomp_1 | STOMP | 12 | 6 | 0.50 | 7.15 |
| stomp_2 | STOMP | 12 | 6 | 0.50 | 3.00 |
| stomp_3_smooth | STOMP | 12 | 6 | 0.50 | 3.00 |
| diff_mppi_3 | Diff-MPPI | 12 | 3 | 0.25 | 3.70 |
| diff_mppi_3_early8 | Diff-MPPI | 12 | 5 | 0.42 | 4.38 |

## Best planner per cell

Best = highest success_rate (ties broken by lowest final_d).

| scenario | speed | radius | best planner | succ | final_d | runner-up | runner succ | final_d gap |
|---|---|---|---|---|---|---|---|---|
| dynamic_crossing | +0.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fine | **1.00** | +0.03 |
| dynamic_crossing | +0.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fine | **1.00** | +0.03 |
| dynamic_crossing | +0.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fine | **1.00** | +0.03 |
| dynamic_crossing | +0.50 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fine | **1.00** | +0.03 |
| dynamic_crossing | +1.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.87 | stomp_2 | **1.00** | +0.01 |
| dynamic_crossing | +1.00 | 1.30 | dwa_fast | **1.00** | 1.82 | stomp_1 | **1.00** | +0.05 |
| dynamic_crossing | +1.50 | 1.00 | dwa_fast | **1.00** | 1.80 | stomp_2 | **1.00** | +0.06 |
| dynamic_crossing | +1.50 | 1.30 | dwa_fine | **1.00** | 1.89 | stomp_3_smooth | **1.00** | +0.01 |
| dynamic_crossing | +2.00 | 1.00 | dwa_fast | **1.00** | 1.81 | dwa_med | **1.00** | +0.02 |
| dynamic_crossing | +2.00 | 1.30 | diff_mppi_3 | **1.00** | 1.85 | stomp_1 | **1.00** | +0.00 |
| dynamic_pincer | +0.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fine | **1.00** | +0.03 |
| dynamic_pincer | +0.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fine | **1.00** | +0.03 |
| dynamic_pincer | +0.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.84 | dwa_fine | **1.00** | +0.01 |
| dynamic_pincer | +0.50 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.83 | stomp_1 | **1.00** | +0.03 |
| dynamic_pincer | +1.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.84 | dwa_fine | **1.00** | +0.02 |
| dynamic_pincer | +1.00 | 1.30 | dwa_med | **1.00** | 1.83 | dwa_fast | **1.00** | +0.01 |
| dynamic_pincer | +1.50 | 1.00 | dwa_med | **1.00** | 1.93 | dwa_fine | **1.00** | +0.05 |
| dynamic_pincer | +1.50 | 1.30 | dwa_fine | **1.00** | 1.97 | dwa_med | **1.00** | +0.01 |
| dynamic_pincer | +2.00 | 1.00 | dwa_med | **1.00** | 1.89 | stomp_1 | **1.00** | +0.01 |
| dynamic_pincer | +2.00 | 1.30 | stomp_2 | **1.00** | 1.88 | dwa_fine | **1.00** | +0.00 |
| dynamic_slalom | +0.00 | 1.00 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.00 | 1.30 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.50 | 1.00 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.50 | 1.30 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_slalom | +1.00 | 1.00 | diff_mppi_3 | **1.00** | 1.86 | dwa_fast | **1.00** | +0.00 |
| dynamic_slalom | +1.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.84 | dwa_med | **1.00** | +0.03 |
| dynamic_slalom | +1.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | dwa_fast | **1.00** | +0.13 |
| dynamic_slalom | +1.50 | 1.30 | dwa_med | **1.00** | 1.83 | dwa_fast | **1.00** | +0.06 |
| dynamic_slalom | +2.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | diff_mppi_3 | **1.00** | +0.05 |
| dynamic_slalom | +2.00 | 1.30 | dwa_fast | **1.00** | 1.79 | diff_mppi_3_early8 | **1.00** | +0.12 |

## Per-cell comparison

### dynamic_crossing | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.65 |

### dynamic_crossing | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.65 |

### dynamic_crossing | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.65 |

### dynamic_crossing | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.65 |

### dynamic_crossing | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.88 | 1.88 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.87 | 1.87 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 0.65 |

### dynamic_crossing | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.82 | 1.82 | 0.04 |
| dwa_fine | DWA | **1.00** | 1.95 | 1.95 | 0.06 |
| dwa_med | DWA | **1.00** | 1.94 | 1.94 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.87 | 1.87 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.97 | 1.97 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.98 | 1.98 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.65 |

### dynamic_crossing | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.80 | 1.80 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.87 | 1.87 | 0.06 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.90 | 1.90 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.86 | 1.86 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.52 | 2.75 | 0.65 |

### dynamic_crossing | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.96 | 2.00 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.98 | 1.98 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.39 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 3.30 | 3.54 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.04 | 3.28 | 0.65 |

### dynamic_crossing | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.81 | 1.81 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.06 |
| dwa_med | DWA | **1.00** | 1.84 | 1.84 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.88 | 1.88 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.97 | 1.97 | 1.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.65 |

### dynamic_crossing | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.86 | 1.86 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.98 | 1.98 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.65 |

### dynamic_pincer | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.65 |

### dynamic_pincer | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.65 |

### dynamic_pincer | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.84 | 1.84 | 0.06 |
| dwa_med | DWA | **1.00** | 1.87 | 1.87 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.93 | 1.93 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.93 | 1.93 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.65 |

### dynamic_pincer | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| dwa_med | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.87 | 1.87 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.95 | 1.95 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.97 | 1.97 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.83 | 1.83 | 0.66 |

### dynamic_pincer | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.88 | 1.88 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.86 | 1.86 | 0.06 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.89 | 1.89 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.94 | 1.94 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.66 |

### dynamic_pincer | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.84 | 1.84 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.93 | 1.93 | 0.06 |
| dwa_med | DWA | **1.00** | 1.83 | 1.83 | 0.06 |
| stomp_1 | STOMP | 0.00 | 6.35 | 6.60 | 0.48 |
| stomp_2 | STOMP | 0.50 | 3.20 | 3.32 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 2.06 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 0.66 |

### dynamic_pincer | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | 0.00 | 3.30 | 3.55 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.06 |
| dwa_med | DWA | **1.00** | 1.93 | 1.93 | 0.06 |
| stomp_1 | STOMP | 0.00 | 2.90 | 3.12 | 0.48 |
| stomp_2 | STOMP | 0.00 | 2.45 | 2.64 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 2.42 | 2.61 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 9.88 | 10.13 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.22 | 3.46 | 0.66 |

### dynamic_pincer | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | 0.00 | 1.92 | 1.92 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.97 | 1.97 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 2.17 | 0.06 |
| stomp_1 | STOMP | 0.00 | 27.31 | 27.38 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.21 | 4.45 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.39 | 4.63 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 10.66 | 10.91 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 4.54 | 4.79 | 0.66 |

### dynamic_pincer | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.90 | 1.90 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.96 | 1.96 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.93 | 1.93 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.44 | 2.66 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.27 | 2.49 | 0.66 |

### dynamic_pincer | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.90 | 2.08 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.88 | 1.88 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.91 | 1.91 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.91 | 3.15 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 0.66 |

### dynamic_slalom | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.43 | 4.68 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.65 |

### dynamic_slalom | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.43 | 4.68 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.65 |

### dynamic_slalom | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.39 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.43 | 4.68 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.67 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 1.35 |

### dynamic_slalom | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.93 | 1.93 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.43 | 4.68 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.66 |

### dynamic_slalom | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.87 | 1.87 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| dwa_med | DWA | **1.00** | 1.96 | 1.96 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.43 | 4.68 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 1.86 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 0.66 |

### dynamic_slalom | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.95 | 1.95 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.92 | 1.92 | 0.06 |
| dwa_med | DWA | **1.00** | 1.87 | 1.87 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.42 | 4.67 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.65 |

### dynamic_slalom | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.94 | 1.94 | 0.05 |
| dwa_fine | DWA | **1.00** | 2.00 | 2.00 | 0.06 |
| dwa_med | DWA | **1.00** | 2.00 | 2.00 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.45 | 4.69 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.42 | 4.66 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.06 | 2.28 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 2.02 | 0.66 |

### dynamic_slalom | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.88 | 1.88 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.06 |
| dwa_med | DWA | **1.00** | 1.83 | 1.83 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.47 | 4.72 | 0.93 |
| stomp_3_smooth | STOMP | 0.00 | 4.44 | 4.69 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.56 | 2.79 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 24.91 | 25.15 | 0.66 |

### dynamic_slalom | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.93 | 1.93 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.86 | 1.86 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.41 | 4.66 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 2.01 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 1.81 | 0.66 |

### dynamic_slalom | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.79 | 1.79 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.06 |
| dwa_med | DWA | **1.00** | 1.97 | 1.97 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.04 | 11.28 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.17 | 2.39 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.91 | 2.12 | 0.66 |

