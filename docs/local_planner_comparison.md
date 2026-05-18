# Local planner cross-comparison

DWA, STOMP and Diff-MPPI variants share the same scenario, bicycle dynamics, cost components and obstacle representation in benchmark_diff_mppi. Each cell is a (scenario, dyn_speed_scale, dyn_radius_scale) tuple; success is averaged across seeds.

## Per-planner summary across all cells

| planner | family | cells | cells solved | mean succ | mean final_d | mean ms |
|---|---|---|---|---|---|---|
| dwa_fast | DWA | 30 | 28 | 0.93 | 2.72 | 0.29 |
| dwa_fine | DWA | 30 | 29 | 0.97 | 1.96 | 0.32 |
| dwa_med | DWA | 30 | 28 | 0.93 | 1.97 | 0.29 |
| stomp_1 | STOMP | 30 | 17 | 0.57 | 2.88 | 1.19 |
| stomp_2 | STOMP | 30 | 16 | 0.53 | 2.99 | 1.71 |
| stomp_3_smooth | STOMP | 30 | 16 | 0.55 | 6.12 | 2.16 |
| diff_mppi_3 | Diff-MPPI | 30 | 21 | 0.70 | 2.64 | 1.41 |
| diff_mppi_3_early8 | Diff-MPPI | 30 | 23 | 0.77 | 2.88 | 1.42 |

## Hard-cell focus (speed >= 1.5)

Filter cells with dyn_speed_scale >= 1.5 to capture the regime where the obstacle moves fast enough to force genuine replanning. Lower bound on success differentiates planners.

| planner | family | hard cells | hard cells solved | mean succ | mean final_d |
|---|---|---|---|---|---|
| dwa_fast | DWA | 12 | 10 | 0.83 | 3.94 |
| dwa_fine | DWA | 12 | 11 | 0.92 | 1.99 |
| dwa_med | DWA | 12 | 10 | 0.83 | 2.05 |
| stomp_1 | STOMP | 12 | 6 | 0.50 | 2.93 |
| stomp_2 | STOMP | 12 | 5 | 0.42 | 3.04 |
| stomp_3_smooth | STOMP | 12 | 5 | 0.46 | 4.62 |
| diff_mppi_3 | Diff-MPPI | 12 | 3 | 0.25 | 3.70 |
| diff_mppi_3_early8 | Diff-MPPI | 12 | 5 | 0.42 | 4.38 |

## Best planner per cell

Best = highest success_rate (ties broken by lowest final_d).

| scenario | speed | radius | best planner | succ | final_d | runner-up | runner succ | final_d gap |
|---|---|---|---|---|---|---|---|---|
| dynamic_crossing | +0.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fast | **1.00** | +0.04 |
| dynamic_crossing | +0.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fast | **1.00** | +0.04 |
| dynamic_crossing | +0.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fast | **1.00** | +0.04 |
| dynamic_crossing | +0.50 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fast | **1.00** | +0.04 |
| dynamic_crossing | +1.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.87 | dwa_fast | **1.00** | +0.03 |
| dynamic_crossing | +1.00 | 1.30 | stomp_3_smooth | **1.00** | 1.87 | diff_mppi_3_early8 | **1.00** | +0.02 |
| dynamic_crossing | +1.50 | 1.00 | dwa_fine | **1.00** | 1.86 | dwa_med | **1.00** | +0.02 |
| dynamic_crossing | +1.50 | 1.30 | stomp_2 | **1.00** | 1.90 | dwa_fine | **1.00** | +0.00 |
| dynamic_crossing | +2.00 | 1.00 | dwa_fast | **1.00** | 1.82 | dwa_fine | **1.00** | +0.03 |
| dynamic_crossing | +2.00 | 1.30 | diff_mppi_3 | **1.00** | 1.85 | dwa_med | **1.00** | +0.00 |
| dynamic_pincer | +0.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.85 | dwa_fast | **1.00** | +0.04 |
| dynamic_pincer | +0.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.85 | stomp_3_smooth | **1.00** | +0.02 |
| dynamic_pincer | +0.50 | 1.00 | dwa_fast | **1.00** | 1.79 | diff_mppi_3_early8 | **1.00** | +0.05 |
| dynamic_pincer | +0.50 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.83 | stomp_1 | **1.00** | +0.05 |
| dynamic_pincer | +1.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.84 | diff_mppi_3 | **1.00** | +0.03 |
| dynamic_pincer | +1.00 | 1.30 | diff_mppi_3 | **1.00** | 1.85 | dwa_fast | **1.00** | +0.01 |
| dynamic_pincer | +1.50 | 1.00 | dwa_fine | **1.00** | 1.95 | dwa_med | 0.00 | +0.27 |
| dynamic_pincer | +1.50 | 1.30 | dwa_fine | 0.00 | 2.76 | dwa_med | 0.00 | +0.50 |
| dynamic_pincer | +2.00 | 1.00 | stomp_2 | **1.00** | 1.87 | dwa_fast | **1.00** | +0.05 |
| dynamic_pincer | +2.00 | 1.30 | stomp_2 | **1.00** | 1.86 | dwa_fine | **1.00** | +0.01 |
| dynamic_slalom | +0.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.88 | dwa_fast | **1.00** | +0.04 |
| dynamic_slalom | +0.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.88 | dwa_fast | **1.00** | +0.04 |
| dynamic_slalom | +0.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.88 | dwa_fast | **1.00** | +0.04 |
| dynamic_slalom | +0.50 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.89 | diff_mppi_3 | **1.00** | +0.06 |
| dynamic_slalom | +1.00 | 1.00 | dwa_med | **1.00** | 1.85 | diff_mppi_3 | **1.00** | +0.02 |
| dynamic_slalom | +1.00 | 1.30 | dwa_fast | **1.00** | 1.82 | diff_mppi_3_early8 | **1.00** | +0.02 |
| dynamic_slalom | +1.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | dwa_fast | **1.00** | +0.01 |
| dynamic_slalom | +1.50 | 1.30 | dwa_fast | **1.00** | 1.83 | dwa_fine | **1.00** | +0.13 |
| dynamic_slalom | +2.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | dwa_med | **1.00** | +0.03 |
| dynamic_slalom | +2.00 | 1.30 | dwa_med | **1.00** | 1.88 | dwa_fast | **1.00** | +0.02 |

## Per-cell comparison

### dynamic_crossing | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.33 |
| dwa_fine | DWA | **1.00** | 1.91 | 1.91 | 0.35 |
| dwa_med | DWA | **1.00** | 1.91 | 1.91 | 0.26 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 1.28 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.92 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 2.38 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.58 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 1.62 |

### dynamic_crossing | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.29 |
| dwa_fine | DWA | **1.00** | 1.91 | 1.91 | 0.30 |
| dwa_med | DWA | **1.00** | 1.91 | 1.91 | 0.23 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 1.44 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 2.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.59 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 1.66 |

### dynamic_crossing | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.16 |
| dwa_fine | DWA | **1.00** | 1.91 | 1.91 | 0.20 |
| dwa_med | DWA | **1.00** | 1.91 | 1.91 | 0.25 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 1.21 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.87 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 2.34 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.53 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 1.67 |

### dynamic_crossing | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.21 |
| dwa_fine | DWA | **1.00** | 1.91 | 1.91 | 0.21 |
| dwa_med | DWA | **1.00** | 1.91 | 1.91 | 0.27 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 1.28 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.86 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 2.35 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.50 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 1.55 |

### dynamic_crossing | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.19 |
| dwa_fine | DWA | **1.00** | 1.92 | 1.92 | 0.26 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.26 |
| stomp_1 | STOMP | **1.00** | 1.90 | 1.90 | 1.23 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.87 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 2.38 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.53 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 1.53 |

### dynamic_crossing | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.99 | 1.99 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.94 | 1.94 | 0.06 |
| dwa_med | DWA | **1.00** | 1.94 | 1.94 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.93 | 1.93 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 0.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.87 | 1.87 | 1.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.98 | 1.98 | 0.69 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.65 |

### dynamic_crossing | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.99 | 1.99 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.86 | 1.86 | 0.06 |
| dwa_med | DWA | **1.00** | 1.88 | 1.88 | 0.05 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 1.36 |
| stomp_3_smooth | STOMP | **1.00** | 1.95 | 1.95 | 2.02 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.52 | 2.75 | 0.69 |

### dynamic_crossing | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.97 | 1.97 | 0.13 |
| dwa_fine | DWA | **1.00** | 1.90 | 1.90 | 0.10 |
| dwa_med | DWA | **1.00** | 1.97 | 1.97 | 0.14 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.91 |
| stomp_2 | STOMP | **1.00** | 1.90 | 1.90 | 1.74 |
| stomp_3_smooth | STOMP | **1.00** | 1.93 | 1.93 | 2.38 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 3.30 | 3.54 | 1.12 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.04 | 3.28 | 1.11 |

### dynamic_crossing | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.82 | 1.82 | 0.28 |
| dwa_fine | DWA | **1.00** | 1.85 | 1.85 | 0.41 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.29 |
| stomp_1 | STOMP | **1.00** | 1.90 | 1.90 | 1.41 |
| stomp_2 | STOMP | **1.00** | 1.87 | 1.87 | 1.93 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 2.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 1.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 1.65 |

### dynamic_crossing | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.99 | 1.99 | 0.42 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.32 |
| dwa_med | DWA | **1.00** | 1.86 | 1.86 | 0.31 |
| stomp_1 | STOMP | **1.00** | 1.88 | 2.03 | 1.45 |
| stomp_2 | STOMP | 0.00 | 2.03 | 2.18 | 1.94 |
| stomp_3_smooth | STOMP | 0.50 | 2.00 | 2.15 | 1.95 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 1.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 1.66 |

### dynamic_pincer | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.74 |
| dwa_fine | DWA | **1.00** | 1.91 | 1.91 | 0.82 |
| dwa_med | DWA | **1.00** | 1.91 | 1.91 | 0.68 |
| stomp_1 | STOMP | **1.00** | 1.91 | 1.91 | 1.44 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 2.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 1.66 |

### dynamic_pincer | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.61 |
| dwa_fine | DWA | **1.00** | 1.91 | 1.91 | 0.26 |
| dwa_med | DWA | **1.00** | 1.91 | 1.91 | 0.23 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 1.44 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 1.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.87 | 1.87 | 2.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 1.66 |

### dynamic_pincer | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.79 | 1.79 | 0.22 |
| dwa_fine | DWA | **1.00** | 1.96 | 1.96 | 0.34 |
| dwa_med | DWA | **1.00** | 1.98 | 1.98 | 0.31 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 1.29 |
| stomp_2 | STOMP | **1.00** | 1.98 | 1.98 | 1.85 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 2.35 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 1.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 1.56 |

### dynamic_pincer | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.95 | 1.95 | 0.29 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.31 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.36 |
| stomp_1 | STOMP | **1.00** | 1.88 | 1.88 | 1.35 |
| stomp_2 | STOMP | **1.00** | 1.95 | 1.95 | 1.87 |
| stomp_3_smooth | STOMP | **1.00** | 1.91 | 1.91 | 2.36 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.55 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.83 | 1.83 | 1.47 |

### dynamic_pincer | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.32 |
| dwa_fine | DWA | **1.00** | 1.94 | 1.94 | 0.37 |
| dwa_med | DWA | **1.00** | 1.92 | 1.92 | 0.35 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 1.37 |
| stomp_2 | STOMP | **1.00** | 1.95 | 1.99 | 1.87 |
| stomp_3_smooth | STOMP | **1.00** | 1.94 | 1.94 | 1.38 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 1.57 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 1.57 |

### dynamic_pincer | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.86 | 1.86 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_med | DWA | **1.00** | 1.87 | 1.87 | 0.06 |
| stomp_1 | STOMP | 0.00 | 3.50 | 3.72 | 0.48 |
| stomp_2 | STOMP | 0.00 | 3.82 | 4.04 | 0.93 |
| stomp_3_smooth | STOMP | 0.00 | 2.38 | 2.55 | 1.44 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 2.06 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 0.66 |

### dynamic_pincer | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | 0.00 | 3.45 | 3.70 | 0.17 |
| dwa_fine | DWA | **1.00** | 1.95 | 1.95 | 0.14 |
| dwa_med | DWA | 0.00 | 2.22 | 2.42 | 0.22 |
| stomp_1 | STOMP | 0.00 | 2.51 | 2.68 | 0.98 |
| stomp_2 | STOMP | 0.00 | 3.36 | 3.57 | 1.53 |
| stomp_3_smooth | STOMP | 0.00 | 2.87 | 3.06 | 2.00 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 9.88 | 10.13 | 1.03 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.22 | 3.46 | 1.16 |

### dynamic_pincer | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | 0.00 | 24.74 | 24.99 | 0.54 |
| dwa_fine | DWA | 0.00 | 2.76 | 2.95 | 0.45 |
| dwa_med | DWA | 0.00 | 3.26 | 3.48 | 0.54 |
| stomp_1 | STOMP | 0.00 | 3.38 | 3.59 | 1.46 |
| stomp_2 | STOMP | 0.00 | 3.84 | 4.06 | 1.92 |
| stomp_3_smooth | STOMP | 0.00 | 19.40 | 19.51 | 2.37 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 10.66 | 10.91 | 1.64 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 4.54 | 4.79 | 1.64 |

### dynamic_pincer | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.92 | 1.92 | 0.39 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.57 |
| dwa_med | DWA | **1.00** | 1.98 | 1.98 | 0.58 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 1.46 |
| stomp_2 | STOMP | **1.00** | 1.87 | 1.87 | 1.92 |
| stomp_3_smooth | STOMP | **1.00** | 1.98 | 1.98 | 2.36 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.44 | 2.66 | 1.64 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.27 | 2.49 | 1.64 |

### dynamic_pincer | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.93 | 1.93 | 0.42 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.75 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.34 |
| stomp_1 | STOMP | **1.00** | 1.91 | 1.91 | 1.47 |
| stomp_2 | STOMP | **1.00** | 1.86 | 1.86 | 1.91 |
| stomp_3_smooth | STOMP | **1.00** | 1.91 | 1.91 | 2.36 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.91 | 3.15 | 1.65 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 1.64 |

### dynamic_slalom | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.92 | 1.92 | 0.09 |
| dwa_fine | DWA | **1.00** | 1.97 | 1.97 | 0.12 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.14 |
| stomp_1 | STOMP | 0.00 | 4.47 | 4.71 | 1.03 |
| stomp_2 | STOMP | 0.00 | 4.67 | 4.91 | 1.75 |
| stomp_3_smooth | STOMP | 0.00 | 17.99 | 17.99 | 2.33 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 1.16 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 1.24 |

### dynamic_slalom | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.92 | 1.92 | 0.24 |
| dwa_fine | DWA | **1.00** | 1.97 | 1.97 | 0.35 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.41 |
| stomp_1 | STOMP | 0.00 | 4.47 | 4.71 | 1.40 |
| stomp_2 | STOMP | 0.00 | 4.67 | 4.91 | 1.94 |
| stomp_3_smooth | STOMP | 0.00 | 17.99 | 17.99 | 2.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 1.62 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 1.63 |

### dynamic_slalom | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.92 | 1.92 | 0.32 |
| dwa_fine | DWA | **1.00** | 1.97 | 1.97 | 0.62 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.57 |
| stomp_1 | STOMP | 0.00 | 4.47 | 4.71 | 1.43 |
| stomp_2 | STOMP | 0.00 | 4.67 | 4.91 | 1.91 |
| stomp_3_smooth | STOMP | 0.00 | 17.99 | 17.99 | 2.27 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 1.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 1.66 |

### dynamic_slalom | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.98 | 1.98 | 0.12 |
| dwa_fine | DWA | **1.00** | 1.97 | 1.97 | 0.20 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.12 |
| stomp_1 | STOMP | 0.00 | 4.47 | 4.71 | 0.84 |
| stomp_2 | STOMP | 0.00 | 4.67 | 4.91 | 1.46 |
| stomp_3_smooth | STOMP | 0.00 | 17.99 | 17.99 | 2.05 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 1.47 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 1.12 |

### dynamic_slalom | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.97 | 1.97 | 0.21 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.25 |
| dwa_med | DWA | **1.00** | 1.85 | 1.85 | 0.25 |
| stomp_1 | STOMP | 0.00 | 4.47 | 4.71 | 1.25 |
| stomp_2 | STOMP | 0.00 | 4.64 | 4.88 | 1.86 |
| stomp_3_smooth | STOMP | 0.00 | 18.01 | 18.01 | 2.35 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 1.86 | 1.33 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 1.55 |

### dynamic_slalom | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.82 | 1.82 | 0.29 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.28 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.32 |
| stomp_1 | STOMP | 0.00 | 4.44 | 4.68 | 1.42 |
| stomp_2 | STOMP | 0.00 | 4.55 | 4.79 | 0.93 |
| stomp_3_smooth | STOMP | 0.00 | 14.82 | 14.88 | 1.39 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 1.57 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 1.58 |

### dynamic_slalom | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.83 | 1.83 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.94 | 1.94 | 0.06 |
| stomp_1 | STOMP | 0.00 | 4.47 | 4.71 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.47 | 4.71 | 0.97 |
| stomp_3_smooth | STOMP | 0.00 | 4.52 | 4.75 | 1.67 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.06 | 2.28 | 0.65 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 2.02 | 0.65 |

### dynamic_slalom | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.83 | 1.83 | 0.19 |
| dwa_fine | DWA | **1.00** | 1.96 | 1.96 | 0.15 |
| dwa_med | DWA | **1.00** | 1.96 | 1.96 | 0.15 |
| stomp_1 | STOMP | 0.00 | 4.48 | 4.72 | 0.98 |
| stomp_2 | STOMP | 0.00 | 4.48 | 4.72 | 1.51 |
| stomp_3_smooth | STOMP | 0.00 | 4.52 | 4.76 | 2.23 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.56 | 2.79 | 1.23 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 24.91 | 25.15 | 1.24 |

### dynamic_slalom | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.36 |
| dwa_fine | DWA | **1.00** | 1.84 | 1.84 | 0.57 |
| dwa_med | DWA | **1.00** | 1.83 | 1.83 | 0.56 |
| stomp_1 | STOMP | 0.00 | 4.39 | 4.63 | 1.47 |
| stomp_2 | STOMP | 0.00 | 4.45 | 4.69 | 1.96 |
| stomp_3_smooth | STOMP | 0.00 | 7.90 | 8.08 | 2.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 2.01 | 1.67 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 1.81 | 1.68 |

### dynamic_slalom | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.83 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.65 |
| dwa_med | DWA | **1.00** | 1.88 | 1.88 | 0.47 |
| stomp_1 | STOMP | 0.00 | 4.39 | 4.63 | 1.44 |
| stomp_2 | STOMP | 0.00 | 4.38 | 4.62 | 1.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.48 | 4.71 | 2.39 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.17 | 2.39 | 1.68 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.91 | 2.12 | 1.67 |

