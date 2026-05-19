# Local planner cross-comparison

DWA, STOMP and Diff-MPPI variants share the same scenario, bicycle dynamics, cost components and obstacle representation in benchmark_diff_mppi. Each cell is a (scenario, dyn_speed_scale, dyn_radius_scale) tuple; success is averaged across seeds.

## Per-planner summary across all cells

| planner | family | cells | cells solved | mean succ | mean final_d | mean coll | mean ms |
|---|---|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | 30 | 30 | 1.00 | 1.92 | 0.00 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | 30 | 20 | 0.67 | 1.90 | 6.37 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | 30 | 21 | 0.70 | 1.91 | 6.23 | 0.05 |
| dwa_fast | DWA | 30 | 28 | 0.93 | 1.94 | 0.60 | 0.10 |
| dwa_fine | DWA | 30 | 30 | 1.00 | 1.92 | 0.00 | 0.12 |
| dwa_med | DWA | 30 | 30 | 1.00 | 1.91 | 0.00 | 0.11 |
| stomp_1 | STOMP | 30 | 17 | 0.57 | 6.22 | 0.00 | 0.55 |
| stomp_2 | STOMP | 30 | 17 | 0.58 | 2.90 | 0.00 | 1.03 |
| stomp_3_smooth | STOMP | 30 | 18 | 0.60 | 2.84 | 0.00 | 1.51 |
| diff_mppi_3 | Diff-MPPI | 30 | 21 | 0.70 | 2.64 | 0.00 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | 30 | 23 | 0.77 | 2.88 | 0.56 | 0.74 |

## Hard-cell focus (speed >= 1.5)

Filter cells with dyn_speed_scale >= 1.5 to capture the regime where the obstacle moves fast enough to force genuine replanning. Lower bound on success differentiates planners; mean collisions per cell exposes the paradigm gap for planners that ignore dynamic obstacles.

| planner | family | hard cells | hard cells solved | mean succ | mean final_d | mean coll |
|---|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | 12 | 12 | 1.00 | 1.89 | 0.00 |
| hybrid_astar_dyn_pp | Hybrid-A* | 12 | 2 | 0.17 | 1.91 | 15.92 |
| hybrid_astar_pp | Hybrid-A* | 12 | 3 | 0.25 | 1.91 | 15.58 |
| dwa_fast | DWA | 12 | 10 | 0.83 | 2.00 | 1.50 |
| dwa_fine | DWA | 12 | 12 | 1.00 | 1.94 | 0.00 |
| dwa_med | DWA | 12 | 12 | 1.00 | 1.91 | 0.00 |
| stomp_1 | STOMP | 12 | 6 | 0.50 | 7.76 | 0.00 |
| stomp_2 | STOMP | 12 | 6 | 0.50 | 3.04 | 0.00 |
| stomp_3_smooth | STOMP | 12 | 6 | 0.50 | 2.98 | 0.00 |
| diff_mppi_3 | Diff-MPPI | 12 | 3 | 0.25 | 3.70 | 0.00 |
| diff_mppi_3_early8 | Diff-MPPI | 12 | 5 | 0.42 | 4.38 | 1.40 |

## Best planner per cell

Best = highest success_rate (ties broken by lowest final_d).

| scenario | speed | radius | best planner | succ | final_d | runner-up | runner succ | final_d gap |
|---|---|---|---|---|---|---|---|---|
| dynamic_crossing | +0.00 | 1.00 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_crossing | +0.00 | 1.30 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_crossing | +0.50 | 1.00 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_crossing | +0.50 | 1.30 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_crossing | +1.00 | 1.00 | hybrid_astar_dyn_pp | **1.00** | 1.85 | stomp_2 | **1.00** | +0.01 |
| dynamic_crossing | +1.00 | 1.30 | dwa_fast | **1.00** | 1.82 | hybrid_astar_dwa | **1.00** | +0.03 |
| dynamic_crossing | +1.50 | 1.00 | dwa_fast | **1.00** | 1.80 | hybrid_astar_dwa | **1.00** | +0.07 |
| dynamic_crossing | +1.50 | 1.30 | hybrid_astar_dwa | **1.00** | 1.84 | dwa_fine | **1.00** | +0.05 |
| dynamic_crossing | +2.00 | 1.00 | dwa_fast | **1.00** | 1.81 | dwa_med | **1.00** | +0.02 |
| dynamic_crossing | +2.00 | 1.30 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3 | **1.00** | +0.01 |
| dynamic_pincer | +0.00 | 1.00 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_pincer | +0.00 | 1.30 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_pincer | +0.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.84 | hybrid_astar_dyn_pp | **1.00** | +0.01 |
| dynamic_pincer | +0.50 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.83 | hybrid_astar_dyn_pp | **1.00** | +0.01 |
| dynamic_pincer | +1.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.84 | dwa_fine | **1.00** | +0.02 |
| dynamic_pincer | +1.00 | 1.30 | dwa_med | **1.00** | 1.83 | hybrid_astar_dyn_pp | **1.00** | +0.00 |
| dynamic_pincer | +1.50 | 1.00 | hybrid_astar_dwa | **1.00** | 1.90 | dwa_med | **1.00** | +0.03 |
| dynamic_pincer | +1.50 | 1.30 | hybrid_astar_dwa | **1.00** | 1.97 | dwa_fine | **1.00** | +0.00 |
| dynamic_pincer | +2.00 | 1.00 | hybrid_astar_dwa | **1.00** | 1.89 | dwa_med | **1.00** | +0.00 |
| dynamic_pincer | +2.00 | 1.30 | hybrid_astar_dwa | **1.00** | 1.86 | stomp_2 | **1.00** | +0.01 |
| dynamic_slalom | +0.00 | 1.00 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.00 | 1.30 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.50 | 1.00 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.50 | 1.30 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_slalom | +1.00 | 1.00 | hybrid_astar_dwa | **1.00** | 1.85 | diff_mppi_3 | **1.00** | +0.02 |
| dynamic_slalom | +1.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.84 | dwa_med | **1.00** | +0.03 |
| dynamic_slalom | +1.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | hybrid_astar_pp | **1.00** | +0.09 |
| dynamic_slalom | +1.50 | 1.30 | dwa_med | **1.00** | 1.83 | hybrid_astar_dwa | **1.00** | +0.04 |
| dynamic_slalom | +2.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | diff_mppi_3 | **1.00** | +0.05 |
| dynamic_slalom | +2.00 | 1.30 | dwa_fast | **1.00** | 1.79 | hybrid_astar_dwa | **1.00** | +0.09 |

## Per-cell comparison

### dynamic_crossing | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| stomp_1 | STOMP | **1.00** | 1.96 | 1.96 | 0.55 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 1.01 |
| stomp_3_smooth | STOMP | **1.00** | 1.88 | 1.88 | 1.48 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.72 |

### dynamic_crossing | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.09 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.96 | 1.96 | 0.54 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 1.01 |
| stomp_3_smooth | STOMP | **1.00** | 1.88 | 1.88 | 1.49 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.72 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.74 |

### dynamic_crossing | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.96 | 1.96 | 0.55 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 1.01 |
| stomp_3_smooth | STOMP | **1.00** | 1.88 | 1.88 | 1.50 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.73 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.72 |

### dynamic_crossing | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.09 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.96 | 1.96 | 0.54 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 1.01 |
| stomp_3_smooth | STOMP | **1.00** | 1.88 | 1.88 | 1.49 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.73 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.73 |

### dynamic_crossing | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.85 | 1.85 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.09 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.90 | 1.90 | 0.54 |
| stomp_2 | STOMP | **1.00** | 1.86 | 1.86 | 1.00 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 1.47 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.72 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 0.73 |

### dynamic_crossing | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.85 | 1.85 | 0.13 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.87 | 1.87 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.82 | 1.82 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.95 | 1.95 | 0.12 |
| dwa_med | DWA | **1.00** | 1.94 | 1.94 | 0.12 |
| stomp_1 | STOMP | **1.00** | 1.88 | 1.88 | 0.54 |
| stomp_2 | STOMP | **1.00** | 1.96 | 1.96 | 1.01 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 1.47 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.98 | 1.98 | 0.72 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.72 |

### dynamic_crossing | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.87 | 1.87 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.84 | 1.84 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.80 | 1.80 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.87 | 1.87 | 0.12 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.91 | 1.91 | 0.55 |
| stomp_2 | STOMP | **1.00** | 1.96 | 1.96 | 1.03 |
| stomp_3_smooth | STOMP | **1.00** | 1.87 | 1.87 | 1.51 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.52 | 2.75 | 0.73 |

### dynamic_crossing | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.13 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.92 | 1.92 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.96 | 2.01 | 0.55 |
| stomp_2 | STOMP | **1.00** | 1.96 | 1.96 | 1.03 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.51 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 3.30 | 3.54 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.04 | 3.28 | 0.74 |

### dynamic_crossing | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.11 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.81 | 1.81 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.12 |
| dwa_med | DWA | **1.00** | 1.84 | 1.84 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.97 | 1.97 | 0.55 |
| stomp_2 | STOMP | **1.00** | 1.87 | 1.87 | 1.04 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 1.52 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.73 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.74 |

### dynamic_crossing | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.89 | 1.89 | 0.11 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.11 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.10 |
| stomp_1 | STOMP | **1.00** | 1.87 | 1.87 | 0.55 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.04 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.50 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.74 |

### dynamic_pincer | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.96 | 1.96 | 0.55 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 1.04 |
| stomp_3_smooth | STOMP | **1.00** | 1.88 | 1.88 | 1.49 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.74 |

### dynamic_pincer | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.96 | 1.96 | 0.54 |
| stomp_2 | STOMP | **1.00** | 1.90 | 1.90 | 1.02 |
| stomp_3_smooth | STOMP | **1.00** | 1.88 | 1.88 | 1.53 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.73 |

### dynamic_pincer | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.94 | 1.94 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.84 | 1.84 | 0.12 |
| dwa_med | DWA | **1.00** | 1.87 | 1.87 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.56 |
| stomp_2 | STOMP | **1.00** | 1.90 | 1.90 | 1.06 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.55 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.75 |

### dynamic_pincer | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.13 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.12 |
| dwa_med | DWA | **1.00** | 1.88 | 1.88 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.87 | 1.87 | 0.56 |
| stomp_2 | STOMP | **1.00** | 1.94 | 1.94 | 1.06 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 1.54 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.83 | 1.83 | 0.75 |

### dynamic_pincer | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 2.00 | 2.00 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.88 | 1.88 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.86 | 1.86 | 0.12 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.11 |
| stomp_1 | STOMP | **1.00** | 1.93 | 1.93 | 0.56 |
| stomp_2 | STOMP | **1.00** | 1.96 | 1.96 | 1.06 |
| stomp_3_smooth | STOMP | **1.00** | 1.97 | 1.97 | 1.55 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 0.76 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.76 |

### dynamic_pincer | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.83 | 1.83 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.84 | 1.84 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.93 | 1.93 | 0.12 |
| dwa_med | DWA | **1.00** | 1.83 | 1.83 | 0.12 |
| stomp_1 | STOMP | 0.00 | 6.31 | 6.56 | 0.57 |
| stomp_2 | STOMP | 0.50 | 2.95 | 3.06 | 1.06 |
| stomp_3_smooth | STOMP | **1.00** | 1.95 | 1.95 | 1.54 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 2.06 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 0.75 |

### dynamic_pincer | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.13 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.98 | 1.98 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.04 |
| dwa_fast | DWA | 0.00 | 3.30 | 3.55 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.12 |
| dwa_med | DWA | **1.00** | 1.93 | 1.93 | 0.11 |
| stomp_1 | STOMP | 0.00 | 2.98 | 3.21 | 0.57 |
| stomp_2 | STOMP | 0.00 | 2.43 | 2.61 | 1.06 |
| stomp_3_smooth | STOMP | 0.00 | 2.41 | 2.60 | 1.54 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 9.88 | 10.13 | 0.76 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.22 | 3.46 | 0.76 |

### dynamic_pincer | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 2.15 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.93 | 1.93 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | 0.00 | 1.92 | 1.92 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.97 | 1.97 | 0.12 |
| dwa_med | DWA | **1.00** | 1.99 | 2.17 | 0.11 |
| stomp_1 | STOMP | 0.00 | 34.54 | 34.54 | 0.56 |
| stomp_2 | STOMP | 0.00 | 4.78 | 5.02 | 1.06 |
| stomp_3_smooth | STOMP | 0.00 | 4.35 | 4.60 | 1.54 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 10.66 | 10.91 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 4.54 | 4.79 | 0.75 |

### dynamic_pincer | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.89 | 1.89 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.84 | 1.84 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.12 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 0.56 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 1.06 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.54 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.44 | 2.66 | 0.76 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.27 | 2.49 | 0.75 |

### dynamic_pincer | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.86 | 1.86 | 0.13 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.98 | 1.98 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.12 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.12 |
| stomp_1 | STOMP | **1.00** | 1.92 | 2.10 | 0.56 |
| stomp_2 | STOMP | **1.00** | 1.87 | 1.87 | 1.05 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.53 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.91 | 3.15 | 0.76 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 0.75 |

### dynamic_slalom | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.12 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.12 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.55 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.68 | 1.05 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.53 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.74 |

### dynamic_slalom | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.13 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.12 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.56 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.68 | 1.06 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.55 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.75 |

### dynamic_slalom | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.13 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.12 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.56 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.68 | 1.04 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.52 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.73 |

### dynamic_slalom | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.93 | 1.93 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.11 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.55 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.68 | 1.04 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.52 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.73 |

### dynamic_slalom | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.85 | 1.85 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.99 | 1.99 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.87 | 1.87 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.11 |
| dwa_med | DWA | **1.00** | 1.96 | 1.96 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.55 |
| stomp_2 | STOMP | 0.00 | 4.44 | 4.68 | 1.04 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.52 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 1.86 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 0.74 |

### dynamic_slalom | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.98 | 1.98 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.95 | 1.95 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.92 | 1.92 | 0.12 |
| dwa_med | DWA | **1.00** | 1.87 | 1.87 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.55 |
| stomp_2 | STOMP | 0.00 | 4.44 | 4.68 | 1.04 |
| stomp_3_smooth | STOMP | 0.00 | 4.42 | 4.66 | 1.52 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.73 |

### dynamic_slalom | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.94 | 1.94 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.94 | 1.94 | 0.04 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.94 | 1.94 | 0.11 |
| dwa_fine | DWA | **1.00** | 2.00 | 2.00 | 0.12 |
| dwa_med | DWA | **1.00** | 2.00 | 2.00 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.56 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.68 | 1.03 |
| stomp_3_smooth | STOMP | 0.00 | 4.39 | 4.63 | 1.52 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.06 | 2.28 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 2.02 | 0.75 |

### dynamic_slalom | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.86 | 1.86 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.98 | 1.98 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.90 | 1.90 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.88 | 1.88 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.12 |
| dwa_med | DWA | **1.00** | 1.83 | 1.83 | 0.12 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.55 |
| stomp_2 | STOMP | 0.00 | 4.44 | 4.69 | 1.03 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.65 | 1.54 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.56 | 2.79 | 0.74 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 24.91 | 25.15 | 0.75 |

### dynamic_slalom | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.14 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.88 | 1.88 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.90 | 1.90 | 0.04 |
| dwa_fast | DWA | **1.00** | 1.93 | 1.93 | 0.10 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.12 |
| dwa_med | DWA | **1.00** | 1.86 | 1.86 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.55 |
| stomp_2 | STOMP | 0.00 | 4.40 | 4.64 | 1.00 |
| stomp_3_smooth | STOMP | 0.00 | 4.37 | 4.61 | 1.47 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 2.01 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 1.81 | 0.75 |

### dynamic_slalom | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.87 | 1.87 | 0.12 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.95 | 1.95 | 0.05 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.90 | 1.90 | 0.05 |
| dwa_fast | DWA | **1.00** | 1.79 | 1.79 | 0.11 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.12 |
| dwa_med | DWA | **1.00** | 1.97 | 1.97 | 0.11 |
| stomp_1 | STOMP | 0.00 | 11.00 | 11.24 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.41 | 4.65 | 0.95 |
| stomp_3_smooth | STOMP | 0.00 | 4.37 | 4.62 | 1.43 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.17 | 2.39 | 0.75 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.91 | 2.12 | 0.74 |

