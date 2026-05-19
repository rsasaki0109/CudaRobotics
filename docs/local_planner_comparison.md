# Local planner cross-comparison

DWA, STOMP and Diff-MPPI variants share the same scenario, bicycle dynamics, cost components and obstacle representation in benchmark_diff_mppi. Each cell is a (scenario, dyn_speed_scale, dyn_radius_scale) tuple; success is averaged across seeds.

## Per-planner summary across all cells

| planner | family | cells | cells solved | mean succ | mean final_d | mean coll | mean ms |
|---|---|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | 30 | 30 | 1.00 | 1.92 | 0.00 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 30 | 20 | 0.67 | 1.90 | 6.37 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | 30 | 29 | 0.97 | 1.94 | 0.00 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 30 | 21 | 0.70 | 1.91 | 6.23 | 0.02 |
| dwa_fast | DWA | 30 | 28 | 0.93 | 1.94 | 0.60 | 0.06 |
| dwa_fine | DWA | 30 | 30 | 1.00 | 1.92 | 0.00 | 0.07 |
| dwa_med | DWA | 30 | 30 | 1.00 | 1.91 | 0.00 | 0.06 |
| stomp_1 | STOMP | 30 | 17 | 0.57 | 5.77 | 0.00 | 0.49 |
| stomp_2 | STOMP | 30 | 18 | 0.60 | 2.86 | 0.00 | 0.95 |
| stomp_3_smooth | STOMP | 30 | 18 | 0.60 | 2.83 | 0.00 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 30 | 21 | 0.70 | 2.64 | 0.00 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 30 | 23 | 0.77 | 2.88 | 0.56 | 0.66 |

## Hard-cell focus (speed >= 1.5)

Filter cells with dyn_speed_scale >= 1.5 to capture the regime where the obstacle moves fast enough to force genuine replanning. Lower bound on success differentiates planners; mean collisions per cell exposes the paradigm gap for planners that ignore dynamic obstacles.

| planner | family | hard cells | hard cells solved | mean succ | mean final_d | mean coll |
|---|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | 12 | 12 | 1.00 | 1.89 | 0.00 |
| hybrid_astar_dyn_pp | Hybrid-A* | 12 | 2 | 0.17 | 1.91 | 15.92 |
| hybrid_astar_mppi | Hybrid-A* | 12 | 11 | 0.92 | 1.97 | 0.00 |
| hybrid_astar_pp | Hybrid-A* | 12 | 3 | 0.25 | 1.91 | 15.58 |
| dwa_fast | DWA | 12 | 10 | 0.83 | 2.00 | 1.50 |
| dwa_fine | DWA | 12 | 12 | 1.00 | 1.94 | 0.00 |
| dwa_med | DWA | 12 | 12 | 1.00 | 1.91 | 0.00 |
| stomp_1 | STOMP | 12 | 6 | 0.50 | 6.63 | 0.00 |
| stomp_2 | STOMP | 12 | 6 | 0.50 | 3.01 | 0.00 |
| stomp_3_smooth | STOMP | 12 | 6 | 0.50 | 2.97 | 0.00 |
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
| dynamic_crossing | +1.50 | 1.30 | hybrid_astar_dwa | **1.00** | 1.84 | stomp_3_smooth | **1.00** | +0.05 |
| dynamic_crossing | +2.00 | 1.00 | dwa_fast | **1.00** | 1.81 | dwa_med | **1.00** | +0.02 |
| dynamic_crossing | +2.00 | 1.30 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3 | **1.00** | +0.01 |
| dynamic_pincer | +0.00 | 1.00 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_pincer | +0.00 | 1.30 | hybrid_astar_dyn_pp | **1.00** | 1.84 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_pincer | +0.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.84 | hybrid_astar_dyn_pp | **1.00** | +0.01 |
| dynamic_pincer | +0.50 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.83 | hybrid_astar_dyn_pp | **1.00** | +0.01 |
| dynamic_pincer | +1.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.84 | dwa_fine | **1.00** | +0.02 |
| dynamic_pincer | +1.00 | 1.30 | dwa_med | **1.00** | 1.83 | hybrid_astar_dyn_pp | **1.00** | +0.00 |
| dynamic_pincer | +1.50 | 1.00 | hybrid_astar_dwa | **1.00** | 1.90 | hybrid_astar_mppi | **1.00** | +0.03 |
| dynamic_pincer | +1.50 | 1.30 | hybrid_astar_dwa | **1.00** | 1.97 | dwa_fine | **1.00** | +0.00 |
| dynamic_pincer | +2.00 | 1.00 | hybrid_astar_dwa | **1.00** | 1.89 | dwa_med | **1.00** | +0.00 |
| dynamic_pincer | +2.00 | 1.30 | hybrid_astar_dwa | **1.00** | 1.86 | stomp_2 | **1.00** | +0.02 |
| dynamic_slalom | +0.00 | 1.00 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.00 | 1.30 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.50 | 1.00 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.00 |
| dynamic_slalom | +0.50 | 1.30 | dwa_fine | **1.00** | 1.88 | diff_mppi_3_early8 | **1.00** | +0.01 |
| dynamic_slalom | +1.00 | 1.00 | hybrid_astar_dwa | **1.00** | 1.85 | diff_mppi_3 | **1.00** | +0.02 |
| dynamic_slalom | +1.00 | 1.30 | diff_mppi_3_early8 | **1.00** | 1.84 | dwa_med | **1.00** | +0.03 |
| dynamic_slalom | +1.50 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | hybrid_astar_mppi | **1.00** | +0.08 |
| dynamic_slalom | +1.50 | 1.30 | dwa_med | **1.00** | 1.83 | hybrid_astar_dwa | **1.00** | +0.04 |
| dynamic_slalom | +2.00 | 1.00 | diff_mppi_3_early8 | **1.00** | 1.81 | diff_mppi_3 | **1.00** | +0.05 |
| dynamic_slalom | +2.00 | 1.30 | dwa_fast | **1.00** | 1.79 | hybrid_astar_dwa | **1.00** | +0.09 |

## Per-cell comparison

### dynamic_crossing | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.08 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.67 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.67 |

### dynamic_crossing | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.42 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.66 |

### dynamic_crossing | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.66 |

### dynamic_crossing | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.08 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.42 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.66 |

### dynamic_crossing | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.85 | 1.85 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.57 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.88 | 1.88 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.86 | 1.86 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.67 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 0.66 |

### dynamic_crossing | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.85 | 1.85 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.87 | 1.87 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.82 | 1.82 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.95 | 1.95 | 0.07 |
| dwa_med | DWA | **1.00** | 1.94 | 1.94 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.91 | 1.91 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.96 | 1.96 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 1.42 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.98 | 1.98 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.66 |

### dynamic_crossing | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.87 | 1.87 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.94 | 1.94 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.80 | 1.80 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.87 | 1.87 | 0.07 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.93 | 1.93 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.92 | 1.92 | 0.96 |
| stomp_3_smooth | STOMP | **1.00** | 1.87 | 1.87 | 1.42 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.52 | 2.75 | 0.66 |

### dynamic_crossing | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.08 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.96 | 1.96 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.92 | 1.92 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.90 | 1.99 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 3.30 | 3.54 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.04 | 3.28 | 0.66 |

### dynamic_crossing | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.81 | 1.81 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.07 |
| dwa_med | DWA | **1.00** | 1.84 | 1.84 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.87 | 1.87 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.66 |

### dynamic_crossing | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.89 | 1.89 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.07 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.86 | 1.86 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.97 | 1.97 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.66 |

### dynamic_pincer | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| stomp_1 | STOMP | **1.00** | 1.95 | 1.95 | 0.50 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.89 | 1.89 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.66 |

### dynamic_pincer | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.08 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.07 |
| stomp_1 | STOMP | **1.00** | 1.94 | 1.94 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.90 | 1.90 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.88 | 1.88 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.85 | 1.85 | 0.66 |

### dynamic_pincer | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.94 | 1.94 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.84 | 1.84 | 0.07 |
| dwa_med | DWA | **1.00** | 1.87 | 1.87 | 0.07 |
| stomp_1 | STOMP | **1.00** | 1.97 | 1.97 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.91 | 1.91 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.66 |

### dynamic_pincer | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.99 | 1.99 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.89 | 1.89 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| dwa_med | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.86 | 1.86 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.95 | 1.95 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.96 | 1.96 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.83 | 1.83 | 0.66 |

### dynamic_pincer | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 2.00 | 2.00 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.86 | 1.86 | 0.07 |
| dwa_med | DWA | **1.00** | 1.95 | 1.95 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.92 | 1.92 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.95 | 1.95 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.97 | 1.97 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.87 | 1.87 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.66 |

### dynamic_pincer | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.83 | 1.83 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.94 | 1.94 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.84 | 1.84 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.93 | 1.93 | 0.07 |
| dwa_med | DWA | **1.00** | 1.83 | 1.83 | 0.07 |
| stomp_1 | STOMP | 0.00 | 6.37 | 6.62 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.98 | 2.02 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.85 | 2.06 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 0.67 |

### dynamic_pincer | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.98 | 1.98 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | 0.00 | 3.30 | 3.55 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.07 |
| dwa_med | DWA | **1.00** | 1.93 | 1.93 | 0.06 |
| stomp_1 | STOMP | 0.00 | 2.89 | 3.11 | 0.49 |
| stomp_2 | STOMP | 0.00 | 2.43 | 2.61 | 0.95 |
| stomp_3_smooth | STOMP | 0.00 | 2.42 | 2.60 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 9.88 | 10.13 | 0.67 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 3.22 | 3.46 | 0.66 |

### dynamic_pincer | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.97 | 2.15 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.93 | 1.93 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | 0.00 | 2.34 | 2.50 | 0.57 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | 0.00 | 1.92 | 1.92 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.97 | 1.97 | 0.07 |
| dwa_med | DWA | **1.00** | 1.99 | 2.17 | 0.07 |
| stomp_1 | STOMP | 0.00 | 21.23 | 21.35 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.37 | 4.61 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.15 | 4.39 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 10.66 | 10.91 | 0.67 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 4.54 | 4.79 | 0.66 |

### dynamic_pincer | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.89 | 1.89 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.84 | 1.84 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| dwa_med | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.89 | 1.89 | 0.49 |
| stomp_2 | STOMP | **1.00** | 1.95 | 1.95 | 0.95 |
| stomp_3_smooth | STOMP | **1.00** | 1.92 | 1.92 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.44 | 2.66 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.27 | 2.49 | 0.66 |

### dynamic_pincer | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.86 | 1.86 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.98 | 1.98 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.97 | 1.97 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.92 | 1.92 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.91 | 1.91 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.89 | 1.89 | 0.06 |
| dwa_med | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| stomp_1 | STOMP | **1.00** | 1.89 | 2.08 | 0.48 |
| stomp_2 | STOMP | **1.00** | 1.87 | 1.87 | 0.94 |
| stomp_3_smooth | STOMP | **1.00** | 1.90 | 1.90 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.91 | 3.15 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 2.75 | 2.98 | 0.66 |

### dynamic_slalom | speed=+0.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.96 | 1.96 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.66 |

### dynamic_slalom | speed=+0.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.96 | 1.96 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.67 | 0.95 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.66 |

### dynamic_slalom | speed=+0.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.96 | 1.96 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.90 | 1.90 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.07 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.67 | 0.95 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.94 | 1.94 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.88 | 1.88 | 0.66 |

### dynamic_slalom | speed=+0.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.93 | 1.93 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.96 | 1.96 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.93 | 1.93 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.41 | 4.66 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.95 | 1.95 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.89 | 1.89 | 0.66 |

### dynamic_slalom | speed=+1.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.85 | 1.85 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.99 | 1.99 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.89 | 1.89 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.87 | 1.87 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.99 | 1.99 | 0.06 |
| dwa_med | DWA | **1.00** | 1.96 | 1.96 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.42 | 4.66 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 1.86 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.97 | 1.97 | 0.66 |

### dynamic_slalom | speed=+1.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.92 | 1.92 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | **1.00** | 1.98 | 1.98 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.95 | 1.95 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.92 | 1.92 | 0.06 |
| dwa_med | DWA | **1.00** | 1.87 | 1.87 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.43 | 4.68 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.42 | 4.67 | 1.40 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.92 | 1.92 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.84 | 1.84 | 0.66 |

### dynamic_slalom | speed=+1.50 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.94 | 1.94 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.94 | 1.94 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.89 | 1.89 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.01 |
| dwa_fast | DWA | **1.00** | 1.94 | 1.94 | 0.05 |
| dwa_fine | DWA | **1.00** | 2.00 | 2.00 | 0.06 |
| dwa_med | DWA | **1.00** | 2.00 | 2.00 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.45 | 4.70 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.40 | 4.65 | 1.40 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.06 | 2.28 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 2.02 | 0.66 |

### dynamic_slalom | speed=+1.50 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.86 | 1.86 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.98 | 1.98 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.90 | 1.90 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.90 | 1.90 | 0.01 |
| dwa_fast | DWA | **1.00** | 1.88 | 1.88 | 0.05 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.06 |
| dwa_med | DWA | **1.00** | 1.83 | 1.83 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.47 | 4.72 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.42 | 4.67 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.56 | 2.79 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | 0.00 | 24.91 | 25.15 | 0.66 |

### dynamic_slalom | speed=+2.00 radius=1.00

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.91 | 1.91 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.88 | 1.88 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.95 | 1.95 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.93 | 1.93 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.88 | 1.88 | 0.06 |
| dwa_med | DWA | **1.00** | 1.86 | 1.86 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.49 |
| stomp_2 | STOMP | 0.00 | 4.40 | 4.65 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.38 | 4.63 | 1.41 |
| diff_mppi_3 | Diff-MPPI | **1.00** | 1.86 | 2.01 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.81 | 1.81 | 0.66 |

### dynamic_slalom | speed=+2.00 radius=1.30

| planner | family | succ | final_d | min_d | ms |
|---|---|---|---|---|---|
| hybrid_astar_dwa | Hybrid-A* | **1.00** | 1.87 | 1.87 | 0.07 |
| hybrid_astar_dyn_pp | Hybrid-A* | 0.00 | 1.95 | 1.95 | 0.02 |
| hybrid_astar_mppi | Hybrid-A* | **1.00** | 1.96 | 1.96 | 0.56 |
| hybrid_astar_pp | Hybrid-A* | 0.00 | 1.90 | 1.90 | 0.02 |
| dwa_fast | DWA | **1.00** | 1.79 | 1.79 | 0.06 |
| dwa_fine | DWA | **1.00** | 1.98 | 1.98 | 0.06 |
| dwa_med | DWA | **1.00** | 1.97 | 1.97 | 0.06 |
| stomp_1 | STOMP | 0.00 | 11.01 | 11.25 | 0.48 |
| stomp_2 | STOMP | 0.00 | 4.42 | 4.67 | 0.94 |
| stomp_3_smooth | STOMP | 0.00 | 4.38 | 4.63 | 1.41 |
| diff_mppi_3 | Diff-MPPI | 0.00 | 2.17 | 2.39 | 0.66 |
| diff_mppi_3_early8 | Diff-MPPI | **1.00** | 1.91 | 2.12 | 0.66 |

