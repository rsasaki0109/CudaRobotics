# MuJoCo Contact-Transfer Evidence

The CUDA planners retain their nominal smooth contact model while MuJoCo executes every selected command and returns the next true state. This is closed-loop sim-to-sim transfer, not open-loop replay or real-robot evidence.

- Summary cells: 105
- Paired comparisons versus MPPI: 70
- Holm-significant positive cells: 3
- Holm-significant negative cells: 0

| Condition | Scenario | Planner | K | N | Success | Wilson 95% CI | Mean ms |
|---|---|---|---:|---:|---:|---:|---:|
| friction_0p3 | box_align_contact_arc | diff_mppi_3 | 256 | 30 | 0.900 | [0.744, 0.965] | 2.425 |
| friction_0p3 | box_align_contact_arc | mppi | 256 | 30 | 0.467 | [0.302, 0.639] | 0.101 |
| friction_0p3 | box_align_contact_arc | soppi_fast | 256 | 30 | 0.667 | [0.488, 0.808] | 1.333 |
| friction_0p3 | box_align_contact_loss | diff_mppi_3 | 256 | 30 | 0.800 | [0.627, 0.905] | 2.431 |
| friction_0p3 | box_align_contact_loss | mppi | 256 | 30 | 0.567 | [0.392, 0.726] | 0.104 |
| friction_0p3 | box_align_contact_loss | soppi_fast | 256 | 30 | 0.533 | [0.361, 0.698] | 1.338 |
| friction_0p3 | box_align_detour | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 3.322 |
| friction_0p3 | box_align_detour | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.148 |
| friction_0p3 | box_align_detour | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.799 |
| friction_0p3 | box_align_strict | diff_mppi_3 | 256 | 30 | 0.933 | [0.787, 0.982] | 2.451 |
| friction_0p3 | box_align_strict | mppi | 256 | 30 | 0.433 | [0.274, 0.608] | 0.097 |
| friction_0p3 | box_align_strict | soppi_fast | 256 | 30 | 0.633 | [0.455, 0.781] | 1.312 |
| friction_0p3 | box_swivel | diff_mppi_3 | 256 | 30 | 0.400 | [0.246, 0.577] | 2.428 |
| friction_0p3 | box_swivel | mppi | 256 | 30 | 0.200 | [0.095, 0.373] | 0.096 |
| friction_0p3 | box_swivel | soppi_fast | 256 | 30 | 0.233 | [0.118, 0.409] | 1.305 |
| friction_0p9 | box_align_contact_arc | diff_mppi_3 | 256 | 30 | 0.633 | [0.455, 0.781] | 2.437 |
| friction_0p9 | box_align_contact_arc | mppi | 256 | 30 | 0.567 | [0.392, 0.726] | 0.107 |
| friction_0p9 | box_align_contact_arc | soppi_fast | 256 | 30 | 0.633 | [0.455, 0.781] | 1.335 |
| friction_0p9 | box_align_contact_loss | diff_mppi_3 | 256 | 30 | 0.633 | [0.455, 0.781] | 2.441 |
| friction_0p9 | box_align_contact_loss | mppi | 256 | 30 | 0.267 | [0.142, 0.444] | 0.102 |
| friction_0p9 | box_align_contact_loss | soppi_fast | 256 | 30 | 0.367 | [0.219, 0.545] | 1.340 |
| friction_0p9 | box_align_detour | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 3.348 |
| friction_0p9 | box_align_detour | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.164 |
| friction_0p9 | box_align_detour | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.914 |
| friction_0p9 | box_align_strict | diff_mppi_3 | 256 | 30 | 0.700 | [0.521, 0.833] | 2.449 |
| friction_0p9 | box_align_strict | mppi | 256 | 30 | 0.467 | [0.302, 0.639] | 0.097 |
| friction_0p9 | box_align_strict | soppi_fast | 256 | 30 | 0.633 | [0.455, 0.781] | 1.312 |
| friction_0p9 | box_swivel | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 2.429 |
| friction_0p9 | box_swivel | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.096 |
| friction_0p9 | box_swivel | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.305 |
| mass_0p75 | box_align_contact_arc | diff_mppi_3 | 256 | 30 | 0.700 | [0.521, 0.833] | 2.434 |
| mass_0p75 | box_align_contact_arc | mppi | 256 | 30 | 0.467 | [0.302, 0.639] | 0.110 |
| mass_0p75 | box_align_contact_arc | soppi_fast | 256 | 30 | 0.767 | [0.591, 0.882] | 1.335 |
| mass_0p75 | box_align_contact_loss | diff_mppi_3 | 256 | 30 | 0.667 | [0.488, 0.808] | 2.438 |
| mass_0p75 | box_align_contact_loss | mppi | 256 | 30 | 0.567 | [0.392, 0.726] | 0.106 |
| mass_0p75 | box_align_contact_loss | soppi_fast | 256 | 30 | 0.533 | [0.361, 0.698] | 1.334 |
| mass_0p75 | box_align_detour | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 3.331 |
| mass_0p75 | box_align_detour | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.159 |
| mass_0p75 | box_align_detour | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.864 |
| mass_0p75 | box_align_strict | diff_mppi_3 | 256 | 30 | 0.867 | [0.703, 0.947] | 2.469 |
| mass_0p75 | box_align_strict | mppi | 256 | 30 | 0.367 | [0.219, 0.545] | 0.110 |
| mass_0p75 | box_align_strict | soppi_fast | 256 | 30 | 0.433 | [0.274, 0.608] | 1.326 |
| mass_0p75 | box_swivel | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 2.443 |
| mass_0p75 | box_swivel | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.111 |
| mass_0p75 | box_swivel | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.323 |
| mass_1p25 | box_align_contact_arc | diff_mppi_3 | 256 | 30 | 0.833 | [0.664, 0.927] | 2.466 |
| mass_1p25 | box_align_contact_arc | mppi | 256 | 30 | 0.433 | [0.274, 0.608] | 0.134 |
| mass_1p25 | box_align_contact_arc | soppi_fast | 256 | 30 | 0.633 | [0.455, 0.781] | 1.383 |
| mass_1p25 | box_align_contact_loss | diff_mppi_3 | 256 | 30 | 0.833 | [0.664, 0.927] | 2.463 |
| mass_1p25 | box_align_contact_loss | mppi | 256 | 30 | 0.467 | [0.302, 0.639] | 0.130 |
| mass_1p25 | box_align_contact_loss | soppi_fast | 256 | 30 | 0.567 | [0.392, 0.726] | 1.384 |
| mass_1p25 | box_align_detour | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 3.373 |
| mass_1p25 | box_align_detour | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.200 |
| mass_1p25 | box_align_detour | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.902 |
| mass_1p25 | box_align_strict | diff_mppi_3 | 256 | 30 | 0.800 | [0.627, 0.905] | 2.470 |
| mass_1p25 | box_align_strict | mppi | 256 | 30 | 0.367 | [0.219, 0.545] | 0.111 |
| mass_1p25 | box_align_strict | soppi_fast | 256 | 30 | 0.467 | [0.302, 0.639] | 1.320 |
| mass_1p25 | box_swivel | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 2.428 |
| mass_1p25 | box_swivel | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.099 |
| mass_1p25 | box_swivel | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.302 |
| nominal | box_align_contact_arc | diff_mppi_3 | 256 | 30 | 0.600 | [0.423, 0.754] | 2.415 |
| nominal | box_align_contact_arc | mppi | 256 | 30 | 0.500 | [0.332, 0.668] | 0.109 |
| nominal | box_align_contact_arc | soppi_fast | 256 | 30 | 0.667 | [0.488, 0.808] | 1.326 |
| nominal | box_align_contact_loss | diff_mppi_3 | 256 | 30 | 0.667 | [0.488, 0.808] | 2.426 |
| nominal | box_align_contact_loss | mppi | 256 | 30 | 0.400 | [0.246, 0.577] | 0.107 |
| nominal | box_align_contact_loss | soppi_fast | 256 | 30 | 0.500 | [0.332, 0.668] | 1.325 |
| nominal | box_align_detour | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 3.312 |
| nominal | box_align_detour | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.159 |
| nominal | box_align_detour | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.845 |
| nominal | box_align_strict | diff_mppi_3 | 256 | 30 | 0.733 | [0.556, 0.858] | 2.433 |
| nominal | box_align_strict | mppi | 256 | 30 | 0.433 | [0.274, 0.608] | 0.106 |
| nominal | box_align_strict | soppi_fast | 256 | 30 | 0.400 | [0.246, 0.577] | 1.301 |
| nominal | box_swivel | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 2.411 |
| nominal | box_swivel | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.096 |
| nominal | box_swivel | soppi_fast | 256 | 30 | 0.033 | [0.006, 0.167] | 1.299 |
| sensor_noise_high | box_align_contact_arc | diff_mppi_3 | 256 | 30 | 0.733 | [0.556, 0.858] | 2.423 |
| sensor_noise_high | box_align_contact_arc | mppi | 256 | 30 | 0.433 | [0.274, 0.608] | 0.103 |
| sensor_noise_high | box_align_contact_arc | soppi_fast | 256 | 30 | 0.667 | [0.488, 0.808] | 1.333 |
| sensor_noise_high | box_align_contact_loss | diff_mppi_3 | 256 | 30 | 0.667 | [0.488, 0.808] | 2.429 |
| sensor_noise_high | box_align_contact_loss | mppi | 256 | 30 | 0.433 | [0.274, 0.608] | 0.107 |
| sensor_noise_high | box_align_contact_loss | soppi_fast | 256 | 30 | 0.367 | [0.219, 0.545] | 1.335 |
| sensor_noise_high | box_align_detour | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 3.337 |
| sensor_noise_high | box_align_detour | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.156 |
| sensor_noise_high | box_align_detour | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.853 |
| sensor_noise_high | box_align_strict | diff_mppi_3 | 256 | 30 | 0.733 | [0.556, 0.858] | 2.410 |
| sensor_noise_high | box_align_strict | mppi | 256 | 30 | 0.600 | [0.423, 0.754] | 0.101 |
| sensor_noise_high | box_align_strict | soppi_fast | 256 | 30 | 0.467 | [0.302, 0.639] | 1.305 |
| sensor_noise_high | box_swivel | diff_mppi_3 | 256 | 30 | 0.067 | [0.018, 0.213] | 2.414 |
| sensor_noise_high | box_swivel | mppi | 256 | 30 | 0.033 | [0.006, 0.167] | 0.106 |
| sensor_noise_high | box_swivel | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.300 |
| sensor_noise_nominal | box_align_contact_arc | diff_mppi_3 | 256 | 30 | 0.633 | [0.455, 0.781] | 2.427 |
| sensor_noise_nominal | box_align_contact_arc | mppi | 256 | 30 | 0.633 | [0.455, 0.781] | 0.103 |
| sensor_noise_nominal | box_align_contact_arc | soppi_fast | 256 | 30 | 0.600 | [0.423, 0.754] | 1.332 |
| sensor_noise_nominal | box_align_contact_loss | diff_mppi_3 | 256 | 30 | 0.700 | [0.521, 0.833] | 2.428 |
| sensor_noise_nominal | box_align_contact_loss | mppi | 256 | 30 | 0.467 | [0.302, 0.639] | 0.107 |
| sensor_noise_nominal | box_align_contact_loss | soppi_fast | 256 | 30 | 0.533 | [0.361, 0.698] | 1.338 |
| sensor_noise_nominal | box_align_detour | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 3.336 |
| sensor_noise_nominal | box_align_detour | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.157 |
| sensor_noise_nominal | box_align_detour | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.855 |
| sensor_noise_nominal | box_align_strict | diff_mppi_3 | 256 | 30 | 0.767 | [0.591, 0.882] | 2.407 |
| sensor_noise_nominal | box_align_strict | mppi | 256 | 30 | 0.533 | [0.361, 0.698] | 0.098 |
| sensor_noise_nominal | box_align_strict | soppi_fast | 256 | 30 | 0.567 | [0.392, 0.726] | 1.301 |
| sensor_noise_nominal | box_swivel | diff_mppi_3 | 256 | 30 | 0.000 | [0.000, 0.114] | 2.409 |
| sensor_noise_nominal | box_swivel | mppi | 256 | 30 | 0.000 | [0.000, 0.114] | 0.098 |
| sensor_noise_nominal | box_swivel | soppi_fast | 256 | 30 | 0.000 | [0.000, 0.114] | 1.293 |

Paired bootstrap intervals, exact McNemar p-values, and Holm-adjusted p-values are retained in `comparisons.csv`. Every failed and negative cell remains in the episode and summary tables.
