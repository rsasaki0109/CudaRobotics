# MPPI Real-Rosbag Evaluation: ERL Prueba2

Date: 2026-07-28

Source: [Navigation Benchmark Rosbags Inspired by ERL Competition Test,
Prueba2](https://doi.org/10.5281/zenodo.10518775), ROS 2 Humble, CC BY 4.0.
Only the selected DB3 member was downloaded through HTTP ranges. SQLite
`PRAGMA integrity_check` returned `ok`.

This is recorded-motion evidence, not a closed-loop CUDA MPPI success claim.
The original robot trajectory does not react to newly computed controller
commands, and this run does not contain a CudaRobotics diagnostics CSV.

## Recorded Evidence

| Metric | Value |
|---|---:|
| Bag messages | 35,806 |
| Recorded duration | 195.02 s |
| Command samples | 2,367 |
| Odometry samples | 6,056 |
| LaserScan samples | 2,925 |
| Path length | 24.42 m |
| Net displacement | 3.04 m |
| Mean observed speed | 0.125 m/s |
| Mean front clearance | 1.077 m |
| Minimum front range | 0.099 m |
| Front below 0.5 m | 12.0% |
| Scan/command pairs within 200 ms | 61.4% |

## Quality-Gate Result

Overall result: **FAIL**

- PASS: positive duration
- PASS: command samples present
- PASS: odometry samples present
- FAIL: at least 90% scan/command pairing coverage
- FAIL: minimum front clearance at least 0.10 m

The 0.099 m minimum is effectively at the scanner lower bound, so it should be
treated as a saturated/lower-bound observation rather than precise collision
distance. The low pairing coverage comes from long periods without nearby
recorded commands; nearest commands outside 200 ms are deliberately excluded.

This negative result demonstrates why recorded bags need explicit evidence
labels and gates. A release-grade closed-loop result still requires a simulator
or live robot recording that includes CUDA MPPI diagnostics and reacts to its
commands.
