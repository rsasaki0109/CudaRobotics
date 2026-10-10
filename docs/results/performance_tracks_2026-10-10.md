# Paired GPU performance improvements (2026-10-10)

GPU: NVIDIA consumer GPU (physical model and device details omitted from public evidence).

Generated from the retained paired run. [Implementation and reproduction](../performance_tracks.md), [all case metrics](performance_tracks_2026-10-10.csv), [source/binary/dataset provenance](performance_tracks_2026-10-10.json).

| Track / method | Outcome | Mean compute ms | Mean per-run p95 ms |
|---|---|---:|---:|
| dynamic / observed_stale | 51/60 goals; 9 collisions; 0 timeouts | 0.496 | 0.533 |
| dynamic / observed_latency | 51/60 goals; 9 collisions; 0 timeouts | 0.486 | 0.520 |
| dynamic / observed_risk | 58/60 goals; 0 collisions; 2 timeouts | 0.527 | 0.555 |
| racing / blind | 0/10 clean 2-lap runs | 0.166 | 0.188 |
| racing / aware | 10/10 clean 2-lap runs | 0.173 | 0.206 |
| fleet / serial | 59/200 delivered in 180 s; 0 contacts | 0.102 | 0.124 |
| fleet / platoon | 200/200 delivered in 180 s; 0 contacts | 0.097 | 0.119 |
| ndt / full | 80/80 recoveries | 328.267 | n/a |
| ndt / cascade | 80/80 recoveries | 67.011 | n/a |

Dynamic/fleet/racing trials use fresh seeds 5000 onward after development on earlier seeds. NDT uses 40 scans with two randomized prior seeds (1/41), a different-session map, and 1,936 initial hypotheses per case. Both NDT paths recover all 80 priors; mean initialization time falls from 328.3 to 67.0 ms (4.90x). Map preparation and initial scan upload are excluded from both timings. NDT timing was rerun after verification tests to avoid GPU contention.

Fleet output increases 3.39x at the same 180-second budget against a conservative serial-reservation baseline. The improved method finishes the 200 deliveries early; throughput still uses the common budget.

The dynamic method retains two timeouts; zero collisions in this batch is not a general safety guarantee. Racing combines a known-grip model, feasible proposals and narrower sampling; it is not an isolated test of model changes. These three tracks use synthetic plants.

Focused CTest verification: fleet completion, clean friction-aware racing laps, and the existing oracle dynamic-MPPI regression all pass (3/3). An additional staged NDT case exercises the one-thread and 12-thread CPU references; the 12-thread reference selects the same winning hypothesis as the GPU.

Local raw runs and comparison artifacts: `build/performance_20261010/final/`. The four GIFs are `media/dynamic_comparison.gif`, `media/ndt_comparison.gif`, `media/fleet_comparison.gif`, and `media/racing_comparison.gif`. The renderer labels playback speed and selects dynamic demonstration seed 5002 by a declared first-improvement rule. Aggregate charts include every case. NDT GIFs show measured initial/final snapshots, not recorded optimizer iterations.

Measured sources were an uncommitted worktree; hashes are retained rather than asserting that the existing HEAD contains these changes.
