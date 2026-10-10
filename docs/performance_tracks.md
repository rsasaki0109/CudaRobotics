# Measured GPU performance improvements

Four runnable development tracks connect compute performance to visible robot
behavior. The retained [2026-10-10 results](results/performance_tracks_2026-10-10.md)
include unsuccessful cases and the scope of each comparison.

## Build, evaluate, and render

```bash
cmake --build build --config Release --target \
  gpu_esdf_mppi_3d benchmark_ndt_localization gpu_fleet_traffic gpu_mppi_racing
python scripts/run_performance_tracks.py --out-dir build/performance_tracks
python scripts/render_performance_tracks.py build/performance_tracks/results.json --videos
```

On Windows, pass `--dll-dir <OpenCV-runtime-bin>` to both Python commands if the
OpenCV DLLs are not on PATH. The runner requires a new output directory, executes
GPU measurements sequentially, and retains commands, logs, per-case metrics,
source and executable hashes, GPU model, and input dataset hashes. Use
`--bin-dir` for another build layout. `--skip-ndt` explicitly omits the external
dataset track; it does not substitute synthetic data.

The default evaluation uses dynamic seeds 5000-5059, racing seeds 5000-5009,
fleet seeds 5000-5002, and NDT prior seeds 1/41 on the same 40 scans. Dynamic
seeds 3000 and 4000 were used during development. `--seed-start` and the trial
count options expose the retained evaluation configuration.

The renderer writes an overview PNG and four paired GIFs to `media/`. Video
capture is separate from timing measurements. Dynamic video selects the first
paired improvement in ascending seed order and declares that seed; aggregate
charts include every case. Playback acceleration is labelled. NDT animation
uses real initial/final pose snapshots and measured elapsed time slowed by 10x;
it does not depict unrecorded optimizer iterations.

## Delayed-observation prediction and feasible control selection

```bash
bin/gpu_esdf_mppi_3d --movers 6 --mover-speed 3 --mode 7 \
  --observation-delay 3 --observation-period 2 --turn-period 20 --seed 5002 --headless
bin/gpu_esdf_mppi_3d --movers 6 --mover-speed 3 --observed-trials --trials 60 \
  --observation-delay 3 --turn-period 20 --seed 5000 --no-video --headless
```

Windows executables live under `bin/Release/` with the `.exe` suffix.

Modes 5-7 consume delayed, noisy position samples. Velocity comes from successive
observations, rather than the simulator's hidden velocity. The evaluation has
300 ms latency, 200 ms sample intervals, 2 cm position noise, and velocity turns
every two seconds. Modes 0-4 keep their existing oracle/static behavior.

| Mode | Method |
|---|---|
| 5 | Constant-velocity prediction, treating stale samples as current |
| 6 | Extrapolate the observed state by its age before predicting |
| 7 | Age compensation, known-boundary reflections, an innovation-dependent uncertainty envelope, and near-term feasibility selection |

After MPPI's weighted update, mode 7 checks the next 1.2 seconds. If the averaged
sequence violates the predicted clearance, a GPU reduction selects the lowest
cost feasible sampled sequence intact. If no sample passes, it requests bounded
braking. This addresses the case where averaging feasible trajectories produces
an infeasible trajectory. Timing includes the observer, transfers, planning,
selection, and fetching the command; it excludes rendering and ground-truth
validation. Collision validation samples eight points along each time step.

Limitations: synthetic position observations, stable object identities, known
motion-region bounds, spherical obstacles, no raw LiDAR association or occlusion,
and a double-integrator vehicle. There is no collision guarantee under arbitrary
unobserved motion. The retained test has two timeouts despite zero collisions.

## Staged NDT initialization

```bash
bin/benchmark_ndt_localization \
  --sequence build/datasets/mcd_ntu_day_02/loc_seq_v025.bin \
  --map-sequence build/datasets/mcd_ntu_night_13/loc_seq_s5_v025.bin \
  --prior-err 10 --grid 11 --grid-step 2 --yaws 16 --tests 40 --cpu-tests 0 \
  --screen-stride 4 --screen-iters 12 --survivors 64 --csv build/ndt_staged.csv
```

The full baseline aligns all 1,936 hypotheses with the full scan. The new path
screens all hypotheses with every fourth point for 12 iterations, retains 64,
aligns those against the full scan, and finely refines the best four. This is
point thinning and candidate screening, not a different-resolution map. GPU and
CPU paths share the same screening strategy. `--screen-stride 1` keeps the full
baseline. CSVs also retain the prior and final pose for visual comparison.

The retained comparison has 40 real scans with two randomized priors each, not
80 independent scans. A second session supplies the map. Recovery means position
error below 0.5 m and yaw error below 2 degrees. Initialization timings include
screening, candidate transfer/selection, and refinement; they exclude building
the resident map and uploading the input scan. One additional case exercised
the staged CPU reference on one and 12 threads.

Limitations: one campus sequence, one other-session map, roll/pitch from ground
truth standing in for gravity, z prior within one metre, and no deskew. This
does not establish Autoware runtime performance or real robot recovery latency.

## Compatible fleet reservations

```bash
bin/gpu_fleet_traffic --robots 200 --steps 1800 --seed 5000 --no-video
bin/gpu_fleet_traffic --robots 200 --steps 1800 --seed 5000 --platoon --no-video
```

This new controlled intersection workload compares a safe serial reservation
baseline with compatible straight-lane platoons. It is separate from
`gpu_multi_robot_planner`'s independent distance fields and repulsion demo.
Both methods use the same queues, vehicle sizes, motion update, and destination.

A GPU scheduler grants a resource to one vehicle or two compatible opposing
lanes. Platoon admission is bounded to three seconds. A reservation is never
released while an admitted vehicle remains in the conflict region; the oldest
waiting eligible vehicle determines the next grant. A parallel car-following
kernel maintains longitudinal gaps. An independent closest-approach calculation
checks swept inter-vehicle segments.

Throughput divides completed deliveries by the same 180-second budget, even
when a method completes early. `sim_s` separately records actual run duration.
`--require-completion` makes an incomplete run fail for smoke testing.

Limitations: four straight lanes, queued initial arrivals, no turns, dispatch,
unstructured obstacles, steering dynamics, or realistic braking dynamics. The
serial baseline is conservative, not a state-of-the-art fleet scheduler.

## Friction-aware racing proposals

```bash
bin/gpu_mppi_racing --grip-plant --seed 5000 --laps 2 --steps 1800 --no-video
bin/gpu_mppi_racing --grip-plant --grip-aware --seed 5000 --laps 2 --steps 1800 --no-video
```

The true vehicle uses a bounded bicycle approximation with a friction circle:
longitudinal acceleration consumes part of the lateral acceleration budget,
and steering yaw rate is limited by the remaining budget. The known surface map
has coefficient 0.9 on dry pavement and 0.3 in the blue slippery quadrant.

The improved controller combines that transition model, friction-demand and
curvature-speed costs, feasible path-following proposals, and narrower control
noise. The benchmark measures this combined controller; it does not isolate
the effect of the transition model alone. GPU softmin and weighted reductions
also replace host cost normalization and a serial per-control reduction.

Non-video runs skip drawing and trajectory downloads. CPU timing is opt-in via
`--cpu-reference` in kinematic mode; it uses independently sampled controls of
the same dimensions, not identical random controls. `--require-success` rejects
incomplete laps or any off-track centre positions. A run stops once its centre
is more than one metre beyond the track edge.

Limitations: known surface grip, no online friction estimator, tyre slip state,
load transfer, or full vehicle dynamics. This is a simulation controller
improvement, not measured real-car lap performance.

## Focused checks

```bash
ctest --test-dir build -C Release \
  -R 'gpu_fleet_traffic_platoon_smoke|gpu_mppi_racing_grip_smoke|demo_headless_gpu_esdf_mppi_3d_dynamic' \
  --output-on-failure -j 1
```

Run verification and video capture after the measurement process has completed
so GPU timings do not contend with other workloads. These are development
benchmarks and smoke tests; they do not replace the existing release contracts.
