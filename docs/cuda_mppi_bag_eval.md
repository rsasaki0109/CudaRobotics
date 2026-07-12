# CUDA MPPI Bag / Real-Data Evaluation Harness

`scripts/run_cuda_mppi_bag_eval.py` is a thin orchestration wrapper for moving
the Nav2 CUDA MPPI controller beyond synthetic maps. It does not assume a
particular robot launch file; instead, pass the launch and mission commands
used by the target platform.

The harness can:

- start a Nav2/controller command,
- play a rosbag2 dataset,
- optionally send goals or waypoints,
- record selected topics into a new rosbag2 output,
- collect `cuda_mppi_controller` diagnostics CSV,
- render the diagnostics CSV into SVG/Markdown with
  `scripts/render_cuda_mppi_diagnostics.py`.

## Typical Bag Replay

```bash
python3 scripts/run_cuda_mppi_bag_eval.py \
  --bag /data/site_run \
  --output-dir build/cuda_mppi_bag_eval/site_run \
  --ros-domain-id 101 \
  --use-sim-time \
  --controller-command 'ros2 launch my_nav bringup.launch.py params_file:=/path/to/cuda_mppi.yaml' \
  --mission-command 'python3 scripts/send_waypoints.py --frame map' \
  --duration 120
```

If the launch file can accept a diagnostics path, use the placeholder:

```bash
--controller-command 'ros2 launch my_nav bringup.launch.py diagnostics_csv:={diagnostics_csv}'
```

The command receives `{out_dir}` and `{diagnostics_csv}` placeholders before it
is started.

## Real Robot / Live Stack

For a live robot, omit `--bag` and provide `--duration`:

```bash
python3 scripts/run_cuda_mppi_bag_eval.py \
  --output-dir build/cuda_mppi_bag_eval/live_site_a \
  --ros-domain-id 42 \
  --controller-command 'ros2 launch my_robot nav2_cuda_mppi.launch.py' \
  --mission-command 'python3 scripts/send_waypoints.py --file route.yaml' \
  --duration 180
```

The runner writes:

- `manifest.json` with commands, return codes, diagnostics path, and recorded topics,
- `controller.log`, `mission.log`, `rosbag_play.log`, and `rosbag_record.log`
  when those subprocesses are used,
- `topics/` rosbag2 output unless `--no-record` is set,
- `diagnostics.svg` and `diagnostics.md` if the diagnostics CSV exists.

## Controller Config

Set these parameters in the CUDA MPPI controller configuration to collect
per-cycle diagnostics:

```yaml
FollowPath:
  diagnostics_log_period: 1.0
  diagnostics_csv_path: /tmp/cuda_mppi_diagnostics.csv
```

For repeatable experiments, prefer an output path passed by the harness via
`{diagnostics_csv}` so logs stay with the recorded topics and manifest.

## Readout

Use the diagnostics plot to inspect:

- solve-time spikes against the 20 Hz budget,
- sustained low valid-rollout ratio,
- all-colliding and retreat cycles,
- command saturation or oscillation.

Treat this as an evaluation harness, not a pass/fail benchmark by itself. The
scenario, map, localization quality, costmap layers, footprint, and waypoint
policy determine whether the run is comparable across commits.

## Public ERL-Inspired Dataset

The open [ERL-inspired navigation benchmark dataset](https://doi.org/10.5281/zenodo.10518775)
contains five ROS 2 Humble runs (6 GB compressed, CC BY 4.0). Download, verify,
extract, and inspect its Nav2 topic compatibility with:

```bash
python3 scripts/prepare_erl_navigation_bags.py --download --extract
```

The command writes `build/datasets/erl_navigation/compatibility_report.json`.
Each bag is classified as `shadow_ready`, `adapter_required`, or
`insufficient_for_nav2_replay`, and its scan, odometry, TF, command, map, and
plan topics are listed. Use the reported topic names to configure remappings
before passing a bag to `run_cuda_mppi_bag_eval.py`.

Recorded motion does not react to newly computed commands. Public bags are
therefore suitable for sensor/costmap replay and shadow-mode command analysis,
not by themselves for claims about closed-loop controller success.

### Offline inspection without ROS

The rosbag2 SQLite databases can be inventoried on a machine without ROS:

```bash
python3 scripts/analyze_rosbag_db3.py /data/erl_navigation/extracted \
  --json build/erl_offline_summary.json \
  --csv build/erl_offline_topics.csv
```

The outputs report per-bag duration and per-topic message count, observed rate,
time coverage, and serialized payload bytes. This inspects recording health and
topic availability; it does not deserialize CDR message payloads.

Twist commands and Odometry poses can also be decoded without ROS:

```bash
python3 scripts/export_rosbag_motion.py /data/erl_navigation/extracted/Prueba5/*.db3 \
  --output-dir build/prueba5_motion
```

This writes `cmd_vel.csv`, `odometry.csv`, and `motion_summary.json`, including
path length, displacement, observed speeds, command speeds, and stop ratio.

## Optional fast-math build

For latency-sensitive, smaller-batch deployments, the CUDA core can be built
with `-DCUDA_MPPI_FAST_MATH=ON`. It is disabled by default because it changes
floating-point behavior and may change an MPPI trajectory. See
[`results/mppi_fast_math_2026-07-12.md`](results/mppi_fast_math_2026-07-12.md)
for the measured speed/quality trade-off and build commands.
