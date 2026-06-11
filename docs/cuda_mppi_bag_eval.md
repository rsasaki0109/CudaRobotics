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
