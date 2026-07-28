# CudaNav Autonomy Evidence Suite

`scripts/run_autonomy_suite.py` is the project-level entry point for CudaNav
release evidence. It orchestrates three deliberately distinct modes:

| Mode | Meaning | Child runner |
|---|---|---|
| `closed_loop` | Commands affect subsequent simulated robot state | `run_cudanav_closed_loop.py` |
| `real_rosbag_shadow` | CUDA controller is evaluated against recorded motion; not closed loop | `run_cudanav_rosbag_replay.py` |
| `multi_gpu` | The closed-loop smoke reproduces across physical GPU models | `run_cudanav_multi_gpu.py` |

The aggregate gate never relabels recorded/shadow evidence as closed-loop
success. It independently reruns every child validator and requires identical
full git commit and controller-config SHA-256 values across all three modes.

## Release run

```bash
python3 scripts/run_autonomy_suite.py \
  --output-dir build/cudanav_autonomy_release \
  --profile release \
  --bag /data/erl_prueba2 \
  --evaluation-db /data/erl_prueba2/rosbag2_0.db3 \
  --controller-config ros2_ws/src/cuda_nav_bringup/config/controller.yaml \
  --controller-command \
    "ros2 launch my_nav shadow_replay.launch.py \
     params_file:={controller_config} \
     diagnostics_csv:={diagnostics_csv}" \
  --multi-gpu-run /evidence/other_gpu/cudanav_smoke
```

The local release closed-loop directory is automatically included in the
cross-machine GPU aggregate. Repeat `--multi-gpu-run` for more imported
machines. Alternatively, use `--multi-gpu-devices 0,1` when two distinct GPU
models are installed in one host.

Interrupted suites retain attempt-numbered child directories and driver logs:

```bash
python3 scripts/run_autonomy_suite.py \
  ...same arguments... \
  --resume
```

Resume is refused if the commit, inputs, profile, controller command, or
hardware-collection plan changes. A previously valid child attempt is
revalidated and reused; an invalid attempt remains visible and a new numbered
attempt is created.

Validate the aggregate independently:

```bash
python3 scripts/validate_autonomy_suite.py \
  build/cudanav_autonomy_release
```

The release suite passes only when:

- the 10-minute closed-loop release policy passes with retained bag and video;
- the content-addressed real rosbag release policy passes and remains labelled
  `shadow_controller_with_recorded_motion`;
- the multi-GPU matrix passes with at least two physical UUIDs and two model
  names;
- all child manifests and their content hashes are valid;
- one clean full commit and one controller configuration bind every mode.

## Development smoke

A closed-loop-only development smoke is available:

```bash
python3 scripts/run_autonomy_suite.py \
  --output-dir build/cudanav_autonomy_smoke \
  --profile smoke
```

This validates orchestration but does not satisfy the release or systems-paper
gate because real-data and multi-GPU modes are absent.
