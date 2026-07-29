# CudaNav ROS 2 Jazzy CI Evidence

The `ROS2 CUDA MPPI` workflow builds the complete ROS 2 Jazzy CudaNav package
set on Ubuntu 24.04 with CUDA 12.6, loads both Nav2 plugins, validates
controller parameters, and runs the package and Python evidence-contract tests.

The final successful step creates `ros_jazzy_ci_evidence.json`. The JSON binds:

- the exact 40-character Git commit;
- GitHub repository, workflow run ID, attempt, and run URL;
- Ubuntu runner image and architecture;
- ROS 2 Jazzy and the observed `nvcc` compiler version;
- the eight CudaNav packages;
- every build, plugin-load, parameter, contract, and package-test gate.

The artifact is created only after all preceding workflow steps succeed.

## Acquire the evidence

Run the workflow on the exact clean paper or release-candidate commit:

```bash
gh workflow run ros2_cuda_mppi.yml --ref PAPER_COMMIT
gh run list --workflow ros2_cuda_mppi.yml --branch PAPER_BRANCH
gh run watch RUN_ID --exit-status
gh run download RUN_ID \
  --name ros-jazzy-ci-evidence-PAPER_COMMIT \
  --dir build/paper/ros_jazzy_ci
```

Validate the downloaded artifact independently:

```bash
python3 scripts/validate_cudanav_ros_ci.py \
  build/paper/ros_jazzy_ci/ros_jazzy_ci_evidence.json
```

Do not treat a compile log, a workflow URL, or a successful unrelated branch as
paper evidence. The downloaded JSON commit must equal the commit shared by the
closed-loop, recorded-shadow, and multi-GPU CudaNav evidence suite.

## Freeze into the paper ledger

After validation, copy the small JSON artifact under `docs/results/`, retain
its workflow run URL, record its SHA-256 in
`paper/artifacts/cudarobotics_systems.json`, and add assertions for:

- `status == "passed"`;
- `git_commit == PAPER_COMMIT`;
- `ros.distro == "jazzy"`;
- `platform.image == "ubuntu-24.04"`;
- all required checks equal `"passed"`.

The systems ledger must remain `ready: false` until the release-profile
closed-loop, real-rosbag shadow, and two-model physical GPU artifacts also
validate on that same commit and controller configuration.
