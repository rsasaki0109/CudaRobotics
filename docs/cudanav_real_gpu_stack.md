# Real-data all-GPU CudaNav core shadow

This experiment sends one content-addressed real PointCloud2 sequence through
all four reusable CudaNav GPU cores in one native process:

```text
PointCloud2 sequence
  -> GPU KISS-ICP pose
  -> rolling GPU voxel map
  -> GPU ESDF and inflated costmap
  -> CUDA MPPI command evaluation
```

The recorded GNSS poses provide the odometry reference and MPPI path. CUDA
MPPI commands are evaluated against the live map but are not applied to the
recorded vehicle or a simulator. This is therefore real-data all-GPU shadow
evidence, not ROS 2 runtime or closed-loop evidence.

## Run

Build the reusable cores and native runner:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release \
  --target cudanav_real_gpu_stack_sequence
```

From a clean commit:

```bash
python3 scripts/run_cudanav_real_gpu_stack.py \
  --database build/datasets/cudanav_localization_smoke/rosbag2_2024_09_12-14_59_58_0.db3 \
  --output-dir build/cudanav_real_gpu_stack_smoke \
  --profile smoke
```

The smoke profile uses 300 frames over 30 seconds. It binds the database,
exported sequence, runner binary, stage report, trajectory, log, source commit,
and commands by SHA-256 and byte count.

Validate a local result:

```bash
python3 scripts/run_cudanav_real_gpu_stack.py \
  --validate build/cudanav_real_gpu_stack_smoke/manifest.json \
  --commit SOURCE_COMMIT
```

Publish and validate portable evidence:

```bash
python3 scripts/run_cudanav_real_gpu_stack.py \
  --publish build/cudanav_real_gpu_stack_smoke/manifest.json \
  --result-id cudanav_real_gpu_stack_YYYY-MM-DD \
  --output-json docs/results/cudanav_real_gpu_stack_YYYY-MM-DD.json \
  --output-markdown docs/results/cudanav_real_gpu_stack_YYYY-MM-DD.md

python3 scripts/run_cudanav_real_gpu_stack.py \
  --validate-portable docs/results/cudanav_real_gpu_stack_YYYY-MM-DD.json \
  --commit SOURCE_COMMIT
```

## Mapping and safety semantics

The 3D mapper retains the complete voxel grid, while its 2D navigation
projection uses a declared height band. CudaNav bringup selects
`[-0.5, 2.0)` m so ground returns below the robot and overhead returns do not
become planar obstacles.

Before ESDF inflation, the robot's declared 0.30 m circular footprint is
cleared. The ROS Nav2 voxel layer applies the same rule. This prevents the
vehicle's own occupied or inflated cell from making every rollout collide;
obstacles outside the footprint remain unchanged.

Real scenes can still contain a genuinely blocked local horizon. The quality
contract permits at most three all-colliding evaluations in the 31-evaluation
smoke profile, but each must produce a safety intervention with linear speed
at most 0.05 m/s. Non-blocked evaluations must retain at least a 1% valid
rollout ratio. The report preserves the number of all-colliding and retreating
evaluations rather than silently removing them.

The 120-second release profile requires at least 100 MPPI evaluations, tightens
ATE and drift gates to 3 m and 5%, and permits at most six bounded safety
interventions. A smoke PASS is not a release-profile claim.
