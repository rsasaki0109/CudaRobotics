# Real-bag GPU KISS-ICP evidence

This harness runs the reusable CUDA KISS-ICP core directly on recorded
`sensor_msgs/msg/PointCloud2` messages from the content-addressed Istanbul
localization bag. It compares the estimated trajectory with the recorded
GNSS `PoseStamped` sequence without requiring a ROS installation.

The result is real-sensor GPU odometry evidence. It is not a CUDA MPPI
controller run, a ROS integration result, or closed-loop autonomy evidence.

## Point timing and deskew contract

The exporter has two explicit sequence formats:

- version 1 stores XYZ only and is accepted by the smoke profile;
- version 2 stores XYZ plus a finite per-point relative timestamp in seconds.

The release profile is fail-closed: its dataset specification must name a
scalar PointCloud2 time field and unit, every selected frame must have a time
span between 1 microsecond and 1 second, the exporter must emit version 2, and
the native runner must GPU-deskew every frame. A scalar ring field may also be
declared and is preserved in the evidence contract, although the current
deskew kernel does not require it.

The localization-only Istanbul smoke topic contains only `x`, `y`, and `z` in
`base_link`; it has no per-point time or ring field. It is therefore useful
for the bounded version-1 smoke result but cannot satisfy the release-profile
deskew gate. A release run requires a raw PointCloud2 source with recorded
per-point timing. The harness reports this limitation instead of inventing
timestamps from point order.

## Build and run

Configure and build the native sequence runner:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target cudanav_kiss_icp_sequence
```

From a clean commit, run the content-bound smoke profile:

```bash
python3 scripts/run_cudanav_kiss_icp_real.py \
  --database build/datasets/cudanav_localization_smoke/rosbag2_2024_09_12-14_59_58_0.db3 \
  --output-dir build/cudanav_kiss_icp_real_smoke \
  --profile smoke
```

On Windows, the default runner is
`bin/Release/cudanav_kiss_icp_sequence.exe`; on Linux it is
`bin/cudanav_kiss_icp_sequence`. Use `--runner` to override it.

The harness refuses a dirty worktree or an existing output directory. It
checks the database filename, byte count, and SHA-256 against
`docs/cudanav_real_dataset_smoke.json`, exports a bounded binary sequence,
runs GPU odometry, and binds the sequence, reports, trajectory, log, and
runner binary by byte count and SHA-256.

The smoke profile uses 300 frames over 30 seconds and requires:

- ATE RMSE at most 5 m;
- final XY drift at most 10% of the reference distance;
- at least 30 scan-to-map inliers on every aligned frame.

The 120-second release profile additionally requires timed version-2 input and
GPU deskew on every frame, and tightens those gates to 3 m, 5%, and 100
inliers respectively. A passing smoke result is not a release-profile claim.

Validate a retained local run against its source commit:

```bash
python3 scripts/run_cudanav_kiss_icp_real.py \
  --validate build/cudanav_kiss_icp_real_smoke/manifest.json \
  --commit SOURCE_COMMIT
```

After a clean-commit run, publish a portable result:

```bash
python3 scripts/run_cudanav_kiss_icp_real.py \
  --publish build/cudanav_kiss_icp_real_smoke/manifest.json \
  --result-id cudanav_kiss_icp_real_2026-07-29 \
  --output-json docs/results/cudanav_kiss_icp_real_2026-07-29.json \
  --output-markdown docs/results/cudanav_kiss_icp_real_2026-07-29.md
```

Portable validation rechecks the declared source commit, dataset and artifact
digests, GPU identity, quality metrics, exact claim scope, and hashes of the
source files that define the experiment:

```bash
python3 scripts/run_cudanav_kiss_icp_real.py \
  --validate-portable docs/results/cudanav_kiss_icp_real_2026-07-29.json \
  --commit SOURCE_COMMIT
```

The checked-in smoke result is
[`results/cudanav_kiss_icp_real_2026-07-29.md`](results/cudanav_kiss_icp_real_2026-07-29.md);
its adjacent JSON is the machine-verifiable source.

## Startup transient

The beginning of this bag contains a GNSS initialization discontinuity: the
first two reference poses jump by roughly 1.9 m and 146 degrees. An initial
five-second experiment kept that discontinuity visible and showed that it
made the reference comparison unsuitable even though GPU correspondence
timing and inlier counts were healthy.

Both declared profiles therefore use an explicit one-second start offset.
The exporter records that offset in its report. This is a documented reference
warmup, not an unreported trajectory trim, and should remain a limitation in
any result derived from this harness.
