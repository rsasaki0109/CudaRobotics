# CudaNav Real-Sensor Dataset Contract

The selected public source for the systems-paper shadow run is the Autoware
Foundation Istanbul mapping-kit ROS 2 dataset. Its official dataset page lists
Hesai Pandar XT32 `sensor_msgs/msg/PointCloud2`, Applanix
`nav_msgs/msg/Odometry`, and `/tf_static`. The checked-in machine-readable
selection is `docs/cudanav_real_dataset.json`.

The source does not claim to contain `nav_msgs/msg/Path`. CudaNav therefore
uses a separate, deterministic sidecar Path derived only from the recorded
Applanix odometry. The first recorded pose is removed as a full planar
translation and yaw transform so the resulting trajectory starts in CudaNav's
`odom` frame. This mode is always labelled
`real_sensor_shadow_with_derived_path`:

- it is not a recorded Path;
- MPPI commands do not affect later sensor or odometry samples;
- it is not closed-loop evidence;
- both the original bag tree and sidecar bag tree are content-addressed;
- provenance binds the source/output topics, algorithm, parameters, and both
  tree hashes.

The source documentation does not grant this repository redistribution rights.
The current official Drive folder contains one bag. It is fixed to Google
Drive ID `1uta5Xr_ftV4jERxPNVqooDvWerK0dn89`, database
`test_20240930_134039_0.db3` (60,179,423,232 bytes), and its separately hosted
`metadata.yaml` (4,854 bytes). Keep both outside release artifacts. Older
localization-evaluation instructions used a different filename; the
machine-readable contract follows the current contents served by the official
dataset folder.

Validate the selection before downloading:

```bash
python3 scripts/validate_cudanav_real_dataset.py
python3 scripts/prepare_cudanav_istanbul_dataset.py --probe-only
```

Download, inspect the SQLite topic table, and regenerate rosbag2 metadata in a
ROS 2 Jazzy environment:

```bash
python3 scripts/prepare_cudanav_istanbul_dataset.py \
  --output-dir build/datasets/cudanav_istanbul \
  --download \
  --reindex
```

The inspection report records the exact DB size/SHA-256 and refuses missing,
zero-count, or wrong-type PointCloud2, Odometry, and static-TF topics.

The same acquisition, derivation, materialization gate, and optional autonomy
run can be planned or executed from one entry point. Dry-run performs no
dataset writes:

```bash
python3 scripts/run_cudanav_real_dataset_pipeline.py \
  --download --reindex --run-autonomy --dry-run

python3 scripts/run_cudanav_real_dataset_pipeline.py \
  --download --reindex --run-autonomy
```

For a release profile, also supply either prior evidence with
`--multi-gpu-run` or distinct local devices with `--multi-gpu-devices`.

Then generate the rosbag2 sidecar and freeze both local inputs:

```bash
python3 scripts/derive_cudanav_path_sidecar.py \
  --source-bag build/datasets/cudanav_istanbul \
  --database \
    build/datasets/cudanav_istanbul/test_20240930_134039_0.db3 \
  --output-bag build/cudanav_real_dataset/path_sidecar \
  --report build/cudanav_real_dataset/path_generator.json \
  --acquisition-report build/datasets/cudanav_istanbul/inspection.json \
  --materialization build/cudanav_real_dataset/materialization.json

python3 scripts/validate_cudanav_real_dataset.py \
  --materialization build/cudanav_real_dataset/materialization.json
```

The default sidecar backend is standard rosbag2 SQLite and does not require
`rosbag2_py`. Use `--storage mcap` on the derivation command, or
`--sidecar-storage mcap` on the pipeline, when an MCAP artifact is preferred
and ROS 2 Python packages are available. ROS Jazzy CI retains an explicit MCAP
write/open round trip in addition to the dependency-free SQLite tests.
The pipeline refreshes the remote Drive identity by default. For an offline
rerun, `--no-probe` reuses a previously bound probe from the existing
inspection report and records that report's hash and inspection time; without
either source of remote identity the materialization gate remains red.

The materialization gate requires positive message counts for every selected
recorded input and for the derived Path. It rehashes both local bag trees, the
generator report, and the acquisition inspection by default. The inspection
binds the selected Drive file ID, exact DB name/size/SHA-256, required-topic
checks, and dataset-spec digest to the source-bag identity. The GTX 1660 Ti
ROS 2 release replay now passes with 793 CUDA MPPI diagnostics, 790/790
point-cloud/command pairs, 4.801 ms solve p95, and 3.535 m minimum measured
front clearance. See
[`results/cudanav_rosbag_shadow_release_2026-07-30.md`](results/cudanav_rosbag_shadow_release_2026-07-30.md).

After a clean-commit materialization passes, publish a portable summary without
redistributing the bag or leaking machine-local paths:

```bash
python3 scripts/publish_cudanav_dataset_evidence.py \
  --spec docs/cudanav_real_dataset_smoke.json \
  --materialization build/cudanav_real_dataset/materialization.json \
  --result-id cudanav_istanbul_materialization_2026-07-29 \
  --output-json docs/results/cudanav_istanbul_materialization_2026-07-29.json \
  --output-markdown \
    docs/results/cudanav_istanbul_materialization_2026-07-29.md
```

The publisher refuses a dirty worktree or existing output. Its JSON retains
the commit, public file identity, database digest, topic counts, derived
sidecar digests and pose statistics, every materialization check, and hashes
of the generating contracts. It explicitly records `gpu_controller_run:
false` and `closed_loop: false`.

For the quality gate, the dependency-free decoder reads PointCloud2 fields by
name and datatype. It measures horizontal clearance inside the declared
front ±30 degree, z `[-0.5, 2.5]` metre, and range `[0.05, 50]` metre filter.
Cloud header timestamps are paired to the nearest real CUDA MPPI diagnostics
command within 200 ms; at least 90% of valid clouds must pair. This replaces
the LaserScan-only evaluator for this dataset without claiming that commands
altered the recorded vehicle motion.

The ROS 2 Jazzy workflow also builds an actual MCAP Path sidecar from a
synthetic DB3, reopens it with `ros2 bag info`, and reruns the materialization
validator. Its attestation cannot pass unless the
`derived_path_sidecar_roundtrip` check ran successfully.

## Smaller real-data smoke contract

The localization-only Istanbul evaluation bag is kept as a separate smoke
contract in `docs/cudanav_real_dataset_smoke.json`. It is not relabelled as
the current full raw mapping-kit bag. The downloaded artifact is
`rosbag2_2024_09_12-14_59_58_0.db3` (1,009,799,168 bytes, SHA-256
`eb80d649a41fd557ff3af5df4424051191fb696d0ebecbeb36b385702d2b4c8d`).
Its SQLite `quick_check` passes and it contains:

- 34,375 `/localization/util/downsample/pointcloud` PointCloud2 messages;
- 343,730 `/sensing/gnss/pose` PoseStamped messages;
- 4 `/tf_static` messages.

The source bag spans about 57 minutes and contains 343,730 recorded poses.
The smoke contract intentionally selects the first 120 seconds and applies a
0.2 m translation threshold, producing 2,778 SE(2)-normalized Path poses in a
222,268-byte message. The dependency-free backend writes that bounded Path as
a standard rosbag2 SQLite sidecar so it remains practical for DDS/controller
smoke tests. Materialize it in one command:

```bash
python3 scripts/run_cudanav_real_dataset_pipeline.py \
  --spec docs/cudanav_real_dataset_smoke.json \
  --dataset-dir build/datasets/cudanav_localization_smoke \
  --work-dir build/cudanav_real_dataset_smoke \
  --download --generate-metadata \
  --sidecar-storage sqlite3
```

Materialization validation independently reopens the SQLite database and
decodes the complete `nav_msgs/msg/Path` CDR payload. It checks the declared
pose count and first/last stamps, strict timestamp ordering, frame IDs, finite
positions, unit quaternions, normalized origin, duration cap, and absence of
trailing bytes. This catches a corrupt or structurally valid but semantically
different sidecar before ROS playback.

This proves real-file acquisition, topic inspection, PoseStamped decoding,
Path derivation, and content-addressed materialization. It does not by itself
prove a GPU controller run; that requires `--run-autonomy` in a sourced ROS 2
CUDA environment.

The bag's only PointCloud2 topic is the already-downsampled
`/localization/util/downsample/pointcloud`. Its schema contains scalar `x`,
`y`, and `z` only; it has neither a per-point timestamp nor a ring field.
Consequently this contract is deliberately `deskew_capable: false`. It may
drive the version-1 GPU smoke harnesses, but it cannot satisfy the timed
version-2 KISS-ICP or all-GPU-stack release gates. Release evidence must use a
separately content-addressed raw PointCloud2 source whose dataset contract
declares the actual time field and unit, physically justified minimum and
maximum scan duration, and whether ring is required. The independent
`inspect_pointcloud2_timing.py` admission gate checks every selected frame and
requires the declared unit to be the unique unit candidate inside those
bounds before either release harness may export it. The exporter then
revalidates every frame's field schema and timing span and never synthesizes
timing from array order.

The admission tool can also be run directly while evaluating a candidate bag:

```bash
python3 scripts/inspect_pointcloud2_timing.py \
  --database RAW_BAG.db3 \
  --pointcloud-topic /points_raw \
  --point-time-field time \
  --point-time-datatype 7 \
  --point-time-unit seconds \
  --ring-field ring --ring-datatype 4 --require-ring \
  --minimum-scan-span-s 0.05 \
  --maximum-scan-span-s 0.15 \
  --require-unambiguous-unit \
  --output build/cudanav_timing_admission.json
```

An admission PASS establishes that the recording matches a declared timing
contract. It does not establish odometry accuracy or authorize a release
claim; those remain downstream GPU quality gates.

The checked-in portable result is
[`results/cudanav_istanbul_materialization_2026-07-29.md`](results/cudanav_istanbul_materialization_2026-07-29.md);
its adjacent JSON is the machine-verifiable source.

The same content-addressed bag can also drive the reusable CUDA KISS-ICP core
directly, without ROS. See
[`cudanav_kiss_icp_real.md`](cudanav_kiss_icp_real.md) for the bounded
PointCloud2 exporter, GPU quality gates, documented one-second GNSS startup
warmup, and portable evidence workflow. That result proves real-sensor GPU
odometry only; it does not prove a controller or closed-loop run.
