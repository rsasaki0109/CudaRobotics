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
The selected raw urban-evaluation file is fixed to Google Drive ID
`1uta5Xr_ftV4jERxPNVqooDvWerK0dn89` and database
`rosbag2_2024_09_11-17_53_54_0.db3`; keep it outside release artifacts.

Validate the selection before downloading:

```bash
python3 scripts/validate_cudanav_real_dataset.py
```

Download, inspect the SQLite topic table, and regenerate rosbag2 metadata in a
ROS 2 Jazzy environment:

```bash
python3 -m pip install gdown
python3 scripts/prepare_cudanav_istanbul_dataset.py \
  --output-dir build/datasets/cudanav_istanbul \
  --download \
  --reindex
```

The inspection report records the exact DB size/SHA-256 and refuses missing,
zero-count, or wrong-type PointCloud2, Odometry, and static-TF topics.

Then generate the rosbag2 sidecar and freeze both local inputs:

```bash
python3 scripts/derive_cudanav_path_sidecar.py \
  --source-bag build/datasets/cudanav_istanbul \
  --database \
    build/datasets/cudanav_istanbul/rosbag2_2024_09_11-17_53_54_0.db3 \
  --output-bag build/cudanav_real_dataset/path_sidecar \
  --report build/cudanav_real_dataset/path_generator.json \
  --materialization build/cudanav_real_dataset/materialization.json

python3 scripts/validate_cudanav_real_dataset.py \
  --materialization build/cudanav_real_dataset/materialization.json
```

The materialization gate requires positive message counts for every selected
recorded input and for the derived Path. It rehashes both local bag trees and
the generator report by default. The actual public-bag GPU shadow replay
remains outstanding; the selection alone is intentionally
`valid: true, ready: false`.

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
