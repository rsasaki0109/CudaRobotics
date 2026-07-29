# CudaNav Real-Sensor Dataset Contract

The selected public source for the systems-paper shadow run is the Autoware
Foundation Istanbul mapping-kit ROS 2 dataset. Its official dataset page lists
Hesai Pandar XT32 `sensor_msgs/msg/PointCloud2`, Applanix
`nav_msgs/msg/Odometry`, and `/tf_static`. The checked-in machine-readable
selection is `docs/cudanav_real_dataset.json`.

The source does not claim to contain `nav_msgs/msg/Path`. CudaNav therefore
uses a separate, deterministic sidecar Path derived only from the recorded
Applanix odometry. This mode is always labelled
`real_sensor_shadow_with_derived_path`:

- it is not a recorded Path;
- MPPI commands do not affect later sensor or odometry samples;
- it is not closed-loop evidence;
- both the original bag tree and sidecar bag tree are content-addressed;
- provenance binds the source/output topics, algorithm, parameters, and both
  tree hashes.

The source documentation does not grant this repository redistribution rights.
Download a bag from the official folder and keep it outside release artifacts.

Validate the selection before downloading:

```bash
python3 scripts/validate_cudanav_real_dataset.py
```

After producing a rosbag2 sidecar containing
`/cuda_nav/derived_plan` (`nav_msgs/msg/Path`), freeze the local inputs:

```bash
python3 scripts/cudanav_real_dataset.py \
  --source-bag /data/istanbul_mapping_bag \
  --derived-path-bag /data/istanbul_mapping_path_sidecar \
  --output build/cudanav_real_dataset/materialization.json

python3 scripts/validate_cudanav_real_dataset.py \
  --materialization build/cudanav_real_dataset/materialization.json
```

The materialization gate requires positive message counts for every selected
recorded input and for the derived Path. It rehashes both local bag trees by
default. Path generation and the actual GPU shadow replay remain outstanding;
the selection alone is intentionally `valid: true, ready: false`.
