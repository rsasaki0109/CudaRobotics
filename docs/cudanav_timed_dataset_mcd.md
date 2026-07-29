# Timed LiDAR release candidate: MCD NTU Day 02

The primary candidate for release-grade CudaNav deskew evidence is the MCD
`ntu_day_02` Ouster OS1-128 sequence. It is 229 seconds long and separates the
Ouster modality from the camera bags, keeping the required download near
5 GB.

The official MCD manual fixes the properties needed by the admission gate:

- `/os_cloud_node/points` is a 10 Hz `sensor_msgs/PointCloud2` stream;
- each Ouster sweep has 128 channels and 1024 points per channel;
- `t` is `uint32` and stores nanoseconds from the start of the sweep;
- the PointCloud2 header stamp denotes the end of the 0.1-second sweep;
- `ring` is `uint8`;
- `pose_inW.csv` supplies 10 Hz `world_T_body` poses;
- `spline.csv` supplies the continuous-time ground-truth trajectory;
- `body.os_sensor.T` in the official calibration transforms the Ouster sensor
  into the body frame.

These are source-declared semantics, not values inferred from the downloaded
samples. The machine-readable candidate is
[`cudanav_timed_dataset_mcd_ntu_day_02.json`](cudanav_timed_dataset_mcd_ntu_day_02.json).

## Required materialization

MCD distributes this sequence as a ROS 1 bag with bz2-compressed chunks.
CudaNav's dependency-free real-data runners consume ROS 2 SQLite/CDR. A
deterministic materializer must therefore:

1. verify the bag, discrete/continuous ground truth, and calibration by exact
   byte count and SHA-256;
2. select the declared 120-second window without modifying point order or
   point fields;
3. convert ROS 1 PointCloud2 serialization to ROS 2 CDR while preserving
   `t:uint32` and `ring:uint8`;
4. transform each `world_T_body` reference pose by `body_T_os_sensor`;
5. emit paired `geometry_msgs/msg/PoseStamped` in the Ouster sensor frame;
6. bind every source and output digest in a materialization report;
7. pass `inspect_pointcloud2_timing.py` before either GPU release harness runs.

Until those checks and the downstream 120-second GPU gates pass, this remains
a candidate and not release evidence.

## Materialized and timing-admitted artifact

The source contract was materialized on 2026-07-29. The resulting ROS 2 bag
contains 1,200 paired point-cloud and sensor-reference poses over 119.902
seconds. Its SQLite database is 7,559,663,616 bytes with SHA-256
`69595af33297924ef36f1cdd0507d4e868d784337bd1249a41e4d4131c30614c`.

The full-window timing admission passed all 1,200 frames:

- observed scan spans were 99.380--100.522 ms;
- `nanoseconds` was the only physically plausible unit;
- all clouds retained the same schema and `os_sensor` frame;
- all frame timestamps were strictly increasing;
- `ring` covered all 128 values from 0 through 127;
- the maximum absolute discrete-GT/LiDAR pairing error was 6.091 ms.

SQLite integrity passed, both topics contain exactly 1,200 messages, and
source-to-ROS2 point-data hashes matched at the first, middle, and last
sampled scans. The portable identity and checks are frozen in
[`cudanav_timed_dataset_mcd_ntu_day_02_materialized.json`](cudanav_timed_dataset_mcd_ntu_day_02_materialized.json).
The 7.56 GB derived bag is not committed or redistributed.

Reproduce the conversion after placing the four content-addressed inputs in
one directory:

```bash
python -m pip install -r scripts/requirements-mcd-materialization.txt
python scripts/materialize_mcd_timed_rosbag.py \
  --source-dir build/datasets/mcd_ntu_day_02 \
  --output build/datasets/mcd_ntu_day_02/ros2_timed_120s \
  --report build/datasets/mcd_ntu_day_02/ros2_timed_120s_materialization.json
```

This artifact is timing-admitted input, not a GPU performance result. It
becomes release evidence only after the 120-second KISS-ICP and all-GPU stack
quality gates pass.

## GPU KISS-ICP release result

The standalone timed-odometry release gate passed on commit `d240161` using an
NVIDIA GeForce GTX 1660 Ti:

- 1,190 deskewed frames over 118.902 seconds;
- 326.021 m reference path;
- 0.815 m ATE RMSE;
- 0.472% final drift;
- 25,004 minimum inliers;
- 327.148 ms mean frame time;
- 177.821 ms GPU nearest-neighbour p95.

The thresholds were ATE RMSE at most 3 m, final drift at most 5%, and at least
100 inliers. The content-bound portable result is
[`results/cudanav_kiss_icp_mcd_ntu_day_02_2026-07-29.md`](results/cudanav_kiss_icp_mcd_ntu_day_02_2026-07-29.md).
This closes the standalone KISS-ICP gate; the all-GPU shadow-stack gate remains
separate.

## All-GPU shadow-stack release result

The four-stage shadow gate passed on commit `541a53d` with the same MCD
sequence:

- GPU KISS-ICP deskewed all 1,190 selected frames;
- ATE RMSE was 0.819 m and final drift was 0.475%;
- rolling voxel mapping integrated rays on all frames and finished with
  1,778,523 observed voxels;
- the peak occupancy projection contained 8,162 occupied cells;
- GPU ESDF p95 was 1.147 ms;
- CUDA MPPI evaluated 120 controls at 0.836 ms solve p95;
- its minimum nonzero valid-rollout ratio was 0.1284 against the 0.01 gate;
- all-colliding evaluations and invalid commands were both zero.

The MPPI shadow configuration uses two optimization iterations per control
evaluation. This was added after the one-iteration configuration exposed a
real sharp-turn negative result (4/2,048 valid rollouts at one evaluation)
without lowering the release threshold.

The content-bound portable result is
[`results/cudanav_all_gpu_mcd_ntu_day_02_2026-07-29.md`](results/cudanav_all_gpu_mcd_ntu_day_02_2026-07-29.md).
Its scope is real-sensor all-GPU shadow execution: commands are evaluated but
not applied, so it is not ROS 2 runtime or closed-loop evidence.

## Source and license

- Dataset/download: <https://mcdviral.github.io/Download.html>
- Sensor and timestamp contract: <https://mcdviral.github.io/UserManual>
- Ground-truth contract: <https://mcdviral.github.io/Groundtruth.html>
- License: CC BY-NC-SA 4.0, non-commercial academic use.

The source files are not redistributed by this repository.
