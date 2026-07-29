# cudanav_istanbul_materialization_2026-07-29

Portable real-file acquisition and derived-Path evidence. This is not a GPU controller or closed-loop result.

- Commit: `1cbd74285044de066a6ed48cd80c240221800e52`
- Dataset: `autoware_istanbul_localization_smoke`
- Database: `rosbag2_2024_09_12-14_59_58_0.db3` (1009799168 bytes)
- Database SHA-256: `eb80d649a41fd557ff3af5df4424051191fb696d0ebecbeb36b385702d2b4c8d`
- Derived poses: 2778 from 343730 recorded samples
- Derived storage: `sqlite3`
- Derived tree SHA-256: `c0090d392d575a250642964dff7a1ed767667d4adfa39daaf0ffb559f71f2806`
- Validation checks: 29 / 29 passed

## Recorded topics

| Topic | Type | Messages |
|---|---|---:|
| `/clock` | `rosgraph_msgs/msg/Clock` | 137491 |
| `/localization/twist_estimator/twist_with_covariance` | `geometry_msgs/msg/TwistWithCovarianceStamped` | 343739 |
| `/localization/util/downsample/pointcloud` | `sensor_msgs/msg/PointCloud2` | 34375 |
| `/sensing/gnss/pose` | `geometry_msgs/msg/PoseStamped` | 343730 |
| `/sensing/gnss/pose_with_covariance` | `geometry_msgs/msg/PoseWithCovarianceStamped` | 343730 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 4 |

## Scope

- Real-file acquisition: yes
- Deterministic derived Path: yes
- GPU controller run: no
- Closed-loop evidence: no
