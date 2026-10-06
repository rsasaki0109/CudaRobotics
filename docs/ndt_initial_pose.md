# Initial pose estimation for map-based 3D LiDAR localization: batched NDT on the GPU

## Why this

Automated vehicles running Autoware localize with NDT against a prior point-cloud map. This includes Japan's level-4 automated buses.

To start, and to recover after a failure, the vehicle needs an initial pose. Autoware's `ndt_scan_matcher` estimates it by running NDT from about 200 candidate poses one after another on the CPU: GNSS gives the position, and the yaw is sampled. That takes seconds, and it relies on the GNSS error being small.

Every candidate is an independent alignment, so they can all run at once. This benchmark measures what that buys on real LiDAR data.

Source: [`src/benchmark_ndt_localization.cu`](../src/benchmark_ndt_localization.cu). Report: [results/ndt_initial_pose_2026-10-07.md](results/ndt_initial_pose_2026-10-07.md).

## Method

- **NDT** (Magnusson 2009, the PCL/Autoware score):
  - 2 m voxels, with covariance eigenvalues clamped at 1 % of the largest, as PCL does.
  - The point's voxel and its 6 face neighbours.
  - Outlier ratio 0.55.
  - Gauss-Newton on the right-perturbed pose, up to 30 iterations, steps clamped to 0.5 m and 0.2 rad.
- **Initial pose:** every hypothesis is aligned. The 4 best by NVTL (Autoware's nearest voxel transformation likelihood) are refined with up to 60 smaller steps, and the best of them is the estimate.
- **GPU:** one CUDA block per hypothesis, 256 threads over the scan points, with a block reduction of the gradient, Hessian and scores. A second kernel solves each hypothesis's 6 × 6 step.
- **CPU:** the same per-point code (`__host__ __device__`), with hypotheses spread over 1 or 12 threads.

## Data

MCD `ntu_day_02`: an Ouster OS1-128 on a campus drive, with ground-truth sensor poses. The 120-second window has 1200 scans, 327 m at 2.7 m/s, with no revisits.

[`scripts/export_lidar_localization_sequence.py`](../scripts/export_lidar_localization_sequence.py) exports it from the ROS 2 bag at a 0.25 m voxel. The benchmark then downsamples each test scan to 1 m within 60 m, about 6600 points.

**Map.** Every 10th scan, placed at its ground-truth pose: 120 scans, 21,955 valid voxels.

**Test.** 40 scans, each halfway between two map scans, 1.35 m from the nearest. Each test starts from:
- a GNSS-like position uniform in a disc of radius r, with z within ±1 m;
- roll and pitch from gravity (an IMU);
- unknown yaw.

Success means within 0.5 m and 2° of the ground truth.

## Results

Criterion and configurations fixed in [`scripts/ndt_initial_pose_eval.py`](../scripts/ndt_initial_pose_eval.py) before the run:

| GNSS error radius | hypotheses | succeeded | GPU ms (mean of 40) | GPU ms on the CPU-timed scans | CPU 1 thread ms | CPU 12 threads ms |
|---|---:|---:|---:|---:|---:|---:|
| 2 m (3 × 3 positions × 16 yaws) | 144 | 40/40 | 46 | 113 | 5,000 | 1,307 |
| 5 m (6 × 6 × 16) | 576 | 40/40 | 115 | 191 | 19,364 | 4,817 |
| 10 m (11 × 11 × 16) | 1936 | 40/40 | 314 | 275 | 57,339 | 13,755 |
| 10 m, CPU-sized budget (3 × 3 at 10 m × 8 yaws) | 72 | **13/40** | 37 | 70 | 2,207 | 559 |

CPU times are means over the first 5 / 3 / 2 / 5 test scans. The GPU column next to them covers the same scans: the first scans include CUDA warm-up, which raises the GPU mean.

- **The GPU aligns the same hypotheses 12-50 times faster than 12 CPU threads, and 44-209 times faster than one.** These are on the same scans and include warm-up.
  - At a 2 m GNSS error, the initial pose takes about 0.05 s on the GPU and 1.3 s on 12 CPU threads.
  - At 10 m, 0.3 s against 14 s.
- **The time buys tolerance to GNSS error.** With a 10 m error (an urban canyon, a parking structure), the full 1936-hypothesis search succeeds on all 40 scans in about 0.3 s. The search a CPU can afford in about half a second (72 hypotheses on 12 threads) succeeds on 13. Paired: 27 / 0, p = 1.5e-8.
- **Accuracy:** NDT started at the ground truth stays within 0.3 m on all 40 scans (mean 0.063 m). Every successful initialization lands at the same accuracy.

**The pre-registered criterion is not met as written.**
- **The failing condition:** condition 1 asked the CPU and the GPU to pick the same hypothesis index, and on 3 of the 5 CPU-timed 2 m scans they did not.
- **Why it does not matter:** many hypotheses converge to the same pose and tie on NVTL up to the order of floating-point summation (a block tree reduction against a sequential sum).
  - A diagnostic rerun prints the gap between the two picks: 0.000 m and 0.00° on all three.
  - The condition should have compared the final poses, not the indices.
- **Condition 2 holds.**

## Map from another session

A deployed vehicle localizes against a map recorded on another day, with other parked cars and people. The same 40 day_02 scans are therefore also tested against a map built from MCD `ntu_night_13`:
- **When and where.** Recorded two days later, at night, on a route that covers the whole day_02 window.
- **How the map was built.** Every 10th scan at its ground-truth pose, in the dataset's common world frame: 44,985 valid voxels.
- **Export.** `--ros1-bag` reads the original ROS 1 bag with `pose_inW.csv` and the calibration.

Before this run, condition 1 was rewritten to compare the CPU's and the GPU's **final poses** (within 0.01 m and 0.1°). Report: [results/ndt_initial_pose_cross_2026-10-07.md](results/ndt_initial_pose_cross_2026-10-07.md).

| GNSS error radius | hypotheses | succeeded | GPU ms (mean of 40) | GPU ms on the CPU-timed scans | CPU 1 thread ms | CPU 12 threads ms |
|---|---:|---:|---:|---:|---:|---:|
| 2 m | 144 | 40/40 | 48 | 98 | 3,691 | 857 |
| 5 m | 576 | 40/40 | 114 | 157 | 16,952 | 3,991 |
| 10 m | 1936 | 40/40 | 333 | 286 | 49,513 | 12,010 |
| 10 m, CPU-sized budget | 72 | **14/40** | 35 | 64 | 2,087 | 590 |

**The criterion is met.**
- **CPU and GPU agree.** They end at the same pose on every CPU-timed scan. On 5 of them they picked different, equivalent hypotheses, all within 0.01 m.
- **The full search beats the CPU budget.** At a 10 m error the full search beats the CPU-sized budget 26 / 0 (p = 3e-8).
- **Speed.** The GPU is 9-42 times faster than 12 CPU threads, and 38-173 times faster than one.

**Accuracy against the other session's map.**
- **Tracking.** NDT started at the ground truth stays within 0.3 m on 39 of 40 scans (median 0.12 m, against 0.06 m on the same-session map).
- **Initialization.** Successful initializations land 0.15 m and 0.3° from the ground truth on average.
- **Part of this is the ground truth.** The 0.12 m includes the disagreement between the two sessions' ground-truth trajectories, so it is an upper bound on NDT's own error.

**Reading.**
- The map from another day and time of day did not cost a single initialization at any GNSS error up to 10 m.
- The GPU's advantage, the full search in about 0.3 s against 12 s on 12 CPU threads, carries over unchanged.

## Limitations

- **One test sequence**, 120 s on a campus. The second map comes from one other session, two days later at night: it has real changes such as parked cars and people, but no construction or seasonal change.
- **Roll and pitch come from the ground truth** (standing in for an IMU), and the z prior is ±1 m.
- **No deskew.**
- **This is a re-implementation of NDT, not Autoware's code.**
  - Autoware parallelizes points within one alignment with OpenMP, and it samples hypotheses sequentially with TPE.
  - The CPU baseline here spreads independent hypotheses over threads, which is the more favourable CPU arrangement.
  - Autoware itself was not run.

## Reproduce

```bash
python scripts/export_lidar_localization_sequence.py \
  --database build/datasets/mcd_ntu_day_02/ros2_timed_120s/mcd_ntu_day_02_timed_0.db3 \
  --output build/datasets/mcd_ntu_day_02/loc_seq_v025.bin --voxel 0.25
python scripts/ndt_initial_pose_eval.py      # all four configurations, CSVs, logs and the report
python scripts/export_lidar_localization_sequence.py --ros1-bag build/datasets/mcd_ntu_night_13/ntu_night_13_os1_128.bag \
  --ground-truth build/datasets/mcd_ntu_night_13/pose_inW.csv --calibration build/datasets/mcd_ntu_night_13/calibration.yaml \
  --output build/datasets/mcd_ntu_night_13/loc_seq_s5_v025.bin --stride 5
python scripts/ndt_initial_pose_eval.py --map-sequence build/datasets/mcd_ntu_night_13/loc_seq_s5_v025.bin   # other-session map
./bin/benchmark_ndt_localization --sequence build/datasets/mcd_ntu_day_02/loc_seq_v025.bin --prior-err 2 --cpu-tests 5
```
