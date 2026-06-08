# GPU Multi-Robot Place Graph SLAM

`src/gpu_multi_robot_place_graph_slam.cu`

This demo connects three drifting robot odometry chains with GPU place
recognition, then runs a lightweight GPU pose-graph optimizer over the combined
graph.

![demo](../gif/gpu_multi_robot_place_graph_slam.gif)

## What It Shows

- Three robots start as separate, biased local odometry graphs.
- A CUDA all-pairs descriptor matcher scores `246 x 246 = 60,516` candidate
  pairs.
- The host accepts the strongest inter-robot place matches as loop edges.
- A CUDA edge-projection optimizer pulls the robot graphs into one shared map.

Latest smoke run:

| Metric | Value |
|---|---:|
| Pose nodes | 246 |
| Descriptor dimensions | 16 |
| GPU descriptor scores | 60,516 |
| Accepted place edges | 42 |
| Exact place matches | 42 / 42 |
| RMSE before optimization | 7.585 m |
| RMSE after optimization | 3.330 m |
| GPU optimizer time | 5.91 ms |

## CUDA Shape

- One CUDA thread scores one descriptor pair in the place-recognition matrix.
- One CUDA thread projects one pose-graph edge and atomically accumulates pose
  corrections.
- One CUDA thread applies the averaged correction for one pose node.

This is intentionally a lightweight visual systems demo, not a full SLAM
frontend. The descriptor is analytic, the accepted matches are high-confidence
synthetic revisits, and the optimizer uses projection-style updates instead of
a full sparse Gauss-Newton solve.

## Build And Run

```bash
cmake --build build --target gpu_multi_robot_place_graph_slam -j$(nproc)
./bin/gpu_multi_robot_place_graph_slam
```

The executable writes `gif/gpu_multi_robot_place_graph_slam.gif`.
