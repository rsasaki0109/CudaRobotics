# CudaNav: A Reproducible End-to-End GPU Autonomy Stack

Draft date: 2026-07-29

## Evidence status

This draft is synchronized with the machine-readable claim ledger in
[`artifacts/cudarobotics_systems.json`](artifacts/cudarobotics_systems.json).
It reports supported native simulation and real-data shadow results, while
keeping ROS 2 release execution and a second physical GPU as explicit pending
evidence. The paper is therefore a current working draft, not a
submission-ready manuscript.

## Abstract

GPU robotics projects often accelerate one stage of a navigation pipeline but
leave data movement, framework integration, and end-to-end evaluation outside
the benchmark. We present CudaNav, an open-source autonomy stack that connects
GPU LiDAR odometry, rolling voxel mapping, a typed Euclidean signed distance
field, and model predictive path integral control through reusable C++/CUDA
cores and ROS 2/Nav2 adapters. The evaluation separates three claims that are
frequently conflated: command-driven closed-loop simulation, recorded-data
shadow execution, and cross-device reproduction. In a continuous native
closed loop on an NVIDIA GeForce GTX 1660 Ti, CudaNav completes 30 alternating
S-course traversals over 1,059.4 simulated seconds and 352.75 m with zero
collisions, 0.0035% final odometry drift, and zero 150 ms frame-deadline
misses. On a content-addressed real PointCloud2 sequence, the GPU KISS-ICP core
processes 300 frames with 0.778 m ATE RMSE and 2.961% final drift; the full
odometry-mapping-ESDF-MPPI shadow pipeline averages 9.999 ms per frame. Every
reported artifact is tied to source, dataset, hardware, thresholds, and raw
outputs by a machine-checked manifest. A release-profile ROS 2 run with MCAP
and video, and reproduction on a second GPU model, remain required before the
broader systems claims are considered supported.

## 1. Introduction

Autonomous navigation is a pipeline, not an isolated kernel. LiDAR odometry
changes the pose used by mapping; mapping changes the cost field used by the
controller; controller commands change the next robot state and therefore the
next scan. Reporting a fast correspondence kernel or a fast rollout kernel
does not establish that this chain runs coherently, meets deadlines, or
produces safe motion.

CudaNav addresses this systems boundary. It retains the small reusable CUDA
cores developed in CudaRobotics, then connects them into one autonomy path:

```text
LiDAR / recorded PointCloud2 / deterministic simulator
                       |
                       v
             GPU KISS-ICP odometry
                       |
                       v
       rolling GPU voxel map and typed ESDF
                       |
                       v
                 CUDA MPPI
                       |
                       v
       command-driven plant or shadow evaluator
```

The central methodological choice is to make evidence type part of the result.
A recorded bag can establish sensor compatibility, accuracy, latency, and
controller diagnostics, but it cannot establish closed-loop success because
computed commands do not alter later measurements. Conversely, synthetic
closed-loop simulation can establish causality and safety under a controlled
scenario, but not real-sensor robustness. CudaNav records these as separate
contracts and refuses to promote one into the other.

The present work makes four contributions:

1. A reusable all-GPU core that connects voxel-hash KISS-ICP-style LiDAR
   odometry, rolling voxel mapping, exact typed 2D ESDF construction, and CUDA
   MPPI without a host-side algorithmic fallback.
2. ROS 2 lifecycle components and a Nav2 costmap/controller path that reuse the
   same tested CUDA sources rather than maintaining framework-specific
   algorithm copies.
3. Distinct content-bound contracts for closed-loop simulation, real-data
   shadow evaluation, and physical multi-GPU reproduction.
4. A consumer-GPU evaluation that retains positive, partial, and negative
   results, including safety stops and missing release evidence.

## 2. Related systems

KISS-ICP demonstrates that a carefully designed point-to-point ICP pipeline can
provide simple and robust LiDAR odometry across sensors and platforms
[[Vizzo et al., 2023](https://arxiv.org/abs/2209.15397)]. CudaNav follows that
minimal point-to-point design philosophy, while replacing correspondence
search with GPU voxel-hash nearest-neighbour lookup and exposing the core to
native and ROS 2 execution paths.

MPPI converts large batches of sampled control sequences into a parallel
receding-horizon update
[[Williams et al., 2017](https://doi.org/10.2514/1.G001921)]. CudaNav uses a
CUDA MPPI implementation with explicit rollout-validity, all-colliding,
retreat, and deadline diagnostics rather than reporting solve time alone.

Navigation2 provides lifecycle-managed navigation components, behavior-tree
orchestration, costmaps, and controller plugins in ROS 2
[[Macenski et al., 2020](https://arxiv.org/abs/2003.00368)]. CudaNav integrates
through these interfaces, but keeps ROS middleware execution distinct from the
native CUDA-core evidence. This distinction prevents source-level integration
from being reported as a runtime result.

## 3. System design

### 3.1 GPU LiDAR odometry

The odometry core downsamples a scan, predicts motion, searches neighbouring
voxel-hash cells for nearest-neighbour correspondences, rejects outliers, and
solves a robust point-to-point rigid update. Device buffers and the local map
persist across frames. Diagnostics include correspondence count, inliers,
nearest-neighbour time, frame time, position error when reference poses are
available, and hash-capacity failures.

The public real-data profile deliberately applies a one-second startup offset.
The excluded region contains a documented GNSS initialization discontinuity
of approximately 1.9 m and 146 degrees. This is declared in the profile and
evidence rather than selected after observing the final metric.

### 3.2 Rolling voxel map and ESDF

Registered points are fused into a bounded rolling voxel representation. A
typed distance-field interface carries grid dimensions, resolution, origin,
storage, and device ownership into the mapping and controller adapters. The
2D navigation field is produced by GPU Euclidean distance propagation and
inflation, with occupancy and observed-voxel counts retained as health
signals.

Separating the typed ESDF from ROS messages is intentional. Native simulation,
recorded-data evaluation, Python bindings, and ROS 2 nodes can share the same
memory and geometry contracts without parsing middleware-specific payloads in
the CUDA kernels.

### 3.3 CUDA MPPI

The controller samples batched trajectories on the GPU, evaluates path,
goal, collision, clearance, curvature-speed, and path-angle terms, then
updates the command sequence by cost-weighted aggregation. The stack supports
DiffDrive, Ackermann, and Omni motion-model adapters. The evidence schema
records sampled and valid rollouts, all-colliding and retreat states, command
finiteness, solve latency, full-frame latency, and whether a command caused
later plant motion.

The native S-course switches between forward-only and bidirectional controller
instances when alternating the path direction. The MPPI warm start resets at a
path reversal; KISS-ICP, the voxel map, ESDF state, and plant continue without
resetting.

### 3.4 ROS 2 and Nav2 boundary

The repository contains lifecycle components for GPU KISS-ICP, voxel mapping,
and ESDF; a typed distance-field message; a Nav2 voxel costmap layer; and the
CUDA MPPI controller plugin. Shared launch and configuration files connect the
components. Source and contract tests are complete. The paper ledger keeps the
integrated ROS 2 claim partial until a green Jazzy build, plugin-load test, and
release-profile GPU runtime attestation are attached from the paper commit.

## 4. Evidence model

### 4.1 Three non-interchangeable modes

| Evidence mode | Commands affect later scans | Real sensor data | What it can establish |
|---|---:|---:|---|
| Native closed-loop simulation | Yes | No | Causal control, mission completion, collisions, drift, deadlines |
| Recorded-data shadow | No | Yes | Parsing, odometry accuracy, mapping/controller diagnostics, latency |
| Physical multi-GPU matrix | Depends on imported run | Depends on imported run | Same-source reproducibility across UUIDs and model names |

Each release manifest records the source commit and digest, clean/dirty state,
GPU name and UUID, driver, scenario/profile, thresholds, commands, and
SHA-256/size for retained artifacts. Validators reopen raw JSON, CSV, bag
metadata, and media instead of trusting report prose.

### 4.2 Claim-to-evidence ledger

The following IDs are authoritative and are checked against the manuscript:

| Claim ID | Current status | Evidence boundary |
|---|---|---|
| `reproducibility_contracts` | Supported | Contract sources and validators exist and are hashed |
| `real_dataset_materialization` | Supported | Public bag and deterministic derived Path are content-addressed |
| `real_gpu_odometry` | Supported | 300-frame GPU KISS-ICP real-data smoke |
| `real_gpu_core_shadow` | Supported | 300-frame native all-GPU shadow smoke |
| `native_gpu_core_closed_loop` | Supported | 30-traversal native release plus bound visual |
| `integrated_gpu_stack` | Partial | Sources exist; final-commit ROS 2 runtime attestation pending |
| `closed_loop_autonomy` | Partial | Native release passes; ROS 2 MCAP/video release pending |
| `real_data_shadow` | Partial | Native smoke passes; ROS 2 release replay pending |
| `multi_gpu_reproduction` | Partial | One UUID-bound GTX 1660 Ti node; second model pending |

## 5. Experimental protocol

### 5.1 Native closed loop

The deterministic S-course occupies a bounded 11 m by 5 m workspace with two
offset wall obstacles, a 0.24 m robot radius, 240 LiDAR rays at three height
levels, and a 0.1 s control period. Ground truth generates LiDAR and scores the
run, but the controller receives only the KISS-ICP estimate and GPU-produced
costmap. The release profile requires:

- 30 completed alternating traversals and at least 600 simulated seconds;
- final goal distance at most 0.30 m and zero collisions;
- odometry drift below 1% of travelled distance;
- fewer than 1% 150 ms frame-deadline misses;
- finite commands, causal command motion, and bounded safety interventions;
- healthy inlier, voxel, occupancy, and rollout-validity signals.

The continuous run is not divided into 30 independent successes. Plant,
odometry, and map state persist for the full mission, making accumulated drift
and stale-map failure visible.

### 5.2 Public recorded data

The real-data source is a content-addressed Autoware Istanbul localization bag.
The retained 1,009,799,168-byte SQLite database contains 34,375 downsampled
PointCloud2 messages and 343,730 GNSS pose messages. A bounded 120-second
selection and deterministic Path sidecar are derived without relabelling
recorded motion as control output.

The current GPU smoke consumes 300 frames over 29.9 seconds. It evaluates
odometry alone and the complete native GPU core shadow path separately. The
release profile remains pending because the full ROS 2 replay and runtime
attestation have not been acquired.

### 5.3 Hardware and reproducibility

The current physical evidence node is an NVIDIA GeForce GTX 1660 Ti with UUID
`GPU-f635286a-d68f-5039-cbc9-22d7f295b3a3`. The matrix validator requires a
clean checkout, identical 40-character source commit and source digest,
distinct physical UUIDs and model names, complete release metrics, and
unchanged result/trajectory hashes. Two runs on the same model do not satisfy
the cross-model claim.

## 6. Results

### 6.1 Continuous native closed loop

![Native all-GPU CudaNav release](../gif/cudanav_gpu_closed_loop_release.gif)

| Metric | Release result |
|---|---:|
| Traversals | 30 / 30 |
| Simulated duration | 1,059.4 s |
| Ground-truth distance | 352.748 m |
| Final goal distance | 0.296 m |
| Collisions | 0 |
| KISS-ICP ATE RMSE | 0.0124 m |
| Final odometry drift | 0.003493% |
| Minimum ICP inliers | 213 |
| Final observed voxels | 53,012 |
| Peak occupied cells | 760 |
| MPPI solve p95 | 0.455 ms |
| Full frame p95 | 5.237 ms |
| Frame-deadline miss rate | 0% |
| All-colliding evaluations | 0 |

The command-effect distance is 352.748 m and agrees with ground-truth travel,
showing that controller outputs cause subsequent plant motion. The animation
is sampled from the 10,594-row trajectory while retaining every traversal
boundary. Its sidecar binds the source result, trajectory, renderer, frame
inventory, and GIF hash.

An independently frozen GTX 1660 Ti matrix node at the same native contract
also completes 30/30 traversals over 1,005.0 seconds and 352.211 m with zero
collisions, 0.00153% final drift, zero deadline misses, 0.617 ms MPPI p95, and
6.417 ms frame p95. This is useful repeat evidence on one physical GPU, not the
required second-model result.

### 6.2 Real PointCloud2 odometry and shadow control

| Metric | GPU odometry | Full GPU shadow stack |
|---|---:|---:|
| Frames / duration | 300 / 29.9 s | 300 / 29.9 s |
| ATE RMSE | 0.778 m | 0.779 m |
| Final drift | 2.961% | 2.985% |
| Mean frame time | 9.669 ms | 9.999 ms |
| GPU NN p95 | 5.582 ms | included in full frame |
| ESDF p95 | not run | 1.153 ms |
| MPPI solve p95 | not run | 0.532 ms |
| Final observed voxels | not run | 107,737 |
| Peak occupied cells | not run | 1,789 |
| Quality gate | Pass | Pass |

The shadow stack performs 31 MPPI evaluations. Two evaluations report all
rollouts colliding; the retained safety behavior commands zero linear speed.
This is a bounded positive diagnostic, not proof of vehicle obstacle avoidance,
because commands are not applied to the recorded sequence.

### 6.3 Current readiness

Supported evidence establishes a coherent native GPU core, causal continuous
simulation, real PointCloud2 execution, and one physical consumer-GPU node. It
does not yet establish:

- a release-profile ROS 2 Jazzy GPU run with positive MCAP topic counts and a
  retained video;
- a release-profile ROS 2 real-bag shadow run;
- reproduction on a second distinct physical GPU model;
- real-robot closed-loop navigation.

The systems ledger remains `ready: false` until the submission-required partial
claims are either supported by the declared evidence or narrowed in the paper.

## 7. Reproduction

The native release and its visual are generated with:

```bash
cmake --build build --target cudanav_gpu_closed_loop_s_course -j
python3 scripts/run_cudanav_gpu_closed_loop.py \
  --profile release \
  --output-dir build/cudanav_gpu_closed_loop_release
python3 scripts/render_cudanav_gpu_closed_loop.py \
  --evidence docs/results/cudanav_gpu_closed_loop_release_2026-07-29.json \
  --trajectory build/cudanav_gpu_closed_loop_release/trajectory.csv \
  --output gif/cudanav_gpu_closed_loop_release.gif \
  --manifest gif/cudanav_gpu_closed_loop_release.json
```

The public dataset and native real-data pipeline are materialized with:

```bash
python3 scripts/run_cudanav_real_dataset_pipeline.py \
  --download --reindex --run-autonomy --profile smoke \
  --output-dir build/cudanav_real_dataset
```

Paper claims and retained hashes are checked with:

```bash
python3 scripts/validate_paper_artifacts.py \
  paper/artifacts/cudarobotics_systems.json
```

Once that command passes with `--require-ready` and this draft no longer
contains its non-ready markers, create the content-complete paper artifact:

```bash
python3 scripts/assemble_systems_paper_bundle.py \
  --output-dir build/cudarobotics_systems_paper_bundle \
  --commit "$(git rev-parse HEAD)"
python3 scripts/archive_systems_paper_bundle.py \
  build/cudarobotics_systems_paper_bundle/submission_manifest.json \
  --output build/cudarobotics-systems-paper-artifact.zip \
  --commit "$(git rev-parse HEAD)"
python3 scripts/validate_systems_paper_archive.py \
  build/cudarobotics-systems-paper-artifact.zip \
  --checksum build/cudarobotics-systems-paper-artifact.zip.sha256 \
  --commit "$(git rev-parse HEAD)"
```

The assembler is fail-closed while any submission-required ledger claim is
partial, so these commands do not turn the current draft into a nominally
ready artifact.

The complete command and threshold documentation is in
[`docs/cudanav_autonomy_suite.md`](../docs/cudanav_autonomy_suite.md),
[`docs/cudanav_real_dataset.md`](../docs/cudanav_real_dataset.md), and
[`docs/cudanav_multi_gpu.md`](../docs/cudanav_multi_gpu.md).

## 8. Limitations and next evidence

The simulator is deterministic and uses synthetic LiDAR geometry. It is a
strong regression oracle for causality, accumulated state, deadlines, and
collision policy, but it does not model all real sensor failure modes or
vehicle dynamics. The Istanbul sequence supplies real PointCloud2 geometry and
reference poses, but shadow commands cannot influence future scans. The
current 29.9-second real-data run is a smoke profile, not the declared
120-second release profile.

The architecture has ROS 2/Nav2 adapters, but source presence and unit tests
are weaker evidence than a release-profile runtime on the exact paper commit.
That run must retain the MCAP, metadata, required topic counts, parameters,
metrics, logs, and video. Finally, all physical evidence currently comes from
one GTX 1660 Ti. The matrix claim requires a second GPU UUID and model at an
identical source commit and digest.

These omissions are release gates, not future-work decoration. The manuscript
and ledger intentionally keep the affected claims partial.

## 9. Availability

- Repository: <https://github.com/rsasaki0109/CudaRobotics>
- Documentation: <https://rsasaki0109.github.io/CudaRobotics/docs/>
- Animated gallery: <https://rsasaki0109.github.io/CudaRobotics/>
- License: MIT
