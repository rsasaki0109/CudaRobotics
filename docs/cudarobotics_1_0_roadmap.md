# CudaRobotics 1.0 Roadmap

This roadmap turns the repository's individual CUDA demos, Python APIs, and
ROS 2 controller into one reproducible end-to-end GPU autonomy stack.

## North Star

The v1.0 demonstration must run this closed loop:

```text
LiDAR / rosbag / simulator
          |
          v
GPU KISS-ICP odometry
          |
          v
GPU voxel map and ESDF
          |
          v
Nav2 CUDA MPPI controller
          |
          v
closed-loop robot motion
          |
          v
JSON metrics, report, replay, and CI evidence
```

The project is not complete when the components merely run independently. A
v1.0 release requires a versioned launch path, reproducible evidence, and
explicit failure reporting across the complete loop.

## Status Snapshot (2026-07-30)

| Area | Implemented | Evidence still required |
|---|---|---|
| v0.3 release | Published `v0.3.0` at `67f134554d054de14f41f86f4a3a800484d886a1`; Build, Python manylinux, ROS 2, systems-paper archive, release-asset re-download, and GHCR publication gates pass | Complete |
| v0.4 Python CudaNav | ROS 2 CudaNav components and reusable CUDA cores exist, while the Python package currently exposes MPPI and registration | One Python navigation API, CLI, public-dataset quickstart, fixed-seed benchmark, and portable wheels |
| v1 release target | The support matrix keeps `1.0.0` as the future target while the published Python and ROS 2 source version is `0.3.0`; readiness remains fail-closed in `development` | Promote a proven v0.4 surface, then acquire immutable-tag fresh-clone, physical-GPU, published-image, and deployed-docs attestations |
| GPU odometry | Reusable voxel-hash KISS-ICP core, lifecycle ROS component, exact-master Jazzy CI, and 804 odometry outputs in the content-addressed Istanbul ROS 2 GPU shadow release | Complete |
| Mapping | Rolling voxel map, exact typed ESDF, lifecycle nodes, exact-master Jazzy CI, and 796 occupancy plus 794 ESDF outputs in the Istanbul ROS 2 GPU shadow release | Complete |
| Nav2 integration | Voxel costmap plugin, CUDA MPPI, deterministic closed-loop bringup; ROS 2 GPU release passes 30/30 traversals over 1325.5 seconds; public real-sensor shadow release passes 793 GPU control cycles with retained MCAP | Complete |
| Reproducibility | Exact-master Jazzy CI passes; MCD 1,190-scan all-GPU release, clean UUID-bound ROS 2 closed-loop release, and content-addressed Istanbul real-rosbag shadow release pass on the GTX 1660 Ti | Acquire the fresh-clone, published-image, and deployed-docs attestations from one immutable tag; multi-GPU is an optional extension |
| Contact paper | Published 32,400-episode robustness, exact 10 ms matched-compute, 3,150-episode closed-loop MuJoCo evidence, ready ledger, anonymous IEEE conference source, generated figures, and content-bound bundle/archive contracts | Select the final venue mode and real anonymous artifact URL; optional independent-hardware replication |
| Papers | Contact-rich Diff-MPPI and CudaNav systems ledgers are `ready: true`; frozen Markdown and anonymous IEEE sources are machine-checked and CI-compilable. CudaNav ROS 2 closed-loop and real-data shadow claims are supported | Publish the final contact bundle after URL selection; multi-GPU remains an optional partial claim |

Implementation status is not evidence status. Machine-readable paper ledgers
under `paper/artifacts/` become ready only when every submission-required
claim is supported; optional partial claims remain visible.

## Program Rules

- Integrate and harden existing work before adding more isolated demos.
- Keep exhaustive or CPU references where they provide a correctness oracle.
- Record the git commit, hardware, driver, CUDA version, command, seeds, and
  raw outputs for every performance claim.
- Treat recorded-motion or shadow-controller bag analysis as such; only a
  controller whose commands affect subsequent robot state is closed loop.
- Keep the systems-paper and contact-rich Diff-MPPI claims separate.
- Preserve negative results and failed quality gates in release evidence.

## Epic 0: v0.2.0 Release Closure

Objective: publish the registration and real-rosbag evaluation release from a
single release-candidate commit.

Required outcomes:

- version consistency passes at `0.2.0`;
- Python CPU tests and CUDA registration consistency pass;
- the registration smoke produces CSV and Markdown artifacts;
- the root CUDA release targets build;
- the ROS 2 CUDA MPPI workflow passes;
- the Python workflow produces the sdist and supported manylinux wheels;
- the documentation site, Colab notebook, and release notes agree on v0.2.0;
- rosbag evidence is labelled recorded/shadow or closed-loop without ambiguity;
- the tag and GitHub release are created only after the evidence is attached.

Exit gate: every row in
[`releases/v0.2.0_smoke_checklist.md`](releases/v0.2.0_smoke_checklist.md) has
a pass, an intentional negative result, or an explicit release blocker. No
required GPU check may be silently skipped.

## Epic 1: Reusable GPU Core

Objective: remove the main scaling and integration barriers without turning
the repository into a framework-heavy codebase.

Initial components:

- reusable GPU voxel-hash indexing with overflow diagnostics;
- CUDA memory ownership and error-handling helpers;
- common event timing and hardware metadata;
- a stable JSON benchmark schema;
- persistent voxel-map normals instead of per-scan exhaustive kNN-PCA;
- Python, C++, and ROS 2 adapters over the same tested kernels.

KISS-ICP gates:

- total odometry time at or below 12 ms per scan on the GTX 1660 Ti reference
  run;
- voxel-hash versus exhaustive-reference translation ATE delta below 0.1 mm;
- no hash overflow at the supported 200k-point local-map capacity;
- deterministic accuracy, correspondence, and schema checks in CTest.

## Epic 2: ROS 2 CudaNav

Objective: connect perception, mapping, and control into a supported ROS 2
navigation path.

Implemented packages or components:

- `cuda_kiss_icp_odometry`;
- `cuda_voxel_mapping`;
- `cuda_esdf` with typed `DistanceField2D`;
- `cuda_voxel_costmap_layer`;
- the existing `cuda_mppi_controller`;
- shared launch, configuration, diagnostics, and bag-replay tools.

Integration contracts:

- `sensor_msgs/PointCloud2` input;
- odometry and TF output;
- voxel-map and ESDF updates;
- Nav2 costmap/controller integration;
- DiffDrive, Ackermann, and Omni configurations;
- controller diagnostics using the common benchmark schema.

Closed-loop exit gate:

- at least 10 minutes of closed-loop simulation;
- zero collisions in the release scenario;
- controller deadline-miss rate below 1%;
- odometry drift below 1% of travelled distance;
- bag, configuration, seeds, JSON metrics, report, and video are retained.

## Epic 3: Reproducible Autonomy Benchmark

Objective: regenerate the project-level evidence with one entry point.

Current evidence entry points:

```bash
python3 scripts/run_autonomy_suite.py \
  --profile release --output-dir build/cudanav_autonomy_release ...
python3 scripts/run_cudanav_real_dataset_pipeline.py \
  --download --reindex --run-autonomy --profile release ...
python3 scripts/run_cudanav_closed_loop.py \
  --profile release --output-dir build/cudanav_closed_loop
python3 scripts/run_cudanav_rosbag_replay.py \
  --profile release --output-dir build/cudanav_rosbag ...
python3 scripts/run_cudanav_multi_gpu.py \
  --output-dir build/cudanav_multi_gpu ...
gh workflow run ros2_cuda_mppi.yml --ref PAPER_COMMIT
python3 scripts/validate_cudanav_ros_ci.py \
  build/paper/ros_jazzy_ci/ros_jazzy_ci_evidence.json
python3 scripts/publish_cudanav_systems_evidence.py \
  --suite-dir build/cudanav_autonomy_release \
  --ros-ci build/paper/ros_jazzy_ci/ros_jazzy_ci_evidence.json \
  --output-dir docs/results --prefix cudanav_systems_YYYY-MM-DD \
  --v1-attestation-name v1_cudanav_systems_release.json
```

`run_autonomy_suite.py` orchestrates these three commands while preserving
their separate evidence modes. Its aggregate validator refuses to collapse
shadow replay into a closed-loop claim and requires one commit and controller
configuration across the complete release suite.

The ROS 2 workflow independently emits a downloadable, commit-bound Jazzy CI
attestation after the build, plugin-load, parameter-validation, contract, and
package-test gates pass. See
[`cudanav_ros_ci_evidence.md`](cudanav_ros_ci_evidence.md).
The systems publisher then independently revalidates both sources and emits
portable JSON summary/provenance plus a Markdown report for the paper ledger.
For a passing release suite, the same invocation can also emit the
content-bound `cudanav_release_evidence` attestation consumed by the v1
support matrix; it cannot be emitted from smoke-only, dirty, single-model, or
non-Jazzy evidence.

Each run must produce:

- a manifest with git and hardware metadata;
- trajectory and controller CSV files;
- ATE, relative pose error, drift, success, collision, and clearance metrics;
- latency distributions and GPU-memory measurements;
- Markdown and machine-readable JSON reports;
- plots or video where applicable;
- commands, logs, skipped checks, and failure reasons.

The release matrix should cover:

- KISS-ICP voxel hash versus exhaustive GPU reference;
- CUDA MPPI versus the Nav2 CPU controller;
- synthetic, recorded/shadow, and closed-loop simulation evidence;
- fixed seeds with at least three repetitions for benchmark claims;
- the reference GTX 1660 Ti and at least one newer desktop GPU;
- Jetson as either a tested target or an explicitly experimental target.

## Epic 4: v0.4 Python CudaNav

Objective: expose the integrated CudaNav path as a small, reusable Python
surface instead of requiring users to assemble ROS 2 components or standalone
demos.

Planned public API:

- `cudarobotics.navigation.KissIcpOdometry`;
- `cudarobotics.navigation.VoxelMap`;
- `cudarobotics.navigation.Esdf2D`;
- one pipeline object connecting odometry, mapping, ESDF, and CUDA MPPI;
- a `cudarobotics-cudanav` CLI for recorded point clouds or rosbag-derived
  inputs.

The first supported workflow must accept a documented public dataset input and
produce:

- a trajectory;
- map and ESDF outputs;
- timing and accuracy metrics;
- a machine-readable manifest;
- a concise visual or replay artifact.

v0.4 exit gates:

- Python API and CLI use the existing tested CUDA cores rather than duplicate
  implementations;
- one command runs the complete pipeline with fixed seeds;
- CPU/no-GPU tests validate schemas and failure handling;
- GPU tests retain hardware, driver, CUDA, latency, memory, and accuracy data;
- CPython 3.10 and 3.12 manylinux wheels contain the complete supported API;
- source distribution, wheels, documentation, and examples report the same
  version;
- multi-GPU remains optional and is not a release blocker.

Website expansion is paused during this epic. Update Markdown documentation
and package examples first; resume site changes only after the Python API and
CLI contracts are stable.

## Epic 5: v1.0 and Publications

Objective: publish a usable toolkit and two reproducible, narrowly scoped paper
packages.

The pre-tag Build, Python package, and ROS 2 Jazzy attestations are combined
by `.github/workflows/v1-release-candidate.yml`. The resulting portable bundle
recomputes every gate, requires three distinct successful runs on the same
`refs/heads/master` commit, binds the Python 1.0.0 artifact/source manifest,
and rejects undeclared or edited files before the immutable tag is created.

v1.0 release gates:

- a new user can install and reach the main demo within 15 minutes;
- Python wheel/source, ROS 2 launch, Colab, Docker, and the documentation site
  describe the same supported release;
- compatibility and support policies are explicit;
- all headline results are regenerated from the release candidate;
- the end-to-end CudaNav gate passes.

The cross-surface versions, commands, and evidence slots are authoritative in
[`v1_support_matrix.json`](v1_support_matrix.json). Its validator may report
`valid: true` during development, but release requires `ready: true`:

```bash
python3 scripts/validate_v1_support_matrix.py \
  --require-ready \
  --evidence-bundle build/v1_release_bundle/bundle.json \
  --release-commit "$(git rev-list -n 1 v1.0.0)"
```

The human operator sequence and candidate release text are frozen in
[`releases/v1.0.0_release_checklist.md`](releases/v1.0.0_release_checklist.md)
and [`releases/v1.0.0_notes.md`](releases/v1.0.0_notes.md).

The four tag-bound attestations are assembled after the immutable tag is
executed and retained as a GitHub Release evidence bundle. This explicitly
avoids trying to commit a file containing the tag commit hash back into the
same immutable Git tree.

The release-evidence sequence is now:

```text
immutable v1.0.0 tag
  -> fresh-clone/no-cache quickstart attestation
  -> CudaNav ROS 2 + real-bag + physical-GPU attestation
  -> published GHCR digest + GPU smoke attestation
  -> deployed public documentation attestation
  -> content-bound bundle.json
  -> validate_v1_support_matrix.py --require-ready
  -> canonical evidence ZIP + SHA-256 sidecar
  -> upload, re-download, and validate the GitHub Release bytes
```

No producer can turn missing Docker, GPU, ROS 2, HTTP, or source
commit evidence into a skipped pass. The post-tag bundle also rejects
undeclared files and non-canonical attestation names; the archive validator
rejects unsafe paths, duplicate members, checksum or CRC failure, oversized
payloads, and non-canonical ZIP metadata.

Publication split:

1. The CudaRobotics systems paper covers the end-to-end stack,
   reproducibility, consumer-GPU deployment, APIs, and benchmark system.
2. The Diff-MPPI paper leads with contact-rich manipulation, includes smooth
   tasks as negative controls, and retains CDF-MPPI and matched-time
   comparisons.

Both publication paths now end in canonical ZIP/checksum artifacts. The
contact-rich path can package a clean ready anonymous ledger after venue/URL
selection. The systems path additionally requires every submission claim,
the final status table, local manuscript links, and every ledger artifact to
pass before its assembler will produce any bundle.

Algorithm counts, benchmark tables, and hardware claims in the paper should be
generated from repository manifests rather than maintained by hand.

## Execution Order

Work proceeds in dependency order:

1. publish v0.3.0 with the integrated CudaNav evidence;
2. define the minimal Python CudaNav API and CLI contracts;
3. connect the existing odometry, mapping, ESDF, and MPPI cores;
4. add the public-dataset quickstart and fixed-seed evidence;
5. publish v0.4.0 without making multi-GPU a blocker;
6. promote the proven surface toward v1.0 and freeze the paper artifacts.

An epic may prototype a downstream interface early, but it cannot be declared
complete before its upstream exit gates pass.
