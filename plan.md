# CudaRobotics Plan / Handoff

Last updated: 2026-04-22 JST

This is the short handoff for the next coding session. The repo is currently on
`master`, synced with `origin/master`, and the working tree was clean before this
file update.

## Current State

- Main branch: `master`
- Latest merged work:
  - PR #1: reproducible benchmark suite runner
  - PR #2: point-cloud CLI demo workflow
- Recent merge commit: `b4f1bb0` from PR #2
- CI status for PR #2: passed
- Deferred branch: `feat/repro-report` exists but was intentionally not merged
  because the user wanted to move to higher-value work.

## What Is Done

### Reproducible Benchmark Suite

The repo now has a reproducible benchmark runner path that can be used to keep
experiments repeatable. This is useful infrastructure, but it is not the most
interesting product-facing value right now.

### Point-Cloud CLI Demo

`bin/benchmark_pointcloud` is now usable as a small CLI demo, not only as a fixed
benchmark.

Supported workflow:

```bash
./bin/benchmark_pointcloud --quick
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op voxel --leaf-size 0.8 --out build/sample_room_voxel.ply
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op ransac --plane-threshold 0.05 --out build/sample_room_plane.ply
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op normals --k 12 --out build/sample_room_normals.ply
```

Useful options:

- `--quick`
- `--input-only`
- `--op voxel|statistical|normals|ransac|gicp|all`
- `--out PATH`
- `--leaf-size`, `--k`, `--std-mul`, `--plane-threshold`, `--ransac-iters`, `--seed`

Bundled sample:

- `examples/pointcloud/sample_room.xyz`

Outputs:

- voxel/statistical/ransac: `.ply` or `.xyz` point rows
- normals: `.ply` with normal properties, or `.xyz` rows as `x y z nx ny nz`

## Validation Already Run

```bash
cmake --build build --target benchmark_pointcloud -j$(nproc)
./bin/benchmark_pointcloud --help
./bin/benchmark_pointcloud --quick --op voxel
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op voxel --leaf-size 0.8 --out build/sample_room_voxel.ply
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op ransac --plane-threshold 0.05 --ransac-iters 256 --out build/sample_room_plane.ply
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op normals --k 12 --out build/sample_room_normals.ply
./bin/benchmark_pointcloud --xyz examples/pointcloud/sample_room.xyz --input-only --op normals --k 12 --out build/sample_room_normals.xyz
ctest --test-dir build -L 'python|cpu' --output-on-failure
git diff --check
```

## Next High-Value Directions

### 1. Make The Point-Cloud Demo More Visual

Best next step if the goal is OSS value:

- add a small `examples/pointcloud/README.md`
- document before/after outputs for voxel, RANSAC plane, and normals
- add a script that runs the sample demo commands and writes all artifacts into
  `build/pointcloud_demo/`
- optionally add screenshots or a simple viewer command path if the repo already
  has a lightweight dependency available

Why: the project becomes easier to understand in 60 seconds. This is more visible
than another internal benchmark report.

### 2. Add A Real Registration Demo

The current external-input path does not write a GICP result because GICP needs a
source and target cloud. A valuable follow-up would add:

- `--source PATH`
- `--target PATH`
- `--op gicp`
- optional synthetic transform mode for the bundled sample cloud
- output of transformed source cloud and final transform matrix

Why: registration is one of the most recognizable point-cloud robotics workflows.

### 3. Keep Diff-MPPI As A Research Track

Diff-MPPI remains the strongest research story in the repo. Return to it when the
goal is paper value:

- final paper wording
- exact-time table cleanup
- standardized benchmark expansion
- stronger MuJoCo or hardware-facing task

Why: this is likely the best academic/research value, but it is heavier than a
quick OSS demo improvement.

### 4. Do Not Resume `feat/repro-report` By Default

The repro report renderer is useful, but the user explicitly called it less
interesting. Treat it as parked unless the next goal is experiment governance or
release engineering.

## Recommended Next Session

Start from `master`, pull, then build the next point-cloud user-facing feature:

```bash
git pull --ff-only
cmake --build build --target benchmark_pointcloud -j$(nproc)
```

Recommended branch name:

```bash
git switch -c feat/pointcloud-registration-demo
```

Recommended target:

Build a source/target GICP demo with generated transform output and a short
example README. That is the clearest next value after the point-cloud CLI merge.
