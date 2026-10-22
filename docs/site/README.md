# CudaRobotics docs site source

This directory contains a static documentation site that can be copied into the
existing `gh-pages` branch under `/docs/` without replacing the animated demo
gallery at the site root.

Preview locally:

```bash
python3 -m http.server 8080 --directory docs/site
```

Then open `http://localhost:8080/`.

## Comparison coverage: every executable browser demo

The inventory in `demos.json` covers all three pages that execute algorithms:
MPPI (`/demo/`), Particle Storm (`/particles/`) and Navigation Workbench
(`/navigation/`). Each has a GPU/CPU comparison panel. Repository checks discover
module-script demo pages and fail if one is missing from the inventory or lacks
comparison controls. The docs homepage links all panels in one place.

The legacy gallery's GIFs/research figures are recorded native runs, not live
browser computations. They cannot honestly be timed by measuring playback.
They remain preserved and are explicitly excluded from the live inventory;
native measurements remain in Results. This does not port all native CUDA
examples into the browser.

All live comparisons share `assets/benchmark.mjs`: three warmups, 15 measured
pairs with alternating order, yielding between calls, median/P10/P90, raw
samples, and full-output agreement. No GPU or failed setup means CPU-only
timings with a reason; output mismatch/zero-resolution timings suppress speedup.
Changing inputs or hiding the page cancels and invalidates comparison results.
The CPU baseline is JavaScript, not optimized native C++; GPU is WebGPU, not
CUDA. Sampling/input creation, compilation and allocation setup are excluded.

- MPPI: seed-311 trajectories at the selected count/state/local target.
  Compare costs only; RNG, softmin and rendering are excluded. Both backends
  receive identical Float32 controls. Tolerance is max(0.001 absolute, 0.1%
  relative), plus identical collision classification.
- Particle Storm: fixed seed-311 Float32 poses and the current 32-beam scan,
  at a separately selected equal count (512–65,536) on both backends. Compare
  likelihood only, using the live filter's shared WGSL likelihood function.
  Prediction, reduction, resampling and drawing are excluded. Tolerance is
  max(0.0001 absolute, 0.1% relative). It never compares different live-view
  populations. All N weights return (N × 4 bytes), unlike the live filter's
  32-byte summary.
- Navigation: the same fixed occupancy distance transform, as detailed below.

MPPI/particle CPU kernels run in a single module worker when available, with
an explicitly labeled main-thread fallback. Worker creation/input setup are
excluded; request/result-message overhead is timed. GPU calls include input
upload, synchronization and all-output readback. GPU resources are reused
within the comparison and released afterwards. Dedicated benchmark devices
do not overwrite live planner/filter state. The exported JSON contains input
parameters/seeds, exact scope, tolerances, raw timings and execution mode;
there are no device identities, external assets or uploads.

```bash
node --test scripts/test_browser_comparisons.mjs scripts/test_browser_demo_inventory.mjs
```

CPU tests include the actual worker message protocol via a Node worker bridge.
The full WebGPU test scripts also run matched comparisons with real compute
pipelines; `--compile-only` validates the new scoring entry point in CI.

## Navigation Workbench: map → clearance → route

Open `/navigation/` to test a warehouse layout or doorway against a circular
robot footprint. Draw/erase obstacles, place the robot and goal, set radius
and clearance margin, and simulate a route. Map edits stop motion immediately;
asynchronous field rebuilds discard stale results and replan from the current
robot position. The page never starts a mission automatically.

Two WebGPU passes compute exact separable squared distances to occupied cell
centres, then subtract a half-cell diagonal and cap by boundary clearance.
This produces a conservative obstacle-distance lower bound. The fixed map is
128 × 96 cells at 0.125 m/cell. CPU A* uses clearance-aware edge costs and
checked shortcuts; segment checks include a half-sampling-interval allowance.
The kinematic disk follower revalidates each complete planned segment against
the current field, avoiding inconsistent subsegment sampling near grid edges.
It is not a native CUDA/Nav2 controller or a physics/safety-certified system.

Save Map writes `cudarobotics.costmap.v1` JSON: binary occupancy rows from bottom
to top, origin `[0,0]`, metre coordinates, radius, margin and endpoints. Load
Map validates this exact format and grid; arbitrary ROS/PGM maps are not
accepted. Radius increments are 0.05 m and margin increments are 0.02 m,
matching the controls. Save Plan writes `cudarobotics.route.v1` with the complete input map,
waypoints, length, clearance lower bound, classified planning status, backend
and a separate simulation execution summary. Blocked plans can also be
exported. No file leaves the browser. These routes are experiment data, not
commands for a real robot.

```bash
node --test scripts/test_browser_navigation.mjs
node scripts/test_browser_navigation_webgpu.mjs --compile-only
node scripts/test_browser_navigation_webgpu.mjs
```

The latter two use the development-only Dawn setup described below. CI checks
shader compilation without a physical GPU. The full test compares all 12,288
GPU field values with the CPU reference across preset, sparse and fully
occupied maps, then checks routes. CPU tests independently check geometric
footprint clearance during full missions, narrow-doorway failures, blocked
routes, map round trips, import rejection, editing pauses and report export.
Browser visual/touch smoke checks remain separate from these DOM-stub tests.

### GPU / CPU comparison

The workbench's **Compare GPU / CPU** panel pauses motion and snapshots the
current map. It runs the same separable distance-field algorithm in the
JavaScript reference and WebGPU implementation, with three warmups per backend
and 15 measured pairs. CPU-first and GPU-first order alternates; the page yields
before every backend call. Median and interpolated P10/P90 wall times, raw
samples, and all-cell output agreement (absolute tolerance 1e-6 m) are retained.
The ratio is CPU median / GPU median, not a predetermined GPU speedup. A result
mismatch or zero-duration median suppresses the ratio. Unsupported devices get
CPU-only timings, never synthetic GPU measurements.

GPU timing includes upload, command submission, synchronization, field readback,
output copy and normal evaluator validation. GPU buffers are reused, while the
CPU reference allocates its temporary/output arrays per call. Compilation,
initialization, warmups, A* and drawing are excluded. This is a browser
end-to-end comparison with a single-threaded JavaScript CPU baseline, not a
CUDA-vs-optimized-native-CPU benchmark or a GPU kernel timer. The small fixed
grid can favour CPU execution; JIT, timer precision, power and other tabs can
affect results. The mission backend selector does not restrict the comparison:
it uses both backends if a WebGPU adapter is available.

Map/settings changes cancel an in-flight comparison, invalidate old results and
serialize replanning after outstanding GPU work finishes. Hiding the page also
cancels measurement. Save Comparison exports the full input map, methodology,
raw timings, percentiles, ratio and error statistics as
`cudarobotics.distance-benchmark.v1` JSON, with no device identity or uploads.

```bash
node --test scripts/test_browser_navigation_benchmark.mjs
```

## Particle Storm: GPU localization

Open `/particles/` for a GPU-heavy companion to the MPPI lab. Up to 65,536
pose hypotheses each transform up to 32 LiDAR endpoints (2,097,152 endpoint
tests per update; invalid poses skip scoring). Four compute passes perform prediction/likelihood scoring,
workgroup prefix scans, a group-level CDF/moment reduction, and stratified
resampling. Instanced sprites render the resulting GPU storage buffer
directly. Only a 32-byte posterior summary is read back each update.

The map and ground-truth LiDAR scan are generated on the CPU. The filter
receives scan ranges and odometry increments, never the ground-truth pose.
Teleporting the robot changes only the simulated ground truth; 3% global
proposals allow recovery. Scatter resets the filter's prior. The known-map
likelihood-field model is MCL, not SLAM or a direct CUDA build. A 512-particle
CPU preview is explicitly labeled when WebGPU is unavailable or fails.
Autoplay respects `prefers-reduced-motion`; paused users can start manually.

```bash
node --test scripts/test_browser_particles.mjs
```

With the development-only Dawn module configured as described below:

```bash
node scripts/test_browser_particles_webgpu.mjs --compile-only
node scripts/test_browser_particles_webgpu.mjs
```

The first command validates all compute and render pipelines without a GPU
and runs in CI. The second checks GPU/CPU likelihood parity (0.1% tolerance
with a 0.0001 absolute floor), posterior reductions, CDF resampling, rendered
pixels, 65,536-particle convergence and recovery after teleporting. It covers
partial workgroups as well as full-size dispatches. Browser smoke checks:
pause/resume, scatter, teleport, particle count and tolerance changes, CPU
preview, a narrow viewport, and a reduced-motion preference. Live update
latency includes compute, drawing and summary readback, not just kernel time.

## Interactive browser MPPI lab

Open `http://localhost:8080/demo/` or the deployed
[MPPI lab](https://rsasaki0109.github.io/CudaRobotics/docs/demo/).
The buildless JavaScript/WGSL demo runs entirely on the visitor's device.
WebGPU evaluates one trajectory per invocation; JavaScript handles seeded
sampling, stable softmin reduction, warm starts and Canvas rendering. If
WebGPU initialization, compilation or execution fails, it uses CPU rollout
costs automatically. HTTPS (or localhost) is required for WebGPU. No npm,
WASM threads, COOP/COEP headers or service worker are needed.
Before enabling WebGPU, a startup check compares 65 GPU rollouts per preset
against the CPU reference (including collision classification and a 0.1%
cost tolerance). A mismatch also selects the CPU fallback.

This follows the browser-local/static-host approach of
[visloc-rs PR #311](https://github.com/rsasaki0109/visloc-rs/pull/311), not its
VIO implementation. CUDA kernels cannot run directly on GitHub Pages; this
is an educational differential-drive MPPI port, not the native controller,
its full critics or a native-vs-browser benchmark. The slalom preset includes
guide waypoints; custom goals use local planning and may need a reset if
trapped. Unsafe commands stop the simulation instead of moving through walls.
There are no application uploads or telemetry.

Run the deterministic CPU tests with Node 22 or later:

```bash
node --test scripts/test_browser_mppi.mjs
```

Validate the real WGSL compiler and compute pipeline without a physical GPU
using Dawn's null backend (also run in repository checks and before deployment):

```bash
npm install --prefix /tmp/browser-mppi --no-audit --no-fund webgpu@0.6.2
export WEBGPU_MODULE=/tmp/browser-mppi/node_modules/webgpu/index.js
node scripts/test_browser_mppi_webgpu.mjs --compile-only
```

Omit `--compile-only` on a machine with a WebGPU adapter to verify GPU/CPU
cost parity, collision classification, buffer resizing and collision-free
goal arrival in all three presets. This uses the same compute implementation
as the page, but does not replace a browser UI/device smoke check. The Dawn
package is a development-only dependency, never shipped to Pages visitors.

Browser smoke check: open `/demo/`, start each preset, pause/reset, set a goal
by tapping the map or entering coordinates, change the sample count and
temperature, and select CPU explicitly. Check WebGPU on a supported HTTPS or
localhost browser as well as automatic CPU fallback on an unsupported device.
The latency display includes sampling, GPU transfers and CPU reduction; it
is not a GPU kernel timer. Each plan advances 0.1 simulation seconds regardless
of wall-clock latency.

The normal source deployment runs automatically when `docs/site/` or its
support-matrix inputs change on `master`. It can also be dispatched manually:

```bash
gh workflow run docs-deploy.yml --ref master
```

The workflow checks out the complete `gh-pages` gallery tree, replaces only
its `/docs/` subtree in the Pages artifact, and synchronizes repository-owned
`gif/*.gif` files into the gallery root without deleting other gallery assets.
It then deploys and re-fetches and byte-compares the public documentation
pages and all browser lab runtime files against the source commit.

For emergency manual deployment to the legacy Pages branch:

```bash
git worktree add /tmp/cudarobotics-pages origin/gh-pages
rsync -a --delete docs/site/ /tmp/cudarobotics-pages/docs/
cd /tmp/cudarobotics-pages
git add docs
git commit -m "Publish docs site"
git push origin HEAD:gh-pages
```

Do not run a Pages workflow that replaces the whole `gh-pages` branch unless
the gallery assets have been migrated first.

For the v1 release, use the manual `v1-docs-deploy.yml` workflow. It checks
out the complete existing `gh-pages` tree, updates `pages/docs/`, synchronizes
the repository-owned GIF inventory, adds a source-tag deployment manifest,
and deploys the preserved complete tree. It then re-fetches the public pages
and uploads content-bound release evidence:

```bash
gh workflow run v1-docs-deploy.yml \
  --ref v1.0.0 -f tag=v1.0.0
```
