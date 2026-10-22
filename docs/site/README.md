# CudaRobotics docs site source

This directory contains a static documentation site that can be copied into the
existing `gh-pages` branch under `/docs/` without replacing the animated demo
gallery at the site root.

Preview locally:

```bash
python3 -m http.server 8080 --directory docs/site
```

Then open `http://localhost:8080/`.

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
