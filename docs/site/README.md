# CudaRobotics docs site source

This directory contains a static documentation site that can be copied into the
existing `gh-pages` branch under `/docs/` without replacing the animated demo
gallery at the site root.

Preview locally:

```bash
python3 -m http.server 8080 --directory docs/site
```

Then open `http://localhost:8080/`.

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
