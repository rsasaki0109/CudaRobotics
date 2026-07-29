# v1.0 Support Matrix Contract

[`v1_support_matrix.json`](v1_support_matrix.json) is the machine-readable
contract connecting the Python source and wheels, ROS 2 launch, Docker image,
Colab notebook, and documentation site.

The current status is deliberately `development`. The matrix records the
versions that exist now without relabelling them as v1.0. It becomes
release-ready only when:

- the Python and all eight ROS packages are versioned `1.0.0`;
- `v1.0.0` is the published supported release;
- a fresh-machine main-demo run reaches its JSON result within 900 seconds;
- the strict CudaNav release evidence and Docker GPU evidence pass;
- the documentation deployment is bound to the same release commit.

The target 15-minute path is:

```bash
docker build --pull --no-cache -f docker/Dockerfile -t cudarobotics .
docker run --rm --gpus all -v "$PWD/out:/out" cudarobotics cudanav
```

This source-build path currently provides the short integration smoke. It
does not replace the retained 10-minute CudaNav release run.

Validate consistency without claiming readiness:

```bash
python3 scripts/validate_v1_support_matrix.py
```

Acquire the 900-second evidence on a clean Docker/NVIDIA host. The timer starts
before a depth-one clone and includes a `--pull --no-cache` image build plus
the CudaNav run:

```bash
python3 scripts/run_v1_quickstart.py \
  --output-dir build/v1_quickstart \
  --ref v1.0.0 \
  --profile release
python3 scripts/validate_v1_quickstart.py \
  build/v1_quickstart \
  --profile release --commit <full-40-character-commit>
```

The runner refuses an existing local `cudarobotics` image. Its manifest binds
the clone/build/run logs, Docker image ID, GPU/driver identity, support matrix,
and CudaNav JSON/log. Missing Docker or NVIDIA tooling is an unavailable gate,
not a skipped pass.

At release freeze, require every readiness field:

```bash
python3 scripts/validate_v1_support_matrix.py --require-ready
```
