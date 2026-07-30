# v1.0 Support Matrix Contract

[`v1_support_matrix.json`](v1_support_matrix.json) is the machine-readable
contract connecting the Python source and wheels, ROS 2 launch, Docker image,
Colab notebook, and documentation site.

The current status is deliberately `development`. The Python source and all
supported ROS 2 packages now carry the `1.0.0` release-candidate version, but
that version alignment alone is not a release claim. The matrix becomes
release-ready only when:

- the Python and all eight ROS packages are versioned `1.0.0`;
- `v1.0.0` is the published supported release;
- a fresh-machine main-demo run reaches its JSON result within 900 seconds;
- the strict CudaNav release evidence and Docker GPU evidence pass;
- the documentation deployment is bound to the same release commit.

The Colab surface and its in-notebook clone command are pinned to the
immutable `v1.0.0` tag. That URL intentionally becomes resolvable only after
the tag is published; it must not fall back to the moving `master` branch.

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

After the release-profile directory passes, publish its content-bound matrix
attestation. The command independently revalidates the retained manifest and
every declared artifact before writing anything:

```bash
python3 scripts/publish_v1_quickstart_attestation.py \
  --evidence-dir build/v1_quickstart \
  --output build/v1_evidence_inputs/v1_quickstart_release.json
```

The publisher prints the content reference later incorporated into the
post-tag release bundle.

After `docker-image.yml` publishes the immutable `v1.0.0` tag, dispatch the
separate self-hosted GPU gate:

```bash
gh workflow run v1-docker-gpu-evidence.yml \
  --ref v1.0.0 -f tag=v1.0.0
```

The runner pulls the published image rather than rebuilding it, requires the
OCI source revision to equal the checked-out tag commit, records the GHCR
digest and physical GPU UUID, runs `cudanav`, and uploads the content-bound
logs, result, manifest, and `v1_docker_gpu_release.json`. Copy the downloaded
attestation without editing it.

Deploy and re-fetch the documentation with the preserved gallery tree:

```bash
gh workflow run v1-docs-deploy.yml \
  --ref v1.0.0 -f tag=v1.0.0
```

The workflow starts from the complete existing `gh-pages` tree, replaces only
its `/docs/` subtree, writes a tag/commit deployment manifest, deploys the
complete tree through GitHub Pages, and then re-fetches the public index,
install, Nav2, and release-manifest URLs. The uploaded
`v1_docs_release.json` is rejected unless all public bodies describe the same
tag commit.

## Release attestation references

The four evidence values are not inline pass/fail claims. Each reference has
exactly:

```json
{
  "path": "<attestation>.json",
  "sha256": "<64 lowercase hex characters>"
}
```

The validator reopens and hashes each referenced file, requires its declared
mode, clean 40-character commit, `v1.0.0` version and tag, upstream payload
hash, successful validator checks, and mode-specific details. The four modes
cover the fresh-clone 15-minute run, CudaNav ROS 2 systems release, published
GPU container, and deployed documentation. They must all identify one release
commit. A legacy inline object containing only `status`, `version`, and
`git_commit` is rejected.

The evidence is necessarily produced after testing the immutable tag, so it
cannot be committed back into that same Git tree. Assemble the four downloaded
attestations into a portable post-tag bundle instead:

```bash
RELEASE_COMMIT="$(git rev-list -n 1 v1.0.0)"
python3 scripts/assemble_v1_release_bundle.py \
  --quickstart build/v1_inputs/quickstart.json \
  --cudanav build/v1_inputs/cudanav.json \
  --docker-gpu build/v1_inputs/docker_gpu.json \
  --documentation build/v1_inputs/documentation.json \
  --output-dir build/v1_release_bundle \
  --commit "$RELEASE_COMMIT"
python3 scripts/validate_v1_support_matrix.py \
  --require-ready \
  --evidence-bundle build/v1_release_bundle/bundle.json \
  --release-commit "$RELEASE_COMMIT"
python3 scripts/archive_v1_release_bundle.py \
  build/v1_release_bundle/bundle.json \
  --output build/cudarobotics-v1.0.0-evidence.zip \
  --commit "$RELEASE_COMMIT"
python3 scripts/validate_v1_release_archive.py \
  build/cudarobotics-v1.0.0-evidence.zip \
  --checksum build/cudarobotics-v1.0.0-evidence.zip.sha256 \
  --commit "$RELEASE_COMMIT"
```

The bundle validator requires its exact five-file inventory, all four hashes,
all four subject commits, and the bundle commit to equal the immutable
`v1.0.0` commit. The canonical ZIP uses sorted members, fixed timestamps and
permissions, and stored payloads. Attach it and its checksum, then re-download
and validate the public bytes:

```bash
gh release upload v1.0.0 \
  build/cudarobotics-v1.0.0-evidence.zip \
  build/cudarobotics-v1.0.0-evidence.zip.sha256
gh release download v1.0.0 \
  --pattern 'cudarobotics-v1.0.0-evidence.zip*' \
  --dir build/v1_downloaded_evidence
python3 scripts/validate_v1_release_archive.py \
  build/v1_downloaded_evidence/cudarobotics-v1.0.0-evidence.zip \
  --checksum build/v1_downloaded_evidence/cudarobotics-v1.0.0-evidence.zip.sha256 \
  --commit "$RELEASE_COMMIT"
```

This avoids an impossible Git self-reference while keeping the tagged source
and its post-tag evidence inseparable and independently downloadable.
