# Python 0.2.0 Windows Release-Artifact Evidence

This is local release-candidate evidence, not a published release or a
substitute for the required CPython 3.10/3.12 manylinux artifacts.

## Result

- Source commit: `1273468f2fe429b4797f225a2a6e3339b535b935`
- Checkout: clean
- Embedded source digest: `4c56949ba56ad33f01512323175faf9b96929ddec3e9068d1010f0006248e6c9`
- Build: CPython 3.12.10, CUDA Toolkit 12.8, Windows amd64
- GPU: NVIDIA GeForce GTX 1660 Ti, driver 596.36
- Artifact verifier: pass for the sdist and wheel
- Fresh installed-wheel test: 13 passed, 1 optional PyTorch DLPack test skipped
- MPPI quickstart: pass, goal reached in 350 steps
- Registration quickstart: pass for RobustTreg, RobustP2Plane, Sinkhorn,
  FGR, and BCPD

The archive verifier reopened both artifacts, required the native extension
and bundled CUDA sources, and compared the embedded per-file source manifest
against the clean checkout. The fresh-venv tests imported
`cudarobotics` from `venv_1273468/Lib/site-packages`, not from the repository.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `cudarobotics-0.2.0.tar.gz` | 46,827 | `df457ec7982cbcbe4121a378f89414a668a611a7d2bd7bda9fe8a2d5357e8d84` |
| `cudarobotics-0.2.0-cp312-cp312-win_amd64.whl` | 787,896 | `a2c2a6179b4a22e15b7ea1fe206a7576dfb8eb4125667674699eafea773e802a` |

Machine-readable artifact and source hashes:
[`python_release_artifacts_windows_2026-07-29.json`](python_release_artifacts_windows_2026-07-29.json).

## Commands

```powershell
python -m build --outdir build/release_v0.2.0/dist_1273468 python
python scripts/verify_python_release_artifacts.py `
  --dist-dir build/release_v0.2.0/dist_1273468 `
  --json build/release_v0.2.0/python_artifacts_1273468.json `
  --require-clean

python -m venv build/release_v0.2.0/venv_1273468
build/release_v0.2.0/venv_1273468/Scripts/python.exe -m pip install `
  build/release_v0.2.0/dist_1273468/cudarobotics-0.2.0-cp312-cp312-win_amd64.whl pytest
build/release_v0.2.0/venv_1273468/Scripts/python.exe -m pytest python/tests -q
```

## Remaining release blockers

- Final CPython 3.10 and 3.12 manylinux x86_64 wheels are not yet available.
- The remote Build, Python package, and ROS 2 gates must all pass on the same
  final release-candidate commit.
- Tagging and publishing remain separate explicit actions.
