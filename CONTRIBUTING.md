# Contributing

CudaRobotics is most useful when each contribution is small, runnable, and easy
to compare against a baseline. Prefer focused PRs over broad refactors.

## Good Contribution Shapes

- A self-contained CUDA demo in `src/` with a clear robotics workload.
- A benchmark variant that shares an existing scenario and reports CSV output.
- A reproduction note under `docs/` that states what was implemented, what was
  not implemented, and which result is positive or negative.
- A script or summary tool that makes existing benchmark output easier to
  regenerate or inspect.

## CUDA Demo Checklist

1. Add or update the executable in `CMakeLists.txt`.
2. Keep a CPU reference, analytic check, or deterministic validation path when
   practical.
3. Write outputs to `build/` or `gif/` rather than requiring manual inspection.
4. Add a short doc under `docs/` when the demo has non-obvious assumptions.
5. Avoid hidden dependencies beyond the repo's documented CUDA, CMake, OpenCV,
   and Eigen setup unless the feature is explicitly optional.

## MPPI Reproduction Checklist

Each reproduction note should include:

- target paper or idea
- implemented scope
- build and benchmark commands
- positive result
- negative or partial result
- paper-faithfulness caveats
- next step for a more faithful reproduction

Use [`docs/mppi_reproduction_zoo.md`](docs/mppi_reproduction_zoo.md) as the
index and add the new entry there when the benchmark is runnable.

## Local Checks

CPU and Python checks:

```bash
cmake -B build
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure --label-regex 'cpu|python' -j$(nproc)
```

CUDA smoke for the MPPI stack:

```bash
docker compose build cudarobotics
docker compose run --rm cudarobotics bash -lc 'python3 scripts/run_mppi_zoo_smoke.py --bin ./bin/benchmark_diff_mppi --out-dir build/mppi_zoo'
```

## PR Notes

- Mention the exact command used to verify the change.
- Include generated CSV or summary paths when the change is benchmark-related.
- Call out limitations. Negative results are welcome when they are measured and
  reproducible.
