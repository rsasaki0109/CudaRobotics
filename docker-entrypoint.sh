#!/bin/bash
set -e
mkdir -p build

case "${1}" in
  benchmark)
    echo "=== Diff-MPPI Benchmark ==="
    ./bin/benchmark_diff_mppi --quick --csv build/benchmark_diff_mppi.csv
    python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi.csv
    ;;
  benchmark-7dof)
    echo "=== 7-DOF Manipulator Benchmark ==="
    ./bin/benchmark_diff_mppi_manipulator_7dof --seed-count 4 --csv build/benchmark_7dof.csv
    ;;
  test)
    echo "=== Running Tests ==="
    ./bin/test_autodiff && echo "test_autodiff: PASSED"
    ./bin/test_gpu_mlp && echo "test_gpu_mlp: PASSED"
    python3 scripts/validate_design_workflow.py && echo "validate_design_workflow: PASSED"
    python3 scripts/check_design_regressions.py && echo "check_design_regressions: PASSED"
    echo "=== All Tests Passed ==="
    ;;
  bash)
    exec /bin/bash "${@:2}"
    ;;
  *)
    exec "$@"
    ;;
esac
