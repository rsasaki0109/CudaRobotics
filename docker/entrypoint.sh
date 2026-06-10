#!/usr/bin/env bash
# Demo entrypoint for the cuda_mppi_controller image.
#   benchmark [wall_gap|narrow_corridor|u_turn|all]  (default)
#   standalone [K] [diff|ackermann|omni|footprint]
#   anything else is exec'd verbatim (e.g. bash)
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ws/install/setup.bash
BIN=/ws/install/cuda_mppi_controller/lib/cuda_mppi_controller
OUT="${OUT_DIR:-/out}"

if [ ! -e /dev/nvidiactl ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: no NVIDIA device visible inside the container."
  echo "         Run with: docker run --rm --gpus all <image>"
fi

case "${1:-benchmark}" in
  benchmark)
    scenario="${2:-wall_gap}"
    mkdir -p "$OUT"
    echo "== plugin_load_test (loads the plugin exactly as controller_server does) =="
    "$BIN/plugin_load_test"
    echo
    echo "== controller_benchmark: Nav2 CPU MPPI vs CUDA GPU MPPI ($scenario) =="
    "$BIN/controller_benchmark" "$OUT" "$scenario"
    echo
    for f in "$OUT"/*/summary.csv; do
      [ -e "$f" ] || continue
      echo "--- $f ---"
      column -s, -t < "$f" 2>/dev/null || cat "$f"
    done
    echo
    echo "CSV + per-run trajectories are under $OUT (mount it with -v \$PWD/out:/out to keep them)."
    ;;
  standalone)
    exec "$BIN/mppi_gpu_standalone" "${2:-16384}" "${3:-diff}"
    ;;
  *)
    exec "$@"
    ;;
esac
