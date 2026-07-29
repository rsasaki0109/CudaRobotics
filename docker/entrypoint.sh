#!/usr/bin/env bash
# Demo entrypoint for the CudaRobotics image.
#   benchmark [wall_gap|narrow_corridor|u_turn|all]  (default)
#   standalone [K] [diff|ackermann|omni|footprint]
#   cudanav [traversals] [mission_timeout_sec]
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
  cudanav)
    traversals="${2:-1}"
    mission_timeout="${3:-90.0}"
    summary="$OUT/cudanav_closed_loop.json"
    launch_log="$OUT/cudanav_closed_loop.log"
    mkdir -p "$OUT"
    rm -f -- "$summary"
    echo "== End-to-end CudaNav closed-loop smoke =="
    echo "GPU KISS-ICP -> voxel map -> ESDF -> Nav2 CUDA MPPI -> simulator"
    ros2 launch cuda_nav_bringup cudanav_closed_loop.launch.py \
      "output_path:=$summary" \
      "traversal_count:=$traversals" \
      "mission_timeout_sec:=$mission_timeout" \
      >"$launch_log" 2>&1 &
    launch_pid=$!
    cleanup() {
      if kill -0 "$launch_pid" 2>/dev/null; then
        kill -INT "$launch_pid" 2>/dev/null || true
        wait "$launch_pid" 2>/dev/null || true
      fi
    }
    trap cleanup EXIT INT TERM
    for _ in $(seq 1 480); do
      [ -s "$summary" ] && break
      if ! kill -0 "$launch_pid" 2>/dev/null; then
        wait "$launch_pid"
        echo "CudaNav launch exited before writing $summary" >&2
        exit 1
      fi
      sleep 0.25
    done
    if [ ! -s "$summary" ]; then
      echo "CudaNav did not produce a summary within 120 seconds" >&2
      exit 1
    fi
    cleanup
    trap - EXIT INT TERM
    python3 - "$summary" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
required = {
    "schema_version",
    "success",
    "smoke_pass",
    "collision_count",
    "odometry_drift_percent",
    "command_deadline_miss_rate",
}
missing = sorted(required - set(summary))
if missing:
    raise SystemExit(f"CudaNav summary missing fields: {missing}")
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["schema_version"] != 1 or not summary["smoke_pass"]:
    raise SystemExit("CudaNav closed-loop smoke failed")
PY
    echo "Summary: $summary"
    echo "Launch log: $launch_log"
    ;;
  *)
    exec "$@"
    ;;
esac
