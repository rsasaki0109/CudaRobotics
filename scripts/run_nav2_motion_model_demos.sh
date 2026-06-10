#!/usr/bin/env bash
# Run nav2 loopback two-waypoint missions for DiffDrive / Ackermann / Omni
# cuda_mppi_controller configs and render GIFs.
#
# Prereq: ros2_ws built, nav2_bringup available, PYTHONNOUSERSITE=1.
# Usage: ./scripts/run_nav2_motion_model_demos.sh [/tmp/nav2_motion] [ROS_DOMAIN_ID]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-/tmp/nav2_motion_models}"
DOMAIN="${2:-101}"
export ROS_DOMAIN_ID="${DOMAIN}"
export PYTHONNOUSERSITE=1
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

set +u
source /opt/ros/jazzy/setup.bash
source "${ROOT}/ros2_ws/install/setup.bash"
set -u

PKG_CFG="$(ros2 pkg prefix cuda_mppi_controller)/share/cuda_mppi_controller/config"

run_one() {
  local model="$1"
  local yaml="$2"
  local out="${OUT_ROOT}/${model}"
  local gif="cuda_mppi_nav2_loopback_${model}.gif"
  local subtitle="loopback sim, GPU MPPI ${model} K=8,192 @ 20 Hz"

  mkdir -p "${out}"
  pkill -f "[t]b3_loopback" 2>/dev/null || true
  pkill -f "[n]av2_bringup" 2>/dev/null || true
  sleep 2

  ros2 launch nav2_bringup tb3_loopback_simulation.launch.py \
    use_rviz:=False \
    params_file:="${yaml}" &
  local launch_pid=$!

  for _ in $(seq 1 90); do
    if ros2 lifecycle get /controller_server 2>/dev/null | grep -q "active \[3\]"; then
      break
    fi
    sleep 1
  done
  if ! ros2 lifecycle get /controller_server 2>/dev/null | grep -q "active \[3\]"; then
    kill "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
    echo "FAIL: nav2 never reached active for ${model}"
    return 1
  fi
  sleep 2

  if ! python3 "${ROOT}/scripts/run_nav2_loopback_demo.py" "${out}"; then
    kill "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
    echo "FAIL: ${model} loopback mission"
    return 1
  fi

  python3 "${ROOT}/scripts/render_nav2_loopback_demo.py" \
    "${out}" "${gif}" "${subtitle}"

  kill "${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
  sleep 2
  echo "OK: ${model} -> gif/${gif}"
}

run_one diff "${PKG_CFG}/nav2_loopback_demo.yaml"
run_one ackermann "${PKG_CFG}/nav2_loopback_demo_ackermann.yaml"
run_one omni "${PKG_CFG}/nav2_loopback_demo_omni.yaml"

echo "All motion-model loopback demos succeeded under ${OUT_ROOT}"
