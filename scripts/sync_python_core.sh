#!/usr/bin/env bash
# Copy the minimal CUDA core consumed by python/ into python/core/ for sdist/wheels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE="${ROOT}/python/core"
SRC="${ROOT}/src"
INC="${ROOT}/include"

mkdir -p "${CORE}/src" "${CORE}/include/cuda_mppi_controller" "${CORE}/include/cudarobotics"

for cu in mppi_gpu filterreg_gpu sinkhorn_gpu fgr_gpu bcpd_gpu robust_treg_gpu robust_p2plane_gpu; do
  install -m 0644 "${SRC}/${cu}.cu" "${CORE}/src/${cu}.cu"
done

install -m 0644 "${INC}/cuda_check.cuh" "${CORE}/include/cuda_check.cuh"
install -m 0644 "${INC}/cuda_mppi_controller/mppi_gpu.hpp" "${CORE}/include/cuda_mppi_controller/mppi_gpu.hpp"
install -m 0644 "${INC}/cudarobotics/"*.hpp "${CORE}/include/cudarobotics/"

python3 "${ROOT}/scripts/python_source_provenance.py"
echo "Synced python/core from ${ROOT}/src and ${ROOT}/include"
