#!/usr/bin/env bash
# Copy the minimal CUDA core consumed by python/ into python/core/ for sdist/wheels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE="${ROOT}/python/core"
SRC="${ROOT}/src"
INC="${ROOT}/include"

mkdir -p "${CORE}/src" "${CORE}/include/cuda_mppi_controller" "${CORE}/include/cudarobotics"

for cu in mppi_gpu filterreg_gpu sinkhorn_gpu fgr_gpu bcpd_gpu robust_treg_gpu robust_p2plane_gpu \
  gpu_kiss_icp voxel_mapping_gpu esdf_2d_gpu cudanav_costmap_gpu; do
  install -m 0644 "${SRC}/${cu}.cu" "${CORE}/src/${cu}.cu"
done

install -m 0644 "${INC}/cuda_check.cuh" "${CORE}/include/cuda_check.cuh"
# Private headers of the CUDA KISS-ICP core (built with CUDAROBOTICS_KISS_ICP_CORE_ONLY).
for header in kiss_icp_spatial.cuh kiss_icp_reduction.cuh kiss_icp_host_map.hpp \
  kiss_icp_downsample.hpp kiss_icp_order.cuh kiss_icp_normal_cache.cuh; do
  install -m 0644 "${INC}/${header}" "${CORE}/include/${header}"
done
install -m 0644 "${INC}/cuda_mppi_controller/mppi_gpu.hpp" "${CORE}/include/cuda_mppi_controller/mppi_gpu.hpp"
install -m 0644 "${INC}/cudarobotics/"*.hpp "${CORE}/include/cudarobotics/"

python3 "${ROOT}/scripts/python_source_provenance.py"
echo "Synced python/core from ${ROOT}/src and ${ROOT}/include"
