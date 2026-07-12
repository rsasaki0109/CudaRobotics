cmake_minimum_required(VERSION 3.18)
project(cudarobotics_python LANGUAGES CXX CUDA)

option(CUDA_MPPI_FAST_MATH "Use CUDA fast math in the MPPI/registration extension (may change trajectories)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Monorepo dev reads repo-root sources; sdist/wheels use bundled python/core/.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../src/mppi_gpu.cu")
  set(CUDAROBOTICS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
else()
  set(CUDAROBOTICS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/core")
endif()

nanobind_add_module(_cudarobotics
  src/cudarobotics/bindings.cpp
  ${CUDAROBOTICS_ROOT}/src/mppi_gpu.cu
  ${CUDAROBOTICS_ROOT}/src/filterreg_gpu.cu
  ${CUDAROBOTICS_ROOT}/src/sinkhorn_gpu.cu
  ${CUDAROBOTICS_ROOT}/src/fgr_gpu.cu
  ${CUDAROBOTICS_ROOT}/src/bcpd_gpu.cu
  ${CUDAROBOTICS_ROOT}/src/robust_treg_gpu.cu
  ${CUDAROBOTICS_ROOT}/src/robust_p2plane_gpu.cu
)

target_include_directories(_cudarobotics PRIVATE
  ${CUDAROBOTICS_ROOT}/include
)
target_link_libraries(_cudarobotics PRIVATE CUDA::cudart CUDA::curand)
target_compile_options(_cudarobotics PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(CUDA_MPPI_FAST_MATH)
  set_source_files_properties(
    ${CUDAROBOTICS_ROOT}/src/mppi_gpu.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math"
  )
endif()
set_target_properties(_cudarobotics PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)

install(TARGETS _cudarobotics LIBRARY DESTINATION cudarobotics)
