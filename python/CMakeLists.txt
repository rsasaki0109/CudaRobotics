cmake_minimum_required(VERSION 3.18)
project(cudarobotics_python LANGUAGES CXX CUDA)

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

set(CUDAROBOTICS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

nanobind_add_module(_cudarobotics
  src/cudarobotics/bindings.cpp
  ${CUDAROBOTICS_ROOT}/src/mppi_gpu.cu
)

target_include_directories(_cudarobotics PRIVATE
  ${CUDAROBOTICS_ROOT}/include
)
target_link_libraries(_cudarobotics PRIVATE CUDA::cudart CUDA::curand)
target_compile_options(_cudarobotics PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(_cudarobotics PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)

install(TARGETS _cudarobotics LIBRARY DESTINATION cudarobotics)
