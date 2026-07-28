cmake_minimum_required(VERSION 3.18)
project(cuda_esdf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(cuda_robotics_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(ROS_DEPS
  cuda_robotics_msgs
  diagnostic_msgs
  geometry_msgs
  lifecycle_msgs
  nav_msgs
  rclcpp
  rclcpp_components
  rclcpp_lifecycle)

add_library(cuda_esdf_component SHARED
  src/cuda_esdf_node.cpp
  "${REPO_ROOT}/src/esdf_2d_gpu.cu")
target_include_directories(cuda_esdf_component
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${REPO_ROOT}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cuda_esdf_component CUDA::cudart)
ament_target_dependencies(cuda_esdf_component ${ROS_DEPS})
rclcpp_components_register_nodes(cuda_esdf_component
  "cuda_esdf::CudaEsdfNode")

add_executable(cuda_esdf_node src/main.cpp)
target_link_libraries(cuda_esdf_node cuda_esdf_component)
ament_target_dependencies(cuda_esdf_node rclcpp)

install(TARGETS cuda_esdf_component cuda_esdf_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(FILES README.md DESTINATION share/${PROJECT_NAME})
install(FILES "${REPO_ROOT}/include/cudarobotics/esdf_2d_gpu.hpp"
  DESTINATION include/cudarobotics)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(lifecycle_configuration_test
    test/lifecycle_configuration_test.cpp)
  if(TARGET lifecycle_configuration_test)
    target_link_libraries(lifecycle_configuration_test cuda_esdf_component)
    ament_target_dependencies(
      lifecycle_configuration_test lifecycle_msgs rclcpp)
  endif()
endif()

ament_export_include_directories(include)
ament_export_dependencies(${ROS_DEPS})
ament_package()
