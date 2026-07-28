cmake_minimum_required(VERSION 3.18)
project(cuda_kiss_icp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(ROS_DEPS
  diagnostic_msgs
  geometry_msgs
  lifecycle_msgs
  nav_msgs
  rclcpp
  rclcpp_components
  rclcpp_lifecycle
  sensor_msgs
  tf2
  tf2_ros
)

add_library(cuda_kiss_icp_component SHARED
  src/cuda_kiss_icp_node.cpp
  src/pointcloud_decoder.cpp
  src/pointcloud_transform.cpp
  "${REPO_ROOT}/src/gpu_kiss_icp.cu"
)
target_compile_definitions(cuda_kiss_icp_component
  PRIVATE CUDAROBOTICS_KISS_ICP_CORE_ONLY)
target_include_directories(cuda_kiss_icp_component
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${REPO_ROOT}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(cuda_kiss_icp_component PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(cuda_kiss_icp_component CUDA::cudart)
ament_target_dependencies(cuda_kiss_icp_component ${ROS_DEPS})
rclcpp_components_register_nodes(cuda_kiss_icp_component
  "cuda_kiss_icp::CudaKissIcpNode")

add_executable(cuda_kiss_icp_node src/main.cpp)
target_link_libraries(cuda_kiss_icp_node cuda_kiss_icp_component)
ament_target_dependencies(cuda_kiss_icp_node rclcpp)

install(TARGETS cuda_kiss_icp_component cuda_kiss_icp_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(FILES "${REPO_ROOT}/include/cudarobotics/kiss_icp_gpu.hpp"
  DESTINATION include/cudarobotics)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(pointcloud_decoder_test
    test/pointcloud_decoder_test.cpp
    src/pointcloud_decoder.cpp)
  if(TARGET pointcloud_decoder_test)
    target_include_directories(pointcloud_decoder_test PRIVATE include)
    ament_target_dependencies(pointcloud_decoder_test sensor_msgs)
  endif()
  ament_add_gtest(lifecycle_configuration_test
    test/lifecycle_configuration_test.cpp)
  if(TARGET lifecycle_configuration_test)
    target_link_libraries(lifecycle_configuration_test cuda_kiss_icp_component)
    ament_target_dependencies(lifecycle_configuration_test lifecycle_msgs rclcpp)
  endif()
  ament_add_gtest(pointcloud_transform_test
    test/pointcloud_transform_test.cpp
    src/pointcloud_transform.cpp)
  if(TARGET pointcloud_transform_test)
    target_include_directories(pointcloud_transform_test PRIVATE include)
    ament_target_dependencies(pointcloud_transform_test geometry_msgs)
  endif()
endif()

ament_export_include_directories(include)
ament_export_dependencies(${ROS_DEPS})
ament_package()
