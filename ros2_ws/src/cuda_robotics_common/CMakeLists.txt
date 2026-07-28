cmake_minimum_required(VERSION 3.18)
project(cuda_robotics_common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(cuda_robotics_common SHARED
  src/pointcloud_decoder.cpp
  src/pointcloud_transform.cpp)
target_include_directories(cuda_robotics_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
ament_target_dependencies(cuda_robotics_common geometry_msgs sensor_msgs)

install(TARGETS cuda_robotics_common
  EXPORT export_cuda_robotics_common
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(pointcloud_decoder_test test/pointcloud_decoder_test.cpp)
  if(TARGET pointcloud_decoder_test)
    target_link_libraries(pointcloud_decoder_test cuda_robotics_common)
  endif()
  ament_add_gtest(pointcloud_transform_test test/pointcloud_transform_test.cpp)
  if(TARGET pointcloud_transform_test)
    target_link_libraries(pointcloud_transform_test cuda_robotics_common)
  endif()
endif()

ament_export_targets(export_cuda_robotics_common HAS_LIBRARY_TARGET)
ament_export_dependencies(geometry_msgs sensor_msgs)
ament_package()
