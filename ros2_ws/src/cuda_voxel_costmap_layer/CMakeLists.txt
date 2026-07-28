cmake_minimum_required(VERSION 3.16)
project(cuda_voxel_costmap_layer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

add_library(cuda_voxel_costmap_layer SHARED
  src/cuda_voxel_costmap_layer.cpp)
target_include_directories(cuda_voxel_costmap_layer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
ament_target_dependencies(cuda_voxel_costmap_layer
  geometry_msgs nav2_costmap_2d nav_msgs pluginlib rclcpp)

pluginlib_export_plugin_description_file(
  nav2_costmap_2d cuda_voxel_costmap_plugin.xml)

add_executable(plugin_load_test test/plugin_load_test.cpp)
ament_target_dependencies(plugin_load_test
  nav2_costmap_2d pluginlib)

install(TARGETS cuda_voxel_costmap_layer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS plugin_load_test
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(FILES cuda_voxel_costmap_plugin.xml README.md
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(occupancy_bridge_test test/occupancy_bridge_test.cpp)
  if(TARGET occupancy_bridge_test)
    target_link_libraries(occupancy_bridge_test cuda_voxel_costmap_layer)
    ament_target_dependencies(
      occupancy_bridge_test geometry_msgs nav2_costmap_2d nav_msgs rclcpp)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(cuda_voxel_costmap_layer)
ament_export_dependencies(geometry_msgs nav2_costmap_2d nav_msgs pluginlib rclcpp)
ament_package()
