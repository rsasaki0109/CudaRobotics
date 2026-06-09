cmake_minimum_required(VERSION 3.18)
project(cuda_mppi_controller LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(ROS_DEPS
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  geometry_msgs
  pluginlib
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_library(${PROJECT_NAME} SHARED
  src/cuda_mppi_controller.cpp
  src/mppi_gpu.cu
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} ${ROS_DEPS})
target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::curand)
set_target_properties(${PROJECT_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Standalone GPU optimizer demo/test (no nav2 needed at runtime):
# synthetic costmap corridor, prints solve time per control cycle.
add_executable(mppi_gpu_standalone test/mppi_gpu_standalone.cpp)
target_link_libraries(mppi_gpu_standalone ${PROJECT_NAME})

# Loads the plugin through pluginlib the same way controller_server does.
add_executable(plugin_load_test test/plugin_load_test.cpp)
ament_target_dependencies(plugin_load_test ${ROS_DEPS})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS mppi_gpu_standalone plugin_load_test
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(FILES cuda_mppi_plugin.xml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

pluginlib_export_plugin_description_file(nav2_core cuda_mppi_plugin.xml)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${ROS_DEPS})
ament_package()
