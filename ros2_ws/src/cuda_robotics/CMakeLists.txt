cmake_minimum_required(VERSION 3.18)
project(cuda_robotics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(ROS_DEPS
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
)

# -------------------------------------------------------------------
# particle_filter_node
# -------------------------------------------------------------------
add_executable(particle_filter_node src/particle_filter_node.cu)
target_include_directories(particle_filter_node PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(particle_filter_node
  ${OpenCV_LIBS}
  CUDA::cudart
  CUDA::curand
)
target_compile_options(particle_filter_node PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
ament_target_dependencies(particle_filter_node ${ROS_DEPS})

# -------------------------------------------------------------------
# dwa_node
# -------------------------------------------------------------------
add_executable(dwa_node src/dwa_node.cu)
target_include_directories(dwa_node PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(dwa_node
  ${OpenCV_LIBS}
  CUDA::cudart
)
ament_target_dependencies(dwa_node ${ROS_DEPS})

# -------------------------------------------------------------------
# Install
# -------------------------------------------------------------------
install(TARGETS
  particle_filter_node
  dwa_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
