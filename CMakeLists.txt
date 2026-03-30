cmake_minimum_required(VERSION 3.18)
project(CPP_ROBOTICS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#############################
#Dependencies
#############################

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(./include)

##############################
#Localization
##############################

# add_executable(ekf src/extended_kalman_filter.cpp)
# target_link_libraries(ekf ${OpenCV_LIBS} )

add_executable(pf src/particle_filter.cu)
target_link_libraries(pf ${OpenCV_LIBS})
target_compile_options(pf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(fastslam1 src/fastslam1.cu)
target_link_libraries(fastslam1 ${OpenCV_LIBS})
target_compile_options(fastslam1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

##############################
#PathPlanning
##############################

# add_executable(dijkstra src/dijkstra.cpp)
# target_link_libraries(dijkstra ${OpenCV_LIBS} )

add_executable(dijkstra_cuda src/dijkstra.cu)
target_link_libraries(dijkstra_cuda ${OpenCV_LIBS})

# add_executable(astar src/a_star.cpp)
# target_link_libraries(astar ${OpenCV_LIBS} )

add_executable(astar_cuda src/a_star.cu)
target_link_libraries(astar_cuda ${OpenCV_LIBS})

# add_executable(rrt src/rrt.cpp)
# target_link_libraries(rrt ${OpenCV_LIBS} )

add_executable(rrt_cuda src/rrt.cu)
target_link_libraries(rrt_cuda ${OpenCV_LIBS})

# add_executable(rrtstar src/rrt_star.cpp)
# target_link_libraries(rrtstar ${OpenCV_LIBS} )

add_executable(rrtstar_cuda src/rrt_star.cu)
target_link_libraries(rrtstar_cuda ${OpenCV_LIBS})

add_executable(rrtstar_rs_cuda src/rrt_star_reeds_shepp.cu)
target_link_libraries(rrtstar_rs_cuda ${OpenCV_LIBS})

add_executable(informed_rrtstar_cuda src/informed_rrt_star.cu)
target_link_libraries(informed_rrtstar_cuda ${OpenCV_LIBS})

add_executable(rrtstar_3d_cuda src/rrt_star_3d.cu)
target_link_libraries(rrtstar_3d_cuda ${OpenCV_LIBS})

add_executable(dwa src/dynamic_window_approach.cu)
target_link_libraries(dwa ${OpenCV_LIBS})

# add_executable(mptg src/model_predictive_trajectory_generator.cpp)
# target_link_libraries(mptg ${OpenCV_LIBS})

# add_executable(slp src/state_lattice_planner.cpp)
# target_link_libraries(slp ${OpenCV_LIBS})

add_executable(slp_cuda src/state_lattice_planner.cu)
target_link_libraries(slp_cuda)

# add_executable(csp src/cubic_spline_planner.cpp)
# target_link_libraries(csp ${OpenCV_LIBS})

add_executable(frenet src/frenet_optimal_trajectory.cu)
target_link_libraries(frenet ${OpenCV_LIBS})
target_compile_options(frenet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(potential_field src/potential_field.cu)
target_link_libraries(potential_field ${OpenCV_LIBS})

add_executable(potential_field_3d src/potential_field_3d.cu)
target_link_libraries(potential_field_3d ${OpenCV_LIBS})

add_executable(prm_cuda src/prm.cu)
target_link_libraries(prm_cuda ${OpenCV_LIBS})

add_executable(voronoi_road_map src/voronoi_road_map.cu)
target_link_libraries(voronoi_road_map ${OpenCV_LIBS})


##############################
#PathTracking
##############################

# add_executable(lqr src/lqr_steer_control.cpp)
# target_link_libraries(lqr ${OpenCV_LIBS})

# add_executable(lqr_full src/lqr_speed_steer_control.cpp)
# target_link_libraries(lqr_full ${OpenCV_LIBS})

#add_executable(mpc src/model_predictive_control.cpp)
#target_link_libraries(mpc ${OpenCV_LIBS} ipopt)

##############################
#Benchmark
##############################

add_executable(benchmark_pf src/benchmark_pf.cu)
target_compile_options(benchmark_pf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(benchmark_dwa src/benchmark_dwa.cu)
