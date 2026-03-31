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

add_executable(ekf src/extended_kalman_filter.cpp)
target_link_libraries(ekf ${OpenCV_LIBS})

add_executable(pf src/particle_filter.cu)
target_link_libraries(pf ${OpenCV_LIBS})
target_compile_options(pf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(fastslam1 src/fastslam1.cu)
target_link_libraries(fastslam1 ${OpenCV_LIBS})
target_compile_options(fastslam1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(amcl src/amcl.cu)
target_link_libraries(amcl ${OpenCV_LIBS})
target_compile_options(amcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(emcl2 src/emcl2.cu)
target_link_libraries(emcl2 ${OpenCV_LIBS})
target_compile_options(emcl2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(pf_on_episode src/pf_on_episode.cu)
target_link_libraries(pf_on_episode ${OpenCV_LIBS})
target_compile_options(pf_on_episode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(graph_slam src/graph_slam.cu)
target_link_libraries(graph_slam ${OpenCV_LIBS})

add_executable(ndt src/ndt.cu)
target_link_libraries(ndt ${OpenCV_LIBS})

##############################
#PathPlanning
##############################

add_executable(dijkstra_cpp src/dijkstra.cpp)
target_link_libraries(dijkstra_cpp ${OpenCV_LIBS})

add_executable(dijkstra_cuda src/dijkstra.cu)
target_link_libraries(dijkstra_cuda ${OpenCV_LIBS})

add_executable(astar_cpp src/a_star.cpp)
target_link_libraries(astar_cpp ${OpenCV_LIBS})

add_executable(astar_cuda src/a_star.cu)
target_link_libraries(astar_cuda ${OpenCV_LIBS})

add_executable(rrt_cpp src/rrt.cpp)
target_link_libraries(rrt_cpp ${OpenCV_LIBS})

add_executable(rrt_cuda src/rrt.cu)
target_link_libraries(rrt_cuda ${OpenCV_LIBS})

add_executable(rrtstar_cpp src/rrt_star.cpp)
target_link_libraries(rrtstar_cpp ${OpenCV_LIBS})

add_executable(rrtstar_cuda src/rrt_star.cu)
target_link_libraries(rrtstar_cuda ${OpenCV_LIBS})

add_executable(rrtstar_rs_cuda src/rrt_star_reeds_shepp.cu)
target_link_libraries(rrtstar_rs_cuda ${OpenCV_LIBS})

add_executable(informed_rrtstar_cuda src/informed_rrt_star.cu)
target_link_libraries(informed_rrtstar_cuda ${OpenCV_LIBS})

add_executable(rrtstar_3d_cuda src/rrt_star_3d.cu)
target_link_libraries(rrtstar_3d_cuda ${OpenCV_LIBS})

add_executable(dwa_cpp src/dynamic_window_approach.cpp)
target_link_libraries(dwa_cpp ${OpenCV_LIBS})

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

add_executable(multi_robot_planner src/multi_robot_planner.cu)
target_link_libraries(multi_robot_planner ${OpenCV_LIBS})

add_executable(orca src/orca.cu)
target_link_libraries(orca ${OpenCV_LIBS})

add_executable(value_iteration src/value_iteration.cu)
target_link_libraries(value_iteration ${OpenCV_LIBS})

add_executable(stomp src/stomp.cu)
target_link_libraries(stomp ${OpenCV_LIBS})

add_executable(hybrid_astar src/hybrid_astar.cu)
target_link_libraries(hybrid_astar ${OpenCV_LIBS})

add_executable(rrt_connect src/rrt_connect.cu)
target_link_libraries(rrt_connect ${OpenCV_LIBS})

add_executable(social_force src/social_force.cu)
target_link_libraries(social_force ${OpenCV_LIBS})

add_executable(mppi src/mppi.cu)
target_link_libraries(mppi ${OpenCV_LIBS})

##############################
#Registration
##############################

add_executable(icp src/icp.cu)
target_link_libraries(icp ${OpenCV_LIBS})

##############################
#Mapping
##############################

add_executable(occupancy_grid src/occupancy_grid_map.cu)
target_link_libraries(occupancy_grid ${OpenCV_LIBS})

##############################
#PathTracking
##############################

add_executable(lqr src/lqr_steer_control.cpp)
target_link_libraries(lqr ${OpenCV_LIBS})

add_executable(lqr_full src/lqr_speed_steer_control.cpp)
target_link_libraries(lqr_full ${OpenCV_LIBS})

#add_executable(mpc src/model_predictive_control.cpp)
#target_link_libraries(mpc ${OpenCV_LIBS} ipopt)

##############################
#Benchmark
##############################

add_executable(benchmark_pf src/benchmark_pf.cu)
target_compile_options(benchmark_pf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(benchmark_rrt src/benchmark_rrt.cu)

add_executable(benchmark_dwa src/benchmark_dwa.cu)

add_executable(comparison_dwa src/comparison_dwa.cu)
target_link_libraries(comparison_dwa ${OpenCV_LIBS})

add_executable(comparison_multi_robot src/comparison_multi_robot.cu)
target_link_libraries(comparison_multi_robot ${OpenCV_LIBS})

add_executable(comparison_multi_robot_visual src/comparison_multi_robot_visual.cu)
target_link_libraries(comparison_multi_robot_visual ${OpenCV_LIBS})

add_executable(comparison_pf src/comparison_pf.cu)
target_link_libraries(comparison_pf ${OpenCV_LIBS})
target_compile_options(comparison_pf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_rrt src/comparison_rrt.cu)
target_link_libraries(comparison_rrt ${OpenCV_LIBS})

add_executable(comparison_astar src/comparison_astar.cu)
target_link_libraries(comparison_astar ${OpenCV_LIBS})

add_executable(comparison_dijkstra src/comparison_dijkstra.cu)
target_link_libraries(comparison_dijkstra ${OpenCV_LIBS})

add_executable(comparison_frenet src/comparison_frenet.cu)
target_link_libraries(comparison_frenet ${OpenCV_LIBS})
target_compile_options(comparison_frenet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_potential_field src/comparison_potential_field.cu)
target_link_libraries(comparison_potential_field ${OpenCV_LIBS})

add_executable(comparison_rrtstar src/comparison_rrtstar.cu)
target_link_libraries(comparison_rrtstar ${OpenCV_LIBS})

add_executable(comparison_prm src/comparison_prm.cu)
target_link_libraries(comparison_prm ${OpenCV_LIBS})

add_executable(comparison_rrt3d src/comparison_rrt3d.cu)
target_link_libraries(comparison_rrt3d ${OpenCV_LIBS})

add_executable(comparison_voronoi src/comparison_voronoi.cu)
target_link_libraries(comparison_voronoi ${OpenCV_LIBS})

add_executable(comparison_occupancy_grid src/comparison_occupancy_grid.cu)
target_link_libraries(comparison_occupancy_grid ${OpenCV_LIBS})

add_executable(comparison_fastslam src/comparison_fastslam.cu)
target_link_libraries(comparison_fastslam ${OpenCV_LIBS})
target_compile_options(comparison_fastslam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_amcl src/comparison_amcl.cu)
target_link_libraries(comparison_amcl ${OpenCV_LIBS})
target_compile_options(comparison_amcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_emcl2 src/comparison_emcl2.cu)
target_link_libraries(comparison_emcl2 ${OpenCV_LIBS})
target_compile_options(comparison_emcl2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_dwa_visual src/comparison_dwa_visual.cu)
target_link_libraries(comparison_dwa_visual ${OpenCV_LIBS})

add_executable(comparison_pf_visual src/comparison_pf_visual.cu)
target_link_libraries(comparison_pf_visual ${OpenCV_LIBS})
target_compile_options(comparison_pf_visual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_stomp src/comparison_stomp.cu)
target_link_libraries(comparison_stomp ${OpenCV_LIBS})

add_executable(comparison_icp src/comparison_icp.cu)
target_link_libraries(comparison_icp ${OpenCV_LIBS})

add_executable(comparison_hybrid_astar src/comparison_hybrid_astar.cu)
target_link_libraries(comparison_hybrid_astar ${OpenCV_LIBS})

add_executable(comparison_orca src/comparison_orca.cu)
target_link_libraries(comparison_orca ${OpenCV_LIBS})

add_executable(comparison_mppi src/comparison_mppi.cu)
target_link_libraries(comparison_mppi ${OpenCV_LIBS})

add_executable(cuda_streams_demo src/cuda_streams_demo.cu)
