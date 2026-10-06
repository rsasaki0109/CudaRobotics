cmake_minimum_required(VERSION 3.18)

# Without this CMake falls back to nvcc's default (sm_52), so modern GPUs run
# JIT-compiled PTX. Build for the local GPU(s) when one is visible; otherwise
# (CI, docker build) emit sm_75 SASS + PTX that JITs forward.
# Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(_cudabot_detected_archs "")
  find_program(_cudabot_nvidia_smi nvidia-smi)
  if(_cudabot_nvidia_smi)
    execute_process(
      COMMAND ${_cudabot_nvidia_smi} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _cudabot_caps
      RESULT_VARIABLE _cudabot_smi_result
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_cudabot_smi_result EQUAL 0)
      string(REPLACE "\n" ";" _cudabot_caps "${_cudabot_caps}")
      foreach(_cap IN LISTS _cudabot_caps)
        string(STRIP "${_cap}" _cap)
        if(_cap MATCHES "^([0-9]+)\\.([0-9]+)$")
          list(APPEND _cudabot_detected_archs "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES _cudabot_detected_archs)
    endif()
  endif()
  if(_cudabot_detected_archs)
    set(CMAKE_CUDA_ARCHITECTURES "${_cudabot_detected_archs}" CACHE STRING "CUDA architectures")
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
  endif()
endif()

project(CPP_ROBOTICS LANGUAGES CXX CUDA)
enable_testing()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

get_property(_cudabot_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _cudabot_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#############################
#Dependencies
#############################

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Prefer the imported target: Eigen 5.x does not set EIGEN3_INCLUDE_DIR, and
# build-tree configs can point it at a non-existent install prefix.
if(TARGET Eigen3::Eigen)
  get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

if(MSVC)
  # M_PI and friends are opt-in with the MSVC CRT.
  add_compile_definitions(_USE_MATH_DEFINES)
  # Sources are UTF-8; without this MSVC reads them in the system code page and
  # warns (C4819) on every file with a non-ASCII comment.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
include_directories(./include)

# cudabot_demo(<name> <source>... [OPENCV] [RELAXED_CONSTEXPR] [FAST_MATH] [NO_FMAD])
#   Adds an executable. OPENCV links ${OpenCV_LIBS}; the other flags pass
#   --expt-relaxed-constexpr / --use_fast_math / --fmad=false to nvcc.
function(cudabot_demo name)
  cmake_parse_arguments(ARG "OPENCV;RELAXED_CONSTEXPR;FAST_MATH;NO_FMAD" "" "" ${ARGN})
  add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_OPENCV)
    target_link_libraries(${name} ${OpenCV_LIBS})
  endif()
  set(cuda_flags "")
  if(ARG_RELAXED_CONSTEXPR)
    list(APPEND cuda_flags --expt-relaxed-constexpr)
  endif()
  if(ARG_FAST_MATH)
    list(APPEND cuda_flags --use_fast_math)
  endif()
  if(ARG_NO_FMAD)
    list(APPEND cuda_flags --fmad=false)
  endif()
  if(cuda_flags)
    target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>")
  endif()
endfunction()

##############################
#Localization
##############################

cudabot_demo(ekf src/extended_kalman_filter.cpp OPENCV)

cudabot_demo(pf src/particle_filter.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(fastslam1 src/fastslam1.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(amcl src/amcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(expansion_reset_mcl src/expansion_reset_mcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(pf_on_episode src/pf_on_episode.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(graph_slam src/graph_slam.cu OPENCV)

cudabot_demo(ndt src/ndt.cu OPENCV)

##############################
#PathPlanning
##############################

cudabot_demo(dijkstra_cpp src/dijkstra.cpp OPENCV)

cudabot_demo(dijkstra_cuda src/dijkstra.cu OPENCV)

cudabot_demo(astar_cpp src/a_star.cpp OPENCV)

cudabot_demo(astar_cuda src/a_star.cu OPENCV)

cudabot_demo(rrt_cpp src/rrt.cpp OPENCV)

cudabot_demo(rrt_cuda src/rrt.cu OPENCV)

cudabot_demo(rrtstar_cpp src/rrt_star.cpp OPENCV)

cudabot_demo(rrtstar_cuda src/rrt_star.cu OPENCV)

cudabot_demo(rrtstar_rs_cuda src/rrt_star_reeds_shepp.cu OPENCV)

cudabot_demo(informed_rrtstar_cuda src/informed_rrt_star.cu OPENCV)

cudabot_demo(rrtstar_3d_cuda src/rrt_star_3d.cu OPENCV)

cudabot_demo(dwa_cpp src/dynamic_window_approach.cpp OPENCV)

cudabot_demo(dwa src/dynamic_window_approach.cu OPENCV)

# add_executable(mptg src/model_predictive_trajectory_generator.cpp)
# target_link_libraries(mptg ${OpenCV_LIBS})

# add_executable(slp src/state_lattice_planner.cpp)
# target_link_libraries(slp ${OpenCV_LIBS})

add_executable(slp_cuda src/state_lattice_planner.cu)
target_link_libraries(slp_cuda)

# add_executable(csp src/cubic_spline_planner.cpp)
# target_link_libraries(csp ${OpenCV_LIBS})

cudabot_demo(frenet src/frenet_optimal_trajectory.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(potential_field src/potential_field.cu OPENCV)

cudabot_demo(potential_field_3d src/potential_field_3d.cu OPENCV)

cudabot_demo(prm_cuda src/prm.cu OPENCV)

cudabot_demo(voronoi_road_map src/voronoi_road_map.cu OPENCV)

cudabot_demo(multi_robot_planner src/multi_robot_planner.cu OPENCV)

cudabot_demo(orca src/orca.cu OPENCV)

cudabot_demo(value_iteration src/value_iteration.cu OPENCV)

cudabot_demo(stomp src/stomp.cu OPENCV)

cudabot_demo(hybrid_astar src/hybrid_astar.cu OPENCV)

cudabot_demo(rrt_connect src/rrt_connect.cu OPENCV)

cudabot_demo(social_force src/social_force.cu OPENCV)

cudabot_demo(mppi src/mppi.cu OPENCV)

cudabot_demo(diff_mppi src/diff_mppi.cu OPENCV)

cudabot_demo(comparison_diff_mppi src/comparison_diff_mppi.cu OPENCV)

cudabot_demo(benchmark_diff_mppi src/benchmark_diff_mppi.cu)
cudabot_demo(benchmark_mppi_control_update src/benchmark_mppi_control_update.cu)
cudabot_demo(benchmark_mppi_softmin src/benchmark_mppi_softmin.cu)
cudabot_demo(benchmark_mppi_diagnostics src/benchmark_mppi_diagnostics.cu)
add_executable(benchmark_mppi_rollout_ablation
  src/benchmark_mppi_rollout_ablation.cpp
  src/mppi_gpu.cu)
target_compile_options(benchmark_mppi_rollout_ablation PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_library(cudarobotics_mppi_gpu STATIC src/mppi_gpu.cu)
add_library(CudaRobotics::mppi_gpu ALIAS cudarobotics_mppi_gpu)
target_include_directories(cudarobotics_mppi_gpu
  PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(cudarobotics_mppi_gpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(cudarobotics_mppi_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
cudabot_demo(benchmark_mppi_rng src/benchmark_mppi_rng.cu)
cudabot_demo(benchmark_diff_mppi_cartpole src/benchmark_diff_mppi_cartpole.cu)
cudabot_demo(benchmark_diff_mppi_dynamic_bicycle src/benchmark_diff_mppi_dynamic_bicycle.cu)
cudabot_demo(benchmark_diff_mppi_manipulator src/benchmark_diff_mppi_manipulator.cu)
cudabot_demo(benchmark_diff_mppi_manipulator_7dof src/benchmark_diff_mppi_manipulator_7dof.cu)

# One-Step CDF-MPPI literature-faithful baseline (arXiv:2509.00836)
cudabot_demo(benchmark_cdf_mppi_7dof src/benchmark_cdf_mppi_7dof.cu OPENCV RELAXED_CONSTEXPR)

# Differentiable planar pushing: autodiff-through-contact test for Diff-MPPI
cudabot_demo(benchmark_diff_mppi_pushing src/benchmark_diff_mppi_pushing.cu RELAXED_CONSTEXPR)

# Differentiable box pushing (position + orientation): contact-point reasoning
cudabot_demo(benchmark_diff_mppi_pushing_box src/benchmark_diff_mppi_pushing_box.cu RELAXED_CONSTEXPR)

cudabot_demo(neural_sdf src/neural_sdf.cu OPENCV)

cudabot_demo(sdf_potential_field src/sdf_potential_field.cu OPENCV)

cudabot_demo(sdf_mppi src/sdf_mppi.cu OPENCV)

cudabot_demo(esdf_mppi src/esdf_mppi.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_esdf_mppi_3d src/gpu_esdf_mppi_3d.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_sdf_nav src/comparison_sdf_nav.cu OPENCV)

cudabot_demo(mini_isaac src/mini_isaac.cu OPENCV)

cudabot_demo(mini_isaac_rl src/mini_isaac_rl.cu OPENCV)

##############################
#Neuroevolution
##############################

cudabot_demo(neuroevo src/neuroevo.cu OPENCV)

cudabot_demo(comparison_neuroevo src/comparison_neuroevo.cu OPENCV)

##############################
#Point Cloud
##############################

add_library(pointcloud_ops STATIC
    src/voxel_grid_filter.cu
    src/statistical_filter.cu
    src/normal_estimation.cu
    src/gicp.cu
    src/ransac_plane.cu
)
target_compile_options(pointcloud_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_executable(benchmark_pointcloud src/benchmark_pointcloud.cu)
target_link_libraries(benchmark_pointcloud pointcloud_ops)

##############################
#Registration
##############################

cudabot_demo(icp src/icp.cu OPENCV)

##############################
#Mapping
##############################

cudabot_demo(occupancy_grid src/occupancy_grid_map.cu OPENCV)

##############################
#PathTracking
##############################

cudabot_demo(lqr src/lqr_steer_control.cpp OPENCV)

cudabot_demo(lqr_full src/lqr_speed_steer_control.cpp OPENCV)

#add_executable(mpc src/model_predictive_control.cpp)
#target_link_libraries(mpc ${OpenCV_LIBS} ipopt)

##############################
#Benchmark
##############################

cudabot_demo(benchmark_pf src/benchmark_pf.cu RELAXED_CONSTEXPR)

cudabot_demo(benchmark_rrt src/benchmark_rrt.cu)

cudabot_demo(benchmark_dwa src/benchmark_dwa.cu)

cudabot_demo(comparison_dwa src/comparison_dwa.cu OPENCV)

cudabot_demo(comparison_multi_robot src/comparison_multi_robot.cu OPENCV)

cudabot_demo(comparison_multi_robot_visual src/comparison_multi_robot_visual.cu OPENCV)

cudabot_demo(comparison_pf src/comparison_pf.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_rrt src/comparison_rrt.cu OPENCV)

cudabot_demo(comparison_astar src/comparison_astar.cu OPENCV)

cudabot_demo(comparison_dijkstra src/comparison_dijkstra.cu OPENCV)

cudabot_demo(comparison_frenet src/comparison_frenet.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_potential_field src/comparison_potential_field.cu OPENCV)

cudabot_demo(comparison_rrtstar src/comparison_rrtstar.cu OPENCV)

cudabot_demo(comparison_prm src/comparison_prm.cu OPENCV)

cudabot_demo(comparison_rrt3d src/comparison_rrt3d.cu OPENCV)

cudabot_demo(comparison_voronoi src/comparison_voronoi.cu OPENCV)

cudabot_demo(comparison_occupancy_grid src/comparison_occupancy_grid.cu OPENCV)

cudabot_demo(comparison_esdf src/comparison_esdf.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_esdf_3d src/comparison_esdf_3d.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(pf_esdf src/pf_esdf.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_rrtstar_rewire src/comparison_rrtstar_rewire.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_bundle_adjustment src/gpu_bundle_adjustment.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_lidar_slam src/gpu_lidar_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(visibility_mppi src/visibility_mppi.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(pf_realistic_obs src/pf_realistic_obs.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_pose_graph_slam src/gpu_pose_graph_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_pose_graph_slam_3d src/gpu_pose_graph_slam_3d.cu OPENCV RELAXED_CONSTEXPR)

add_executable(gpu_pose_graph_slam_3d_robust src/gpu_pose_graph_slam_3d.cu)
target_link_libraries(gpu_pose_graph_slam_3d_robust ${OpenCV_LIBS})
target_compile_definitions(gpu_pose_graph_slam_3d_robust PRIVATE GPU_POSE_GRAPH_3D_ROBUST=1)
target_compile_options(gpu_pose_graph_slam_3d_robust PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

cudabot_demo(gpu_pose_graph_slam_3d_switchable src/gpu_pose_graph_slam_3d_switchable.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_global_localization_mcl src/gpu_global_localization_mcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_megaparticles_stein_mcl src/gpu_megaparticles_stein_mcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_megaparticles_lsh src/gpu_megaparticles_lsh.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_megaparticles_6dof src/gpu_megaparticles_6dof.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_megaparticles_gicp_mcl src/gpu_megaparticles_gicp_mcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_megaparticles_smoother src/gpu_megaparticles_smoother.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_kld_amcl src/gpu_kld_amcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_online_slam src/gpu_online_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_online_slam_3d_switchable src/gpu_online_slam_3d_switchable.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_ndt_2d src/gpu_ndt_2d.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_correlative_scan_matching src/gpu_correlative_scan_matching.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_csm_loop_closure_slam src/gpu_csm_loop_closure_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_csm_submap_slam src/gpu_csm_submap_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_branch_and_bound_csm src/gpu_branch_and_bound_csm.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_bnb_loop_closure_slam src/gpu_bnb_loop_closure_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_ndt_3d src/gpu_ndt_3d.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_ndt_3d_multires src/gpu_ndt_3d_multires.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_gicp_2d src/gpu_gicp_2d.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_gicp_3d src/gpu_gicp_3d.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_gaussian_splatting src/gpu_gaussian_splatting.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_gaussian_splatting_slam src/gpu_gaussian_splatting_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_nerf_volume src/gpu_nerf_volume.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_multi_robot_planner src/gpu_multi_robot_planner.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_multi_robot_place_graph_slam src/gpu_multi_robot_place_graph_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_crowd_swarm src/gpu_crowd_swarm.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_gnn_swarm_controller src/gpu_gnn_swarm_controller.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_sfm_mini src/gpu_sfm_mini.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_pcg_solver src/gpu_pcg_solver.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_em_gmm src/gpu_em_gmm.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_spectral_clustering src/gpu_spectral_clustering.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_label_propagation src/gpu_label_propagation.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_label_propagation_traversability src/gpu_label_propagation_traversability.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_graph_crf_traversability src/gpu_graph_crf_traversability.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_gat_traversability_policy src/gpu_gat_traversability_policy.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_diff_value_iteration_traversability src/gpu_diff_value_iteration_traversability.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_neural_astar_traversability src/gpu_neural_astar_traversability.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_anytime_neural_astar_traversability src/gpu_anytime_neural_astar_traversability.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_multigoal_neural_astar_traversability src/gpu_multigoal_neural_astar_traversability.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_spatiotemporal_neural_astar_traversability src/gpu_spatiotemporal_neural_astar_traversability.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_experience_graph_neural_planner src/gpu_experience_graph_neural_planner.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_graph_guided_neural_mppi src/gpu_graph_guided_neural_mppi.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_kinodynamic_graph_neural_mppi src/gpu_kinodynamic_graph_neural_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_interaction_graph_neural_mppi src/gpu_interaction_graph_neural_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_multiagent_graph_neural_mppi src/gpu_multiagent_graph_neural_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_priority_graph_neural_mppi src/gpu_priority_graph_neural_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_intent_graph_neural_mppi src/gpu_intent_graph_neural_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_belief_risk_graph_mppi src/gpu_belief_risk_graph_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_best_response_graph_mppi src/gpu_best_response_graph_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_iterative_game_graph_mppi src/gpu_iterative_game_graph_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_noregret_game_graph_mppi src/gpu_noregret_game_graph_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_safe_noregret_game_graph_mppi src/gpu_safe_noregret_game_graph_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_learned_safety_dual_graph_mppi src/gpu_learned_safety_dual_graph_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_trainable_safety_dual_graph_mppi src/gpu_trainable_safety_dual_graph_mppi.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_planner_showdown_benchmark src/gpu_planner_showdown_benchmark.cu OPENCV RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_planner_falsifier_benchmark src/gpu_planner_falsifier_benchmark.cu RELAXED_CONSTEXPR FAST_MATH)

cudabot_demo(gpu_interaction_graph_risk src/gpu_interaction_graph_risk.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_reciprocal_risk_planner src/gpu_reciprocal_risk_planner.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_hungarian_assignment src/gpu_hungarian_assignment.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_assignment_tracking src/gpu_assignment_tracking.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_cma_es src/gpu_cma_es.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_mcts_planner src/gpu_mcts_planner.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_diffusion_planner src/gpu_diffusion_planner.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_diffusion_policy src/gpu_diffusion_policy.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(diff_e2e_slam src/diff_e2e_slam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_voxel_map src/comparison_voxel_map.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_collision_check src/comparison_collision_check.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_fastslam src/comparison_fastslam.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_amcl src/comparison_amcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_expansion_reset_mcl src/comparison_expansion_reset_mcl.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_lidar_sim src/comparison_lidar_sim.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_lidar3d_sim src/comparison_lidar3d_sim.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_lidar3d_realistic src/comparison_lidar3d_realistic.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_gaussian_splatting src/comparison_gaussian_splatting.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_batched_ilqr src/gpu_batched_ilqr.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(gpu_batched_ik src/gpu_batched_ik.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(gpu_parallel_ilqr src/gpu_parallel_ilqr.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

add_executable(gpu_filterreg src/gpu_filterreg.cu src/filterreg_gpu.cu)
target_include_directories(gpu_filterreg PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_filterreg ${OpenCV_LIBS})
target_compile_options(gpu_filterreg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

cudabot_demo(gpu_bcpd src/gpu_bcpd.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_sinkhorn_reg src/gpu_sinkhorn_reg.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_robust_treg src/gpu_robust_treg.cu OPENCV RELAXED_CONSTEXPR)

add_library(cudarobotics_kiss_icp_gpu STATIC src/gpu_kiss_icp.cu)
add_library(CudaRobotics::kiss_icp_gpu ALIAS cudarobotics_kiss_icp_gpu)
target_compile_definitions(cudarobotics_kiss_icp_gpu
  PRIVATE CUDAROBOTICS_KISS_ICP_CORE_ONLY)
target_include_directories(cudarobotics_kiss_icp_gpu
  PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(cudarobotics_kiss_icp_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(cudarobotics_kiss_icp_gpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

cudabot_demo(gpu_kiss_icp src/gpu_kiss_icp.cu OPENCV RELAXED_CONSTEXPR)

add_executable(kiss_icp_gpu_streaming_smoke tests/kiss_icp_gpu_streaming_smoke.cu)
target_link_libraries(kiss_icp_gpu_streaming_smoke cudarobotics_kiss_icp_gpu)
target_compile_options(kiss_icp_gpu_streaming_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(cudanav_kiss_icp_sequence tools/cudanav_kiss_icp_sequence.cu)
target_link_libraries(cudanav_kiss_icp_sequence cudarobotics_kiss_icp_gpu)
target_compile_options(cudanav_kiss_icp_sequence PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(cudanav_real_gpu_stack_sequence
  tools/cudanav_real_gpu_stack_sequence.cu)
target_link_libraries(cudanav_real_gpu_stack_sequence
  cudarobotics_kiss_icp_gpu
  cudarobotics_voxel_mapping_gpu
  cudarobotics_esdf_2d_gpu
  cudarobotics_mppi_gpu)
target_compile_options(cudanav_real_gpu_stack_sequence PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(cudanav_gpu_closed_loop_s_course
  tools/cudanav_gpu_closed_loop_s_course.cu)
target_link_libraries(cudanav_gpu_closed_loop_s_course
  cudarobotics_kiss_icp_gpu
  cudarobotics_voxel_mapping_gpu
  cudarobotics_esdf_2d_gpu
  cudarobotics_mppi_gpu)
target_compile_options(cudanav_gpu_closed_loop_s_course PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_library(cudarobotics_voxel_mapping_gpu STATIC src/voxel_mapping_gpu.cu)
add_library(CudaRobotics::voxel_mapping_gpu ALIAS cudarobotics_voxel_mapping_gpu)
target_include_directories(cudarobotics_voxel_mapping_gpu
  PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(cudarobotics_voxel_mapping_gpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_executable(voxel_mapping_gpu_smoke tests/voxel_mapping_gpu_smoke.cu)
target_link_libraries(voxel_mapping_gpu_smoke cudarobotics_voxel_mapping_gpu)

add_library(cudarobotics_esdf_2d_gpu STATIC src/esdf_2d_gpu.cu)
add_library(CudaRobotics::esdf_2d_gpu ALIAS cudarobotics_esdf_2d_gpu)
target_include_directories(cudarobotics_esdf_2d_gpu
  PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(cudarobotics_esdf_2d_gpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_executable(esdf_2d_gpu_smoke tests/esdf_2d_gpu_smoke.cu)
target_link_libraries(esdf_2d_gpu_smoke cudarobotics_esdf_2d_gpu)

cudabot_demo(mppi_reduction_smoke tests/mppi_reduction_smoke.cu)

cudabot_demo(gpu_mpc_qp src/gpu_mpc_qp.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_constrained_mpc src/gpu_constrained_mpc.cu OPENCV RELAXED_CONSTEXPR)

add_executable(gpu_cudampc src/gpu_cudampc.cu)
target_link_libraries(gpu_cudampc ${OpenCV_LIBS})
target_compile_options(gpu_cudampc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(gpu_cudampc PROPERTIES CUDA_ARCHITECTURES 75)

cudabot_demo(gpu_diff_contact_push src/gpu_diff_contact_push.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_robust_p2plane_reg src/gpu_robust_p2plane_reg.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_real_bunny_reg src/gpu_real_bunny_reg.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_fgr src/gpu_fgr.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_filterreg_p2plane src/gpu_filterreg_p2plane.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_tsdf_fusion src/gpu_tsdf_fusion.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(gpu_dbscan src/gpu_dbscan.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

add_library(cudarobotics_lidar_objects_gpu STATIC src/lidar_objects_gpu.cu)
add_library(CudaRobotics::lidar_objects_gpu ALIAS cudarobotics_lidar_objects_gpu)
target_include_directories(cudarobotics_lidar_objects_gpu
  PUBLIC ${CMAKE_SOURCE_DIR}/include)
# --fmad=false: the GPU boxes must equal the CPU references bit for bit
target_compile_options(cudarobotics_lidar_objects_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)
set_target_properties(cudarobotics_lidar_objects_gpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_executable(lidar_objects_gpu_smoke tests/lidar_objects_gpu_smoke.cu)
target_link_libraries(lidar_objects_gpu_smoke cudarobotics_lidar_objects_gpu)

cudabot_demo(gpu_ground_segmentation src/gpu_ground_segmentation.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)
target_link_libraries(gpu_ground_segmentation cudarobotics_lidar_objects_gpu)

cudabot_demo(gpu_sgm_stereo src/gpu_sgm_stereo.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(gpu_wavefront_planner src/gpu_wavefront_planner.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(gpu_marching_cubes src/gpu_marching_cubes.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(gpu_mppi_racing src/gpu_mppi_racing.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(gpu_direct_vo src/gpu_direct_vo.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(comparison_reeds_shepp_fan src/comparison_reeds_shepp_fan.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(diff_pf src/diff_pf.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(diff_pf_mlp src/diff_pf_mlp.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_dwa_visual src/comparison_dwa_visual.cu OPENCV)

cudabot_demo(comparison_pf_visual src/comparison_pf_visual.cu OPENCV RELAXED_CONSTEXPR)

cudabot_demo(comparison_stomp src/comparison_stomp.cu OPENCV)

cudabot_demo(comparison_icp src/comparison_icp.cu OPENCV)

cudabot_demo(comparison_hybrid_astar src/comparison_hybrid_astar.cu OPENCV)

cudabot_demo(comparison_orca src/comparison_orca.cu OPENCV)

cudabot_demo(comparison_mppi src/comparison_mppi.cu OPENCV)

##############################
#SwarmOptimization
##############################

cudabot_demo(pso_cuda src/pso.cu OPENCV)

cudabot_demo(differential_evolution src/differential_evolution.cu OPENCV)

cudabot_demo(cma_es src/cma_es.cu OPENCV)

cudabot_demo(aco_tsp src/aco_tsp.cu OPENCV)

cudabot_demo(comparison_swarm src/comparison_swarm.cu OPENCV)

cudabot_demo(cuda_streams_demo src/cuda_streams_demo.cu)

cudabot_demo(gpu_frontier_exploration src/gpu_frontier_exploration.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

##############################
# Common Foundation Tests
##############################
cudabot_demo(gpu_lk_optical_flow src/gpu_lk_optical_flow.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(test_autodiff src/test_autodiff.cu)
cudabot_demo(test_gpu_mlp src/test_gpu_mlp.cu)

cudabot_demo(test_host_math src/test_host_math.cpp OPENCV)

cudabot_demo(test_lie_group_math tests/lie_group_math_cpu.cpp)
cudabot_demo(test_imu_preintegration tests/imu_preintegration_cpu.cpp)
cudabot_demo(test_robust_loss tests/robust_loss_cpu.cpp)
cudabot_demo(test_g2o_io tests/g2o_io_cpu.cpp)
cudabot_demo(test_graph_optimization tests/graph_optimization_cpu.cpp)
cudabot_demo(test_mathr_native tests/mathr_native_cpu.cpp)
cudabot_demo(test_imu_graph tests/imu_graph_cpu.cpp)
cudabot_demo(test_graph_g2o tests/graph_g2o_cpu.cpp)
cudabot_demo(test_projection_gpu tests/projection_gpu_smoke.cu)
add_executable(test_imu_preintegration_gpu tests/imu_preintegration_gpu_smoke.cu)
target_compile_options(test_imu_preintegration_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

##############################
# CTest Registration
##############################

# GPU tests (require NVIDIA GPU)
add_test(NAME test_autodiff COMMAND $<TARGET_FILE:test_autodiff>)
add_test(NAME test_gpu_mlp COMMAND $<TARGET_FILE:test_gpu_mlp>)
set_tests_properties(test_autodiff test_gpu_mlp PROPERTIES
  TIMEOUT 60
  LABELS "gpu")

add_test(NAME gpu_kiss_icp_gate
  COMMAND $<TARGET_FILE:gpu_kiss_icp>
    --check
    --no-video
    --frames 64
    --nn voxel
    --json ${CMAKE_BINARY_DIR}/gpu_kiss_icp_check.json
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_kiss_icp_gate PROPERTIES
  TIMEOUT 120
  LABELS "gpu;odometry;benchmark")

add_test(NAME kiss_icp_gpu_streaming_smoke
  COMMAND $<TARGET_FILE:kiss_icp_gpu_streaming_smoke>)
set_tests_properties(kiss_icp_gpu_streaming_smoke PROPERTIES
  TIMEOUT 60
  LABELS "gpu;odometry;api")

add_test(NAME voxel_mapping_gpu_smoke
  COMMAND $<TARGET_FILE:voxel_mapping_gpu_smoke>)
set_tests_properties(voxel_mapping_gpu_smoke PROPERTIES
  TIMEOUT 60
  LABELS "gpu;mapping;api")

add_test(NAME gpu_batched_ik_gate
  COMMAND $<TARGET_FILE:gpu_batched_ik> --check --no-video
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_batched_ik_gate PROPERTIES
  TIMEOUT 300
  LABELS "gpu;manipulation;ik")

add_test(NAME lidar_objects_gpu_smoke
  COMMAND $<TARGET_FILE:lidar_objects_gpu_smoke>)
set_tests_properties(lidar_objects_gpu_smoke PROPERTIES
  TIMEOUT 120
  LABELS "gpu;pointcloud;ground")

add_test(NAME gpu_ground_segmentation_gate
  COMMAND $<TARGET_FILE:gpu_ground_segmentation> --check --no-video
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_ground_segmentation_gate PROPERTIES
  TIMEOUT 300
  LABELS "gpu;pointcloud;ground")

add_test(NAME gpu_ground_segmentation_track_gate
  COMMAND $<TARGET_FILE:gpu_ground_segmentation> --sequence --check --no-video
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_ground_segmentation_track_gate PROPERTIES
  TIMEOUT 600
  LABELS "gpu;pointcloud;ground;tracking")

add_test(NAME gpu_ground_segmentation_motion_gate
  COMMAND $<TARGET_FILE:gpu_ground_segmentation> --sequence --moving --check --no-video
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_ground_segmentation_motion_gate PROPERTIES
  TIMEOUT 600
  LABELS "gpu;pointcloud;ground;tracking")

add_test(NAME esdf_2d_gpu_smoke
  COMMAND $<TARGET_FILE:esdf_2d_gpu_smoke>)
set_tests_properties(esdf_2d_gpu_smoke PROPERTIES
  TIMEOUT 60
  LABELS "gpu;mapping;esdf;reference")

add_test(NAME mppi_reduction_smoke
  COMMAND $<TARGET_FILE:mppi_reduction_smoke>)
set_tests_properties(mppi_reduction_smoke PROPERTIES
  TIMEOUT 60
  LABELS "gpu;mppi;reference")

# Run interactive demos end-to-end without a display (see include/display.h).
set(CUDABOT_DEMO_SMOKE_DIR ${CMAKE_BINARY_DIR}/demo_smoke)
file(MAKE_DIRECTORY ${CUDABOT_DEMO_SMOKE_DIR}/gif)
foreach(demo mppi diff_mppi comparison_mppi esdf_mppi gpu_esdf_mppi_3d ndt occupancy_grid amcl)
  add_test(NAME demo_headless_${demo}
    COMMAND $<TARGET_FILE:${demo}>
    WORKING_DIRECTORY ${CUDABOT_DEMO_SMOKE_DIR})
  set_tests_properties(demo_headless_${demo} PROPERTIES
    ENVIRONMENT "CUDABOT_HEADLESS=1"
    TIMEOUT 300
    LABELS "gpu;demo")
endforeach()

add_test(NAME demo_headless_gpu_esdf_mppi_3d_dynamic
  COMMAND $<TARGET_FILE:gpu_esdf_mppi_3d> --movers 6 --mover-speed 2 --mode 2 --no-video
  WORKING_DIRECTORY ${CUDABOT_DEMO_SMOKE_DIR})
set_tests_properties(demo_headless_gpu_esdf_mppi_3d_dynamic PROPERTIES
  ENVIRONMENT "CUDABOT_HEADLESS=1"
  TIMEOUT 300
  LABELS "gpu;demo")

add_test(NAME gpu_planner_showdown_gate
  COMMAND $<TARGET_FILE:gpu_planner_showdown_benchmark>
    --check
    --no-video
    --scenario baseline
    --json ${CMAKE_BINARY_DIR}/gpu_planner_showdown_benchmark.json
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_planner_showdown_gate PROPERTIES
  TIMEOUT 90
  LABELS "gpu;showdown;benchmark")

add_test(NAME gpu_planner_falsifier_gate
  COMMAND $<TARGET_FILE:gpu_planner_falsifier_benchmark>
    --check
    --json ${CMAKE_BINARY_DIR}/gpu_planner_falsifier_benchmark.json
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_planner_falsifier_gate PROPERTIES
  TIMEOUT 30
  LABELS "gpu;falsifier;benchmark")

# Python validation tests (no GPU required)
add_test(NAME validate_design_workflow
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/validate_design_workflow.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_design_regressions
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/check_design_regressions.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_scaffold
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/check_scaffold_design_problem.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_repro_suite_dry_run
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/run_repro_suite.py
    --dry-run
    --suite smoke
    --output-dir ${CMAKE_BINARY_DIR}/repro_suite_dry_run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_mppi_zoo_smoke_dry_run
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/run_mppi_zoo_smoke.py
    --dry-run
    --out-dir ${CMAKE_BINARY_DIR}/mppi_zoo_dry_run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_mppi_zoo_chart_render
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/render_mppi_zoo_chart.py
    --csv ${CMAKE_SOURCE_DIR}/docs/results/mppi_zoo_smoke_2026-06-05.csv
    --svg-out ${CMAKE_BINARY_DIR}/mppi_zoo_smoke_chart.svg
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_planner_showdown_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/summarize_planner_showdown.py
    --json ${CMAKE_SOURCE_DIR}/gif/gpu_planner_showdown_benchmark.json
    --markdown-out ${CMAKE_BINARY_DIR}/gpu_planner_showdown_benchmark.md
    --strict
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_planner_showdown_matrix_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/summarize_planner_showdown.py
    --json ${CMAKE_SOURCE_DIR}/gif/gpu_planner_showdown_benchmark.json
    --json ${CMAKE_SOURCE_DIR}/gif/gpu_planner_showdown_benchmark.json
    --markdown-out ${CMAKE_BINARY_DIR}/gpu_planner_showdown_matrix.md
    --strict
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_planner_falsifier_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/summarize_planner_falsifier.py
    --json ${CMAKE_SOURCE_DIR}/gif/gpu_planner_falsifier_benchmark.json
    --markdown-out ${CMAKE_BINARY_DIR}/gpu_planner_falsifier_benchmark.md
    --strict
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_erl_navigation_bag_inspector
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_prepare_erl_navigation_bags.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_offline_rosbag_db3_analyzer
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_analyze_rosbag_db3.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_offline_rosbag_motion_decoder
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_export_rosbag_motion.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_offline_rosbag_motion_comparison
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_compare_rosbag_motion.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_offline_rosbag_clearance_decoder
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_analyze_rosbag_clearance.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_cuda_acceleration_auditor
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_audit_cuda_acceleration.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_registration_suite
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_registration_suite.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_mppi_rosbag_evaluation
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_evaluate_mppi_rosbag.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_version_consistency
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_version_consistency.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_cudanav_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_contract.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_cudanav_autonomy_suite_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_autonomy_suite.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_cudanav_autonomy_suite_release_dry_run
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/run_autonomy_suite.py
    --output-dir ${CMAKE_BINARY_DIR}/cudanav_autonomy_release
    --profile release
    --bag ${CMAKE_BINARY_DIR}/fixture_bag
    --evaluation-db ${CMAKE_BINARY_DIR}/fixture_bag/run.db3
    --controller-config ${CMAKE_SOURCE_DIR}/ros2_ws/src/cuda_nav_bringup/config/controller.yaml
    --controller-command "ros2 launch cuda_nav_bringup cudanav_closed_loop.launch.py"
    --dry-run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_paper_artifact_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_paper_artifact_contract.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_paper_artifact_manifests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/validate_paper_artifacts.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_cudarobotics_systems_paper
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudarobotics_systems_paper.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_systems_paper_bundle
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_systems_paper_bundle.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_systems_paper_archive
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_systems_paper_archive.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_contact_rich_diff_mppi_paper
  COMMAND python3
    ${CMAKE_SOURCE_DIR}/scripts/test_contact_rich_diff_mppi_paper.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_submission_bundle
  COMMAND python3
    ${CMAKE_SOURCE_DIR}/scripts/test_contact_submission_bundle.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_submission_archive
  COMMAND python3
    ${CMAKE_SOURCE_DIR}/scripts/test_contact_submission_archive.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_ros_ci_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_ros_ci_evidence.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_cudanav_systems_publisher_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_publish_cudanav_systems_evidence.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_paper_results
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/render_contact_paper_results.py
    --check
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_paper_results_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_render_contact_paper_results.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_robustness_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_contact_robustness.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_robustness_release_dry_run
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/run_contact_robustness.py
    --output-dir ${CMAKE_BINARY_DIR}/contact_robustness_release
    --profile release
    --dry-run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_matched_compute_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_contact_matched_compute.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_matched_compute_release_dry_run
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/run_contact_matched_compute.py
    --output-dir ${CMAKE_BINARY_DIR}/contact_matched_compute_release
    --profile release
    --dry-run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_external_fidelity_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_contact_external_fidelity.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_contact_external_fidelity_release_dry_run
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/run_contact_external_fidelity.py
    --output-dir ${CMAKE_BINARY_DIR}/contact_external_fidelity_release
    --profile release
    --dry-run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_python_core_sync
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_python_core_sync.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_python_release_artifacts
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_verify_python_release_artifacts.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_release_preflight
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_release_preflight.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_release_ci_evidence
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_release_ci_evidence.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v0_2_release_evidence
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v0_2_release_evidence.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v0_2_release_bundle
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v0_2_release_bundle.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v0_2_release_archive
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v0_2_release_archive.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_distribution
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_distribution.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_support_matrix
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_support_matrix.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_release_candidate
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_release_candidate.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_release_attestation
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_release_attestation.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_publish_v1_quickstart_attestation
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_publish_v1_quickstart_attestation.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_docker_gpu_evidence
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_docker_gpu_evidence.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_release_bundle
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_release_bundle.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_release_archive
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_release_archive.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_documentation_evidence
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_documentation_evidence.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_stage_v1_docs
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_stage_v1_docs.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_v1_quickstart_evidence
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_v1_quickstart_evidence.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_real_dataset
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_real_dataset.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_derived_path
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_derive_cudanav_path_sidecar.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_pointcloud2_clearance_decoder
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_analyze_pointcloud2_clearance.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_pointcloud2_timing_admission
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_inspect_pointcloud2_timing.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_mcd_timed_rosbag_materializer
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_materialize_mcd_timed_rosbag.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_istanbul_dataset_preparer
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_prepare_cudanav_istanbul_dataset.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_real_dataset_pipeline
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_run_cudanav_real_dataset_pipeline.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_dataset_publisher
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_publish_cudanav_dataset_evidence.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_kiss_icp_real
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_kiss_icp_real.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_real_gpu_stack
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_real_gpu_stack.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_gpu_closed_loop
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_gpu_closed_loop.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_gpu_closed_loop_visual
  COMMAND python3
    ${CMAKE_SOURCE_DIR}/scripts/test_render_cudanav_gpu_closed_loop.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_multi_gpu
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_multi_gpu.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
add_test(NAME check_cudanav_multi_gpu_workflow
  COMMAND python3
    ${CMAKE_SOURCE_DIR}/scripts/test_cudanav_multi_gpu_workflow.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
set_tests_properties(validate_design_workflow check_design_regressions check_scaffold check_repro_suite_dry_run check_mppi_zoo_smoke_dry_run check_mppi_zoo_chart_render check_planner_showdown_summary check_planner_showdown_matrix_summary check_planner_falsifier_summary check_erl_navigation_bag_inspector check_offline_rosbag_db3_analyzer check_offline_rosbag_motion_decoder check_offline_rosbag_motion_comparison check_offline_rosbag_clearance_decoder check_cuda_acceleration_auditor check_registration_suite check_mppi_rosbag_evaluation check_version_consistency check_cudanav_contract check_cudanav_autonomy_suite_contract check_cudanav_autonomy_suite_release_dry_run check_cudanav_ros_ci_contract check_cudanav_systems_publisher_contract check_paper_artifact_contract check_paper_artifact_manifests check_cudarobotics_systems_paper check_systems_paper_bundle check_systems_paper_archive check_contact_rich_diff_mppi_paper check_contact_submission_bundle check_contact_submission_archive check_contact_paper_results check_contact_paper_results_contract check_contact_robustness_contract check_contact_robustness_release_dry_run check_contact_matched_compute_contract check_contact_matched_compute_release_dry_run check_contact_external_fidelity_contract check_contact_external_fidelity_release_dry_run check_python_core_sync check_python_release_artifacts check_release_preflight check_release_ci_evidence check_v0_2_release_evidence check_v0_2_release_bundle check_v0_2_release_archive check_cudanav_distribution check_v1_support_matrix check_v1_release_candidate check_v1_release_attestation check_publish_v1_quickstart_attestation check_v1_docker_gpu_evidence check_v1_release_bundle check_v1_release_archive check_v1_documentation_evidence check_stage_v1_docs check_v1_quickstart_evidence check_cudanav_real_dataset check_cudanav_derived_path check_pointcloud2_clearance_decoder check_pointcloud2_timing_admission check_cudanav_istanbul_dataset_preparer check_cudanav_real_dataset_pipeline check_cudanav_dataset_publisher check_cudanav_kiss_icp_real check_cudanav_real_gpu_stack check_cudanav_gpu_closed_loop check_cudanav_gpu_closed_loop_visual check_cudanav_multi_gpu check_cudanav_multi_gpu_workflow
  PROPERTIES
  TIMEOUT 30
  LABELS "python")

# CPU tests (no GPU required)
add_test(NAME test_host_math COMMAND $<TARGET_FILE:test_host_math>)
set_tests_properties(test_host_math PROPERTIES
  TIMEOUT 10
  LABELS "cpu")

add_test(NAME test_lie_group_math COMMAND $<TARGET_FILE:test_lie_group_math>)
set_tests_properties(test_lie_group_math PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math")

add_test(NAME test_imu_preintegration COMMAND $<TARGET_FILE:test_imu_preintegration>)
set_tests_properties(test_imu_preintegration PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math;imu")

add_test(NAME test_robust_loss COMMAND $<TARGET_FILE:test_robust_loss>)
set_tests_properties(test_robust_loss PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math;robust")

add_test(NAME test_g2o_io COMMAND $<TARGET_FILE:test_g2o_io>)
set_tests_properties(test_g2o_io PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math;graph")

add_test(NAME test_graph_optimization COMMAND $<TARGET_FILE:test_graph_optimization>)
set_tests_properties(test_graph_optimization PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math;graph")

add_test(NAME test_mathr_native COMMAND $<TARGET_FILE:test_mathr_native>)
set_tests_properties(test_mathr_native PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math;geometry;filter")

add_test(NAME test_imu_graph COMMAND $<TARGET_FILE:test_imu_graph>)
set_tests_properties(test_imu_graph PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math;imu;graph")

add_test(NAME test_graph_g2o COMMAND $<TARGET_FILE:test_graph_g2o>)
set_tests_properties(test_graph_g2o PROPERTIES
  TIMEOUT 10
  LABELS "cpu;math;graph;g2o")

add_test(NAME test_projection_gpu COMMAND $<TARGET_FILE:test_projection_gpu>)
set_tests_properties(test_projection_gpu PROPERTIES
  TIMEOUT 60
  LABELS "gpu;math;geometry"
  SKIP_RETURN_CODE 77)

add_test(NAME test_imu_preintegration_gpu
  COMMAND $<TARGET_FILE:test_imu_preintegration_gpu>)
set_tests_properties(test_imu_preintegration_gpu PROPERTIES
  TIMEOUT 60
  LABELS "gpu;math;imu;graph"
  SKIP_RETURN_CODE 77)

##############################
# MuJoCo Benchmarks
##############################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(mujoco QUIET)

if(mujoco_FOUND)
  add_executable(benchmark_diff_mppi_mujoco src/benchmark_diff_mppi_mujoco.cu)
  target_link_libraries(benchmark_diff_mppi_mujoco ${OpenCV_LIBS} mujoco::mujoco)
  target_compile_options(benchmark_diff_mppi_mujoco PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(benchmark_diff_mppi_mujoco PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD_REQUIRED YES)
  target_compile_definitions(benchmark_diff_mppi_mujoco PRIVATE
    CUDAROBOTICS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  if(DEFINED mujoco_LIBRARY_DIR)
    set_target_properties(benchmark_diff_mppi_mujoco PROPERTIES
      BUILD_RPATH "${mujoco_LIBRARY_DIR}")
  endif()

  add_executable(benchmark_diff_mppi_pushing_box_mujoco
    src/benchmark_diff_mppi_pushing_box_mujoco.cu)
  target_link_libraries(benchmark_diff_mppi_pushing_box_mujoco
    mujoco::mujoco)
  target_compile_options(benchmark_diff_mppi_pushing_box_mujoco PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_compile_definitions(benchmark_diff_mppi_pushing_box_mujoco PRIVATE
    CUDAROBOTICS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(benchmark_diff_mppi_pushing_box_mujoco PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD_REQUIRED YES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON)
  add_custom_command(TARGET benchmark_diff_mppi_pushing_box_mujoco POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:mujoco::mujoco>"
      "$<TARGET_FILE_DIR:benchmark_diff_mppi_pushing_box_mujoco>"
    COMMENT "Staging the MuJoCo runtime beside the contact benchmark")
  if(DEFINED mujoco_LIBRARY_DIR)
    set_target_properties(benchmark_diff_mppi_pushing_box_mujoco PROPERTIES
      BUILD_RPATH "${mujoco_LIBRARY_DIR}")
  endif()

  add_executable(benchmark_diff_mppi_mujoco_reacher src/benchmark_diff_mppi_mujoco_reacher.cu)
  target_link_libraries(benchmark_diff_mppi_mujoco_reacher ${OpenCV_LIBS} mujoco::mujoco)
  target_compile_options(benchmark_diff_mppi_mujoco_reacher PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(benchmark_diff_mppi_mujoco_reacher PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD_REQUIRED YES)
  target_compile_definitions(benchmark_diff_mppi_mujoco_reacher PRIVATE
    CUDAROBOTICS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  if(DEFINED mujoco_LIBRARY_DIR)
    set_target_properties(benchmark_diff_mppi_mujoco_reacher PROPERTIES
      BUILD_RPATH "${mujoco_LIBRARY_DIR}")
  endif()
endif()

cudabot_demo(gpu_jfa_edt src/gpu_jfa_edt.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

cudabot_demo(gpu_kdtree_nn src/gpu_kdtree_nn.cu OPENCV RELAXED_CONSTEXPR NO_FMAD)

##############################
#Partial builds
##############################

# -DCUDABOT_DEMO_FILTER=<regex> keeps only matching executables in the default
# build (e.g. "mppi|esdf"); the others stay available via --target <name>.
# Libraries leave the default build too and are built on demand as deps.
set(CUDABOT_DEMO_FILTER "" CACHE STRING
  "Regex of executables built by default (empty: all)")
if(CUDABOT_DEMO_FILTER)
  get_property(_cudabot_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  set(_cudabot_kept 0)
  foreach(_t IN LISTS _cudabot_targets)
    get_target_property(_type ${_t} TYPE)
    if(_type STREQUAL "EXECUTABLE" AND _t MATCHES "${CUDABOT_DEMO_FILTER}")
      math(EXPR _cudabot_kept "${_cudabot_kept} + 1")
    elseif(_type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$")
      set_target_properties(${_t} PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
  endforeach()
  message(STATUS "CUDABOT_DEMO_FILTER '${CUDABOT_DEMO_FILTER}': ${_cudabot_kept} executables in the default build")
endif()
