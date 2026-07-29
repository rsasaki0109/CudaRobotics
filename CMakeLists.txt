cmake_minimum_required(VERSION 3.18)
project(CPP_ROBOTICS LANGUAGES CXX CUDA)
enable_testing()

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

add_executable(expansion_reset_mcl src/expansion_reset_mcl.cu)
target_link_libraries(expansion_reset_mcl ${OpenCV_LIBS})
target_compile_options(expansion_reset_mcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

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

add_executable(diff_mppi src/diff_mppi.cu)
target_link_libraries(diff_mppi ${OpenCV_LIBS})

add_executable(comparison_diff_mppi src/comparison_diff_mppi.cu)
target_link_libraries(comparison_diff_mppi ${OpenCV_LIBS})

add_executable(benchmark_diff_mppi src/benchmark_diff_mppi.cu)
add_executable(benchmark_mppi_control_update src/benchmark_mppi_control_update.cu)
add_executable(benchmark_mppi_softmin src/benchmark_mppi_softmin.cu)
add_executable(benchmark_mppi_diagnostics src/benchmark_mppi_diagnostics.cu)
add_executable(benchmark_mppi_rollout_ablation
  src/benchmark_mppi_rollout_ablation.cpp
  src/mppi_gpu.cu)
target_compile_options(benchmark_mppi_rollout_ablation PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_executable(benchmark_mppi_rng src/benchmark_mppi_rng.cu)
add_executable(benchmark_diff_mppi_cartpole src/benchmark_diff_mppi_cartpole.cu)
add_executable(benchmark_diff_mppi_dynamic_bicycle src/benchmark_diff_mppi_dynamic_bicycle.cu)
add_executable(benchmark_diff_mppi_manipulator src/benchmark_diff_mppi_manipulator.cu)
add_executable(benchmark_diff_mppi_manipulator_7dof src/benchmark_diff_mppi_manipulator_7dof.cu)

# One-Step CDF-MPPI literature-faithful baseline (arXiv:2509.00836)
add_executable(benchmark_cdf_mppi_7dof src/benchmark_cdf_mppi_7dof.cu)
target_link_libraries(benchmark_cdf_mppi_7dof ${OpenCV_LIBS})
target_compile_options(benchmark_cdf_mppi_7dof PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Differentiable planar pushing: autodiff-through-contact test for Diff-MPPI
add_executable(benchmark_diff_mppi_pushing src/benchmark_diff_mppi_pushing.cu)
target_compile_options(benchmark_diff_mppi_pushing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Differentiable box pushing (position + orientation): contact-point reasoning
add_executable(benchmark_diff_mppi_pushing_box src/benchmark_diff_mppi_pushing_box.cu)
target_compile_options(benchmark_diff_mppi_pushing_box PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(neural_sdf src/neural_sdf.cu)
target_link_libraries(neural_sdf ${OpenCV_LIBS})

add_executable(sdf_potential_field src/sdf_potential_field.cu)
target_link_libraries(sdf_potential_field ${OpenCV_LIBS})

add_executable(sdf_mppi src/sdf_mppi.cu)
target_link_libraries(sdf_mppi ${OpenCV_LIBS})

add_executable(esdf_mppi src/esdf_mppi.cu)
target_link_libraries(esdf_mppi ${OpenCV_LIBS})
target_compile_options(esdf_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_sdf_nav src/comparison_sdf_nav.cu)
target_link_libraries(comparison_sdf_nav ${OpenCV_LIBS})

add_executable(mini_isaac src/mini_isaac.cu)
target_link_libraries(mini_isaac ${OpenCV_LIBS})

add_executable(mini_isaac_rl src/mini_isaac_rl.cu)
target_link_libraries(mini_isaac_rl ${OpenCV_LIBS})

##############################
#Neuroevolution
##############################

add_executable(neuroevo src/neuroevo.cu)
target_link_libraries(neuroevo ${OpenCV_LIBS})

add_executable(comparison_neuroevo src/comparison_neuroevo.cu)
target_link_libraries(comparison_neuroevo ${OpenCV_LIBS})

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

add_executable(comparison_esdf src/comparison_esdf.cu)
target_link_libraries(comparison_esdf ${OpenCV_LIBS})
target_compile_options(comparison_esdf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(pf_esdf src/pf_esdf.cu)
target_link_libraries(pf_esdf ${OpenCV_LIBS})
target_compile_options(pf_esdf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_rrtstar_rewire src/comparison_rrtstar_rewire.cu)
target_link_libraries(comparison_rrtstar_rewire ${OpenCV_LIBS})
target_compile_options(comparison_rrtstar_rewire PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_bundle_adjustment src/gpu_bundle_adjustment.cu)
target_link_libraries(gpu_bundle_adjustment ${OpenCV_LIBS})
target_compile_options(gpu_bundle_adjustment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_lidar_slam src/gpu_lidar_slam.cu)
target_link_libraries(gpu_lidar_slam ${OpenCV_LIBS})
target_compile_options(gpu_lidar_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(visibility_mppi src/visibility_mppi.cu)
target_link_libraries(visibility_mppi ${OpenCV_LIBS})
target_compile_options(visibility_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(pf_realistic_obs src/pf_realistic_obs.cu)
target_link_libraries(pf_realistic_obs ${OpenCV_LIBS})
target_compile_options(pf_realistic_obs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_pose_graph_slam src/gpu_pose_graph_slam.cu)
target_link_libraries(gpu_pose_graph_slam ${OpenCV_LIBS})
target_compile_options(gpu_pose_graph_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_pose_graph_slam_3d src/gpu_pose_graph_slam_3d.cu)
target_link_libraries(gpu_pose_graph_slam_3d ${OpenCV_LIBS})
target_compile_options(gpu_pose_graph_slam_3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_pose_graph_slam_3d_robust src/gpu_pose_graph_slam_3d.cu)
target_link_libraries(gpu_pose_graph_slam_3d_robust ${OpenCV_LIBS})
target_compile_definitions(gpu_pose_graph_slam_3d_robust PRIVATE GPU_POSE_GRAPH_3D_ROBUST=1)
target_compile_options(gpu_pose_graph_slam_3d_robust PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_pose_graph_slam_3d_switchable src/gpu_pose_graph_slam_3d_switchable.cu)
target_link_libraries(gpu_pose_graph_slam_3d_switchable ${OpenCV_LIBS})
target_compile_options(gpu_pose_graph_slam_3d_switchable PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_global_localization_mcl src/gpu_global_localization_mcl.cu)
target_link_libraries(gpu_global_localization_mcl ${OpenCV_LIBS})
target_compile_options(gpu_global_localization_mcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_megaparticles_stein_mcl src/gpu_megaparticles_stein_mcl.cu)
target_link_libraries(gpu_megaparticles_stein_mcl ${OpenCV_LIBS})
target_compile_options(gpu_megaparticles_stein_mcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_megaparticles_lsh src/gpu_megaparticles_lsh.cu)
target_link_libraries(gpu_megaparticles_lsh ${OpenCV_LIBS})
target_compile_options(gpu_megaparticles_lsh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_megaparticles_6dof src/gpu_megaparticles_6dof.cu)
target_link_libraries(gpu_megaparticles_6dof ${OpenCV_LIBS})
target_compile_options(gpu_megaparticles_6dof PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_megaparticles_gicp_mcl src/gpu_megaparticles_gicp_mcl.cu)
target_link_libraries(gpu_megaparticles_gicp_mcl ${OpenCV_LIBS})
target_compile_options(gpu_megaparticles_gicp_mcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_megaparticles_smoother src/gpu_megaparticles_smoother.cu)
target_link_libraries(gpu_megaparticles_smoother ${OpenCV_LIBS})
target_compile_options(gpu_megaparticles_smoother PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_kld_amcl src/gpu_kld_amcl.cu)
target_link_libraries(gpu_kld_amcl ${OpenCV_LIBS})
target_compile_options(gpu_kld_amcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_online_slam src/gpu_online_slam.cu)
target_link_libraries(gpu_online_slam ${OpenCV_LIBS})
target_compile_options(gpu_online_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_online_slam_3d_switchable src/gpu_online_slam_3d_switchable.cu)
target_link_libraries(gpu_online_slam_3d_switchable ${OpenCV_LIBS})
target_compile_options(gpu_online_slam_3d_switchable PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_ndt_2d src/gpu_ndt_2d.cu)
target_link_libraries(gpu_ndt_2d ${OpenCV_LIBS})
target_compile_options(gpu_ndt_2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_correlative_scan_matching src/gpu_correlative_scan_matching.cu)
target_link_libraries(gpu_correlative_scan_matching ${OpenCV_LIBS})
target_compile_options(gpu_correlative_scan_matching PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_csm_loop_closure_slam src/gpu_csm_loop_closure_slam.cu)
target_link_libraries(gpu_csm_loop_closure_slam ${OpenCV_LIBS})
target_compile_options(gpu_csm_loop_closure_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_csm_submap_slam src/gpu_csm_submap_slam.cu)
target_link_libraries(gpu_csm_submap_slam ${OpenCV_LIBS})
target_compile_options(gpu_csm_submap_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_branch_and_bound_csm src/gpu_branch_and_bound_csm.cu)
target_link_libraries(gpu_branch_and_bound_csm ${OpenCV_LIBS})
target_compile_options(gpu_branch_and_bound_csm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_bnb_loop_closure_slam src/gpu_bnb_loop_closure_slam.cu)
target_link_libraries(gpu_bnb_loop_closure_slam ${OpenCV_LIBS})
target_compile_options(gpu_bnb_loop_closure_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_ndt_3d src/gpu_ndt_3d.cu)
target_link_libraries(gpu_ndt_3d ${OpenCV_LIBS})
target_compile_options(gpu_ndt_3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_ndt_3d_multires src/gpu_ndt_3d_multires.cu)
target_link_libraries(gpu_ndt_3d_multires ${OpenCV_LIBS})
target_compile_options(gpu_ndt_3d_multires PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_gicp_2d src/gpu_gicp_2d.cu)
target_link_libraries(gpu_gicp_2d ${OpenCV_LIBS})
target_compile_options(gpu_gicp_2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_gicp_3d src/gpu_gicp_3d.cu)
target_link_libraries(gpu_gicp_3d ${OpenCV_LIBS})
target_compile_options(gpu_gicp_3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_gaussian_splatting src/gpu_gaussian_splatting.cu)
target_link_libraries(gpu_gaussian_splatting ${OpenCV_LIBS})
target_compile_options(gpu_gaussian_splatting PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_gaussian_splatting_slam src/gpu_gaussian_splatting_slam.cu)
target_link_libraries(gpu_gaussian_splatting_slam ${OpenCV_LIBS})
target_compile_options(gpu_gaussian_splatting_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_nerf_volume src/gpu_nerf_volume.cu)
target_link_libraries(gpu_nerf_volume ${OpenCV_LIBS})
target_compile_options(gpu_nerf_volume PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_multi_robot_planner src/gpu_multi_robot_planner.cu)
target_link_libraries(gpu_multi_robot_planner ${OpenCV_LIBS})
target_compile_options(gpu_multi_robot_planner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_multi_robot_place_graph_slam src/gpu_multi_robot_place_graph_slam.cu)
target_link_libraries(gpu_multi_robot_place_graph_slam ${OpenCV_LIBS})
target_compile_options(gpu_multi_robot_place_graph_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_crowd_swarm src/gpu_crowd_swarm.cu)
target_link_libraries(gpu_crowd_swarm ${OpenCV_LIBS})
target_compile_options(gpu_crowd_swarm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_gnn_swarm_controller src/gpu_gnn_swarm_controller.cu)
target_link_libraries(gpu_gnn_swarm_controller ${OpenCV_LIBS})
target_compile_options(gpu_gnn_swarm_controller PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_sfm_mini src/gpu_sfm_mini.cu)
target_link_libraries(gpu_sfm_mini ${OpenCV_LIBS})
target_compile_options(gpu_sfm_mini PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_pcg_solver src/gpu_pcg_solver.cu)
target_link_libraries(gpu_pcg_solver ${OpenCV_LIBS})
target_compile_options(gpu_pcg_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_em_gmm src/gpu_em_gmm.cu)
target_link_libraries(gpu_em_gmm ${OpenCV_LIBS})
target_compile_options(gpu_em_gmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_spectral_clustering src/gpu_spectral_clustering.cu)
target_link_libraries(gpu_spectral_clustering ${OpenCV_LIBS})
target_compile_options(gpu_spectral_clustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_label_propagation src/gpu_label_propagation.cu)
target_link_libraries(gpu_label_propagation ${OpenCV_LIBS})
target_compile_options(gpu_label_propagation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_label_propagation_traversability src/gpu_label_propagation_traversability.cu)
target_link_libraries(gpu_label_propagation_traversability ${OpenCV_LIBS})
target_compile_options(gpu_label_propagation_traversability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_graph_crf_traversability src/gpu_graph_crf_traversability.cu)
target_link_libraries(gpu_graph_crf_traversability ${OpenCV_LIBS})
target_compile_options(gpu_graph_crf_traversability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_gat_traversability_policy src/gpu_gat_traversability_policy.cu)
target_link_libraries(gpu_gat_traversability_policy ${OpenCV_LIBS})
target_compile_options(gpu_gat_traversability_policy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_diff_value_iteration_traversability src/gpu_diff_value_iteration_traversability.cu)
target_link_libraries(gpu_diff_value_iteration_traversability ${OpenCV_LIBS})
target_compile_options(gpu_diff_value_iteration_traversability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_neural_astar_traversability src/gpu_neural_astar_traversability.cu)
target_link_libraries(gpu_neural_astar_traversability ${OpenCV_LIBS})
target_compile_options(gpu_neural_astar_traversability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_anytime_neural_astar_traversability src/gpu_anytime_neural_astar_traversability.cu)
target_link_libraries(gpu_anytime_neural_astar_traversability ${OpenCV_LIBS})
target_compile_options(gpu_anytime_neural_astar_traversability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_multigoal_neural_astar_traversability src/gpu_multigoal_neural_astar_traversability.cu)
target_link_libraries(gpu_multigoal_neural_astar_traversability ${OpenCV_LIBS})
target_compile_options(gpu_multigoal_neural_astar_traversability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_spatiotemporal_neural_astar_traversability src/gpu_spatiotemporal_neural_astar_traversability.cu)
target_link_libraries(gpu_spatiotemporal_neural_astar_traversability ${OpenCV_LIBS})
target_compile_options(gpu_spatiotemporal_neural_astar_traversability PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_experience_graph_neural_planner src/gpu_experience_graph_neural_planner.cu)
target_link_libraries(gpu_experience_graph_neural_planner ${OpenCV_LIBS})
target_compile_options(gpu_experience_graph_neural_planner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_graph_guided_neural_mppi src/gpu_graph_guided_neural_mppi.cu)
target_link_libraries(gpu_graph_guided_neural_mppi ${OpenCV_LIBS})
target_compile_options(gpu_graph_guided_neural_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_kinodynamic_graph_neural_mppi src/gpu_kinodynamic_graph_neural_mppi.cu)
target_link_libraries(gpu_kinodynamic_graph_neural_mppi ${OpenCV_LIBS})
target_compile_options(gpu_kinodynamic_graph_neural_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_interaction_graph_neural_mppi src/gpu_interaction_graph_neural_mppi.cu)
target_link_libraries(gpu_interaction_graph_neural_mppi ${OpenCV_LIBS})
target_compile_options(gpu_interaction_graph_neural_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_multiagent_graph_neural_mppi src/gpu_multiagent_graph_neural_mppi.cu)
target_link_libraries(gpu_multiagent_graph_neural_mppi ${OpenCV_LIBS})
target_compile_options(gpu_multiagent_graph_neural_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_priority_graph_neural_mppi src/gpu_priority_graph_neural_mppi.cu)
target_link_libraries(gpu_priority_graph_neural_mppi ${OpenCV_LIBS})
target_compile_options(gpu_priority_graph_neural_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_intent_graph_neural_mppi src/gpu_intent_graph_neural_mppi.cu)
target_link_libraries(gpu_intent_graph_neural_mppi ${OpenCV_LIBS})
target_compile_options(gpu_intent_graph_neural_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_belief_risk_graph_mppi src/gpu_belief_risk_graph_mppi.cu)
target_link_libraries(gpu_belief_risk_graph_mppi ${OpenCV_LIBS})
target_compile_options(gpu_belief_risk_graph_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_best_response_graph_mppi src/gpu_best_response_graph_mppi.cu)
target_link_libraries(gpu_best_response_graph_mppi ${OpenCV_LIBS})
target_compile_options(gpu_best_response_graph_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_iterative_game_graph_mppi src/gpu_iterative_game_graph_mppi.cu)
target_link_libraries(gpu_iterative_game_graph_mppi ${OpenCV_LIBS})
target_compile_options(gpu_iterative_game_graph_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_noregret_game_graph_mppi src/gpu_noregret_game_graph_mppi.cu)
target_link_libraries(gpu_noregret_game_graph_mppi ${OpenCV_LIBS})
target_compile_options(gpu_noregret_game_graph_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_safe_noregret_game_graph_mppi src/gpu_safe_noregret_game_graph_mppi.cu)
target_link_libraries(gpu_safe_noregret_game_graph_mppi ${OpenCV_LIBS})
target_compile_options(gpu_safe_noregret_game_graph_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_learned_safety_dual_graph_mppi src/gpu_learned_safety_dual_graph_mppi.cu)
target_link_libraries(gpu_learned_safety_dual_graph_mppi ${OpenCV_LIBS})
target_compile_options(gpu_learned_safety_dual_graph_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_trainable_safety_dual_graph_mppi src/gpu_trainable_safety_dual_graph_mppi.cu)
target_link_libraries(gpu_trainable_safety_dual_graph_mppi ${OpenCV_LIBS})
target_compile_options(gpu_trainable_safety_dual_graph_mppi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_planner_showdown_benchmark src/gpu_planner_showdown_benchmark.cu)
target_link_libraries(gpu_planner_showdown_benchmark ${OpenCV_LIBS})
target_compile_options(gpu_planner_showdown_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_planner_falsifier_benchmark src/gpu_planner_falsifier_benchmark.cu)
target_compile_options(gpu_planner_falsifier_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)

add_executable(gpu_interaction_graph_risk src/gpu_interaction_graph_risk.cu)
target_link_libraries(gpu_interaction_graph_risk ${OpenCV_LIBS})
target_compile_options(gpu_interaction_graph_risk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_reciprocal_risk_planner src/gpu_reciprocal_risk_planner.cu)
target_link_libraries(gpu_reciprocal_risk_planner ${OpenCV_LIBS})
target_compile_options(gpu_reciprocal_risk_planner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_hungarian_assignment src/gpu_hungarian_assignment.cu)
target_link_libraries(gpu_hungarian_assignment ${OpenCV_LIBS})
target_compile_options(gpu_hungarian_assignment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_assignment_tracking src/gpu_assignment_tracking.cu)
target_link_libraries(gpu_assignment_tracking ${OpenCV_LIBS})
target_compile_options(gpu_assignment_tracking PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_cma_es src/gpu_cma_es.cu)
target_link_libraries(gpu_cma_es ${OpenCV_LIBS})
target_compile_options(gpu_cma_es PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_mcts_planner src/gpu_mcts_planner.cu)
target_link_libraries(gpu_mcts_planner ${OpenCV_LIBS})
target_compile_options(gpu_mcts_planner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_diffusion_planner src/gpu_diffusion_planner.cu)
target_link_libraries(gpu_diffusion_planner ${OpenCV_LIBS})
target_compile_options(gpu_diffusion_planner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_diffusion_policy src/gpu_diffusion_policy.cu)
target_link_libraries(gpu_diffusion_policy ${OpenCV_LIBS})
target_compile_options(gpu_diffusion_policy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(diff_e2e_slam src/diff_e2e_slam.cu)
target_link_libraries(diff_e2e_slam ${OpenCV_LIBS})
target_compile_options(diff_e2e_slam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_voxel_map src/comparison_voxel_map.cu)
target_link_libraries(comparison_voxel_map ${OpenCV_LIBS})
target_compile_options(comparison_voxel_map PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_collision_check src/comparison_collision_check.cu)
target_link_libraries(comparison_collision_check ${OpenCV_LIBS})
target_compile_options(comparison_collision_check PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_fastslam src/comparison_fastslam.cu)
target_link_libraries(comparison_fastslam ${OpenCV_LIBS})
target_compile_options(comparison_fastslam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_amcl src/comparison_amcl.cu)
target_link_libraries(comparison_amcl ${OpenCV_LIBS})
target_compile_options(comparison_amcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_expansion_reset_mcl src/comparison_expansion_reset_mcl.cu)
target_link_libraries(comparison_expansion_reset_mcl ${OpenCV_LIBS})
target_compile_options(comparison_expansion_reset_mcl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_lidar_sim src/comparison_lidar_sim.cu)
target_link_libraries(comparison_lidar_sim ${OpenCV_LIBS})
target_compile_options(comparison_lidar_sim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_lidar3d_sim src/comparison_lidar3d_sim.cu)
target_link_libraries(comparison_lidar3d_sim ${OpenCV_LIBS})
target_compile_options(comparison_lidar3d_sim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_lidar3d_realistic src/comparison_lidar3d_realistic.cu)
target_link_libraries(comparison_lidar3d_realistic ${OpenCV_LIBS})
target_compile_options(comparison_lidar3d_realistic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(comparison_gaussian_splatting src/comparison_gaussian_splatting.cu)
target_link_libraries(comparison_gaussian_splatting ${OpenCV_LIBS})
target_compile_options(comparison_gaussian_splatting PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_batched_ilqr src/gpu_batched_ilqr.cu)
target_link_libraries(gpu_batched_ilqr ${OpenCV_LIBS})
target_compile_options(gpu_batched_ilqr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_parallel_ilqr src/gpu_parallel_ilqr.cu)
target_link_libraries(gpu_parallel_ilqr ${OpenCV_LIBS})
target_compile_options(gpu_parallel_ilqr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_filterreg src/gpu_filterreg.cu src/filterreg_gpu.cu)
target_include_directories(gpu_filterreg PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_filterreg ${OpenCV_LIBS})
target_compile_options(gpu_filterreg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_bcpd src/gpu_bcpd.cu)
target_link_libraries(gpu_bcpd ${OpenCV_LIBS})
target_compile_options(gpu_bcpd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_sinkhorn_reg src/gpu_sinkhorn_reg.cu)
target_link_libraries(gpu_sinkhorn_reg ${OpenCV_LIBS})
target_compile_options(gpu_sinkhorn_reg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_robust_treg src/gpu_robust_treg.cu)
target_link_libraries(gpu_robust_treg ${OpenCV_LIBS})
target_compile_options(gpu_robust_treg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

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

add_executable(gpu_kiss_icp src/gpu_kiss_icp.cu)
target_link_libraries(gpu_kiss_icp ${OpenCV_LIBS})
target_compile_options(gpu_kiss_icp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(kiss_icp_gpu_streaming_smoke tests/kiss_icp_gpu_streaming_smoke.cu)
target_link_libraries(kiss_icp_gpu_streaming_smoke cudarobotics_kiss_icp_gpu)
target_compile_options(kiss_icp_gpu_streaming_smoke PRIVATE
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

add_executable(gpu_mpc_qp src/gpu_mpc_qp.cu)
target_link_libraries(gpu_mpc_qp ${OpenCV_LIBS})
target_compile_options(gpu_mpc_qp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_constrained_mpc src/gpu_constrained_mpc.cu)
target_link_libraries(gpu_constrained_mpc ${OpenCV_LIBS})
target_compile_options(gpu_constrained_mpc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_diff_contact_push src/gpu_diff_contact_push.cu)
target_link_libraries(gpu_diff_contact_push ${OpenCV_LIBS})
target_compile_options(gpu_diff_contact_push PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_robust_p2plane_reg src/gpu_robust_p2plane_reg.cu)
target_link_libraries(gpu_robust_p2plane_reg ${OpenCV_LIBS})
target_compile_options(gpu_robust_p2plane_reg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_real_bunny_reg src/gpu_real_bunny_reg.cu)
target_link_libraries(gpu_real_bunny_reg ${OpenCV_LIBS})
target_compile_options(gpu_real_bunny_reg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_fgr src/gpu_fgr.cu)
target_link_libraries(gpu_fgr ${OpenCV_LIBS})
target_compile_options(gpu_fgr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_filterreg_p2plane src/gpu_filterreg_p2plane.cu)
target_link_libraries(gpu_filterreg_p2plane ${OpenCV_LIBS})
target_compile_options(gpu_filterreg_p2plane PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_tsdf_fusion src/gpu_tsdf_fusion.cu)
target_link_libraries(gpu_tsdf_fusion ${OpenCV_LIBS})
target_compile_options(gpu_tsdf_fusion PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_dbscan src/gpu_dbscan.cu)
target_link_libraries(gpu_dbscan ${OpenCV_LIBS})
target_compile_options(gpu_dbscan PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_sgm_stereo src/gpu_sgm_stereo.cu)
target_link_libraries(gpu_sgm_stereo ${OpenCV_LIBS})
target_compile_options(gpu_sgm_stereo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_wavefront_planner src/gpu_wavefront_planner.cu)
target_link_libraries(gpu_wavefront_planner ${OpenCV_LIBS})
target_compile_options(gpu_wavefront_planner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_marching_cubes src/gpu_marching_cubes.cu)
target_link_libraries(gpu_marching_cubes ${OpenCV_LIBS})
target_compile_options(gpu_marching_cubes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_mppi_racing src/gpu_mppi_racing.cu)
target_link_libraries(gpu_mppi_racing ${OpenCV_LIBS})
target_compile_options(gpu_mppi_racing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(gpu_direct_vo src/gpu_direct_vo.cu)
target_link_libraries(gpu_direct_vo ${OpenCV_LIBS})
target_compile_options(gpu_direct_vo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(comparison_reeds_shepp_fan src/comparison_reeds_shepp_fan.cu)
target_link_libraries(comparison_reeds_shepp_fan ${OpenCV_LIBS})
target_compile_options(comparison_reeds_shepp_fan PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(diff_pf src/diff_pf.cu)
target_link_libraries(diff_pf ${OpenCV_LIBS})
target_compile_options(diff_pf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(diff_pf_mlp src/diff_pf_mlp.cu)
target_link_libraries(diff_pf_mlp ${OpenCV_LIBS})
target_compile_options(diff_pf_mlp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

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

##############################
#SwarmOptimization
##############################

add_executable(pso_cuda src/pso.cu)
target_link_libraries(pso_cuda ${OpenCV_LIBS})

add_executable(differential_evolution src/differential_evolution.cu)
target_link_libraries(differential_evolution ${OpenCV_LIBS})

add_executable(cma_es src/cma_es.cu)
target_link_libraries(cma_es ${OpenCV_LIBS})

add_executable(aco_tsp src/aco_tsp.cu)
target_link_libraries(aco_tsp ${OpenCV_LIBS})

add_executable(comparison_swarm src/comparison_swarm.cu)
target_link_libraries(comparison_swarm ${OpenCV_LIBS})

add_executable(cuda_streams_demo src/cuda_streams_demo.cu)

add_executable(gpu_frontier_exploration src/gpu_frontier_exploration.cu)
target_link_libraries(gpu_frontier_exploration ${OpenCV_LIBS})
target_compile_options(gpu_frontier_exploration PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

##############################
# Common Foundation Tests
##############################
add_executable(gpu_lk_optical_flow src/gpu_lk_optical_flow.cu)
target_link_libraries(gpu_lk_optical_flow ${OpenCV_LIBS})
target_compile_options(gpu_lk_optical_flow PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(test_autodiff src/test_autodiff.cu)
add_executable(test_gpu_mlp src/test_gpu_mlp.cu)

add_executable(test_host_math src/test_host_math.cpp)
target_link_libraries(test_host_math ${OpenCV_LIBS})

##############################
# CTest Registration
##############################

# GPU tests (require NVIDIA GPU)
add_test(NAME test_autodiff COMMAND ${PROJECT_SOURCE_DIR}/bin/test_autodiff)
add_test(NAME test_gpu_mlp COMMAND ${PROJECT_SOURCE_DIR}/bin/test_gpu_mlp)
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

add_test(NAME esdf_2d_gpu_smoke
  COMMAND $<TARGET_FILE:esdf_2d_gpu_smoke>)
set_tests_properties(esdf_2d_gpu_smoke PROPERTIES
  TIMEOUT 60
  LABELS "gpu;mapping;esdf;reference")

add_test(NAME gpu_planner_showdown_gate
  COMMAND ${PROJECT_SOURCE_DIR}/bin/gpu_planner_showdown_benchmark
    --check
    --no-video
    --scenario baseline
    --json ${CMAKE_BINARY_DIR}/gpu_planner_showdown_benchmark.json
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gpu_planner_showdown_gate PROPERTIES
  TIMEOUT 90
  LABELS "gpu;showdown;benchmark")

add_test(NAME gpu_planner_falsifier_gate
  COMMAND ${PROJECT_SOURCE_DIR}/bin/gpu_planner_falsifier_benchmark
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
    --multi-gpu-run ${CMAKE_BINARY_DIR}/fixture_other_gpu
    --dry-run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_paper_artifact_contract
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/test_paper_artifact_contract.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME check_paper_artifact_manifests
  COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/validate_paper_artifacts.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
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
set_tests_properties(validate_design_workflow check_design_regressions check_scaffold check_repro_suite_dry_run check_mppi_zoo_smoke_dry_run check_mppi_zoo_chart_render check_planner_showdown_summary check_planner_showdown_matrix_summary check_planner_falsifier_summary check_erl_navigation_bag_inspector check_offline_rosbag_db3_analyzer check_offline_rosbag_motion_decoder check_offline_rosbag_motion_comparison check_offline_rosbag_clearance_decoder check_cuda_acceleration_auditor check_registration_suite check_mppi_rosbag_evaluation check_version_consistency check_cudanav_contract check_cudanav_autonomy_suite_contract check_cudanav_autonomy_suite_release_dry_run check_cudanav_ros_ci_contract check_cudanav_systems_publisher_contract check_paper_artifact_contract check_paper_artifact_manifests check_contact_paper_results check_contact_paper_results_contract check_contact_robustness_contract check_contact_robustness_release_dry_run check_contact_matched_compute_contract check_contact_matched_compute_release_dry_run check_contact_external_fidelity_contract check_contact_external_fidelity_release_dry_run check_python_core_sync check_python_release_artifacts check_release_preflight check_release_ci_evidence check_v0_2_release_evidence
  PROPERTIES
  TIMEOUT 30
  LABELS "python")

# CPU tests (no GPU required)
add_test(NAME test_host_math COMMAND ${PROJECT_SOURCE_DIR}/bin/test_host_math)
set_tests_properties(test_host_math PROPERTIES
  TIMEOUT 10
  LABELS "cpu")

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

add_executable(gpu_jfa_edt src/gpu_jfa_edt.cu)
target_link_libraries(gpu_jfa_edt ${OpenCV_LIBS})
target_compile_options(gpu_jfa_edt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)

add_executable(gpu_kdtree_nn src/gpu_kdtree_nn.cu)
target_link_libraries(gpu_kdtree_nn ${OpenCV_LIBS})
target_compile_options(gpu_kdtree_nn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false>)
