/*************************************************************************
    Benchmark: MPPI vs Diff-MPPI
    - Runs multiple navigation scenarios across configurable sample sweeps
    - Compares sampling-only MPPI against gradient-refined variants
    - Supports both fixed-budget and cap-based wall-clock analyses downstream
    - Writes per-episode CSV to build/benchmark_diff_mppi.csv by default
 ************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "diff_cost.cuh"
#include "diff_dynamics.cuh"
#include "hybrid_astar_pp.h"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;
using namespace cudabot;

static const float WORKSPACE = 50.0f;
static const int MAX_OBSTACLES = 16;
static const int MAX_DYNAMIC_OBSTACLES = 8;
static const float DEFAULT_LAMBDA = 8.0f;
static const int DEFAULT_T_HORIZON = 30;
static const int BENCH_WARMUP_ITERS = 4;

__constant__ Obstacle d_obstacles_bench[MAX_OBSTACLES];

struct DynamicObstacle {
    float x;
    float y;
    float vx;
    float vy;
    float r;
};

__constant__ DynamicObstacle d_dynamic_obstacles_bench[MAX_DYNAMIC_OBSTACLES];

struct Scenario {
    string name;
    float start_x = 5.0f;
    float start_y = 5.0f;
    float start_theta = 0.0f;
    float start_v = 0.0f;
    float goal_tol = 2.0f;
    int max_steps = 220;
    BicycleParams params;
    CostParams cost_params;
    float grad_alpha_scale = 1.0f;
    int n_obs = 0;
    Obstacle obstacles[MAX_OBSTACLES];
    int n_dyn_obs = 0;
    DynamicObstacle dynamic_obstacles[MAX_DYNAMIC_OBSTACLES];
    bool use_dynamic_mismatch = false;
    float dyn_time_offset_max = 0.0f;
    float dyn_speed_scale_max = 0.0f;
    float dyn_lateral_jitter = 0.0f;
    bool use_model_mismatch = false;
    float eval_wheelbase_scale = 1.0f;
    float eval_max_speed_scale = 1.0f;
    float eval_max_steer_scale = 1.0f;
};

struct PlannerVariant {
    string name;
    // planner_kind: 0 = MPPI / gradient / feedback (legacy path),
    //               1 = DWA (discrete dynamic-window grid search),
    //               2 = STOMP (cost-weighted noise with smoothness projection).
    // For kinds 1 and 2 the legacy MPPI flags (use_sampling/use_gradient/use_feedback)
    // are ignored and a dedicated controller branch runs instead.
    int planner_kind = 0;
    // Per-variant override for the MPPI / Diff-MPPI / STOMP / hybrid_astar_mppi
    // rollout horizon. 0 means "use DEFAULT_T_HORIZON (or the --t-horizon CLI
    // override if supplied)". A positive value pins this variant to a
    // specific T so the default sweep uses the right horizon without the
    // caller needing to know the flag. Mirrors the per-variant
    // dwa_predict_steps mechanism for DWA-family planners.
    int t_horizon = 0;
    // DWA parameters (only used when planner_kind == 1).
    int dwa_n_accel = 9;
    int dwa_n_steer = 13;
    int dwa_predict_steps = 20;
    float dwa_accel_min = -3.0f;
    float dwa_accel_max = 3.0f;
    // Weights are scaled to roughly match MPPI's cost_params so DWA's per-step
    // contributions are comparable. dwa_w_terminal=20 was selected by grid
    // search against dynamic_pincer hard cells: bumping from 12 to 20 lifts
    // the hard-cell success rate from 0.50 to 0.83 (collision-free) with
    // negligible impact on the easier dynamic_crossing / dynamic_slalom cells.
    // See scripts/grid_search_dwa_weights.py for the search procedure.
    float dwa_w_goal = 5.0f;
    float dwa_w_speed = 0.20f;
    float dwa_w_obs = 11.5f;
    float dwa_w_heading = 0.50f;
    float dwa_w_terminal = 20.0f;
    // STOMP parameters (only used when planner_kind == 2).
    int stomp_iterations = 2;
    int stomp_smoothing_passes = 1;
    float stomp_h = 10.0f;        // STOMP weight sharpness
    float stomp_sigma_accel = 1.5f;
    float stomp_sigma_steer = 0.18f;
    // Hybrid A* + Pure Pursuit parameters (only used when planner_kind == 3).
    // Hybrid A* plans once at episode start against the STATIC obstacles of
    // the scenario; dynamic obstacles are deliberately ignored to make
    // the global-planner-blind-to-dynamic-obstacles paradigm gap explicit.
    // Pure pursuit then tracks the planned path with a fixed lookahead.
    int hap_n_steer = 7;
    // dt * v_search gives the metric step size per node expansion; we want
    // it > cell_size = 1.0m so that children land in distinct (x,y,theta)
    // cells. With v_search=2.5, dt=1.0 we move 2.5 m per expansion. sub_steps
    // controls the integration fidelity inside the bicycle update.
    int hap_sub_steps = 8;
    float hap_dt = 1.0f;
    float hap_v_search = 2.5f;
    float hap_steer_penalty = 0.05f;
    float hap_robot_radius = 0.6f;
    float hap_lookahead = 4.0f;
    float hap_target_speed = 5.0f;
    float hap_speed_gain = 1.5f;
    int hap_max_expansions = 100000;
    // Safety inflation only used by the dyn-aware variant (planner_kind=5).
    // ~1 m of buffer is the empirical sweet spot on the hard cells: enough
    // to absorb the constant-speed-search vs. acceleration-from-rest timing
    // mismatch while still letting the search find a path through the
    // pincer convergence funnel. Larger values (2 m+) over-inflate and the
    // search either degrades into the static path or fails entirely on
    // pincer.
    float hap_dyn_inflation = 1.0f;
    // Hybrid A* + DWA hybrid (planner_kind == 4). Reuses hap_* search params
    // and dwa_* grid params; DWA cost replaces goal-distance/heading with
    // path-follow terms (nearest path point + lookahead heading).
    float had_w_path = 5.0f;
    float had_w_heading = 0.5f;
    float had_w_speed = 0.20f;
    float had_w_obs = 11.5f;
    float had_w_terminal = 20.0f;
    int had_lookahead_idx = 2;
    // Hybrid A* + MPPI hybrid (planner_kind == 6). Same path-follow cost
    // shape as the dwa hybrid but the per-step controller is the MPPI
    // sampling pipeline; ham_w_* mirror had_w_* but stay separately
    // tunable so the MPPI noise (which integrates over a longer horizon)
    // can be balanced against the path-follow pull independently of DWA.
    float ham_w_path = 5.0f;
    float ham_w_heading = 0.5f;
    float ham_w_speed = 0.20f;
    float ham_w_obs = 11.5f;
    // MPPI's noise dilutes deterministic goal-pull (vs. DWA's argmin), so
    // a larger terminal weight is needed to keep samples coherent toward
    // the goal in the last few metres.
    float ham_w_terminal = 50.0f;
    int ham_lookahead_idx = 2;
    bool use_sampling = true;
    bool use_feedback = false;
    bool use_gradient = false;
    int feedback_mode = 0;
    int feedback_passes = 1;
    int replan_stride = 1;
    int grad_steps = 0;
    int grad_update_horizon = 0;
    float alpha = 0.0f;
    float grad_skip_threshold = 0.0f;  // skip gradient step if norm < threshold (0 = never skip)
    float sampling_lambda = DEFAULT_LAMBDA;
    float feedback_gain_scale = 1.0f;
    float feedback_noise_accel = 0.9f;
    float feedback_noise_steer = 0.10f;
    float feedback_longitudinal_gain = 0.0f;
    float feedback_speed_gain = 0.0f;
    float feedback_lateral_gain = 0.0f;
    float feedback_heading_gain = 0.0f;
    float feedback_setpoint_blend = 0.0f;
    float feedback_q_position = 0.0f;
    float feedback_q_heading = 0.0f;
    float feedback_q_speed = 0.0f;
    float feedback_r_accel = 0.0f;
    float feedback_r_steer = 0.0f;
    float feedback_terminal_scale = 0.0f;
    float feedback_ref_blend = 1.0f;
    float feedback_cov_regularization = 0.0f;
    float feedback_cov_blend = 1.0f;
    float feedback_lqr_blend = 0.0f;
    // Step-MPPI: learned sampling distribution
    bool use_learned_sampling = false;
    int mlp_hidden_size = 32;   // unused in lightweight mode; kept for API compat
    float mlp_lr = 0.001f;      // EMA learning rate for sampling bias
    bool use_learned_sigma = false;
    float learned_sigma_lr = 0.12f;
    float learned_min_accel_sigma = 0.20f;
    float learned_min_steer_sigma = 0.020f;
    float learned_max_accel_sigma = 3.50f;
    float learned_max_steer_sigma = 0.35f;
    float learned_init_accel_sigma = 1.50f;
    float learned_init_steer_sigma = 0.18f;
    // LP-MPPI: low-pass filtering of sampled control perturbations
    // (arXiv:2503.11717). This lightweight reproduction uses a one-pole IIR
    // filter along the horizon; alpha=1.0 recovers vanilla MPPI noise.
    bool use_low_pass_sampling = false;
    float lp_alpha = 0.35f;
    // DBaS-Log-MPPI: Efficient and Safe Trajectory Optimization via Barrier States
    // (arXiv:2504.06437). Lightweight reproduction: sample MPPI controls
    // with a symmetric normal-lognormal heavy-tailed perturbation, adapt the
    // exploration scale from the current barrier state, and add a continuous
    // obstacle-clearance barrier cost during rollout.
    bool use_dbas_log_sampling = false;
    float dbas_safe_margin = 0.45f;
    float dbas_barrier_eps = 0.40f;
    float dbas_barrier_cap = 30.0f;
    float dbas_barrier_weight = 180.0f;
    float dbas_gamma = 0.25f;
    float dbas_mu = 0.70f;
    float dbas_log_sigma = 0.45f;
    float dbas_lognormal_clip = 4.0f;
    float dbas_noise_scale = 1.0f;
    float dbas_speed_damping = 0.15f;
    // dsMPPI: deterministic sampling MPPI (arXiv:2601.03893).
    // Lightweight reproduction: replace per-rollout random samples with
    // antithetic low-discrepancy samples, apply temporal low-pass coloring,
    // and optionally run multiple MPPI-style proposal updates per control step.
    bool use_deterministic_sampling = false;
    int ds_iterations = 2;
    float ds_alpha = 0.35f;
    float ds_noise_scale = 1.0f;
    float ds_momentum = 0.20f;
    int ds_stride = 4093;
    bool ds_adapt_sigma = false;
    float ds_sigma_blend = 0.35f;
    float ds_min_accel_sigma = 0.20f;
    float ds_min_steer_sigma = 0.020f;
    float ds_max_accel_sigma = 4.00f;
    float ds_max_steer_sigma = 0.35f;
    bool ds_elite_update = false;
    int ds_elite_count = 16;
    float ds_elite_sigma_blend = 0.25f;
    // pi-MPPI: projection-filtered MPPI (arXiv:2504.10962).
    // Lightweight reproduction: project sampled control sequences onto box
    // constraints for controls plus first/second finite differences before
    // rollout, then project the weighted-average sequence again.
    bool use_projection_sampling = false;
    int projection_passes = 2;
    float projection_max_accel_delta = 1.20f;
    float projection_max_steer_delta = 0.10f;
    float projection_max_accel_ddelta = 1.00f;
    float projection_max_steer_ddelta = 0.08f;
    // CDF-MPPI: Configuration-space Distance Field MPPI
    // (arXiv:2509.00836). Lightweight reproduction: use the existing
    // obstacle layout as a differentiable 2D C-space distance field, seed the
    // nominal with the negative distance-field potential, and add a smooth CDF
    // margin cost so shorter horizons can still react to obstacles.
    bool use_cdf_guidance = false;
    float cdf_seed_blend = 0.30f;
    float cdf_goal_pull = 1.0f;
    float cdf_obs_pull = 3.5f;
    float cdf_dyn_pull = 1.0f;
    float cdf_safe_margin = 3.0f;
    float cdf_obs_cost = 1.2f;
    float cdf_dyn_cost = 0.6f;
    // PA-MPPI: Perception-Aware MPPI for unknown-environment navigation
    // (arXiv:2509.14978). Lightweight reproduction: when the goal is
    // occluded, bias MPPI weights toward trajectories that improve line of
    // sight to the goal/frontier direction and align heading with that view.
    bool use_pa_perception_cost = false;
    float pa_safe_margin = 0.35f;
    float pa_poi_weight = 120.0f;
    float pa_occlusion_weight = 900.0f;
    float pa_frontier_reward = 500.0f;
    float pa_forward_occ_weight = 180.0f;
    float pa_goal_gate = 3.0f;
    float pa_activation = 0.08f;
    float pa_ray_length = 8.0f;
    float pa_score_cap = 300.0f;
    // SOPPI: Stein-Optimized Path-Integral Inference (arXiv:2511.02015).
    // Lightweight reproduction: apply SVGD in per-timestep action space after
    // Gaussian MPPI sampling, then re-rollout the moved samples before weighting.
    bool use_soppi_sampling = false;
    int soppi_svgd_iters = 1;
    float soppi_step_size = 0.045f;
    float soppi_bandwidth = 2.0f;
    int soppi_neighbor_count = 0;  // 0 = all particles; >0 = deterministic particle subset
    // SVG-MPPI: Stein Variational Guided MPPI (arXiv:2309.11040).
    // Lightweight reproduction: estimate the target mode as the best rollout,
    // then reweight MPPI samples by a trajectory-space RBF kernel around that
    // mode before the closed-form MPPI control update.
    bool use_svg_mode_guidance = false;
    float svg_bandwidth = 30.0f;
    float svg_mode_weight = 1.0f;
    int svg_stride = 2;
    // PR/EMPPI: parameter-robust MPPI (arXiv:2601.02948 / arXiv:2006.03106).
    // Lightweight reproduction: each sampled control sequence is evaluated
    // under a small deterministic set of bicycle-parameter particles, and the
    // MPPI weight uses an average/worst-case blend of their trajectory costs.
    bool use_parameter_robust_sampling = false;
    int pr_param_particles = 3;
    float pr_wheelbase_span = 0.45f;
    float pr_max_speed_span = 0.20f;
    float pr_max_steer_span = 0.18f;
    float pr_worst_blend = 0.50f;
    // Shield-MPPI: CBF-augmented MPPI plus local repair
    // (arXiv:2302.11719). Lightweight reproduction: add a discrete CBF
    // margin-violation penalty to rollout costs, then repair the first
    // control by a small host-side safe-action grid before execution.
    bool use_shield_cost = false;
    bool use_shield_repair = false;
    float shield_safe_margin = 1.2f;
    float shield_cbf_alpha = 0.40f;
    float shield_cbf_weight = 90.0f;
    int shield_repair_steps = 8;
    int shield_repair_grid = 5;
    float shield_repair_accel_delta = 2.0f;
    float shield_repair_steer_delta = 0.30f;
    float shield_repair_safety_weight = 250.0f;
    float shield_repair_control_weight = 0.25f;
    // SC-MPPI: Safe Importance Sampling / safety-controlled MPPI
    // (arXiv:2303.03441). Lightweight reproduction: embed a local
    // obstacle-margin safety controller inside each sampled rollout before
    // the dynamics step, so the sampled trajectories themselves are safer.
    bool use_safety_controlled_sampling = false;
    float sc_safe_margin = 1.0f;
    float sc_avoid_gain = 0.55f;
    float sc_speed_gain = 0.80f;
    float sc_max_steer_delta = 0.28f;
    float sc_max_accel_delta = 1.8f;
    float sc_control_weight = 0.05f;
    // CSC-MPPI: constrained sampling cluster MPPI (arXiv:2506.16386).
    // Lightweight reproduction: use constrained/safety-controlled rollouts,
    // then avoid MPPI's mode-averaging failure by selecting a low-cost
    // representative trajectory from coarse trajectory-space clusters.
    bool use_cluster_representative_update = false;
    int csc_cluster_count = 4;
    float csc_safe_margin = 0.25f;
    float csc_constraint_weight = 4000.0f;
    float csc_update_blend = 0.85f;
    // DM-MPPI: Datamodel for efficient/safe MPPI (arXiv:2512.00759).
    // Lightweight reproduction: replace the learned influence predictor with
    // a cost/margin feature surrogate, prune low-influence samples during the
    // control update, and boost the score of constraint-violating rollouts.
    bool use_datamodel_influence_pruning = false;
    float dm_keep_fraction = 0.40f;
    float dm_cost_temperature = 8.0f;
    float dm_safe_margin = 0.75f;
    float dm_prob_sigma = 0.65f;
    float dm_violation_weight = 3000.0f;
    float dm_safety_power = 1.0f;
    // Tsallis VI-MPC / Tsallis-MPPI (arXiv:2104.00241).
    // Lightweight reproduction: replace the exponential MPPI optimality
    // likelihood with a q-exponential over normalized rollout costs.
    bool use_tsallis_weights = false;
    float tsallis_q = 0.70f;
    float tsallis_temperature = 8.0f;
    float tsallis_min_weight = 1.0e-8f;
    // CC-MPPI: covariance-controlled trajectory distribution
    // (arXiv:2109.12147). Lightweight reproduction: penalize terminal
    // rollout dispersion around the current cost-weighted terminal mode
    // before the MPPI control update, approximating terminal covariance
    // steering without solving the full LTV covariance-control subproblem.
    bool use_covariance_control_weights = false;
    float cc_terminal_weight = 1.0f;
    float cc_terminal_target_radius = 4.0f;
    float cc_heading_weight = 0.35f;
    float cc_speed_weight = 0.10f;
    float cc_min_weight = 1.0e-10f;
    // TD-CD-MPPI: Temporal-Difference Constraint-Discounted MPPI
    // (IEEE RAL 2026). Lightweight reproduction: replace the learned
    // terminal value function with an analytic value-to-go surrogate, and
    // modulate a trajectory survival discount from constraint margins.
    bool use_td_cd_weights = false;
    float td_terminal_value_scale = 2.0f;
    float td_safe_margin = 0.85f;
    float td_discount_sigma = 0.65f;
    float td_discount_power = 1.0f;
    float td_failure_cost = 35000.0f;
    // C2U-MPPI: chance-constrained unscented MPPI
    // (arXiv:2501.08520). Lightweight reproduction: propagate dynamic
    // obstacle prediction uncertainty with a small sigma-point set and
    // convert the resulting margin distribution into a deterministic
    // chance-constraint backoff before the MPPI weight update.
    bool use_c2u_chance_constraints = false;
    float c2u_safe_margin = 0.0f;
    float c2u_robot_sigma = 0.12f;
    float c2u_dyn_sigma0 = 0.30f;
    float c2u_dyn_sigma_growth = 0.06f;
    float c2u_risk_z = 1.28f;
    float c2u_prob_sigma = 0.85f;
    float c2u_probability_power = 0.80f;
    float c2u_violation_weight = 1200.0f;
    float c2u_min_probability = 1.0e-6f;
    // DUCCT-MPPI: Dual-Uncertainty Chance-Constrained Tube MPPI
    // (arXiv:2605.28330). Lightweight reproduction: share a simple
    // localization uncertainty tube across rollouts, aggregate static and
    // dynamic obstacle collision risks with 1 - prod(1 - p_i), and inject
    // the joint risk as a soft cost plus hard-threshold rejection penalty.
    bool use_ducct_risk = false;
    float ducct_loc_sigma0 = 0.12f;
    float ducct_loc_sigma_growth = 0.04f;
    float ducct_pred_sigma0 = 0.35f;
    float ducct_pred_sigma_growth = 0.06f;
    float ducct_static_sigma = 0.12f;
    float ducct_risk_weight = 1800.0f;
    float ducct_hard_threshold = 0.65f;
    float ducct_reject_cost = 25000.0f;
    float ducct_survival_power = 0.60f;
    float ducct_min_survival = 1.0e-6f;
    // DRA-MPPI: Dynamic Risk-Aware MPPI
    // (arXiv:2506.21205). Lightweight reproduction: estimate dynamic
    // obstacle collision probability with a fixed low-discrepancy Monte
    // Carlo stencil over each rollout state's collision disk, then inject
    // CP as a soft risk cost plus hard threshold rejection.
    bool use_dra_risk = false;
    int dra_mc_samples = 12;
    float dra_robot_radius = 0.60f;
    float dra_pred_sigma0 = 0.35f;
    float dra_pred_sigma_growth = 0.06f;
    float dra_mode_weight = 0.0f;
    float dra_mode_lateral_offset = 1.40f;
    float dra_soft_weight = 1200.0f;
    float dra_hard_threshold = 0.45f;
    float dra_reject_cost = 18000.0f;
    float dra_survival_power = 0.30f;
    float dra_min_survival = 1.0e-6f;
    // BC-MPPI: Bayesian/probabilistic constraint layer
    // (arXiv:2510.00272). Lightweight reproduction: estimate a rollout's
    // feasibility probability from obstacle margins and multiply MPPI's
    // trajectory weight by that scalar instead of adding a penalty cost.
    bool use_bc_safety_layer = false;
    float bc_safe_margin = 1.0f;
    float bc_prob_sigma = 0.75f;
    float bc_probability_power = 1.0f;
    float bc_min_probability = 1.0e-5f;
};

struct EpisodeMetrics {
    string scenario;
    string planner;
    int seed = 0;
    int k_samples = 0;
    int t_horizon = 0;
    int grad_steps = 0;
    float alpha = 0.0f;
    int reached_goal = 0;
    int collision_free = 0;
    int success = 0;
    int steps = 0;
    float final_distance = 0.0f;
    float min_goal_distance = 0.0f;
    float cumulative_cost = 0.0f;
    int collisions = 0;
    float mean_control_delta = 0.0f;
    float control_roughness = 0.0f;
    float avg_control_ms = 0.0f;
    float total_control_ms = 0.0f;
    float episode_ms = 0.0f;
    long long sample_budget = 0;
};

struct TrajectoryRow {
    string scenario;
    string planner;
    int seed = 0;
    int k_samples = 0;
    int episode_step = 0;
    float x = 0.0f;
    float y = 0.0f;
    float theta = 0.0f;
    float v = 0.0f;
    float goal_distance = 0.0f;
};

struct TraceRow {
    string scenario;
    string planner;
    int seed = 0;
    int k_samples = 0;
    int grad_steps = 0;
    float alpha = 0.0f;
    int episode_step = 0;
    int horizon_step = 0;
    float goal_distance = 0.0f;
    float min_obstacle_margin = 0.0f;
    float control_ms = 0.0f;
    float sampled_accel = 0.0f;
    float sampled_steer = 0.0f;
    float final_accel = 0.0f;
    float final_steer = 0.0f;
    float delta_accel = 0.0f;
    float delta_steer = 0.0f;
    float delta_norm = 0.0f;
    float grad_accel = 0.0f;
    float grad_steer = 0.0f;
    float grad_norm = 0.0f;
};

struct SummaryStats {
    int episodes = 0;
    int successes = 0;
    float sum_steps = 0.0f;
    float sum_final_distance = 0.0f;
    float sum_min_goal_distance = 0.0f;
    float sum_cumulative_cost = 0.0f;
    float sum_avg_control_ms = 0.0f;
    float sum_total_control_ms = 0.0f;
    float sum_collisions = 0.0f;
    float sum_mean_control_delta = 0.0f;
    float sum_control_roughness = 0.0f;
};

__global__ void init_curand_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rollout_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float accel = d_nominal[t * 2 + 0] + curand_normal(&local_rng) * 1.5f;
        float steer = d_nominal[t * 2 + 1] + curand_normal(&local_rng) * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void rollout_low_pass_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float lp_alpha)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = sqrtf((2.0f - alpha) / alpha);

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        filt_accel = beta * filt_accel + alpha * curand_normal(&local_rng);
        filt_steer = beta * filt_steer + alpha * curand_normal(&local_rng);
        float accel = d_nominal[t * 2 + 0] + filt_accel * variance_gain * 1.5f;
        float steer = d_nominal[t * 2 + 1] + filt_steer * variance_gain * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__device__ inline float min_obstacle_margin_device(
    float x,
    float y,
    int n_obs,
    int n_dyn_obs,
    float tau)
{
    float best = 1.0e9f;
    for (int i = 0; i < n_obs; i++) {
        float dx = x - d_obstacles_bench[i].x;
        float dy = y - d_obstacles_bench[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
        best = fminf(best, margin);
    }
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        float dx = x - ox;
        float dy = y - oy;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
        best = fminf(best, margin);
    }
    return best;
}

__device__ inline float dbas_barrier_state_device(
    float signed_clearance,
    float eps,
    float cap)
{
    float e = fmaxf(0.05f, eps);
    float c = fmaxf(1.0f, cap);
    float z = signed_clearance / e;
    float barrier;
    if (z >= 0.0f) {
        float denom = 1.0f + z;
        barrier = 1.0f / (denom * denom);
    } else {
        barrier = 1.0f + z * z;
    }
    return fminf(c, barrier);
}

__device__ inline float dbas_log_noise_device(
    curandState* rng,
    float log_sigma,
    float clip)
{
    float s = fmaxf(0.0f, log_sigma);
    float log_gain = -0.5f * s * s + s * curand_normal(rng);
    float gain = expf(clampf(log_gain, -4.0f, logf(fmaxf(1.0f, clip))));
    return curand_normal(rng) * gain;
}

__global__ void rollout_dbas_log_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    bool use_low_pass,
    float lp_alpha,
    float safe_margin,
    float barrier_eps,
    float barrier_cap,
    float barrier_weight,
    float dbas_gamma,
    float dbas_mu,
    float log_sigma,
    float lognormal_clip,
    float noise_scale,
    float speed_damping)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = use_low_pass ? sqrtf((2.0f - alpha) / alpha) : 1.0f;
    float gamma = clampf(dbas_gamma, 0.0f, 0.95f);
    float mu = fmaxf(0.01f, dbas_mu);
    float scale0 = fmaxf(0.05f, noise_scale);
    float barrier_scale = fmaxf(0.0f, barrier_weight);
    float damping = fmaxf(0.0f, speed_damping);
    float tau0 = start_step * params.dt;
    float beta_state = dbas_barrier_state_device(
        min_obstacle_margin_device(sx, sy, n_obs, n_dyn_obs, tau0) - safe_margin,
        barrier_eps, barrier_cap);

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float exploration = sqrtf(fmaxf(0.05f, mu * logf(2.7182818f + beta_state)));
        exploration *= scale0;
        float na = dbas_log_noise_device(&local_rng, log_sigma, lognormal_clip) * exploration;
        float ns = dbas_log_noise_device(&local_rng, log_sigma, lognormal_clip) * exploration;
        if (use_low_pass) {
            filt_accel = beta * filt_accel + alpha * na;
            filt_steer = beta * filt_steer + alpha * ns;
            na = filt_accel * variance_gain;
            ns = filt_steer * variance_gain;
        }
        float accel = d_nominal[t * 2 + 0] + na * 1.5f;
        float steer = d_nominal[t * 2 + 1] + ns * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float clearance = min_obstacle_margin_device(x, y, n_obs, n_dyn_obs, tau) - safe_margin;
        float barrier = dbas_barrier_state_device(clearance, barrier_eps, barrier_cap);
        beta_state = (1.0f - gamma) * barrier + gamma * beta_state;
        total_cost += barrier_scale * beta_state * params.dt;
        total_cost += damping * barrier_scale * beta_state * v * v * params.dt;

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__device__ inline float c2u_chance_margin_device(
    float x,
    float y,
    int n_obs,
    int n_dyn_obs,
    float tau,
    float rel_tau,
    float safe_margin,
    float robot_sigma,
    float dyn_sigma0,
    float dyn_sigma_growth,
    float risk_z)
{
    float best = 1.0e9f;
    float z = fmaxf(0.0f, risk_z);
    float robot_std = fmaxf(0.0f, robot_sigma);
    float robot_backoff = z * robot_std;
    for (int i = 0; i < n_obs; i++) {
        float dx = x - d_obstacles_bench[i].x;
        float dy = y - d_obstacles_bench[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
        best = fminf(best, margin - safe_margin - robot_backoff);
    }

    float dyn_std = fmaxf(0.0f, dyn_sigma0) + fmaxf(0.0f, dyn_sigma_growth) * fmaxf(0.0f, rel_tau);
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        float r = d_dynamic_obstacles_bench[i].r;
        float offsets[5][2] = {
            {0.0f, 0.0f},
            {dyn_std, 0.0f},
            {-dyn_std, 0.0f},
            {0.0f, dyn_std},
            {0.0f, -dyn_std},
        };
        float margins[5];
        float mean = 0.0f;
        for (int j = 0; j < 5; j++) {
            float dx = x - (ox + offsets[j][0]);
            float dy = y - (oy + offsets[j][1]);
            margins[j] = sqrtf(dx * dx + dy * dy + 1e-6f) - r;
            mean += margins[j] * 0.2f;
        }
        float var = 0.0f;
        for (int j = 0; j < 5; j++) {
            float d = margins[j] - mean;
            var += 0.2f * d * d;
        }
        float chance_std = sqrtf(fmaxf(0.0f, var + robot_std * robot_std));
        best = fminf(best, mean - z * chance_std - safe_margin);
    }
    return best;
}

__device__ inline float ducct_margin_risk_device(float margin, float radius, float sigma)
{
    float r = fmaxf(0.25f, radius);
    float s = fmaxf(0.03f, sigma);
    float s2 = s * s;
    float r2 = r * r;
    float dilution = r2 / (r2 + s2);
    if (margin <= 0.0f) {
        float penetration = clampf((-margin) / r, 0.0f, 1.0f);
        return clampf(dilution + (1.0f - dilution) * penetration, 0.0f, 1.0f);
    }
    float denom = fmaxf(1.0e-4f, s2 + 0.10f * r2);
    float tail = expf(-0.5f * margin * margin / denom);
    return clampf(dilution * tail, 0.0f, 1.0f);
}

__device__ inline float ducct_joint_risk_device(
    float x,
    float y,
    int n_obs,
    int n_dyn_obs,
    float tau,
    float rel_tau,
    float loc_sigma0,
    float loc_sigma_growth,
    float pred_sigma0,
    float pred_sigma_growth,
    float static_sigma)
{
    float loc_sigma = fmaxf(0.0f, loc_sigma0)
        + fmaxf(0.0f, loc_sigma_growth) * sqrtf(fmaxf(0.0f, rel_tau));
    float static_total_sigma = sqrtf(loc_sigma * loc_sigma + fmaxf(0.0f, static_sigma) * fmaxf(0.0f, static_sigma));
    float survival = 1.0f;

    for (int i = 0; i < n_obs; i++) {
        float dx = x - d_obstacles_bench[i].x;
        float dy = y - d_obstacles_bench[i].y;
        float r = d_obstacles_bench[i].r;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - r;
        float risk = ducct_margin_risk_device(margin, r, static_total_sigma);
        survival *= fmaxf(0.0f, 1.0f - risk);
    }

    float pred_sigma = fmaxf(0.0f, pred_sigma0)
        + fmaxf(0.0f, pred_sigma_growth) * fmaxf(0.0f, rel_tau);
    float dyn_total_sigma = sqrtf(loc_sigma * loc_sigma + pred_sigma * pred_sigma);
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        float dx = x - ox;
        float dy = y - oy;
        float r = d_dynamic_obstacles_bench[i].r;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - r;
        float risk = ducct_margin_risk_device(margin, r, dyn_total_sigma);
        survival *= fmaxf(0.0f, 1.0f - risk);
    }
    return clampf(1.0f - survival, 0.0f, 1.0f);
}

__device__ inline float dra_gaussian_occupancy_device(
    float px,
    float py,
    float mx,
    float my,
    float sigma)
{
    float s = fmaxf(0.05f, sigma);
    float dx = px - mx;
    float dy = py - my;
    return expf(-0.5f * (dx * dx + dy * dy) / (s * s));
}

__device__ inline float dra_dynamic_occupancy_device(
    float px,
    float py,
    const DynamicObstacle& obstacle,
    float tau,
    float sigma,
    float mode_weight,
    float mode_lateral_offset)
{
    float mx = obstacle.x + obstacle.vx * tau;
    float my = obstacle.y + obstacle.vy * tau;
    float w = clampf(mode_weight, 0.0f, 0.45f);
    float base_w = fmaxf(0.0f, 1.0f - 2.0f * w);
    float occupancy = base_w * dra_gaussian_occupancy_device(px, py, mx, my, sigma);

    if (w > 1.0e-6f) {
        float speed = sqrtf(obstacle.vx * obstacle.vx + obstacle.vy * obstacle.vy);
        float nx = 0.0f;
        float ny = 1.0f;
        if (speed > 1.0e-3f) {
            nx = -obstacle.vy / speed;
            ny = obstacle.vx / speed;
        }
        float offset = fmaxf(0.0f, mode_lateral_offset);
        occupancy += w * dra_gaussian_occupancy_device(
            px, py, mx + offset * nx, my + offset * ny, sigma);
        occupancy += w * dra_gaussian_occupancy_device(
            px, py, mx - offset * nx, my - offset * ny, sigma);
    }

    return clampf(occupancy, 0.0f, 1.0f);
}

__device__ inline float dra_collision_probability_device(
    float x,
    float y,
    int n_dyn_obs,
    float tau,
    float rel_tau,
    int mc_samples,
    float robot_radius,
    float pred_sigma0,
    float pred_sigma_growth,
    float mode_weight,
    float mode_lateral_offset)
{
    if (n_dyn_obs <= 0) return 0.0f;

    int samples = max(1, min(mc_samples, 32));
    float pred_sigma = fmaxf(0.05f, pred_sigma0)
        + fmaxf(0.0f, pred_sigma_growth) * fmaxf(0.0f, rel_tau);
    float robot_r = fmaxf(0.05f, robot_radius);
    const float golden_angle = 2.39996323f;
    float survival = 1.0f;

    for (int i = 0; i < n_dyn_obs; i++) {
        const DynamicObstacle& obstacle = d_dynamic_obstacles_bench[i];
        float disk_r = robot_r + fmaxf(0.05f, obstacle.r);
        float cp = 0.0f;
        for (int j = 0; j < samples; j++) {
            float u = (static_cast<float>(j) + 0.5f) / static_cast<float>(samples);
            float sample_r = sqrtf(u) * disk_r;
            float angle = golden_angle * static_cast<float>(j)
                + 0.41f * static_cast<float>(i);
            float px = x + sample_r * cosf(angle);
            float py = y + sample_r * sinf(angle);
            cp += dra_dynamic_occupancy_device(
                px, py, obstacle, tau, pred_sigma, mode_weight, mode_lateral_offset);
        }
        cp = clampf(cp / static_cast<float>(samples), 0.0f, 1.0f);
        survival *= fmaxf(0.0f, 1.0f - cp);
    }

    return clampf(1.0f - survival, 0.0f, 1.0f);
}

__global__ void rollout_shield_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    bool use_low_pass,
    float lp_alpha,
    float safe_margin,
    float cbf_alpha,
    float cbf_weight)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = use_low_pass ? sqrtf((2.0f - alpha) / alpha) : 1.0f;
    float prev_h = min_obstacle_margin_device(
        x, y, n_obs, n_dyn_obs, start_step * params.dt) - safe_margin;

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float na = curand_normal(&local_rng);
        float ns = curand_normal(&local_rng);
        if (use_low_pass) {
            filt_accel = beta * filt_accel + alpha * na;
            filt_steer = beta * filt_steer + alpha * ns;
            na = filt_accel * variance_gain;
            ns = filt_steer * variance_gain;
        }
        float accel = d_nominal[t * 2 + 0] + na * 1.5f;
        float steer = d_nominal[t * 2 + 1] + ns * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float next_h = min_obstacle_margin_device(x, y, n_obs, n_dyn_obs, tau) - safe_margin;
        float violation = fmaxf(0.0f, (1.0f - cbf_alpha) * prev_h - next_h);
        total_cost += cbf_weight * violation * violation;
        if (next_h < 0.0f) total_cost += 0.50f * cbf_weight * next_h * next_h;
        prev_h = next_h;

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__device__ inline BicycleParams robust_param_particle(
    BicycleParams base,
    int particle,
    int particle_count,
    float wheelbase_span,
    float max_speed_span,
    float max_steer_span)
{
    BicycleParams p = base;
    if (particle_count <= 1) return p;

    float wb_scale = 1.0f;
    float speed_scale = 1.0f;
    float steer_scale = 1.0f;
    switch (particle % 5) {
        case 0:
            break;
        case 1:
            wb_scale = 1.0f + wheelbase_span;
            speed_scale = 1.0f - max_speed_span;
            steer_scale = 1.0f - max_steer_span;
            break;
        case 2:
            wb_scale = 1.0f - 0.50f * wheelbase_span;
            speed_scale = 1.0f + 0.50f * max_speed_span;
            steer_scale = 1.0f;
            break;
        case 3:
            wb_scale = 1.0f + 0.70f * wheelbase_span;
            speed_scale = 1.0f;
            steer_scale = 1.0f - 0.70f * max_steer_span;
            break;
        default:
            wb_scale = 1.0f;
            speed_scale = 1.0f - max_speed_span;
            steer_scale = 1.0f - 0.35f * max_steer_span;
            break;
    }

    p.L = fmaxf(0.5f, base.L * wb_scale);
    p.max_speed = fmaxf(0.5f, base.max_speed * speed_scale);
    p.max_steer = fmaxf(0.05f, base.max_steer * steer_scale);
    return p;
}

__device__ inline float nav_stage_cost_float(
    float x,
    float y,
    float theta,
    float v,
    float accel,
    float steer,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    float tau)
{
    float total_cost = 0.0f;
    float dxg = x - cost_params.goal_x;
    float dyg = y - cost_params.goal_y;
    total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
    total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
    float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
    float heading_err = theta - desired_heading;
    total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
    float speed_err = v - cost_params.target_speed;
    total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

    for (int i = 0; i < n_obs; i++) {
        float dx = x - d_obstacles_bench[i].x;
        float dy = y - d_obstacles_bench[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
        if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
        else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
    }

    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        float dx = x - ox;
        float dy = y - oy;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
        if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
        else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
    }

    if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    return total_cost;
}

__global__ void rollout_parameter_robust_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    bool use_low_pass,
    float lp_alpha,
    int param_particles,
    float wheelbase_span,
    float max_speed_span,
    float max_steer_span,
    float worst_blend)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = use_low_pass ? sqrtf((2.0f - alpha) / alpha) : 1.0f;

    for (int t = 0; t < T; t++) {
        float na = curand_normal(&local_rng);
        float ns = curand_normal(&local_rng);
        if (use_low_pass) {
            filt_accel = beta * filt_accel + alpha * na;
            filt_steer = beta * filt_steer + alpha * ns;
            na = filt_accel * variance_gain;
            ns = filt_steer * variance_gain;
        }
        float accel = d_nominal[t * 2 + 0] + na * 1.5f;
        float steer = d_nominal[t * 2 + 1] + ns * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);
        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;
    }

    int n_particles = max(1, min(param_particles, 5));
    float sum_cost = 0.0f;
    float worst_cost = -FLT_MAX;
    for (int pidx = 0; pidx < n_particles; pidx++) {
        BicycleParams rp = robust_param_particle(
            params, pidx, n_particles,
            wheelbase_span, max_speed_span, max_steer_span);
        float x = sx;
        float y = sy;
        float theta = stheta;
        float v = sv;
        float total_cost = 0.0f;

        if (pidx == 0 && d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + 3] = v;
        }

        for (int t = 0; t < T; t++) {
            float accel = d_perturbed[k * T * 2 + t * 2 + 0];
            float steer = clampf(d_perturbed[k * T * 2 + t * 2 + 1],
                                 -rp.max_steer, rp.max_steer);
            bicycle_step(x, y, theta, v, accel, steer, rp);

            if (pidx == 0 && d_rollout_states != nullptr) {
                d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
                d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
                d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
                d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
            }

            float tau = (start_step + t + 1) * rp.dt;
            total_cost += nav_stage_cost_float(
                x, y, theta, v, accel, steer, rp, cost_params,
                n_obs, n_dyn_obs, tau);
        }

        float dx = x - cost_params.goal_x;
        float dy = y - cost_params.goal_y;
        total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
        sum_cost += total_cost;
        worst_cost = fmaxf(worst_cost, total_cost);
    }

    float avg_cost = sum_cost / static_cast<float>(n_particles);
    float blend = clampf(worst_blend, 0.0f, 1.0f);
    d_costs[k] = (1.0f - blend) * avg_cost + blend * worst_cost;
    d_rng[k] = local_rng;
}

__global__ void rollout_learned_sampling_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    const float* d_sigma,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    bool use_low_pass,
    float lp_alpha)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = use_low_pass ? sqrtf((2.0f - alpha) / alpha) : 1.0f;

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float na = curand_normal(&local_rng);
        float ns = curand_normal(&local_rng);
        if (use_low_pass) {
            filt_accel = beta * filt_accel + alpha * na;
            filt_steer = beta * filt_steer + alpha * ns;
            na = filt_accel * variance_gain;
            ns = filt_steer * variance_gain;
        }
        float accel_sigma = d_sigma ? d_sigma[t * 2 + 0] : 1.5f;
        float steer_sigma = d_sigma ? d_sigma[t * 2 + 1] : 0.18f;
        float accel = d_nominal[t * 2 + 0] + na * accel_sigma;
        float steer = d_nominal[t * 2 + 1] + ns * steer_sigma;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__device__ inline float cdf_margin_cost(
    float x, float y, int n_obs, int n_dyn_obs, int start_step, int t,
    const BicycleParams& params, float safe_margin, float obs_cost, float dyn_cost)
{
    float total = 0.0f;
    float safe = fmaxf(0.25f, safe_margin);
    for (int i = 0; i < n_obs; i++) {
        float dx = x - d_obstacles_bench[i].x;
        float dy = y - d_obstacles_bench[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1.0e-6f) - d_obstacles_bench[i].r;
        if (margin < safe) {
            float v = safe - margin;
            total += fmaxf(0.0f, obs_cost) * v * v * params.dt;
            if (margin <= 0.0f) total += fmaxf(0.0f, obs_cost) * 50.0f;
        }
    }
    float tau = (start_step + t + 1) * params.dt;
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        float dx = x - ox;
        float dy = y - oy;
        float margin = sqrtf(dx * dx + dy * dy + 1.0e-6f) - d_dynamic_obstacles_bench[i].r;
        if (margin < safe) {
            float v = safe - margin;
            total += fmaxf(0.0f, dyn_cost) * v * v * params.dt;
            if (margin <= 0.0f) total += fmaxf(0.0f, dyn_cost) * 50.0f;
        }
    }
    return total;
}

__global__ void rollout_cdf_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    bool use_low_pass,
    float lp_alpha,
    float cdf_safe_margin,
    float cdf_obs_cost,
    float cdf_dyn_cost)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = use_low_pass ? sqrtf((2.0f - alpha) / alpha) : 1.0f;

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float na = curand_normal(&local_rng);
        float ns = curand_normal(&local_rng);
        if (use_low_pass) {
            filt_accel = beta * filt_accel + alpha * na;
            filt_steer = beta * filt_steer + alpha * ns;
            na = filt_accel * variance_gain;
            ns = filt_steer * variance_gain;
        }
        float accel = d_nominal[t * 2 + 0] + na * 1.5f;
        float steer = d_nominal[t * 2 + 1] + ns * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;
        total_cost += cdf_margin_cost(x, y, n_obs, n_dyn_obs, start_step, t,
                                      params, cdf_safe_margin, cdf_obs_cost, cdf_dyn_cost);

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__device__ inline float halton01(int index, int base) {
    float f = 1.0f / static_cast<float>(base);
    float result = 0.0f;
    int n = max(index, 1);
    while (n > 0) {
        result += f * static_cast<float>(n % base);
        n /= base;
        f /= static_cast<float>(base);
    }
    return fminf(fmaxf(result, 1.0e-6f), 1.0f - 1.0e-6f);
}

__device__ inline float deterministic_normal(int index, int base_u, int base_v) {
    float u = halton01(index, base_u);
    float v = halton01(index + 7919, base_v);
    return sqrtf(-2.0f * logf(u)) * cosf(6.28318530718f * v);
}

__global__ void rollout_deterministic_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    const float* d_sigma,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    int sample_seed,
    int pass_index,
    float ds_alpha,
    float ds_noise_scale,
    int ds_stride)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(ds_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = sqrtf((2.0f - alpha) / alpha);
    int stride = max(ds_stride, 1);
    int pair_count = max((K + 1) / 2, 1);

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float raw_accel = 0.0f;
        float raw_steer = 0.0f;
        if (k > 0) {
            int sample_slot = k - 1;
            int pair = sample_slot / 2;
            float sign = (sample_slot & 1) ? -1.0f : 1.0f;
            float steer_sign = ((sample_slot + pass_index) & 2) ? -sign : sign;
            int idx = 1 + (start_step + 1) * stride
                    + (sample_seed + 1) * 65537
                    + (pass_index + 1) * 104729
                    + (t + 1) * pair_count
                    + pair;
            raw_accel = sign * deterministic_normal(idx, 2, 3);
            raw_steer = steer_sign * deterministic_normal(idx + 3571, 5, 7);
        }

        filt_accel = beta * filt_accel + alpha * raw_accel;
        filt_steer = beta * filt_steer + alpha * raw_steer;
        float accel_sigma = d_sigma ? d_sigma[t * 2 + 0] : 1.5f * ds_noise_scale;
        float steer_sigma = d_sigma ? d_sigma[t * 2 + 1] : 0.18f * ds_noise_scale;
        float accel = d_nominal[t * 2 + 0] + filt_accel * variance_gain * accel_sigma;
        float steer = d_nominal[t * 2 + 1] + filt_steer * variance_gain * steer_sigma;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
}

__device__ inline void project_control_component(
    float* d_controls,
    int base,
    int T,
    int component,
    float lo,
    float hi,
    float max_delta,
    float max_ddelta,
    int passes)
{
    float du = fmaxf(max_delta, 1.0e-4f);
    float ddu = fmaxf(max_ddelta, 1.0e-4f);
    int pcount = min(max(passes, 1), 8);
    for (int t = 0; t < T; t++) {
        int idx = base + t * 2 + component;
        d_controls[idx] = clampf(d_controls[idx], lo, hi);
    }
    for (int p = 0; p < pcount; p++) {
        for (int t = 1; t < T; t++) {
            int idx = base + t * 2 + component;
            float prev = d_controls[base + (t - 1) * 2 + component];
            d_controls[idx] = clampf(clampf(d_controls[idx], prev - du, prev + du), lo, hi);
        }
        for (int t = T - 2; t >= 0; t--) {
            int idx = base + t * 2 + component;
            float next = d_controls[base + (t + 1) * 2 + component];
            d_controls[idx] = clampf(clampf(d_controls[idx], next - du, next + du), lo, hi);
        }
        for (int t = 2; t < T; t++) {
            int idx = base + t * 2 + component;
            float prev = d_controls[base + (t - 1) * 2 + component];
            float prev2 = d_controls[base + (t - 2) * 2 + component];
            float pred = 2.0f * prev - prev2;
            d_controls[idx] = clampf(clampf(d_controls[idx], pred - ddu, pred + ddu), lo, hi);
        }
        for (int t = T - 3; t >= 0; t--) {
            int idx = base + t * 2 + component;
            float next = d_controls[base + (t + 1) * 2 + component];
            float next2 = d_controls[base + (t + 2) * 2 + component];
            float pred = 2.0f * next - next2;
            d_controls[idx] = clampf(clampf(d_controls[idx], pred - ddu, pred + ddu), lo, hi);
        }
    }
}

__global__ void rollout_projection_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    int projection_passes,
    float max_accel_delta,
    float max_steer_delta,
    float max_accel_ddelta,
    float max_steer_ddelta)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    int base = k * T * 2;
    for (int t = 0; t < T; t++) {
        float accel = d_nominal[t * 2 + 0] + curand_normal(&local_rng) * 1.5f;
        float steer = d_nominal[t * 2 + 1] + curand_normal(&local_rng) * 0.18f;
        d_perturbed[base + t * 2 + 0] = accel;
        d_perturbed[base + t * 2 + 1] = steer;
    }

    project_control_component(d_perturbed, base, T, 0, -4.0f, 4.0f,
                              max_accel_delta, max_accel_ddelta, projection_passes);
    project_control_component(d_perturbed, base, T, 1, -params.max_steer, params.max_steer,
                              max_steer_delta, max_steer_ddelta, projection_passes);

    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float accel = d_perturbed[base + t * 2 + 0];
        float steer = d_perturbed[base + t * 2 + 1];

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void project_nominal_controls_kernel(
    float* d_nominal,
    int T,
    float max_steer,
    int projection_passes,
    float max_accel_delta,
    float max_steer_delta,
    float max_accel_ddelta,
    float max_steer_ddelta)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    project_control_component(d_nominal, 0, T, 0, -4.0f, 4.0f,
                              max_accel_delta, max_accel_ddelta, projection_passes);
    project_control_component(d_nominal, 0, T, 1, -max_steer, max_steer,
                              max_steer_delta, max_steer_ddelta, projection_passes);
}

__global__ void rollout_fixed_controls_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_controls,
    float* d_costs,
    float* d_rollout_states,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float accel = clampf(d_controls[k * T * 2 + t * 2 + 0], -4.0f, 4.0f);
        float steer = clampf(d_controls[k * T * 2 + t * 2 + 1],
                             -params.max_steer, params.max_steer);

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        total_cost += cost_params.heading_weight * heading_err * heading_err * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
}

// Hybrid A* + MPPI hybrid rollout: same sampling pipeline as rollout_kernel
// but the cost replaces goal-distance / heading with path-follow terms
// (nearest waypoint + lookahead heading) computed against a pre-planned
// Hybrid A* path. Obstacle / speed / terminal terms are kept verbatim so
// the dynamic obstacle reaction stays the same as vanilla MPPI.
__global__ void hybrid_astar_mppi_rollout_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float w_path, float w_speed, float w_obs, float w_heading,
    float w_terminal,
    const float* __restrict__ d_path,
    int path_n,
    int lookahead_idx)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    curandState local_rng = d_rng[k];
    float x = sx, y = sy, theta = stheta, v = sv;
    float total_cost = 0.0f;

    for (int t = 0; t < T; t++) {
        float accel = d_nominal[t * 2 + 0] + curand_normal(&local_rng) * 1.5f;
        float steer = d_nominal[t * 2 + 1] + curand_normal(&local_rng) * 0.18f;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);
        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (path_n > 0) {
            float best_d2 = 1.0e30f;
            int best_idx = 0;
            for (int p = 0; p < path_n; p++) {
                float dxp = x - d_path[p * 3 + 0];
                float dyp = y - d_path[p * 3 + 1];
                float d2 = dxp * dxp + dyp * dyp;
                if (d2 < best_d2) { best_d2 = d2; best_idx = p; }
            }
            total_cost += w_path * sqrtf(best_d2 + 0.01f) * params.dt;
            int look = best_idx + lookahead_idx;
            if (look >= path_n) look = path_n - 1;
            float lx = d_path[look * 3 + 0];
            float ly = d_path[look * 3 + 1];
            float dxL = lx - x, dyL = ly - y;
            float dL2 = dxL * dxL + dyL * dyL;
            float desired = (dL2 > 0.25f)
                ? atan2f(dyL, dxL)
                : d_path[look * 3 + 2];
            float herr = theta - desired;
            while (herr >  3.14159265f) herr -= 6.28318531f;
            while (herr < -3.14159265f) herr += 6.28318531f;
            total_cost += w_heading * herr * herr * params.dt;
        } else {
            float dxg = x - cost_params.goal_x;
            float dyg = y - cost_params.goal_y;
            total_cost += w_path * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
            float desired = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
            float herr = theta - desired;
            total_cost += w_heading * herr * herr * params.dt;
        }

        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float speed_err = v - cost_params.target_speed;
        total_cost += w_speed * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += w_obs * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += w_obs / (margin * margin);
        }
        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += w_obs * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += w_obs / (margin * margin);
        }
        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    // Path-aware terminal: soft target a few indices ahead of rollout-
    // end's nearest path node, plus a remaining-arclength penalty.
    // Matches hybrid_astar_dwa_grid_kernel's formulation. Tuning the
    // multiplier didn't fix the open-dynamic regression observed when
    // hybrid_astar_mppi defaulted to T=60 (regression came from the
    // longer horizon, not the terminal). hybrid_astar_mppi now defaults
    // to T=30 again; hybrid_astar_mppi_long uses T=60 for the topology
    // suite.
    if (path_n > 0) {
        float best_d2 = 1.0e30f;
        int best_idx = 0;
        for (int p = 0; p < path_n; p++) {
            float dxp = x - d_path[p * 3 + 0];
            float dyp = y - d_path[p * 3 + 1];
            float d2 = dxp * dxp + dyp * dyp;
            if (d2 < best_d2) { best_d2 = d2; best_idx = p; }
        }
        int term_idx = best_idx + lookahead_idx;
        if (term_idx >= path_n) term_idx = path_n - 1;
        float tdx = x - d_path[term_idx * 3 + 0];
        float tdy = y - d_path[term_idx * 3 + 1];
        float remaining = static_cast<float>(path_n - 1 - term_idx) * 2.5f;
        total_cost += w_terminal * (sqrtf(tdx * tdx + tdy * tdy + 0.01f) + remaining);
    } else {
        float gdx = x - cost_params.goal_x;
        float gdy = y - cost_params.goal_y;
        total_cost += w_terminal * sqrtf(gdx * gdx + gdy * gdy + 0.01f);
    }
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__host__ __device__ inline float wrap_angle(float angle) {
    while (angle > 3.14159265f) angle -= 6.28318531f;
    while (angle < -3.14159265f) angle += 6.28318531f;
    return angle;
}

__device__ inline float nearest_obstacle_away_direction_device(
    float x,
    float y,
    int n_obs,
    int n_dyn_obs,
    float tau,
    float& away_x,
    float& away_y)
{
    float best = 1.0e9f;
    away_x = 0.0f;
    away_y = 0.0f;

    for (int i = 0; i < n_obs; i++) {
        float dx = x - d_obstacles_bench[i].x;
        float dy = y - d_obstacles_bench[i].y;
        float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
        float margin = dist - d_obstacles_bench[i].r;
        if (margin < best) {
            best = margin;
            away_x = dx / dist;
            away_y = dy / dist;
        }
    }

    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        float dx = x - ox;
        float dy = y - oy;
        float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
        float margin = dist - d_dynamic_obstacles_bench[i].r;
        if (margin < best) {
            best = margin;
            away_x = dx / dist;
            away_y = dy / dist;
        }
    }

    return best;
}

__global__ void rollout_safety_controlled_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    float* d_costs,
    float* d_perturbed,
    float* d_rollout_states,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    bool use_low_pass,
    float lp_alpha,
    float safe_margin,
    float avoid_gain,
    float speed_gain,
    float max_steer_delta,
    float max_accel_delta,
    float control_weight)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;
    float filt_accel = 0.0f;
    float filt_steer = 0.0f;
    float alpha = clampf(lp_alpha, 0.02f, 1.0f);
    float beta = 1.0f - alpha;
    float variance_gain = use_low_pass ? sqrtf((2.0f - alpha) / alpha) : 1.0f;
    float safe = fmaxf(0.10f, safe_margin);

    if (d_rollout_states != nullptr) {
        d_rollout_states[k * (T + 1) * 4 + 0] = x;
        d_rollout_states[k * (T + 1) * 4 + 1] = y;
        d_rollout_states[k * (T + 1) * 4 + 2] = theta;
        d_rollout_states[k * (T + 1) * 4 + 3] = v;
    }

    for (int t = 0; t < T; t++) {
        float na = curand_normal(&local_rng);
        float ns = curand_normal(&local_rng);
        if (use_low_pass) {
            filt_accel = beta * filt_accel + alpha * na;
            filt_steer = beta * filt_steer + alpha * ns;
            na = filt_accel * variance_gain;
            ns = filt_steer * variance_gain;
        }

        float raw_accel = clampf(d_nominal[t * 2 + 0] + na * 1.5f, -4.0f, 4.0f);
        float raw_steer = clampf(d_nominal[t * 2 + 1] + ns * 0.18f,
                                 -params.max_steer, params.max_steer);
        float accel = raw_accel;
        float steer = raw_steer;

        float px = x;
        float py = y;
        float ptheta = theta;
        float pv = v;
        bicycle_step(px, py, ptheta, pv, raw_accel, raw_steer, params);

        float tau = (start_step + t + 1) * params.dt;
        float away_x = 0.0f;
        float away_y = 0.0f;
        float predicted_margin = nearest_obstacle_away_direction_device(
            px, py, n_obs, n_dyn_obs, tau, away_x, away_y);

        if (predicted_margin < safe) {
            float danger = clampf((safe - predicted_margin) / safe, 0.0f, 1.5f);
            float away_heading = atan2f(away_y, away_x);
            float heading_err = wrap_angle(away_heading - theta);
            float steer_delta = clampf(
                avoid_gain * danger * heading_err,
                -max_steer_delta,
                max_steer_delta);

            float obstacle_heading = atan2f(-away_y, -away_x);
            float closing = fmaxf(0.0f, cosf(wrap_angle(theta - obstacle_heading)));
            float accel_delta = -clampf(
                speed_gain * danger * (0.35f + closing + 0.15f * fmaxf(v, 0.0f)),
                0.0f,
                max_accel_delta);

            steer = clampf(steer + steer_delta, -params.max_steer, params.max_steer);
            accel = clampf(accel + accel_delta, -4.0f, 4.0f);
            total_cost += control_weight *
                ((accel - raw_accel) * (accel - raw_accel)
               + 4.0f * (steer - raw_steer) * (steer - raw_steer));
        }

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        if (d_rollout_states != nullptr) {
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 0] = x;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 1] = y;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 2] = theta;
            d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4 + 3] = v;
        }

        total_cost += nav_stage_cost_float(
            x, y, theta, v, accel, steer,
            params, cost_params, n_obs, n_dyn_obs, tau);
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__device__ void terminal_grad(float x, float y, const CostParams& cp, float grad[4]);
__device__ void stage_cost_grad(
    float x, float y, float theta, float v, float accel, float steer,
    const CostParams& cp, int n_obs, int n_dyn_obs, float tau, float grad[6]);

__device__ inline bool invert_2x2(const float A[2][2], float invA[2][2]) {
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (fabsf(det) < 1.0e-8f) return false;
    float inv_det = 1.0f / det;
    invA[0][0] = A[1][1] * inv_det;
    invA[0][1] = -A[0][1] * inv_det;
    invA[1][0] = -A[1][0] * inv_det;
    invA[1][1] = A[0][0] * inv_det;
    return true;
}

__device__ inline bool invert_4x4(const float A[4][4], float invA[4][4]) {
    float aug[4][8];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) aug[row][col] = A[row][col];
        for (int col = 0; col < 4; col++) aug[row][4 + col] = (row == col) ? 1.0f : 0.0f;
    }

    for (int pivot = 0; pivot < 4; pivot++) {
        int best_row = pivot;
        float best_value = fabsf(aug[pivot][pivot]);
        for (int row = pivot + 1; row < 4; row++) {
            float value = fabsf(aug[row][pivot]);
            if (value > best_value) {
                best_value = value;
                best_row = row;
            }
        }
        if (best_value < 1.0e-8f) return false;
        if (best_row != pivot) {
            for (int col = 0; col < 8; col++) {
                float tmp = aug[pivot][col];
                aug[pivot][col] = aug[best_row][col];
                aug[best_row][col] = tmp;
            }
        }

        float diag = aug[pivot][pivot];
        float inv_diag = 1.0f / diag;
        for (int col = 0; col < 8; col++) aug[pivot][col] *= inv_diag;

        for (int row = 0; row < 4; row++) {
            if (row == pivot) continue;
            float factor = aug[row][pivot];
            if (fabsf(factor) < 1.0e-12f) continue;
            for (int col = 0; col < 8; col++) aug[row][col] -= factor * aug[pivot][col];
        }
    }

    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) invA[row][col] = aug[row][4 + col];
    }
    return true;
}

__global__ void compute_feedback_gains_kernel(
    const float* d_states,
    const float* d_nominal,
    float* d_feedback_gains,
    BicycleParams params,
    CostParams cost_params,
    int T,
    float q_position_scale,
    float q_heading_scale,
    float q_speed_scale,
    float r_accel_scale,
    float r_steer_scale,
    float terminal_scale)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float q_position = fmaxf(0.05f, cost_params.goal_weight * params.dt * q_position_scale);
    float q_heading = fmaxf(0.02f, cost_params.heading_weight * params.dt * q_heading_scale);
    float q_speed = fmaxf(0.02f, cost_params.speed_weight * params.dt * q_speed_scale);
    float r_accel = fmaxf(0.05f, cost_params.control_weight * params.dt * r_accel_scale);
    float r_steer = fmaxf(0.03f, cost_params.control_weight * params.dt * r_steer_scale);

    float Q[4][4] = {};
    Q[0][0] = q_position;
    Q[1][1] = q_position;
    Q[2][2] = q_heading;
    Q[3][3] = q_speed;

    float P[4][4] = {};
    P[0][0] = fmaxf(0.25f, cost_params.terminal_weight * terminal_scale);
    P[1][1] = fmaxf(0.25f, cost_params.terminal_weight * terminal_scale);
    P[2][2] = fmaxf(0.10f, cost_params.heading_weight * terminal_scale);
    P[3][3] = fmaxf(0.10f, cost_params.speed_weight * terminal_scale);

    for (int t = T - 1; t >= 0; t--) {
        float x = d_states[t * 4 + 0];
        float y = d_states[t * 4 + 1];
        float theta = d_states[t * 4 + 2];
        float v = d_states[t * 4 + 3];
        float accel = d_nominal[t * 2 + 0];
        float steer = d_nominal[t * 2 + 1];

        float J[4][6];
        bicycle_jacobian(x, y, theta, v, accel, steer, params, J);

        float A[4][4];
        float B[4][2];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) A[row][col] = J[row][col];
            B[row][0] = J[row][4];
            B[row][1] = J[row][5];
        }

        float PB[4][2] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 2; col++) {
                for (int k = 0; k < 4; k++) PB[row][col] += P[row][k] * B[k][col];
            }
        }

        float BtPB[2][2] = {};
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 2; col++) {
                for (int k = 0; k < 4; k++) BtPB[row][col] += B[k][row] * PB[k][col];
            }
        }

        float S[2][2] = {
            {BtPB[0][0] + r_accel, BtPB[0][1]},
            {BtPB[1][0], BtPB[1][1] + r_steer},
        };
        float S_inv[2][2];
        if (!invert_2x2(S, S_inv)) {
            S[0][0] += 0.10f;
            S[1][1] += 0.10f;
            invert_2x2(S, S_inv);
        }

        float PA[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 4; k++) PA[row][col] += P[row][k] * A[k][col];
            }
        }

        float BtPA[2][4] = {};
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 4; k++) BtPA[row][col] += B[k][row] * PA[k][col];
            }
        }

        float K[2][4] = {};
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 2; k++) K[row][col] += S_inv[row][k] * BtPA[k][col];
                d_feedback_gains[t * 8 + row * 4 + col] = K[row][col];
            }
        }

        float AtPA[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 4; k++) AtPA[row][col] += A[k][row] * PA[k][col];
            }
        }

        float KtBtPA[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                for (int k = 0; k < 2; k++) KtBtPA[row][col] += K[k][row] * BtPA[k][col];
            }
        }

        float P_next[4][4] = {};
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                P_next[row][col] = Q[row][col] + AtPA[row][col] - KtBtPA[row][col];
            }
        }

        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                P[row][col] = 0.5f * (P_next[row][col] + P_next[col][row]);
            }
        }
    }
}

__global__ void rollout_feedback_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal,
    const float* d_nominal_states,
    const float* d_feedback_gains,
    float* d_costs,
    float* d_perturbed,
    curandState* d_rng,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float gain_scale,
    float noise_accel_sigma,
    float noise_steer_sigma,
    float longitudinal_gain,
    float speed_gain,
    float lateral_gain,
    float heading_gain,
    float setpoint_blend)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    curandState local_rng = d_rng[k];
    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    float total_cost = 0.0f;

    for (int t = 0; t < T; t++) {
        int t_next = min(t + 1, T);
        float x_nom0 = d_nominal_states[t * 4 + 0];
        float y_nom0 = d_nominal_states[t * 4 + 1];
        float theta_nom0 = d_nominal_states[t * 4 + 2];
        float v_nom0 = d_nominal_states[t * 4 + 3];
        float x_nom1 = d_nominal_states[t_next * 4 + 0];
        float y_nom1 = d_nominal_states[t_next * 4 + 1];
        float theta_nom1 = d_nominal_states[t_next * 4 + 2];
        float v_nom1 = d_nominal_states[t_next * 4 + 3];
        float x_nom = (1.0f - setpoint_blend) * x_nom0 + setpoint_blend * x_nom1;
        float y_nom = (1.0f - setpoint_blend) * y_nom0 + setpoint_blend * y_nom1;
        float theta_nom = wrap_angle((1.0f - setpoint_blend) * theta_nom0 + setpoint_blend * theta_nom1);
        float v_nom = (1.0f - setpoint_blend) * v_nom0 + setpoint_blend * v_nom1;
        float dx = x_nom - x;
        float dy = y_nom - y;
        float ex = x - x_nom;
        float ey = y - y_nom;
        float etheta = wrap_angle(theta - theta_nom);
        float ev = v - v_nom;
        float ct = cosf(theta_nom);
        float st = sinf(theta_nom);
        float longitudinal_err = ct * dx + st * dy;
        float lateral_err = -st * dx + ct * dy;
        float heading_err = wrap_angle(theta_nom - theta);
        float speed_err = v_nom - v;

        const float* K_t = &d_feedback_gains[t * 8];
        float accel_feedback =
            K_t[0] * ex + K_t[1] * ey + K_t[2] * etheta + K_t[3] * ev;
        float steer_feedback =
            K_t[4] * ex + K_t[5] * ey + K_t[6] * etheta + K_t[7] * ev;

        float accel = d_nominal[t * 2 + 0]
                    + curand_normal(&local_rng) * noise_accel_sigma
                    - gain_scale * accel_feedback
                    + longitudinal_gain * longitudinal_err
                    + speed_gain * speed_err;
        float steer = d_nominal[t * 2 + 1]
                    + curand_normal(&local_rng) * noise_steer_sigma
                    - gain_scale * steer_feedback
                    + lateral_gain * lateral_err
                    + heading_gain * heading_err;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -params.max_steer, params.max_steer);

        d_perturbed[k * T * 2 + t * 2 + 0] = accel;
        d_perturbed[k * T * 2 + t * 2 + 1] = steer;

        bicycle_step(x, y, theta, v, accel, steer, params);

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        total_cost += cost_params.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
        total_cost += cost_params.control_weight * (accel * accel + steer * steer) * params.dt;
        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float goal_heading_err = wrap_angle(theta - desired_heading);
        total_cost += cost_params.heading_weight * goal_heading_err * goal_heading_err * params.dt;
        float speed_goal_err = v - cost_params.target_speed;
        total_cost += cost_params.speed_weight * speed_goal_err * speed_goal_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float odx = x - d_obstacles_bench[i].x;
            float ody = y - d_obstacles_bench[i].y;
            float margin = sqrtf(odx * odx + ody * ody + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float odx = x - ox;
            float ody = y - oy;
            float margin = sqrtf(odx * odx + ody * ody + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) total_cost += cost_params.obs_weight * 100.0f;
            else if (margin < cost_params.obs_influence) total_cost += cost_params.obs_weight / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) total_cost += 500.0f;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    total_cost += cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
    d_costs[k] = total_cost;
    d_rng[k] = local_rng;
}

__global__ void compute_rollout_initial_gradients_kernel(
    const float* d_rollout_states,
    const float* d_perturbed,
    float* d_rollout_init_grads,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const float* rollout_states = &d_rollout_states[k * (T + 1) * 4];
    const float* rollout_actions = &d_perturbed[k * T * 2];

    float adj[4];
    terminal_grad(rollout_states[T * 4 + 0], rollout_states[T * 4 + 1], cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        float x = rollout_states[t * 4 + 0];
        float y = rollout_states[t * 4 + 1];
        float theta = rollout_states[t * 4 + 2];
        float v = rollout_states[t * 4 + 3];
        float accel = rollout_actions[t * 2 + 0];
        float steer = rollout_actions[t * 2 + 1];

        float J[4][6];
        float stage_grad_vec[6];
        float next_adj[4];
        float tau = (start_step + t) * params.dt;

        bicycle_jacobian(x, y, theta, v, accel, steer, params, J);
        stage_cost_grad(x, y, theta, v, accel, steer, cost_params, n_obs, n_dyn_obs, tau, stage_grad_vec);

        for (int col = 0; col < 4; col++) {
            next_adj[col] = stage_grad_vec[col];
            for (int row = 0; row < 4; row++) next_adj[col] += J[row][col] * adj[row];
        }
        for (int i = 0; i < 4; i++) adj[i] = next_adj[i];
    }

    for (int i = 0; i < 4; i++) d_rollout_init_grads[k * 4 + i] = adj[i];
}

__global__ void compute_sensitivity_feedback_gains_kernel(
    const float* d_nominal,
    const float* d_perturbed,
    const float* d_weights,
    const float* d_rollout_init_grads,
    float* d_feedback_gains,
    float lambda,
    int K,
    int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float weighted_grad[4] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < 4; j++) weighted_grad[j] += w * d_rollout_init_grads[k * 4 + j];
    }

    for (int t = 0; t < T; t++) {
        float accel_mean = d_nominal[t * 2 + 0];
        float steer_mean = d_nominal[t * 2 + 1];
        for (int j = 0; j < 4; j++) {
            float accel_cov = 0.0f;
            float steer_cov = 0.0f;
            for (int k = 0; k < K; k++) {
                float w = d_weights[k];
                float g = d_rollout_init_grads[k * 4 + j];
                accel_cov += w * d_perturbed[k * T * 2 + t * 2 + 0] * g;
                steer_cov += w * d_perturbed[k * T * 2 + t * 2 + 1] * g;
            }
            d_feedback_gains[t * 8 + 0 * 4 + j] = -(accel_cov - accel_mean * weighted_grad[j]) / lambda;
            d_feedback_gains[t * 8 + 1 * 4 + j] = -(steer_cov - steer_mean * weighted_grad[j]) / lambda;
        }
    }
}

__global__ void compute_reference_feedback_gain_kernel(
    const float* d_nominal,
    const float* d_perturbed,
    const float* d_weights,
    const float* d_rollout_init_grads,
    float* d_feedback_gains,
    float lambda,
    int K,
    int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const float inv_lambda = 1.0f / fmaxf(1.0e-6f, lambda);
    float weighted_grad[4] = {};
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        for (int j = 0; j < 4; j++) weighted_grad[j] += w * d_rollout_init_grads[k * 4 + j];
    }

    for (int i = 0; i < T * 8; i++) d_feedback_gains[i] = 0.0f;

    float nominal_accel = d_nominal[0];
    float nominal_steer = d_nominal[1];
    for (int j = 0; j < 4; j++) {
        float accel_gain = 0.0f;
        float steer_gain = 0.0f;
        for (int k = 0; k < K; k++) {
            float w = d_weights[k];
            float centered_grad = d_rollout_init_grads[k * 4 + j] - weighted_grad[j];
            float delta_accel = d_perturbed[k * T * 2 + 0] - nominal_accel;
            float delta_steer = d_perturbed[k * T * 2 + 1] - nominal_steer;
            accel_gain += -inv_lambda * w * centered_grad * delta_accel;
            steer_gain += -inv_lambda * w * centered_grad * delta_steer;
        }
        d_feedback_gains[0 * 4 + j] = accel_gain;
        d_feedback_gains[1 * 4 + j] = steer_gain;
    }
}

__global__ void compute_covariance_feedback_gains_kernel(
    const float* d_nominal,
    const float* d_nominal_states,
    const float* d_perturbed,
    const float* d_rollout_states,
    const float* d_weights,
    float* d_feedback_gains,
    int K,
    int T,
    float regularization)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const float eps = fmaxf(1.0e-4f, regularization);
    for (int t = 0; t < T; t++) {
        float Sigma_xx[4][4] = {};
        float Sigma_ux[2][4] = {};
        float x_nom = d_nominal_states[t * 4 + 0];
        float y_nom = d_nominal_states[t * 4 + 1];
        float theta_nom = d_nominal_states[t * 4 + 2];
        float v_nom = d_nominal_states[t * 4 + 3];
        float accel_nom = d_nominal[t * 2 + 0];
        float steer_nom = d_nominal[t * 2 + 1];

        for (int k = 0; k < K; k++) {
            float w = d_weights[k];
            const float* rollout_state = &d_rollout_states[k * (T + 1) * 4 + t * 4];
            float x_dev[4];
            x_dev[0] = rollout_state[0] - x_nom;
            x_dev[1] = rollout_state[1] - y_nom;
            x_dev[2] = wrap_angle(rollout_state[2] - theta_nom);
            x_dev[3] = rollout_state[3] - v_nom;

            float u_dev[2];
            u_dev[0] = d_perturbed[k * T * 2 + t * 2 + 0] - accel_nom;
            u_dev[1] = d_perturbed[k * T * 2 + t * 2 + 1] - steer_nom;

            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) Sigma_xx[row][col] += w * x_dev[row] * x_dev[col];
            }
            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 4; col++) Sigma_ux[row][col] += w * u_dev[row] * x_dev[col];
            }
        }

        for (int i = 0; i < 4; i++) Sigma_xx[i][i] += eps;

        float Sigma_xx_inv[4][4];
        if (!invert_4x4(Sigma_xx, Sigma_xx_inv)) {
            for (int i = 0; i < 4; i++) Sigma_xx[i][i] += 10.0f * eps;
            invert_4x4(Sigma_xx, Sigma_xx_inv);
        }

        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
                float gain = 0.0f;
                for (int k = 0; k < 4; k++) gain += Sigma_ux[row][k] * Sigma_xx_inv[k][col];
                d_feedback_gains[t * 8 + row * 4 + col] = -gain;
            }
        }
    }
}

__global__ void blend_feedback_gains_kernel(
    float* d_out,
    const float* d_aux,
    int T,
    float out_scale,
    float aux_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * 8;
    if (idx >= total) return;
    d_out[idx] = out_scale * d_out[idx] + aux_scale * d_aux[idx];
}

// Single-thread kernel: sequential min-reduce + normalize (K is small enough that
// a single-thread scan is faster than a parallel reduction with launch overhead).
__global__ void compute_weights_kernel(const float* d_costs, float* d_weights, int K, float lambda) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float min_cost = FLT_MAX;
    for (int k = 0; k < K; k++) min_cost = fminf(min_cost, d_costs[k]);

    float sum_w = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = expf(-(d_costs[k] - min_cost) / lambda);
        d_weights[k] = w;
        sum_w += w;
    }
    if (sum_w > 0.0f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    }
}

__global__ void compute_tsallis_weights_kernel(
    const float* d_costs,
    float* d_weights,
    int K,
    float q,
    float temperature,
    float min_weight)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float min_cost = FLT_MAX;
    float max_cost = -FLT_MAX;
    for (int k = 0; k < K; k++) {
        min_cost = fminf(min_cost, d_costs[k]);
        max_cost = fmaxf(max_cost, d_costs[k]);
    }

    float span = fmaxf(1.0e-3f, max_cost - min_cost);
    float temp = fmaxf(1.0e-3f, temperature);
    float qv = clampf(q, 0.05f, 2.50f);
    float floor_w = clampf(min_weight, 0.0f, 1.0f);
    float sum_w = 0.0f;

    for (int k = 0; k < K; k++) {
        float normalized = clampf((d_costs[k] - min_cost) / span, 0.0f, 1.0f);
        float x = -temp * normalized;
        float w;
        if (fabsf(qv - 1.0f) < 1.0e-4f) {
            w = expf(x);
        } else {
            float base = 1.0f + (1.0f - qv) * x;
            w = base > 0.0f ? powf(base, 1.0f / (1.0f - qv)) : 0.0f;
        }
        w = fmaxf(floor_w, w);
        d_weights[k] = w;
        sum_w += w;
    }

    if (sum_w > 1.0e-12f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    } else {
        float uniform = 1.0f / static_cast<float>(K);
        for (int k = 0; k < K; k++) d_weights[k] = uniform;
    }
}

__global__ void compute_covariance_control_weights_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_weights,
    int K,
    int T,
    float lambda,
    float terminal_weight,
    float target_radius,
    float heading_weight,
    float speed_weight,
    float min_weight)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float min_cost = FLT_MAX;
    for (int k = 0; k < K; k++) min_cost = fminf(min_cost, d_costs[k]);

    float sum_pre = 0.0f;
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    float mean_v = 0.0f;
    float mean_sin = 0.0f;
    float mean_cos = 0.0f;
    float inv_lambda = 1.0f / fmaxf(lambda, 1.0e-3f);
    for (int k = 0; k < K; k++) {
        float w = expf(-(d_costs[k] - min_cost) * inv_lambda);
        const float* terminal = &d_rollout_states[k * (T + 1) * 4 + T * 4];
        mean_x += w * terminal[0];
        mean_y += w * terminal[1];
        mean_sin += w * sinf(terminal[2]);
        mean_cos += w * cosf(terminal[2]);
        mean_v += w * terminal[3];
        sum_pre += w;
    }

    if (sum_pre > 1.0e-12f) {
        float inv_sum = 1.0f / sum_pre;
        mean_x *= inv_sum;
        mean_y *= inv_sum;
        mean_sin *= inv_sum;
        mean_cos *= inv_sum;
        mean_v *= inv_sum;
    } else {
        float inv_k = 1.0f / static_cast<float>(K);
        for (int k = 0; k < K; k++) {
            const float* terminal = &d_rollout_states[k * (T + 1) * 4 + T * 4];
            mean_x += inv_k * terminal[0];
            mean_y += inv_k * terminal[1];
            mean_sin += inv_k * sinf(terminal[2]);
            mean_cos += inv_k * cosf(terminal[2]);
            mean_v += inv_k * terminal[3];
        }
    }
    float mean_theta = atan2f(mean_sin, mean_cos);

    float target2 = fmaxf(0.0f, target_radius) * fmaxf(0.0f, target_radius);
    float terminal_scale = fmaxf(0.0f, terminal_weight);
    float h_weight = fmaxf(0.0f, heading_weight);
    float v_weight = fmaxf(0.0f, speed_weight);
    float floor_w = clampf(min_weight, 0.0f, 1.0f);

    float min_adjusted = FLT_MAX;
    for (int k = 0; k < K; k++) {
        const float* terminal = &d_rollout_states[k * (T + 1) * 4 + T * 4];
        float dx = terminal[0] - mean_x;
        float dy = terminal[1] - mean_y;
        float dtheta = wrap_angle(terminal[2] - mean_theta);
        float dv = terminal[3] - mean_v;
        float dispersion = dx * dx + dy * dy + h_weight * dtheta * dtheta + v_weight * dv * dv;
        float excess = fmaxf(0.0f, dispersion - target2);
        float adjusted = d_costs[k] + terminal_scale * excess;
        min_adjusted = fminf(min_adjusted, adjusted);
    }

    float sum_w = 0.0f;
    for (int k = 0; k < K; k++) {
        const float* terminal = &d_rollout_states[k * (T + 1) * 4 + T * 4];
        float dx = terminal[0] - mean_x;
        float dy = terminal[1] - mean_y;
        float dtheta = wrap_angle(terminal[2] - mean_theta);
        float dv = terminal[3] - mean_v;
        float dispersion = dx * dx + dy * dy + h_weight * dtheta * dtheta + v_weight * dv * dv;
        float excess = fmaxf(0.0f, dispersion - target2);
        float adjusted = d_costs[k] + terminal_scale * excess;
        float w = expf(-(adjusted - min_adjusted) * inv_lambda);
        w = fmaxf(floor_w, w);
        d_weights[k] = w;
        sum_w += w;
    }

    if (sum_w > 1.0e-12f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    } else {
        float uniform = 1.0f / static_cast<float>(K);
        for (int k = 0; k < K; k++) d_weights[k] = uniform;
    }
}

__global__ void compute_td_cd_scores_kernel(
    const float* d_rollout_states,
    const float* d_perturbed,
    float* d_scores,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float terminal_value_scale,
    float safe_margin,
    float discount_sigma,
    float discount_power,
    float failure_cost)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float sigma = fmaxf(1.0e-3f, discount_sigma);
    float power = fmaxf(0.0f, discount_power);
    float fail_cost = fmaxf(0.0f, failure_cost);
    float value_scale = fmaxf(0.0f, terminal_value_scale);
    float survival = 1.0f;
    float score = 0.0f;

    for (int t = 0; t < T; t++) {
        const float* s = &d_rollout_states[k * (T + 1) * 4 + (t + 1) * 4];
        const float* u = &d_perturbed[k * T * 2 + t * 2];
        float tau = (start_step + t + 1) * params.dt;
        float stage = nav_stage_cost_float(
            s[0], s[1], s[2], s[3], u[0], u[1],
            params, cost_params, n_obs, n_dyn_obs, tau);

        float margin = min_obstacle_margin_device(s[0], s[1], n_obs, n_dyn_obs, tau);
        float z = clampf((margin - safe_margin) / sigma, -20.0f, 20.0f);
        float feasibility = 1.0f / (1.0f + expf(-z));
        float step_discount = powf(clampf(feasibility, 1.0e-6f, 1.0f), power);
        float next_survival = survival * step_discount;
        float failure_mass = fmaxf(0.0f, survival - next_survival);

        score += survival * stage + failure_mass * fail_cost;
        survival = next_survival;
    }

    const float* terminal = &d_rollout_states[k * (T + 1) * 4 + T * 4];
    float dx = terminal[0] - cost_params.goal_x;
    float dy = terminal[1] - cost_params.goal_y;
    float dist = sqrtf(dx * dx + dy * dy + 0.01f);
    float target_speed = fmaxf(0.5f, cost_params.target_speed);
    float value_goal = cost_params.terminal_weight * dist
        + 0.5f * cost_params.goal_weight * dist * dist / target_speed;
    float desired_heading = atan2f(cost_params.goal_y - terminal[1],
                                   cost_params.goal_x - terminal[0]);
    float heading_err = wrap_angle(terminal[2] - desired_heading);
    float speed_err = terminal[3] - cost_params.target_speed;
    float value_shape =
        2.0f * cost_params.heading_weight * heading_err * heading_err
        + 2.0f * cost_params.speed_weight * speed_err * speed_err;
    score += survival * value_scale * (value_goal + value_shape);

    d_scores[k] = score;
}

__global__ void compute_svg_mode_weights_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_weights,
    int K,
    int T,
    float lambda,
    float bandwidth,
    float mode_weight,
    int stride)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float min_cost = FLT_MAX;
    int best = 0;
    for (int k = 0; k < K; k++) {
        if (d_costs[k] < min_cost) {
            min_cost = d_costs[k];
            best = k;
        }
    }

    int step_stride = max(1, stride);
    float bw = fmaxf(1.0e-3f, bandwidth);
    float sum_w = 0.0f;
    for (int k = 0; k < K; k++) {
        float d2 = 0.0f;
        int count = 0;
        for (int t = step_stride; t <= T; t += step_stride) {
            int ib = best * (T + 1) * 4 + t * 4;
            int ik = k * (T + 1) * 4 + t * 4;
            float dx = d_rollout_states[ik + 0] - d_rollout_states[ib + 0];
            float dy = d_rollout_states[ik + 1] - d_rollout_states[ib + 1];
            float dtheta = wrap_angle(d_rollout_states[ik + 2] - d_rollout_states[ib + 2]);
            float dv = d_rollout_states[ik + 3] - d_rollout_states[ib + 3];
            d2 += dx * dx + dy * dy + 0.25f * dtheta * dtheta + 0.10f * dv * dv;
            count++;
        }
        if (count > 0) d2 /= static_cast<float>(count);
        float cost_w = expf(-(d_costs[k] - min_cost) / fmaxf(lambda, 1.0e-3f));
        float mode_w = expf(-d2 / bw);
        float w = cost_w * (1.0f + fmaxf(0.0f, mode_weight) * mode_w);
        d_weights[k] = w;
        sum_w += w;
    }
    if (sum_w > 1.0e-12f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    } else {
        float uniform = 1.0f / static_cast<float>(K);
        for (int k = 0; k < K; k++) d_weights[k] = uniform;
    }
}

__device__ inline float pa_segment_occlusion_device(
    float x0,
    float y0,
    float x1,
    float y1,
    int n_obs,
    int n_dyn_obs,
    float tau,
    float safe_margin)
{
    float vx = x1 - x0;
    float vy = y1 - y0;
    float len2 = vx * vx + vy * vy;
    if (len2 < 1.0e-6f) return 0.0f;

    float best = 0.0f;
    for (int i = 0; i < n_obs; i++) {
        float wx = d_obstacles_bench[i].x - x0;
        float wy = d_obstacles_bench[i].y - y0;
        float u = clampf((wx * vx + wy * vy) / len2, 0.0f, 1.0f);
        if (u < 0.02f || u > 0.98f) continue;
        float px = x0 + u * vx;
        float py = y0 + u * vy;
        float dx = d_obstacles_bench[i].x - px;
        float dy = d_obstacles_bench[i].y - py;
        float r = fmaxf(0.05f, d_obstacles_bench[i].r + safe_margin);
        float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
        float depth = clampf((r - dist) / r, 0.0f, 1.0f);
        float along_weight = 1.0f - 0.35f * u;
        best = fmaxf(best, depth * along_weight);
    }

    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        float wx = ox - x0;
        float wy = oy - y0;
        float u = clampf((wx * vx + wy * vy) / len2, 0.0f, 1.0f);
        if (u < 0.02f || u > 0.98f) continue;
        float px = x0 + u * vx;
        float py = y0 + u * vy;
        float dx = ox - px;
        float dy = oy - py;
        float r = fmaxf(0.05f, d_dynamic_obstacles_bench[i].r + safe_margin);
        float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
        float depth = clampf((r - dist) / r, 0.0f, 1.0f);
        float along_weight = 1.0f - 0.35f * u;
        best = fmaxf(best, depth * along_weight);
    }
    return clampf(best, 0.0f, 1.0f);
}

__global__ void compute_pa_perception_scores_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_scores,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float safe_margin,
    float poi_weight,
    float occlusion_weight,
    float frontier_reward,
    float forward_occ_weight,
    float goal_gate,
    float activation,
    float ray_length,
    float score_cap)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const float* start = &d_rollout_states[k * (T + 1) * 4];
    float gx = cost_params.goal_x;
    float gy = cost_params.goal_y;
    float tau0 = start_step * params.dt;
    float start_occ = pa_segment_occlusion_device(
        start[0], start[1], gx, gy, n_obs, n_dyn_obs, tau0, safe_margin);
    float active_threshold = fmaxf(0.0f, activation);
    float score_delta = 0.0f;
    float count = 0.0f;

    for (int t = 1; t <= T; t++) {
        const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
        float x = s[0];
        float y = s[1];
        float theta = s[2];
        float dxg = gx - x;
        float dyg = gy - y;
        float dist_goal = sqrtf(dxg * dxg + dyg * dyg + 1.0e-6f);
        if (dist_goal <= goal_gate) continue;

        float tau = (start_step + t) * params.dt;
        float occ = pa_segment_occlusion_device(
            x, y, gx, gy, n_obs, n_dyn_obs, tau, safe_margin);
        float active = clampf((fmaxf(start_occ, occ) - active_threshold)
                              / fmaxf(1.0e-3f, 1.0f - active_threshold),
                              0.0f, 1.0f);
        if (active <= 1.0e-5f) continue;

        float bearing = atan2f(dyg, dxg);
        float heading_err = wrap_angle(theta - bearing);
        float poi = 1.0f - cosf(heading_err);

        float look_x = x + cosf(theta) * fminf(ray_length, dist_goal);
        float look_y = y + sinf(theta) * fminf(ray_length, dist_goal);
        float forward_occ = pa_segment_occlusion_device(
            x, y, look_x, look_y, n_obs, n_dyn_obs, tau, safe_margin);

        float exposed_gain = fmaxf(0.0f, start_occ - occ);
        float late_weight = 0.75f + 0.25f * static_cast<float>(t) / fmaxf(1.0f, static_cast<float>(T));
        score_delta += active * (
            fmaxf(0.0f, occlusion_weight) * occ * occ
            + fmaxf(0.0f, poi_weight) * poi * poi
            + fmaxf(0.0f, forward_occ_weight) * forward_occ * forward_occ
            - fmaxf(0.0f, frontier_reward) * exposed_gain * late_weight);
        count += 1.0f;
    }

    if (count > 0.0f) score_delta /= count;
    float cap = fmaxf(0.0f, score_cap);
    if (cap > 0.0f) score_delta = clampf(score_delta, -cap, cap);
    d_scores[k] = d_costs[k] + score_delta;
}

__global__ void compute_bc_safety_weights_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_weights,
    BicycleParams params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float lambda,
    float safe_margin,
    float prob_sigma,
    float probability_power,
    float min_probability)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float min_cost = FLT_MAX;
    for (int k = 0; k < K; k++) min_cost = fminf(min_cost, d_costs[k]);

    float sigma = fmaxf(1.0e-3f, prob_sigma);
    float power = fmaxf(0.0f, probability_power);
    float min_prob = clampf(min_probability, 1.0e-12f, 1.0f);
    float sum_w = 0.0f;

    for (int k = 0; k < K; k++) {
        float log_prob = 0.0f;
        float min_margin_seen = 1.0e9f;
        for (int t = 1; t <= T; t++) {
            const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
            float tau = (start_step + t) * params.dt;
            float margin = min_obstacle_margin_device(s[0], s[1], n_obs, n_dyn_obs, tau);
            min_margin_seen = fminf(min_margin_seen, margin);
            float z = clampf((margin - safe_margin) / sigma, -20.0f, 20.0f);
            float p = 1.0f / (1.0f + expf(-z));
            log_prob += logf(fmaxf(min_prob, p));
        }

        // Lightweight surrogate for trajectory feasibility: the geometric
        // mean keeps scores comparable across horizons, while min-step
        // probability still suppresses trajectories with one close pass.
        float mean_prob = expf(log_prob / fmaxf(1.0f, static_cast<float>(T)));
        float min_z = clampf((min_margin_seen - safe_margin) / sigma, -20.0f, 20.0f);
        float min_step_prob = 1.0f / (1.0f + expf(-min_z));
        float feasibility = sqrtf(fmaxf(min_prob, mean_prob) * fmaxf(min_prob, min_step_prob));
        feasibility = powf(fmaxf(min_prob, feasibility), power);

        float cost_w = expf(-(d_costs[k] - min_cost) / fmaxf(lambda, 1.0e-3f));
        float w = cost_w * feasibility;
        d_weights[k] = w;
        sum_w += w;
    }

    if (sum_w > 1.0e-12f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    } else {
        float uniform = 1.0f / static_cast<float>(K);
        for (int k = 0; k < K; k++) d_weights[k] = uniform;
    }
}

__global__ void compute_c2u_chance_scores_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_scores,
    BicycleParams params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float lambda,
    float safe_margin,
    float robot_sigma,
    float dyn_sigma0,
    float dyn_sigma_growth,
    float risk_z,
    float prob_sigma,
    float probability_power,
    float violation_weight,
    float min_probability)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float sigma = fmaxf(1.0e-3f, prob_sigma);
    float power = fmaxf(0.0f, probability_power);
    float min_prob = clampf(min_probability, 1.0e-12f, 1.0f);
    float penalty_scale = fmaxf(0.0f, violation_weight);

    float violation_cost = 0.0f;
    float log_prob = 0.0f;
    float min_h = 1.0e9f;
    for (int t = 1; t <= T; t++) {
        const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
        float rel_tau = t * params.dt;
        float tau = (start_step + t) * params.dt;
        float h = c2u_chance_margin_device(
            s[0], s[1], n_obs, n_dyn_obs, tau, rel_tau, safe_margin,
            robot_sigma, dyn_sigma0, dyn_sigma_growth, risk_z);
        min_h = fminf(min_h, h);
        float violation = fmaxf(0.0f, -h);
        violation_cost += violation * violation;
        float p = 1.0f / (1.0f + expf(-clampf(h / sigma, -20.0f, 20.0f)));
        log_prob += logf(fmaxf(min_prob, p));
    }

    float mean_prob = expf(log_prob / fmaxf(1.0f, static_cast<float>(T)));
    float min_prob_step = 1.0f / (1.0f + expf(-clampf(min_h / sigma, -20.0f, 20.0f)));
    float feasibility = sqrtf(fmaxf(min_prob, mean_prob) * fmaxf(min_prob, min_prob_step));
    feasibility = powf(fmaxf(min_prob, feasibility), power);

    float score = d_costs[k]
        + penalty_scale * violation_cost / fmaxf(1.0f, static_cast<float>(T))
        - fmaxf(lambda, 1.0e-3f) * logf(fmaxf(min_prob, feasibility));
    d_scores[k] = score;
}

__global__ void compute_ducct_risk_scores_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_scores,
    BicycleParams params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float lambda,
    float loc_sigma0,
    float loc_sigma_growth,
    float pred_sigma0,
    float pred_sigma_growth,
    float static_sigma,
    float risk_weight,
    float hard_threshold,
    float reject_cost,
    float survival_power,
    float min_survival)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float threshold = clampf(hard_threshold, 0.0f, 1.0f);
    float risk_scale = fmaxf(0.0f, risk_weight);
    float reject_scale = fmaxf(0.0f, reject_cost);
    float power = fmaxf(0.0f, survival_power);
    float min_surv = clampf(min_survival, 1.0e-12f, 1.0f);

    float risk_sum = 0.0f;
    float reject_sum = 0.0f;
    float log_survival = 0.0f;
    float max_risk = 0.0f;
    for (int t = 1; t <= T; t++) {
        const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
        float rel_tau = t * params.dt;
        float tau = (start_step + t) * params.dt;
        float risk = ducct_joint_risk_device(
            s[0], s[1], n_obs, n_dyn_obs, tau, rel_tau,
            loc_sigma0, loc_sigma_growth,
            pred_sigma0, pred_sigma_growth, static_sigma);
        risk_sum += risk;
        max_risk = fmaxf(max_risk, risk);
        float excess = fmaxf(0.0f, risk - threshold);
        reject_sum += excess * excess;
        log_survival += logf(fmaxf(min_surv, 1.0f - risk));
    }

    float denom = fmaxf(1.0f, static_cast<float>(T));
    float mean_risk = risk_sum / denom;
    float survival = expf(log_survival / denom);
    float score = d_costs[k]
        + risk_scale * mean_risk
        + reject_scale * reject_sum / denom
        - fmaxf(lambda, 1.0e-3f) * power * logf(fmaxf(min_surv, survival));
    if (max_risk > threshold) {
        score += reject_scale * (max_risk - threshold);
    }
    d_scores[k] = score;
}

__global__ void compute_dra_risk_scores_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_scores,
    BicycleParams params,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float lambda,
    int mc_samples,
    float robot_radius,
    float pred_sigma0,
    float pred_sigma_growth,
    float mode_weight,
    float mode_lateral_offset,
    float soft_weight,
    float hard_threshold,
    float reject_cost,
    float survival_power,
    float min_survival)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float threshold = clampf(hard_threshold, 0.0f, 1.0f);
    float soft_scale = fmaxf(0.0f, soft_weight);
    float reject_scale = fmaxf(0.0f, reject_cost);
    float power = fmaxf(0.0f, survival_power);
    float min_surv = clampf(min_survival, 1.0e-12f, 1.0f);

    float cp_sum = 0.0f;
    float reject_sum = 0.0f;
    float log_survival = 0.0f;
    float max_cp = 0.0f;
    for (int t = 1; t <= T; t++) {
        const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
        float rel_tau = t * params.dt;
        float tau = (start_step + t) * params.dt;
        float cp = dra_collision_probability_device(
            s[0], s[1], n_dyn_obs, tau, rel_tau, mc_samples, robot_radius,
            pred_sigma0, pred_sigma_growth, mode_weight, mode_lateral_offset);
        cp_sum += cp;
        max_cp = fmaxf(max_cp, cp);
        float excess = fmaxf(0.0f, cp - threshold);
        reject_sum += excess * excess;
        log_survival += logf(fmaxf(min_surv, 1.0f - cp));
    }

    float denom = fmaxf(1.0f, static_cast<float>(T));
    float mean_cp = cp_sum / denom;
    float survival = expf(log_survival / denom);
    float score = d_costs[k]
        + soft_scale * mean_cp
        + reject_scale * reject_sum / denom
        - fmaxf(lambda, 1.0e-3f) * power * logf(fmaxf(min_surv, survival));
    if (max_cp > threshold) {
        score += reject_scale * (max_cp - threshold);
    }
    d_scores[k] = score;
}

__global__ void compute_dm_influence_weights_kernel(
    const float* d_costs,
    const float* d_rollout_states,
    float* d_weights,
    BicycleParams params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float keep_fraction,
    float cost_temperature,
    float safe_margin,
    float prob_sigma,
    float violation_weight,
    float safety_power)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float min_score = FLT_MAX;
    float max_score = -FLT_MAX;
    for (int k = 0; k < K; k++) {
        float min_margin = 1.0e9f;
        for (int t = 1; t <= T; t++) {
            const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
            float tau = (start_step + t) * params.dt;
            min_margin = fminf(min_margin, min_obstacle_margin_device(
                s[0], s[1], n_obs, n_dyn_obs, tau));
        }
        float violation = fmaxf(0.0f, safe_margin - min_margin);
        float score = d_costs[k] + fmaxf(0.0f, violation_weight) * violation * violation;
        min_score = fminf(min_score, score);
        max_score = fmaxf(max_score, score);
    }

    float score_span = fmaxf(1.0e-3f, max_score - min_score);
    float sigma = fmaxf(1.0e-3f, prob_sigma);
    float temp = fmaxf(0.1f, cost_temperature);
    float power = fmaxf(0.0f, safety_power);

    for (int k = 0; k < K; k++) {
        float min_margin = 1.0e9f;
        for (int t = 1; t <= T; t++) {
            const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
            float tau = (start_step + t) * params.dt;
            min_margin = fminf(min_margin, min_obstacle_margin_device(
                s[0], s[1], n_obs, n_dyn_obs, tau));
        }
        float violation = fmaxf(0.0f, safe_margin - min_margin);
        float score = d_costs[k] + fmaxf(0.0f, violation_weight) * violation * violation;
        float normalized_score = clampf((score - min_score) / score_span, 0.0f, 1.0f);
        float z = clampf((min_margin - safe_margin) / sigma, -20.0f, 20.0f);
        float safety_prob = 1.0f / (1.0f + expf(-z));
        float influence = expf(-temp * normalized_score)
            * powf(fmaxf(1.0e-6f, safety_prob), power);
        d_weights[k] = influence;
    }

    int keep = max(1, min(K, static_cast<int>(ceilf(
        clampf(keep_fraction, 1.0f / static_cast<float>(K), 1.0f)
        * static_cast<float>(K)))));
    float cutoff = -FLT_MAX;
    float previous = FLT_MAX;
    for (int r = 0; r < keep; r++) {
        float best = -FLT_MAX;
        for (int k = 0; k < K; k++) {
            float value = d_weights[k];
            if (value <= previous + 1.0e-12f && value > best) best = value;
        }
        cutoff = best;
        previous = best - 1.0e-12f;
    }

    float sum_w = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = (d_weights[k] + 1.0e-12f >= cutoff) ? d_weights[k] : 0.0f;
        d_weights[k] = w;
        sum_w += w;
    }

    if (sum_w > 1.0e-12f) {
        for (int k = 0; k < K; k++) d_weights[k] /= sum_w;
    } else {
        float uniform = 1.0f / static_cast<float>(K);
        for (int k = 0; k < K; k++) d_weights[k] = uniform;
    }
}

__global__ void update_controls_from_cluster_representative_kernel(
    float* d_nominal,
    const float* d_perturbed,
    const float* d_costs,
    const float* d_rollout_states,
    BicycleParams params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    int cluster_count,
    float safe_margin,
    float constraint_weight,
    float update_blend)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int max_clusters = 8;
    int clusters = max(2, min(cluster_count, max_clusters));
    float best_score[max_clusters];
    int best_idx[max_clusters];
    for (int c = 0; c < max_clusters; c++) {
        best_score[c] = FLT_MAX;
        best_idx[c] = -1;
    }

    float global_best_score = FLT_MAX;
    int global_best_idx = 0;
    int mid = max(1, T / 2);
    float safe = fmaxf(-1.0f, safe_margin);
    float penalty_weight = fmaxf(0.0f, constraint_weight);

    for (int k = 0; k < K; k++) {
        float min_margin = 1.0e9f;
        for (int t = 1; t <= T; t++) {
            const float* s = &d_rollout_states[k * (T + 1) * 4 + t * 4];
            float tau = (start_step + t) * params.dt;
            min_margin = fminf(min_margin, min_obstacle_margin_device(
                s[0], s[1], n_obs, n_dyn_obs, tau));
        }
        float violation = fmaxf(0.0f, safe - min_margin);
        float score = d_costs[k] + penalty_weight * violation * violation;

        int mid_base = k * (T + 1) * 4 + mid * 4;
        int end_base = k * (T + 1) * 4 + T * 4;
        float y_feature = 0.55f * d_rollout_states[mid_base + 1]
                        + 0.45f * d_rollout_states[end_base + 1];
        int cid = static_cast<int>(floorf(clampf(y_feature / WORKSPACE, 0.0f, 0.9999f)
                                        * static_cast<float>(clusters)));
        cid = max(0, min(cid, clusters - 1));

        if (score < best_score[cid]) {
            best_score[cid] = score;
            best_idx[cid] = k;
        }
        if (score < global_best_score) {
            global_best_score = score;
            global_best_idx = k;
        }
    }

    int selected = global_best_idx;
    float selected_score = global_best_score;
    for (int c = 0; c < clusters; c++) {
        if (best_idx[c] >= 0 && best_score[c] < selected_score) {
            selected_score = best_score[c];
            selected = best_idx[c];
        }
    }

    float blend = clampf(update_blend, 0.0f, 1.0f);
    float keep = 1.0f - blend;
    for (int t = 0; t < T; t++) {
        d_nominal[t * 2 + 0] = keep * d_nominal[t * 2 + 0]
            + blend * d_perturbed[selected * T * 2 + t * 2 + 0];
        d_nominal[t * 2 + 1] = keep * d_nominal[t * 2 + 1]
            + blend * d_perturbed[selected * T * 2 + t * 2 + 1];
    }
}

__global__ void update_controls_kernel(float* d_nominal, const float* d_perturbed, const float* d_weights, int K, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float accel = 0.0f;
    float steer = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = d_weights[k];
        accel += w * d_perturbed[k * T * 2 + t * 2 + 0];
        steer += w * d_perturbed[k * T * 2 + t * 2 + 1];
    }
    d_nominal[t * 2 + 0] = accel;
    d_nominal[t * 2 + 1] = steer;
}

__global__ void blend_controls_with_previous_kernel(float* d_nominal, const float* d_previous, int T, float momentum) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float keep = clampf(momentum, 0.0f, 0.95f);
    float take = 1.0f - keep;
    d_nominal[t * 2 + 0] = keep * d_previous[t * 2 + 0] + take * d_nominal[t * 2 + 0];
    d_nominal[t * 2 + 1] = keep * d_previous[t * 2 + 1] + take * d_nominal[t * 2 + 1];
}

__global__ void init_deterministic_sigma_kernel(float* d_sigma, int T, float accel_sigma, float steer_sigma) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_sigma[t * 2 + 0] = accel_sigma;
    d_sigma[t * 2 + 1] = steer_sigma;
}

__global__ void shift_deterministic_sigma_kernel(float* d_sigma, int T, float accel_sigma, float steer_sigma) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || T <= 0) return;
    for (int t = 0; t + 1 < T; t++) {
        d_sigma[t * 2 + 0] = d_sigma[(t + 1) * 2 + 0];
        d_sigma[t * 2 + 1] = d_sigma[(t + 1) * 2 + 1];
    }
    d_sigma[(T - 1) * 2 + 0] = accel_sigma;
    d_sigma[(T - 1) * 2 + 1] = steer_sigma;
}

__global__ void update_deterministic_sigma_kernel(
    float* d_sigma,
    const float* d_perturbed,
    const float* d_weights,
    const float* d_nominal,
    int K,
    int T,
    float blend,
    float min_accel_sigma,
    float min_steer_sigma,
    float max_accel_sigma,
    float max_steer_sigma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * 2) return;
    int t = idx / 2;
    int c = idx & 1;
    float mean = d_nominal[idx];
    float var = 0.0f;
    for (int k = 0; k < K; k++) {
        float diff = d_perturbed[k * T * 2 + t * 2 + c] - mean;
        var += d_weights[k] * diff * diff;
    }
    float lo = (c == 0) ? min_accel_sigma : min_steer_sigma;
    float hi = (c == 0) ? max_accel_sigma : max_steer_sigma;
    float sigma_new = clampf(sqrtf(fmaxf(var, lo * lo)), lo, hi);
    float keep = 1.0f - clampf(blend, 0.0f, 1.0f);
    d_sigma[idx] = clampf(keep * d_sigma[idx] + (1.0f - keep) * sigma_new, lo, hi);
}

__global__ void update_deterministic_elite_kernel(
    float* d_nominal,
    float* d_sigma,
    const float* d_perturbed,
    const float* d_costs,
    int K,
    int T,
    int elite_count,
    float sigma_blend,
    float max_steer,
    float min_accel_sigma,
    float min_steer_sigma,
    float max_accel_sigma,
    float max_steer_sigma)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int elite_n = min(max(elite_count, 1), min(K, 64));
    int elite_idx[64];
    float elite_cost[64];
    for (int i = 0; i < elite_n; i++) {
        elite_idx[i] = -1;
        elite_cost[i] = FLT_MAX;
    }

    for (int k = 0; k < K; k++) {
        float c = d_costs[k];
        int pos = -1;
        for (int i = 0; i < elite_n; i++) {
            if (c < elite_cost[i]) {
                pos = i;
                break;
            }
        }
        if (pos >= 0) {
            for (int j = elite_n - 1; j > pos; j--) {
                elite_cost[j] = elite_cost[j - 1];
                elite_idx[j] = elite_idx[j - 1];
            }
            elite_cost[pos] = c;
            elite_idx[pos] = k;
        }
    }

    float keep_sigma = 1.0f - clampf(sigma_blend, 0.0f, 1.0f);
    for (int t = 0; t < T; t++) {
        for (int comp = 0; comp < 2; comp++) {
            float mean = 0.0f;
            for (int i = 0; i < elite_n; i++) {
                int k = max(elite_idx[i], 0);
                mean += d_perturbed[k * T * 2 + t * 2 + comp];
            }
            mean /= static_cast<float>(elite_n);
            float var = 0.0f;
            for (int i = 0; i < elite_n; i++) {
                int k = max(elite_idx[i], 0);
                float diff = d_perturbed[k * T * 2 + t * 2 + comp] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(elite_n);
            int out = t * 2 + comp;
            d_nominal[out] = (comp == 0)
                           ? clampf(mean, -4.0f, 4.0f)
                           : clampf(mean, -max_steer, max_steer);
            if (d_sigma != nullptr) {
                float lo = (comp == 0) ? min_accel_sigma : min_steer_sigma;
                float hi = (comp == 0) ? max_accel_sigma : max_steer_sigma;
                float sigma_new = clampf(sqrtf(fmaxf(var, lo * lo)), lo, hi);
                d_sigma[out] = clampf(keep_sigma * d_sigma[out] + (1.0f - keep_sigma) * sigma_new, lo, hi);
            }
        }
    }
}

// ---- Step-MPPI kernels: learned sampling bias ----

// Apply per-timestep bias shifts to d_nominal before sampling.
// d_bias has T*2 elements: [bias_accel_0, bias_steer_0, bias_accel_1, ...]
__global__ void apply_sampling_bias_kernel(float* d_nominal, const float* d_bias, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_nominal[t * 2 + 0] += d_bias[t * 2 + 0];
    d_nominal[t * 2 + 1] += d_bias[t * 2 + 1];
}

// Remove per-timestep bias shifts from d_nominal after sampling (restore original).
__global__ void remove_sampling_bias_kernel(float* d_nominal, const float* d_bias, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_nominal[t * 2 + 0] -= d_bias[t * 2 + 0];
    d_nominal[t * 2 + 1] -= d_bias[t * 2 + 1];
}

// Update the sampling bias using cost-weighted EMA of control deviations.
// After the MPPI update, d_nominal holds the new weighted-average controls.
// d_nominal_pre holds the pre-bias nominal (original before bias was added).
// The "target shift" is: (new_nominal - nominal_pre), i.e. what MPPI wanted to shift toward.
// We update: bias <- (1-lr)*bias + lr*(new_nominal - nominal_pre)
// with a decay to prevent unbounded drift.
__global__ void update_sampling_bias_kernel(
    float* d_bias,
    const float* d_nominal_new,    // post-MPPI-update nominal
    const float* d_nominal_pre,    // pre-bias nominal (before apply_sampling_bias)
    int T, float lr, float decay)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float target_accel = d_nominal_new[t * 2 + 0] - d_nominal_pre[t * 2 + 0];
    float target_steer = d_nominal_new[t * 2 + 1] - d_nominal_pre[t * 2 + 1];
    d_bias[t * 2 + 0] = decay * ((1.0f - lr) * d_bias[t * 2 + 0] + lr * target_accel);
    d_bias[t * 2 + 1] = decay * ((1.0f - lr) * d_bias[t * 2 + 1] + lr * target_steer);
}

// Shift the sampling bias by one timestep (called when the horizon shifts).
__global__ void shift_sampling_bias_kernel(float* d_bias, int T) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int t = 0; t < T - 1; t++) {
        d_bias[t * 2 + 0] = d_bias[(t + 1) * 2 + 0];
        d_bias[t * 2 + 1] = d_bias[(t + 1) * 2 + 1];
    }
    d_bias[(T - 1) * 2 + 0] = 0.0f;
    d_bias[(T - 1) * 2 + 1] = 0.0f;
}

// ---- End Step-MPPI kernels ----

// Single-thread kernel: sequential forward rollout (each state depends on the previous).
__global__ void rollout_nominal_kernel(
    float sx, float sy, float stheta, float sv,
    const float* d_nominal, float* d_states,
    BicycleParams params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float x = sx;
    float y = sy;
    float theta = stheta;
    float v = sv;
    d_states[0] = x;
    d_states[1] = y;
    d_states[2] = theta;
    d_states[3] = v;

    for (int t = 0; t < T; t++) {
        bicycle_step(x, y, theta, v, d_nominal[t * 2 + 0], d_nominal[t * 2 + 1], params);
        d_states[(t + 1) * 4 + 0] = x;
        d_states[(t + 1) * 4 + 1] = y;
        d_states[(t + 1) * 4 + 2] = theta;
        d_states[(t + 1) * 4 + 3] = v;
    }
}

__device__ void terminal_grad(float x, float y, const CostParams& cp, float grad[4]) {
    for (int i = 0; i < 4; i++) grad[i] = 0.0f;
    for (int var = 0; var < 4; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (var == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf cost = goal_cost_diff(dx, dy, cp.goal_x, cp.goal_y, cp.terminal_weight);
        grad[var] = cost.deriv;
    }
}

__device__ inline Dualf dynamic_obstacle_cost_diff(
    Dualf px, Dualf py, float tau, int n_dyn_obs, float influence, float weight)
{
    Dualf cost = Dualf::constant(0.0f);
    for (int i = 0; i < n_dyn_obs; i++) {
        float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
        float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
        Dualf dx = px - Dualf::constant(ox);
        Dualf dy = py - Dualf::constant(oy);
        Dualf d = cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(1e-6f))
                - Dualf::constant(d_dynamic_obstacles_bench[i].r);
        if (d.val < influence && d.val > 0.1f) {
            cost = cost + Dualf::constant(weight) / (d * d);
        } else if (d.val <= 0.1f) {
            cost = cost + Dualf::constant(weight * 100.0f);
        }
    }
    return cost;
}

__device__ void stage_cost_grad(
    float x, float y, float theta, float v, float accel, float steer,
    const CostParams& cp, int n_obs, int n_dyn_obs, float tau, float grad[6])
{
    for (int var = 0; var < 6; var++) {
        Dualf dx = (var == 0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (var == 1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dtheta = (var == 2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dv = (var == 3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf da = (var == 4) ? Dualf::variable(accel) : Dualf::constant(accel);
        Dualf ds = (var == 5) ? Dualf::variable(steer) : Dualf::constant(steer);

        Dualf cost = goal_cost_diff(dx, dy, cp.goal_x, cp.goal_y, cp.goal_weight)
                   + obstacle_cost_diff(dx, dy, d_obstacles_bench, n_obs, cp.obs_influence, cp.obs_weight)
                   + dynamic_obstacle_cost_diff(dx, dy, tau, n_dyn_obs, cp.obs_influence, cp.obs_weight)
                   + control_cost_diff(da, ds, cp.control_weight)
                   + speed_cost_diff(dv, cp.target_speed, cp.speed_weight)
                   + heading_cost_diff(dx, dy, dtheta, cp.goal_x, cp.goal_y, cp.heading_weight);
        grad[var] = cost.deriv;
    }
}

// Phase 1: Parallel cost gradient + Jacobian computation (T threads)
__global__ void precompute_nav_gradients_kernel(
    const float* d_states, const float* d_nominal,
    float* d_stage_grads,   // T * 6
    float* d_jacobians,     // T * 4 * 6
    BicycleParams params, CostParams cost_params,
    int n_obs, int n_dyn_obs, int start_step, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float x = d_states[t * 4 + 0];
    float y = d_states[t * 4 + 1];
    float theta = d_states[t * 4 + 2];
    float v = d_states[t * 4 + 3];
    float accel = d_nominal[t * 2 + 0];
    float steer = d_nominal[t * 2 + 1];
    float tau = (start_step + t) * params.dt;

    float sg[6];
    stage_cost_grad(x, y, theta, v, accel, steer, cost_params, n_obs, n_dyn_obs, tau, sg);
    for (int i = 0; i < 6; i++) d_stage_grads[t * 6 + i] = sg[i];

    float J[4][6];
    bicycle_jacobian(x, y, theta, v, accel, steer, params, J);
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 6; c++)
            d_jacobians[t * 24 + r * 6 + c] = J[r][c];
}

// Phase 2: Sequential backward adjoint pass (1 thread, matrix ops only)
// Single-thread kernel: sequential backward adjoint pass (each timestep depends on
// the next). Cost gradients and Jacobians are precomputed in parallel by
// precompute_nav_gradients_kernel, so this kernel does matrix ops only.
__global__ void backward_nav_adjoint_kernel(
    const float* d_states,
    const float* d_stage_grads,
    const float* d_jacobians,
    float* d_grad,
    CostParams cost_params, int T)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float adj[4];
    terminal_grad(d_states[T * 4 + 0], d_states[T * 4 + 1], cost_params, adj);

    for (int t = T - 1; t >= 0; t--) {
        const float* sg = &d_stage_grads[t * 6];
        const float* Jf = &d_jacobians[t * 24];

        d_grad[t * 2 + 0] = sg[4];
        d_grad[t * 2 + 1] = sg[5];
        for (int row = 0; row < 4; row++) {
            d_grad[t * 2 + 0] += Jf[row * 6 + 4] * adj[row];
            d_grad[t * 2 + 1] += Jf[row * 6 + 5] * adj[row];
        }

        float next_adj[4];
        for (int col = 0; col < 4; col++) {
            next_adj[col] = sg[col];
            for (int row = 0; row < 4; row++) next_adj[col] += Jf[row * 6 + col] * adj[row];
        }
        for (int i = 0; i < 4; i++) adj[i] = next_adj[i];
    }
}

// Compute total gradient norm across all timesteps (for adaptive skip).
__global__ void gradient_norm_kernel(const float* d_grad, float* d_grad_norm, int T) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float norm_sq = 0.0f;
    for (int t = 0; t < T; t++) {
        float a = d_grad[t * 2 + 0];
        float s = d_grad[t * 2 + 1];
        norm_sq += a * a + s * s;
    }
    d_grad_norm[0] = sqrtf(norm_sq);
}

__global__ void gradient_step_kernel(float* d_nominal, const float* d_grad, int T, float alpha, float max_steer) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    d_nominal[t * 2 + 0] = clampf(d_nominal[t * 2 + 0] - alpha * d_grad[t * 2 + 0], -4.0f, 4.0f);
    d_nominal[t * 2 + 1] = clampf(d_nominal[t * 2 + 1] - alpha * d_grad[t * 2 + 1], -max_steer, max_steer);
}

__global__ void soppi_stage_score_kernel(
    const float* d_controls,
    const float* d_rollout_states,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int K,
    int T,
    float lambda,
    float* d_scores)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * T;
    if (idx >= total) return;

    int k = idx / T;
    int t = idx - k * T;
    int base = k * T * 2 + t * 2;
    const float* state = &d_rollout_states[k * (T + 1) * 4 + t * 4];
    float tau = (start_step + t) * params.dt;
    float grad[6];
    stage_cost_grad(state[0], state[1], state[2], state[3],
                    d_controls[base + 0], d_controls[base + 1],
                    cost_params, n_obs, n_dyn_obs, tau, grad);
    float inv_lambda = 1.0f / fmaxf(lambda, 1.0e-3f);
    d_scores[base + 0] = -clampf(grad[4] * inv_lambda, -25.0f, 25.0f);
    d_scores[base + 1] = -clampf(grad[5] * inv_lambda, -25.0f, 25.0f);
}

__global__ void soppi_svgd_step_kernel(
    const float* d_controls,
    float* d_controls_next,
    const float* d_scores,
    BicycleParams params,
    int K,
    int T,
    int neighbor_count,
    float bandwidth,
    float step_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * T;
    if (idx >= total) return;

    const float noise_accel = 1.5f;
    const float noise_steer = 0.18f;
    const float h = fmaxf(0.10f, bandwidth);
    int k = idx / T;
    int t = idx - k * T;
    int base = k * T * 2 + t * 2;
    float accel_i = d_controls[base + 0];
    float steer_i = d_controls[base + 1];
    int neighbor_samples = K;
    if (neighbor_count > 0 && neighbor_count < K) neighbor_samples = neighbor_count;
    int stride = K / neighbor_samples;
    if (stride < 1) stride = 1;

    float phi_accel = 0.0f;
    float phi_steer = 0.0f;
    for (int m = 0; m < neighbor_samples; m++) {
        int j = neighbor_count > 0 ? (k + m * stride) % K : m;
        int jbase = j * T * 2 + t * 2;
        float accel_j = d_controls[jbase + 0];
        float steer_j = d_controls[jbase + 1];
        float da = (accel_j - accel_i) / noise_accel;
        float ds = (steer_j - steer_i) / noise_steer;
        float k_rbf = expf(-(da * da + ds * ds) / h);

        float repel_accel = -2.0f * k_rbf * da / (h * noise_accel);
        float repel_steer = -2.0f * k_rbf * ds / (h * noise_steer);
        phi_accel += k_rbf * d_scores[jbase + 0] + repel_accel;
        phi_steer += k_rbf * d_scores[jbase + 1] + repel_steer;
    }
    phi_accel /= fmaxf(1.0f, static_cast<float>(neighbor_samples));
    phi_steer /= fmaxf(1.0f, static_cast<float>(neighbor_samples));

    float delta_accel = clampf(step_size * phi_accel, -0.40f, 0.40f);
    float delta_steer = clampf(step_size * phi_steer, -0.06f, 0.06f);
    d_controls_next[base + 0] = clampf(accel_i + delta_accel, -4.0f, 4.0f);
    d_controls_next[base + 1] = clampf(steer_i + delta_steer, -params.max_steer, params.max_steer);
}

// One thread = one (accel, steer) grid point. Each thread holds its (accel, steer)
// constant over T_dwa steps and integrates the bicycle dynamics, returning the
// trajectory cost. The host then takes the argmin and uses (accel, steer) as the
// next control. Cost terms mirror the MPPI cost so DWA and MPPI are comparable.
__global__ void dwa_grid_kernel(
    float sx, float sy, float stheta, float sv,
    float* d_grid_costs,
    float* d_grid_accels,
    float* d_grid_steers,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int T_dwa,
    int n_accel,
    int n_steer,
    float accel_min, float accel_max,
    float w_goal, float w_speed, float w_obs, float w_heading,
    float w_terminal)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_accel * n_steer;
    if (tid >= total) return;

    int i_accel = tid / n_steer;
    int i_steer = tid % n_steer;
    float accel = (n_accel > 1)
        ? accel_min + (accel_max - accel_min) * i_accel / (float)(n_accel - 1)
        : 0.5f * (accel_min + accel_max);
    float steer_range = 2.0f * params.max_steer;
    float steer = (n_steer > 1)
        ? -params.max_steer + steer_range * i_steer / (float)(n_steer - 1)
        : 0.0f;

    d_grid_accels[tid] = accel;
    d_grid_steers[tid] = steer;

    float x = sx, y = sy, theta = stheta, v = sv;
    float cost = 0.0f;
    bool collided = false;

    for (int t = 0; t < T_dwa; t++) {
        bicycle_step(x, y, theta, v, accel, steer, params);

        float dxg = x - cost_params.goal_x;
        float dyg = y - cost_params.goal_y;
        cost += w_goal * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;

        float desired_heading = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
        float heading_err = theta - desired_heading;
        cost += w_heading * heading_err * heading_err * params.dt;

        float speed_err = v - cost_params.target_speed;
        cost += w_speed * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) { cost += w_obs * 100.0f; collided = true; }
            else if (margin < cost_params.obs_influence) cost += w_obs / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) { cost += w_obs * 100.0f; collided = true; }
            else if (margin < cost_params.obs_influence) cost += w_obs / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) cost += 500.0f;
        if (collided) break;
    }

    float dx = x - cost_params.goal_x;
    float dy = y - cost_params.goal_y;
    cost += w_terminal * sqrtf(dx * dx + dy * dy + 0.01f);

    d_grid_costs[tid] = cost;
}

// Hybrid A* + DWA hybrid: DWA grid search but cost is dominated by tracking a
// pre-computed Hybrid A* path. Replaces goal-distance/heading with
// (nearest-path-point lateral error) and (heading vs lookahead-point bearing).
// Static obstacles are baked into the global path so the local term mostly
// shapes around dynamic obstacles; obstacle and speed terms remain as in DWA.
// If path_n == 0 (planning failed) the kernel falls back to vanilla DWA cost.
__global__ void hybrid_astar_dwa_grid_kernel(
    float sx, float sy, float stheta, float sv,
    float* d_grid_costs,
    float* d_grid_accels,
    float* d_grid_steers,
    BicycleParams params,
    CostParams cost_params,
    int n_obs,
    int n_dyn_obs,
    int start_step,
    int T_dwa,
    int n_accel,
    int n_steer,
    float accel_min, float accel_max,
    float w_path, float w_speed, float w_obs, float w_heading,
    float w_terminal,
    const float* __restrict__ d_path,
    int path_n,
    int lookahead_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_accel * n_steer;
    if (tid >= total) return;

    int i_accel = tid / n_steer;
    int i_steer = tid % n_steer;
    float accel = (n_accel > 1)
        ? accel_min + (accel_max - accel_min) * i_accel / (float)(n_accel - 1)
        : 0.5f * (accel_min + accel_max);
    float steer_range = 2.0f * params.max_steer;
    float steer = (n_steer > 1)
        ? -params.max_steer + steer_range * i_steer / (float)(n_steer - 1)
        : 0.0f;

    d_grid_accels[tid] = accel;
    d_grid_steers[tid] = steer;

    float x = sx, y = sy, theta = stheta, v = sv;
    float cost = 0.0f;
    bool collided = false;

    for (int t = 0; t < T_dwa; t++) {
        bicycle_step(x, y, theta, v, accel, steer, params);

        if (path_n > 0) {
            float best_d2 = 1.0e30f;
            int best_idx = 0;
            for (int p = 0; p < path_n; p++) {
                float dxp = x - d_path[p * 3 + 0];
                float dyp = y - d_path[p * 3 + 1];
                float d2 = dxp * dxp + dyp * dyp;
                if (d2 < best_d2) { best_d2 = d2; best_idx = p; }
            }
            cost += w_path * sqrtf(best_d2 + 0.01f) * params.dt;
            int look = best_idx + lookahead_idx;
            if (look >= path_n) look = path_n - 1;
            float lx = d_path[look * 3 + 0];
            float ly = d_path[look * 3 + 1];
            float dxL = lx - x;
            float dyL = ly - y;
            float dL2 = dxL * dxL + dyL * dyL;
            // Within ~0.5m of the lookahead point the bearing atan2(dyL,dxL)
            // is numerically unstable and degenerates to 0 when robot sits on
            // the path end — that traps the robot by penalizing any theta.
            // Fall back to the path waypoint's stored tangent in that regime.
            float desired = (dL2 > 0.25f)
                ? atan2f(dyL, dxL)
                : d_path[look * 3 + 2];
            float herr = theta - desired;
            while (herr >  3.14159265f) herr -= 6.28318531f;
            while (herr < -3.14159265f) herr += 6.28318531f;
            cost += w_heading * herr * herr * params.dt;
        } else {
            float dxg = x - cost_params.goal_x;
            float dyg = y - cost_params.goal_y;
            cost += w_path * sqrtf(dxg * dxg + dyg * dyg + 0.01f) * params.dt;
            float desired = atan2f(cost_params.goal_y - y, cost_params.goal_x - x);
            float herr = theta - desired;
            cost += w_heading * herr * herr * params.dt;
        }

        float speed_err = v - cost_params.target_speed;
        cost += w_speed * speed_err * speed_err * params.dt;

        for (int i = 0; i < n_obs; i++) {
            float dx = x - d_obstacles_bench[i].x;
            float dy = y - d_obstacles_bench[i].y;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_obstacles_bench[i].r;
            if (margin <= 0.1f) { cost += w_obs * 100.0f; collided = true; }
            else if (margin < cost_params.obs_influence) cost += w_obs / (margin * margin);
        }

        float tau = (start_step + t + 1) * params.dt;
        for (int i = 0; i < n_dyn_obs; i++) {
            float ox = d_dynamic_obstacles_bench[i].x + d_dynamic_obstacles_bench[i].vx * tau;
            float oy = d_dynamic_obstacles_bench[i].y + d_dynamic_obstacles_bench[i].vy * tau;
            float dx = x - ox;
            float dy = y - oy;
            float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - d_dynamic_obstacles_bench[i].r;
            if (margin <= 0.1f) { cost += w_obs * 100.0f; collided = true; }
            else if (margin < cost_params.obs_influence) cost += w_obs / (margin * margin);
        }

        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) cost += 500.0f;
        if (collided) break;
    }

    // Path-aware terminal cost. The earlier "w_terminal * dist(robot_end,
    // goal_x/y)" formulation pulls the robot toward the abstract goal at
    // the end of every rollout, which on local-minima scenes (U-trap,
    // S-corridor) drags the robot off the path and into the trap. With
    // a path available we instead pull toward a soft target a few indices
    // ahead of the rollout-end's nearest path node, plus a small
    // remaining-path-length penalty so paths with less left to cover are
    // preferred. The constant-step approximation (2.5 m / step) is
    // independent of v_search and works as a relative-cost shape.
    if (path_n > 0) {
        float best_d2 = 1.0e30f;
        int best_idx = 0;
        for (int p = 0; p < path_n; p++) {
            float dxp = x - d_path[p * 3 + 0];
            float dyp = y - d_path[p * 3 + 1];
            float d2 = dxp * dxp + dyp * dyp;
            if (d2 < best_d2) { best_d2 = d2; best_idx = p; }
        }
        int term_idx = best_idx + lookahead_idx;
        if (term_idx >= path_n) term_idx = path_n - 1;
        float tdx = x - d_path[term_idx * 3 + 0];
        float tdy = y - d_path[term_idx * 3 + 1];
        float remaining = static_cast<float>(path_n - 1 - term_idx) * 2.5f;
        cost += w_terminal * (sqrtf(tdx * tdx + tdy * tdy + 0.01f) + remaining);
    } else {
        float gdx = x - cost_params.goal_x;
        float gdy = y - cost_params.goal_y;
        cost += w_terminal * sqrtf(gdx * gdx + gdy * gdy + 0.01f);
    }

    d_grid_costs[tid] = cost;
}

// STOMP weight kernel: P(k) ∝ exp(-h * (S(k) - S_min) / (S_max - S_min)).
// This normalises into [0, 1] before the exponential, so the sharpness parameter
// h has the same effect regardless of cost scale -- different from MPPI's λ.
__global__ void compute_stomp_weights_kernel(const float* d_costs, float* d_weights, int K, float h) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float min_cost = d_costs[0];
    float max_cost = d_costs[0];
    for (int k = 1; k < K; k++) {
        float c = d_costs[k];
        if (c < min_cost) min_cost = c;
        if (c > max_cost) max_cost = c;
    }
    float range = max_cost - min_cost;
    if (range < 1.0e-6f) {
        float u = 1.0f / (float)K;
        for (int k = 0; k < K; k++) d_weights[k] = u;
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = expf(-h * (d_costs[k] - min_cost) / range);
        d_weights[k] = w;
        sum += w;
    }
    float inv = 1.0f / sum;
    for (int k = 0; k < K; k++) d_weights[k] *= inv;
}

// Smoothness projection by a 3-tap moving average over the horizon. This is a
// STOMP smoothness projection M = (R^T R)^-1, column-normalised so the max
// |entry| per column is 1. R is the T x T tridiagonal second-difference
// operator with Dirichlet boundaries (rows [-2 1 ...], [1 -2 1 ...], ...,
// [... 1 -2]), so A = R^T R is the squared-acceleration penalty. Applying
// u_smooth = M @ u yields the STOMP-paper smoothness projection that
// guarantees the update lives in the null-space of the high-frequency
// modes. Replaces the previous 3-tap moving-average approximation.
//
// The matrix is small (T <= ~50 in practice), so build_stomp_M just
// inverts A on the host via Gauss-Jordan and ships the result to device
// constant-ish memory once per episode. Per-call cost is T*T fmas per
// time step, which is negligible compared to the rollout kernel.
static std::vector<float> build_stomp_M(int T) {
    if (T <= 0) return {};
    std::vector<float> R(T * T, 0.0f);
    for (int i = 0; i < T; i++) {
        R[i * T + i] = -2.0f;
        if (i > 0)     R[i * T + (i - 1)] = 1.0f;
        if (i < T - 1) R[i * T + (i + 1)] = 1.0f;
    }
    std::vector<double> A(T * T, 0.0);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            double s = 0.0;
            for (int k = 0; k < T; k++) {
                s += static_cast<double>(R[k * T + i])
                   * static_cast<double>(R[k * T + j]);
            }
            A[i * T + j] = s;
        }
    }
    // Gauss-Jordan inversion of A in place into M_inv.
    std::vector<double> M_inv(T * T, 0.0);
    for (int i = 0; i < T; i++) M_inv[i * T + i] = 1.0;
    for (int col = 0; col < T; col++) {
        int pivot = col;
        double best = std::abs(A[col * T + col]);
        for (int r = col + 1; r < T; r++) {
            double v = std::abs(A[r * T + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-12) {
            // Singular -- fall back to identity (callers will still smooth
            // via the column-normalised matrix; this should not happen for
            // the Dirichlet second-difference operator).
            std::vector<float> I(T * T, 0.0f);
            for (int i = 0; i < T; i++) I[i * T + i] = 1.0f;
            return I;
        }
        if (pivot != col) {
            for (int j = 0; j < T; j++) {
                std::swap(A[col * T + j], A[pivot * T + j]);
                std::swap(M_inv[col * T + j], M_inv[pivot * T + j]);
            }
        }
        double diag = A[col * T + col];
        for (int j = 0; j < T; j++) {
            A[col * T + j] /= diag;
            M_inv[col * T + j] /= diag;
        }
        for (int r = 0; r < T; r++) {
            if (r == col) continue;
            double factor = A[r * T + col];
            if (factor == 0.0) continue;
            for (int j = 0; j < T; j++) {
                A[r * T + j]     -= factor * A[col * T + j];
                M_inv[r * T + j] -= factor * M_inv[col * T + j];
            }
        }
    }
    // STOMP column normalisation: each column scaled so max |entry| = 1/N.
    // Concretely M.col(j) /= (N * max(|A^{-1}.col(j)|)). This bounds the row
    // sums of M by ~1, so M @ δu acts as a proper smoothing average rather
    // than amplifying the update (which is what a column-max=1 scaling
    // would do — row sums grow to O(T) for the squared-Laplacian inverse).
    std::vector<float> M(T * T, 0.0f);
    double N = static_cast<double>(T);
    for (int j = 0; j < T; j++) {
        double maxabs = 0.0;
        for (int i = 0; i < T; i++) {
            double v = std::abs(M_inv[i * T + j]);
            if (v > maxabs) maxabs = v;
        }
        double scale = (maxabs > 1e-12) ? (1.0 / (N * maxabs)) : 1.0;
        for (int i = 0; i < T; i++) {
            M[i * T + j] = static_cast<float>(M_inv[i * T + j] * scale);
        }
    }
    return M;
}

// STOMP applies M to the update delta (u_new - u_old), not to the absolute
// trajectory. The column-normalised M does not preserve constants, so
// projecting u itself would shrink the nominal toward zero each call.
// Instead we compute u_smooth = u_old + M @ (u_new - u_old), which keeps
// the previously committed trajectory and only smooths the cost-weighted
// noise average that drove this iteration's change.
__global__ void stomp_delta_project_kernel(const float* __restrict__ u_old,
                                            const float* __restrict__ u_new,
                                            float* __restrict__ u_out,
                                            const float* __restrict__ d_M,
                                            int T,
                                            float max_steer) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float a_sum = 0.0f;
    float s_sum = 0.0f;
    for (int j = 0; j < T; j++) {
        float w = d_M[t * T + j];
        float du_a = u_new[j * 2 + 0] - u_old[j * 2 + 0];
        float du_s = u_new[j * 2 + 1] - u_old[j * 2 + 1];
        a_sum += w * du_a;
        s_sum += w * du_s;
    }
    u_out[t * 2 + 0] = clampf(u_old[t * 2 + 0] + a_sum, -4.0f, 4.0f);
    u_out[t * 2 + 1] = clampf(u_old[t * 2 + 1] + s_sum, -max_steer, max_steer);
}

__global__ void copy_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

static float dynamic_obstacle_margin(float x, float y, const DynamicObstacle& obs, float tau) {
    float ox = obs.x + obs.vx * tau;
    float oy = obs.y + obs.vy * tau;
    float dx = x - ox;
    float dy = y - oy;
    return sqrtf(dx * dx + dy * dy + 1e-6f) - obs.r;
}

static float host_step_cost(
    float x, float y, float theta, float v, float accel, float steer,
    const Scenario& scenario, int step_index)
{
    const CostParams& cp = scenario.cost_params;
    float tau = step_index * scenario.params.dt;
    float dxg = x - cp.goal_x;
    float dyg = y - cp.goal_y;
    float cost = cp.goal_weight * sqrtf(dxg * dxg + dyg * dyg + 0.01f);
    cost += cp.control_weight * (accel * accel + steer * steer);
    float desired_heading = atan2f(cp.goal_y - y, cp.goal_x - x);
    float heading_err = theta - desired_heading;
    cost += cp.heading_weight * heading_err * heading_err;
    float speed_err = v - cp.target_speed;
    cost += cp.speed_weight * speed_err * speed_err;

    for (int i = 0; i < scenario.n_obs; i++) {
        float dx = x - scenario.obstacles[i].x;
        float dy = y - scenario.obstacles[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - scenario.obstacles[i].r;
        if (margin <= 0.1f) cost += cp.obs_weight * 100.0f;
        else if (margin < cp.obs_influence) cost += cp.obs_weight / (margin * margin);
    }

    for (int i = 0; i < scenario.n_dyn_obs; i++) {
        float margin = dynamic_obstacle_margin(x, y, scenario.dynamic_obstacles[i], tau);
        if (margin <= 0.1f) cost += cp.obs_weight * 100.0f;
        else if (margin < cp.obs_influence) cost += cp.obs_weight / (margin * margin);
    }

    if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) cost += 500.0f;
    return cost;
}

static float min_obstacle_margin(float x, float y, const Scenario& scenario, int step_index) {
    float best = 1.0e9f;
    float tau = step_index * scenario.params.dt;
    for (int i = 0; i < scenario.n_obs; i++) {
        float dx = x - scenario.obstacles[i].x;
        float dy = y - scenario.obstacles[i].y;
        float margin = sqrtf(dx * dx + dy * dy + 1e-6f) - scenario.obstacles[i].r;
        best = std::min(best, margin);
    }
    for (int i = 0; i < scenario.n_dyn_obs; i++) {
        best = std::min(best, dynamic_obstacle_margin(x, y, scenario.dynamic_obstacles[i], tau));
    }
    return best;
}

class EpisodeRunner {
public:
    EpisodeRunner(const PlannerVariant& variant, const Scenario& planning_scenario, const Scenario& eval_scenario,
                  int k_samples, int t_horizon, int seed,
                  vector<TraceRow>* trace_rows = nullptr, int trace_max_steps = 0,
                  vector<TrajectoryRow>* trajectory_rows = nullptr)
        : variant_(variant), planning_scenario_(planning_scenario), eval_scenario_(eval_scenario),
          k_samples_(k_samples), t_horizon_(t_horizon), seed_(seed),
          trace_rows_(trace_rows), trace_max_steps_(trace_max_steps),
          trajectory_rows_(trajectory_rows) {
        reset_state();

        h_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_costs_.assign(k_samples_, 0.0f);
        h_grad_.assign(t_horizon_ * 2, 0.0f);
        h_states_.assign((t_horizon_ + 1) * 4, 0.0f);
        h_feedback_gains_host_.assign(t_horizon_ * 2 * 4, 0.0f);
        h_sample_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_final_nominal_.assign(t_horizon_ * 2, 0.0f);
        h_grad_snapshot_.assign(t_horizon_ * 2, 0.0f);

        CUDA_CHECK(cudaMalloc(&d_nominal_, h_nominal_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_costs_, h_costs_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_, k_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed_, k_samples_ * t_horizon_ * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_states_, k_samples_ * (t_horizon_ + 1) * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rollout_init_grads_, k_samples_ * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states_, h_states_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_, h_grad_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nav_stage_grads_, t_horizon_ * 6 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nav_jacobians_, t_horizon_ * 24 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_norm_, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_, t_horizon_ * 2 * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_feedback_gains_aux_, t_horizon_ * 2 * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, k_samples_ * sizeof(curandState)));

        // Step-MPPI / dsMPPI: allocate per-horizon auxiliary buffers
        if (variant_.use_learned_sampling) {
            CUDA_CHECK(cudaMalloc(&d_sampling_bias_, t_horizon_ * 2 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_sampling_bias_, 0, t_horizon_ * 2 * sizeof(float)));
            if (variant_.use_learned_sigma) {
                CUDA_CHECK(cudaMalloc(&d_step_sigma_, t_horizon_ * 2 * sizeof(float)));
            }
        }
        if (variant_.use_learned_sampling || variant_.use_deterministic_sampling) {
            CUDA_CHECK(cudaMalloc(&d_nominal_pre_bias_, t_horizon_ * 2 * sizeof(float)));
        }
        if (variant_.use_deterministic_sampling && (variant_.ds_adapt_sigma || variant_.ds_elite_update)) {
            CUDA_CHECK(cudaMalloc(&d_ds_sigma_, t_horizon_ * 2 * sizeof(float)));
        }
        if (variant_.use_soppi_sampling) {
            CUDA_CHECK(cudaMalloc(&d_soppi_scratch_, k_samples_ * t_horizon_ * 2 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_soppi_scores_, k_samples_ * t_horizon_ * 2 * sizeof(float)));
        }

        if (variant_.planner_kind == 1 || variant_.planner_kind == 4) {
            dwa_grid_size_ = max(1, variant_.dwa_n_accel) * max(1, variant_.dwa_n_steer);
            h_dwa_costs_.assign(dwa_grid_size_, 0.0f);
            h_dwa_accels_.assign(dwa_grid_size_, 0.0f);
            h_dwa_steers_.assign(dwa_grid_size_, 0.0f);
            CUDA_CHECK(cudaMalloc(&d_dwa_costs_, dwa_grid_size_ * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_dwa_accels_, dwa_grid_size_ * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_dwa_steers_, dwa_grid_size_ * sizeof(float)));
        }
        if (variant_.planner_kind == 2) {
            CUDA_CHECK(cudaMalloc(&d_stomp_scratch_, t_horizon_ * 2 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_stomp_old_, t_horizon_ * 2 * sizeof(float)));
            std::vector<float> M_host = build_stomp_M(t_horizon_);
            CUDA_CHECK(cudaMalloc(&d_stomp_M_, t_horizon_ * t_horizon_ * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_stomp_M_, M_host.data(),
                                  t_horizon_ * t_horizon_ * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        reset_rng();
    }

    ~EpisodeRunner() {
        CUDA_CHECK(cudaFree(d_nominal_));
        CUDA_CHECK(cudaFree(d_costs_));
        CUDA_CHECK(cudaFree(d_weights_));
        CUDA_CHECK(cudaFree(d_perturbed_));
        CUDA_CHECK(cudaFree(d_rollout_states_));
        CUDA_CHECK(cudaFree(d_rollout_init_grads_));
        CUDA_CHECK(cudaFree(d_states_));
        CUDA_CHECK(cudaFree(d_grad_));
        CUDA_CHECK(cudaFree(d_nav_stage_grads_));
        CUDA_CHECK(cudaFree(d_nav_jacobians_));
        CUDA_CHECK(cudaFree(d_grad_norm_));
        CUDA_CHECK(cudaFree(d_feedback_gains_));
        CUDA_CHECK(cudaFree(d_feedback_gains_aux_));
        CUDA_CHECK(cudaFree(d_rng_));
        if (d_sampling_bias_) CUDA_CHECK(cudaFree(d_sampling_bias_));
        if (d_step_sigma_) CUDA_CHECK(cudaFree(d_step_sigma_));
        if (d_nominal_pre_bias_) CUDA_CHECK(cudaFree(d_nominal_pre_bias_));
        if (d_ds_sigma_) CUDA_CHECK(cudaFree(d_ds_sigma_));
        if (d_soppi_scratch_) CUDA_CHECK(cudaFree(d_soppi_scratch_));
        if (d_soppi_scores_) CUDA_CHECK(cudaFree(d_soppi_scores_));
        if (d_dwa_costs_) CUDA_CHECK(cudaFree(d_dwa_costs_));
        if (d_dwa_accels_) CUDA_CHECK(cudaFree(d_dwa_accels_));
        if (d_dwa_steers_) CUDA_CHECK(cudaFree(d_dwa_steers_));
        if (d_stomp_scratch_) CUDA_CHECK(cudaFree(d_stomp_scratch_));
        if (d_stomp_old_) CUDA_CHECK(cudaFree(d_stomp_old_));
        if (d_stomp_M_) CUDA_CHECK(cudaFree(d_stomp_M_));
        if (d_had_path_) CUDA_CHECK(cudaFree(d_had_path_));
    }

    EpisodeMetrics run() {
        reset_state();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        reset_ds_sigma();
        reset_step_sigma();
        warmup_controller();
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        reset_ds_sigma();
        reset_step_sigma();
        reset_rng();

        if (trajectory_rows_ != nullptr) {
            float goal_dx = rx_ - eval_scenario_.cost_params.goal_x;
            float goal_dy = ry_ - eval_scenario_.cost_params.goal_y;
            append_trajectory_row(0, sqrtf(goal_dx * goal_dx + goal_dy * goal_dy));
        }

        auto episode_begin = chrono::steady_clock::now();
        float total_control_ms = 0.0f;
        int controller_updates = 0;
        float prev_accel = 0.0f;
        float prev_steer = 0.0f;
        bool have_prev_control = false;
        float control_delta_sum = 0.0f;
        float control_roughness_sum = 0.0f;
        int control_delta_count = 0;

        for (int step = 0; step < eval_scenario_.max_steps; step++) {
            float goal_dx = rx_ - eval_scenario_.cost_params.goal_x;
            float goal_dy = ry_ - eval_scenario_.cost_params.goal_y;
            float goal_dist = sqrtf(goal_dx * goal_dx + goal_dy * goal_dy);
            float margin_before = min_obstacle_margin(rx_, ry_, eval_scenario_, step);
            min_goal_distance_ = std::min(min_goal_distance_, goal_dist);
            if (goal_dist < eval_scenario_.goal_tol) {
                reached_goal_ = true;
                steps_taken_ = step;
                break;
            }

            auto t0 = chrono::steady_clock::now();
            bool replan = should_replan(step);
            if (replan) {
                controller_update(rx_, ry_, rtheta_, rv_, step);
                controller_updates++;
                sync_nominal_from_device();
                if (uses_feedback_local_action()) sync_feedback_policy_from_device();
                CUDA_CHECK(cudaMemcpy(h_costs_.data(), d_costs_, h_costs_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            }
            float accel = h_nominal_[0];
            float steer = h_nominal_[1];
            if (uses_feedback_local_action()) {
                compute_feedback_inner_action(accel, steer);
            }
            if (variant_.use_shield_repair) {
                apply_shield_repair(accel, steer, step);
            }
            if (have_prev_control) {
                float da = accel - prev_accel;
                float ds = steer - prev_steer;
                control_delta_sum += sqrtf(da * da + ds * ds);
                control_roughness_sum += da * da + ds * ds;
                control_delta_count++;
            }
            prev_accel = accel;
            prev_steer = steer;
            have_prev_control = true;
            auto t1 = chrono::steady_clock::now();
            float control_ms = chrono::duration<float, milli>(t1 - t0).count();
            total_control_ms += control_ms;

            if (trace_rows_ != nullptr && step < trace_max_steps_) {
                append_trace_rows(step, goal_dist, margin_before, control_ms);
            }

            bicycle_step(rx_, ry_, rtheta_, rv_, accel, steer, eval_scenario_.params);
            cumulative_cost_ += host_step_cost(rx_, ry_, rtheta_, rv_, accel, steer, eval_scenario_, step + 1);

            float margin = min_obstacle_margin(rx_, ry_, eval_scenario_, step + 1);
            if (margin <= 0.0f || rx_ < 0.0f || rx_ > WORKSPACE || ry_ < 0.0f || ry_ > WORKSPACE) collisions_++;

            shift_host_policy();
            steps_taken_ = step + 1;

            if (trajectory_rows_ != nullptr) {
                float gdx = rx_ - eval_scenario_.cost_params.goal_x;
                float gdy = ry_ - eval_scenario_.cost_params.goal_y;
                append_trajectory_row(step + 1, sqrtf(gdx * gdx + gdy * gdy));
            }
        }

        auto episode_end = chrono::steady_clock::now();
        float final_dx = rx_ - eval_scenario_.cost_params.goal_x;
        float final_dy = ry_ - eval_scenario_.cost_params.goal_y;
        float final_distance = sqrtf(final_dx * final_dx + final_dy * final_dy);
        if (final_distance < eval_scenario_.goal_tol) reached_goal_ = true;

        EpisodeMetrics metrics;
        metrics.scenario = eval_scenario_.name;
        metrics.planner = variant_.name;
        metrics.seed = seed_;
        metrics.k_samples = k_samples_;
        metrics.t_horizon = t_horizon_;
        metrics.grad_steps = variant_.grad_steps;
        metrics.alpha = variant_.alpha;
        metrics.reached_goal = reached_goal_ ? 1 : 0;
        metrics.collision_free = collisions_ == 0 ? 1 : 0;
        metrics.success = (metrics.reached_goal && metrics.collision_free) ? 1 : 0;
        metrics.steps = steps_taken_;
        metrics.final_distance = final_distance;
        metrics.min_goal_distance = min_goal_distance_;
        metrics.cumulative_cost = cumulative_cost_;
        metrics.collisions = collisions_;
        metrics.mean_control_delta = control_delta_count > 0 ? control_delta_sum / control_delta_count : 0.0f;
        metrics.control_roughness = control_delta_count > 0 ? control_roughness_sum / control_delta_count : 0.0f;
        metrics.total_control_ms = total_control_ms;
        metrics.avg_control_ms = steps_taken_ > 0 ? total_control_ms / steps_taken_ : 0.0f;
        metrics.episode_ms = chrono::duration<float, milli>(episode_end - episode_begin).count();
        int sampling_passes = variant_.use_deterministic_sampling ? max(1, variant_.ds_iterations) : 1;
        metrics.sample_budget = static_cast<long long>(controller_updates) * sampling_passes * k_samples_ * t_horizon_;
        return metrics;
    }

private:
    void reset_rng() {
        int block = 256;
        init_curand_kernel<<<(k_samples_ + block - 1) / block, block>>>(d_rng_, k_samples_, static_cast<unsigned long long>(seed_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void reset_ds_sigma() {
        if (!d_ds_sigma_) return;
        int block = 256;
        init_deterministic_sigma_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_ds_sigma_, t_horizon_,
            1.5f * variant_.ds_noise_scale,
            0.18f * variant_.ds_noise_scale);
    }

    void reset_step_sigma() {
        if (!d_step_sigma_) return;
        int block = 256;
        init_deterministic_sigma_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_step_sigma_, t_horizon_,
            variant_.learned_init_accel_sigma,
            variant_.learned_init_steer_sigma);
    }

    bool uses_feedback_local_action() const {
        return variant_.use_feedback && (variant_.feedback_mode == 5 || variant_.feedback_mode == 6 || variant_.feedback_mode == 7 || variant_.feedback_mode == 8 || variant_.feedback_mode == 9 || variant_.feedback_mode == 10);
    }

    bool should_replan(int step) const {
        if (variant_.feedback_mode != 6 && variant_.feedback_mode != 8) return true;
        int stride = max(1, variant_.replan_stride);
        return (step % stride) == 0;
    }

    void sync_nominal_from_device() {
        CUDA_CHECK(cudaMemcpy(h_nominal_.data(), d_nominal_, h_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void sync_feedback_policy_from_device() {
        CUDA_CHECK(cudaMemcpy(h_states_.data(), d_states_, h_states_.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_feedback_gains_host_.data(), d_feedback_gains_, h_feedback_gains_host_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float shield_candidate_score(
        float first_accel,
        float first_steer,
        float base_accel,
        float base_steer,
        int start_step) const
    {
        const Scenario& sc = planning_scenario_;
        float x = rx_;
        float y = ry_;
        float theta = rtheta_;
        float v = rv_;
        float safe = fmaxf(0.05f, variant_.shield_safe_margin);
        float alpha = clampf(variant_.shield_cbf_alpha, 0.02f, 0.98f);
        float prev_h = min_obstacle_margin(x, y, sc, start_step) - safe;
        float score = variant_.shield_repair_control_weight
            * ((first_accel - base_accel) * (first_accel - base_accel)
               + 12.0f * (first_steer - base_steer) * (first_steer - base_steer));
        int steps = max(1, min(variant_.shield_repair_steps, t_horizon_));

        for (int i = 0; i < steps; i++) {
            float accel = (i == 0) ? first_accel : h_nominal_[i * 2 + 0];
            float steer = (i == 0) ? first_steer : h_nominal_[i * 2 + 1];
            accel = clampf(accel, -4.0f, 4.0f);
            steer = clampf(steer, -sc.params.max_steer, sc.params.max_steer);
            bicycle_step(x, y, theta, v, accel, steer, sc.params);

            int eval_step = start_step + i + 1;
            float h = min_obstacle_margin(x, y, sc, eval_step) - safe;
            float violation = fmaxf(0.0f, (1.0f - alpha) * prev_h - h);
            score += host_step_cost(x, y, theta, v, accel, steer, sc, eval_step);
            score += variant_.shield_repair_safety_weight * violation * violation;
            if (h < 0.0f) score += 2.0f * variant_.shield_repair_safety_weight * h * h;
            prev_h = h;
        }

        float dx = x - sc.cost_params.goal_x;
        float dy = y - sc.cost_params.goal_y;
        score += 0.25f * sc.cost_params.terminal_weight * sqrtf(dx * dx + dy * dy + 0.01f);
        if (x < 0.0f || x > WORKSPACE || y < 0.0f || y > WORKSPACE) score += 1.0e5f;
        return score;
    }

    void apply_shield_repair(float& accel, float& steer, int start_step) {
        float base_accel = clampf(accel, -4.0f, 4.0f);
        float base_steer = clampf(steer, -planning_scenario_.params.max_steer,
                                  planning_scenario_.params.max_steer);
        float best_accel = base_accel;
        float best_steer = base_steer;
        float best_score = shield_candidate_score(
            best_accel, best_steer, base_accel, base_steer, start_step);

        int grid = max(3, variant_.shield_repair_grid);
        if ((grid % 2) == 0) grid++;
        float accel_delta = fmaxf(0.0f, variant_.shield_repair_accel_delta);
        float steer_delta = fmaxf(0.0f, variant_.shield_repair_steer_delta);
        for (int ia = 0; ia < grid; ia++) {
            float fa = (grid == 1) ? 0.0f : (2.0f * ia / static_cast<float>(grid - 1) - 1.0f);
            float ca = clampf(base_accel + fa * accel_delta, -4.0f, 4.0f);
            for (int is = 0; is < grid; is++) {
                float fs = (grid == 1) ? 0.0f : (2.0f * is / static_cast<float>(grid - 1) - 1.0f);
                float cs = clampf(base_steer + fs * steer_delta,
                                  -planning_scenario_.params.max_steer,
                                  planning_scenario_.params.max_steer);
                float score = shield_candidate_score(ca, cs, base_accel, base_steer, start_step);
                if (score < best_score) {
                    best_score = score;
                    best_accel = ca;
                    best_steer = cs;
                }
            }
        }

        // Explicit brake candidates matter when the current nominal is already
        // near the edge of the repair grid.
        for (int is = 0; is < grid; is++) {
            float fs = (grid == 1) ? 0.0f : (2.0f * is / static_cast<float>(grid - 1) - 1.0f);
            float cs = clampf(base_steer + fs * steer_delta,
                              -planning_scenario_.params.max_steer,
                              planning_scenario_.params.max_steer);
            float score = shield_candidate_score(-4.0f, cs, base_accel, base_steer, start_step);
            if (score < best_score) {
                best_score = score;
                best_accel = -4.0f;
                best_steer = cs;
            }
        }

        accel = best_accel;
        steer = best_steer;
        h_nominal_[0] = best_accel;
        h_nominal_[1] = best_steer;
    }

    void seed_cdf_nominal(float sx, float sy, float stheta, float sv, int start_step) {
        if (!variant_.use_cdf_guidance || variant_.cdf_seed_blend <= 0.0f) return;
        const BicycleParams& bp = planning_scenario_.params;
        const CostParams& cp = planning_scenario_.cost_params;
        float x = sx, y = sy, theta = stheta, v = sv;
        float blend0 = clampf(variant_.cdf_seed_blend, 0.0f, 1.0f);
        float safe = fmaxf(0.25f, variant_.cdf_safe_margin);
        for (int t = 0; t < t_horizon_; t++) {
            float vx = 0.0f;
            float vy = 0.0f;
            float dxg = cp.goal_x - x;
            float dyg = cp.goal_y - y;
            float dg = sqrtf(dxg * dxg + dyg * dyg + 1.0e-6f);
            vx += variant_.cdf_goal_pull * dxg / dg;
            vy += variant_.cdf_goal_pull * dyg / dg;

            float nearest_margin = safe;
            for (int i = 0; i < planning_scenario_.n_obs; i++) {
                float dx = x - planning_scenario_.obstacles[i].x;
                float dy = y - planning_scenario_.obstacles[i].y;
                float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
                float margin = dist - planning_scenario_.obstacles[i].r;
                nearest_margin = std::min(nearest_margin, margin);
                if (margin < safe) {
                    float strength = variant_.cdf_obs_pull * (safe - margin) / (safe * fmaxf(dist, 1.0e-3f));
                    vx += strength * dx / dist;
                    vy += strength * dy / dist;
                }
            }

            float tau = (start_step + t + 1) * bp.dt;
            for (int i = 0; i < planning_scenario_.n_dyn_obs; i++) {
                float ox = planning_scenario_.dynamic_obstacles[i].x + planning_scenario_.dynamic_obstacles[i].vx * tau;
                float oy = planning_scenario_.dynamic_obstacles[i].y + planning_scenario_.dynamic_obstacles[i].vy * tau;
                float dx = x - ox;
                float dy = y - oy;
                float dist = sqrtf(dx * dx + dy * dy + 1.0e-6f);
                float margin = dist - planning_scenario_.dynamic_obstacles[i].r;
                nearest_margin = std::min(nearest_margin, margin);
                if (margin < safe) {
                    float strength = variant_.cdf_dyn_pull * (safe - margin) / (safe * fmaxf(dist, 1.0e-3f));
                    vx += strength * dx / dist;
                    vy += strength * dy / dist;
                }
            }

            float desired_heading = atan2f(vy, vx);
            float heading_err = wrap_angle(desired_heading - theta);
            float steer = clampf(0.85f * heading_err, -bp.max_steer, bp.max_steer);
            float slow = nearest_margin < safe
                ? clampf(0.65f + 0.35f * nearest_margin / safe, 0.55f, 1.0f)
                : 1.0f;
            float target_speed = cp.target_speed * slow;
            float accel = clampf((target_speed - v) * 1.4f, -4.0f, 4.0f);
            float blend = blend0 * (1.0f - 0.015f * static_cast<float>(t));
            blend = clampf(blend, 0.05f, blend0);
            int base = t * 2;
            h_nominal_[base + 0] = (1.0f - blend) * h_nominal_[base + 0] + blend * accel;
            h_nominal_[base + 1] = (1.0f - blend) * h_nominal_[base + 1] + blend * steer;
            bicycle_step(x, y, theta, v, h_nominal_[base + 0], h_nominal_[base + 1], bp);
        }
    }

    void compute_feedback_inner_action(float& accel, float& steer) {
        int t_next = min(1, t_horizon_);
        float x_nom0 = h_states_[0];
        float y_nom0 = h_states_[1];
        float theta_nom0 = h_states_[2];
        float v_nom0 = h_states_[3];
        float x_nom1 = h_states_[t_next * 4 + 0];
        float y_nom1 = h_states_[t_next * 4 + 1];
        float theta_nom1 = h_states_[t_next * 4 + 2];
        float v_nom1 = h_states_[t_next * 4 + 3];
        float blend = variant_.feedback_setpoint_blend;
        float x_nom = (1.0f - blend) * x_nom0 + blend * x_nom1;
        float y_nom = (1.0f - blend) * y_nom0 + blend * y_nom1;
        float theta_nom = wrap_angle((1.0f - blend) * theta_nom0 + blend * theta_nom1);
        float v_nom = (1.0f - blend) * v_nom0 + blend * v_nom1;

        float dx = x_nom - rx_;
        float dy = y_nom - ry_;
        float ex = rx_ - x_nom;
        float ey = ry_ - y_nom;
        float etheta = wrap_angle(rtheta_ - theta_nom);
        float ev = rv_ - v_nom;
        float ct = cosf(theta_nom);
        float st = sinf(theta_nom);
        float longitudinal_err = ct * dx + st * dy;
        float lateral_err = -st * dx + ct * dy;
        float heading_err = wrap_angle(theta_nom - rtheta_);
        float speed_err = v_nom - rv_;

        const float* K_t = h_feedback_gains_host_.data();
        float accel_feedback = K_t[0] * ex + K_t[1] * ey + K_t[2] * etheta + K_t[3] * ev;
        float steer_feedback = K_t[4] * ex + K_t[5] * ey + K_t[6] * etheta + K_t[7] * ev;

        accel = h_nominal_[0]
              - variant_.feedback_gain_scale * accel_feedback
              + variant_.feedback_longitudinal_gain * longitudinal_err
              + variant_.feedback_speed_gain * speed_err;
        steer = h_nominal_[1]
              - variant_.feedback_gain_scale * steer_feedback
              + variant_.feedback_lateral_gain * lateral_err
              + variant_.feedback_heading_gain * heading_err;
        accel = clampf(accel, -4.0f, 4.0f);
        steer = clampf(steer, -eval_scenario_.params.max_steer, eval_scenario_.params.max_steer);

        if (trace_rows_ != nullptr) {
            h_sample_nominal_ = h_nominal_;
            h_final_nominal_ = h_nominal_;
            h_final_nominal_[0] = accel;
            h_final_nominal_[1] = steer;
            fill(h_grad_snapshot_.begin(), h_grad_snapshot_.end(), 0.0f);
        }
    }

    void shift_host_policy() {
        for (int t = 0; t < t_horizon_ - 1; t++) {
            h_nominal_[t * 2 + 0] = h_nominal_[(t + 1) * 2 + 0];
            h_nominal_[t * 2 + 1] = h_nominal_[(t + 1) * 2 + 1];
        }
        h_nominal_[(t_horizon_ - 1) * 2 + 0] = 0.0f;
        h_nominal_[(t_horizon_ - 1) * 2 + 1] = 0.0f;

        // Step-MPPI: shift the sampling bias to match the shifted horizon
        if (variant_.use_learned_sampling) {
            shift_sampling_bias_kernel<<<1, 1>>>(d_sampling_bias_, t_horizon_);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (d_step_sigma_) {
            shift_deterministic_sigma_kernel<<<1, 1>>>(
                d_step_sigma_, t_horizon_,
                variant_.learned_init_accel_sigma,
                variant_.learned_init_steer_sigma);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (d_ds_sigma_) {
            shift_deterministic_sigma_kernel<<<1, 1>>>(
                d_ds_sigma_, t_horizon_,
                1.5f * variant_.ds_noise_scale,
                0.18f * variant_.ds_noise_scale);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (!uses_feedback_local_action()) return;

        for (int t = 0; t < t_horizon_; t++) {
            for (int i = 0; i < 4; i++) {
                h_states_[t * 4 + i] = h_states_[(t + 1) * 4 + i];
            }
        }
        for (int i = 0; i < 4; i++) {
            h_states_[t_horizon_ * 4 + i] = h_states_[(t_horizon_ - 1) * 4 + i];
        }

        if (variant_.feedback_mode != 8) {
            for (int t = 0; t < t_horizon_ - 1; t++) {
                for (int i = 0; i < 8; i++) {
                    h_feedback_gains_host_[t * 8 + i] = h_feedback_gains_host_[(t + 1) * 8 + i];
                }
            }
            for (int i = 0; i < 8; i++) {
                h_feedback_gains_host_[(t_horizon_ - 1) * 8 + i] = 0.0f;
            }
        }
    }

    void controller_update(float sx, float sy, float stheta, float sv, int start_step) {
        if (variant_.planner_kind == 1) {
            dwa_controller_update(sx, sy, stheta, sv, start_step);
            return;
        }
        if (variant_.planner_kind == 2) {
            stomp_controller_update(sx, sy, stheta, sv, start_step);
            return;
        }
        if (variant_.planner_kind == 3) {
            hybrid_astar_pp_controller_update(sx, sy, stheta, sv, start_step);
            return;
        }
        if (variant_.planner_kind == 4) {
            hybrid_astar_dwa_controller_update(sx, sy, stheta, sv, start_step);
            return;
        }
        if (variant_.planner_kind == 5) {
            hybrid_astar_dyn_pp_controller_update(sx, sy, stheta, sv, start_step);
            return;
        }
        if (variant_.planner_kind == 6) {
            hybrid_astar_mppi_controller_update(sx, sy, stheta, sv, start_step);
            return;
        }
        seed_cdf_nominal(sx, sy, stheta, sv, start_step);
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(), h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        int block = 256;
        if (variant_.use_sampling) {
            // Step-MPPI: save pre-bias nominal and apply learned bias before sampling
            if (variant_.use_learned_sampling) {
                CUDA_CHECK(cudaMemcpy(d_nominal_pre_bias_, d_nominal_, t_horizon_ * 2 * sizeof(float), cudaMemcpyDeviceToDevice));
                apply_sampling_bias_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_nominal_, d_sampling_bias_, t_horizon_);
            }

            int open_loop_passes = 1;
            if (variant_.use_feedback && (variant_.feedback_mode == 1 || variant_.feedback_mode == 3 || variant_.feedback_mode == 4 || variant_.feedback_mode == 6 || variant_.feedback_mode == 9)) {
                open_loop_passes = 2;
            }
            if (variant_.use_deterministic_sampling) {
                open_loop_passes = max(open_loop_passes, max(1, variant_.ds_iterations));
            }
            for (int pass = 0; pass < open_loop_passes; pass++) {
                if (variant_.use_deterministic_sampling && d_nominal_pre_bias_ != nullptr) {
                    CUDA_CHECK(cudaMemcpy(d_nominal_pre_bias_, d_nominal_,
                                          t_horizon_ * 2 * sizeof(float),
                                          cudaMemcpyDeviceToDevice));
                }

                if (variant_.use_safety_controlled_sampling) {
                    rollout_safety_controlled_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.use_low_pass_sampling, variant_.lp_alpha,
                        variant_.sc_safe_margin,
                        variant_.sc_avoid_gain,
                        variant_.sc_speed_gain,
                        variant_.sc_max_steer_delta,
                        variant_.sc_max_accel_delta,
                        variant_.sc_control_weight);
                } else if (variant_.use_shield_cost) {
                    rollout_shield_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.use_low_pass_sampling, variant_.lp_alpha,
                        variant_.shield_safe_margin,
                        variant_.shield_cbf_alpha,
                        variant_.shield_cbf_weight);
                } else if (variant_.use_parameter_robust_sampling) {
                    rollout_parameter_robust_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.use_low_pass_sampling, variant_.lp_alpha,
                        variant_.pr_param_particles,
                        variant_.pr_wheelbase_span,
                        variant_.pr_max_speed_span,
                        variant_.pr_max_steer_span,
                        variant_.pr_worst_blend);
                } else if (variant_.use_learned_sigma) {
                    rollout_learned_sampling_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_step_sigma_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.use_low_pass_sampling, variant_.lp_alpha);
                } else if (variant_.use_cdf_guidance) {
                    rollout_cdf_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.use_low_pass_sampling, variant_.lp_alpha,
                        variant_.cdf_safe_margin,
                        variant_.cdf_obs_cost,
                        variant_.cdf_dyn_cost);
                } else if (variant_.use_deterministic_sampling) {
                    rollout_deterministic_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_ds_sigma_, d_costs_, d_perturbed_, d_rollout_states_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_, seed_, pass,
                        variant_.ds_alpha, variant_.ds_noise_scale, variant_.ds_stride);
                } else if (variant_.use_projection_sampling) {
                    rollout_projection_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.projection_passes,
                        variant_.projection_max_accel_delta,
                        variant_.projection_max_steer_delta,
                        variant_.projection_max_accel_ddelta,
                        variant_.projection_max_steer_ddelta);
                } else if (variant_.use_dbas_log_sampling) {
                    rollout_dbas_log_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.use_low_pass_sampling, variant_.lp_alpha,
                        variant_.dbas_safe_margin,
                        variant_.dbas_barrier_eps,
                        variant_.dbas_barrier_cap,
                        variant_.dbas_barrier_weight,
                        variant_.dbas_gamma,
                        variant_.dbas_mu,
                        variant_.dbas_log_sigma,
                        variant_.dbas_lognormal_clip,
                        variant_.dbas_noise_scale,
                        variant_.dbas_speed_damping);
                } else if (variant_.use_low_pass_sampling) {
                    rollout_low_pass_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_, variant_.lp_alpha);
                } else {
                    rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                }
                if (variant_.use_soppi_sampling) {
                    int total_particles = k_samples_ * t_horizon_;
                    float* d_controls_src = d_perturbed_;
                    float* d_controls_dst = d_soppi_scratch_;
                    for (int iter = 0; iter < max(1, variant_.soppi_svgd_iters); iter++) {
                        soppi_stage_score_kernel<<<(total_particles + block - 1) / block, block>>>(
                            d_controls_src, d_rollout_states_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_, variant_.sampling_lambda,
                            d_soppi_scores_);
                        soppi_svgd_step_kernel<<<(total_particles + block - 1) / block, block>>>(
                            d_controls_src, d_controls_dst, d_soppi_scores_,
                            planning_scenario_.params, k_samples_, t_horizon_,
                            variant_.soppi_neighbor_count,
                            variant_.soppi_bandwidth, variant_.soppi_step_size);
                        rollout_fixed_controls_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_controls_dst, d_costs_, d_rollout_states_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        float* tmp = d_controls_src;
                        d_controls_src = d_controls_dst;
                        d_controls_dst = tmp;
                    }
                    if (d_controls_src != d_perturbed_) {
                        CUDA_CHECK(cudaMemcpy(d_perturbed_, d_controls_src,
                                              k_samples_ * t_horizon_ * 2 * sizeof(float),
                                              cudaMemcpyDeviceToDevice));
                    }
                }
                if (variant_.use_deterministic_sampling && variant_.ds_elite_update) {
                    update_deterministic_elite_kernel<<<1, 1>>>(
                        d_nominal_, d_ds_sigma_, d_perturbed_, d_costs_,
                        k_samples_, t_horizon_, variant_.ds_elite_count,
                        variant_.ds_elite_sigma_blend,
                        planning_scenario_.params.max_steer,
                        variant_.ds_min_accel_sigma,
                        variant_.ds_min_steer_sigma,
                        variant_.ds_max_accel_sigma,
                        variant_.ds_max_steer_sigma);
                } else {
                    if (variant_.use_cluster_representative_update) {
                        update_controls_from_cluster_representative_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_costs_, d_rollout_states_,
                            planning_scenario_.params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_,
                            variant_.csc_cluster_count,
                            variant_.csc_safe_margin,
                            variant_.csc_constraint_weight,
                            variant_.csc_update_blend);
                    } else {
                        if (variant_.use_datamodel_influence_pruning) {
                            compute_dm_influence_weights_kernel<<<1, 1>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                planning_scenario_.params,
                                planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                                start_step, k_samples_, t_horizon_,
                                variant_.dm_keep_fraction,
                                variant_.dm_cost_temperature,
                                variant_.dm_safe_margin,
                                variant_.dm_prob_sigma,
                                variant_.dm_violation_weight,
                                variant_.dm_safety_power);
                        } else if (variant_.use_tsallis_weights) {
                            compute_tsallis_weights_kernel<<<1, 1>>>(
                                d_costs_, d_weights_, k_samples_,
                                variant_.tsallis_q,
                                variant_.tsallis_temperature,
                                variant_.tsallis_min_weight);
                        } else if (variant_.use_covariance_control_weights) {
                            compute_covariance_control_weights_kernel<<<1, 1>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                k_samples_, t_horizon_,
                                variant_.sampling_lambda,
                                variant_.cc_terminal_weight,
                                variant_.cc_terminal_target_radius,
                                variant_.cc_heading_weight,
                                variant_.cc_speed_weight,
                                variant_.cc_min_weight);
                        } else if (variant_.use_td_cd_weights) {
                            compute_td_cd_scores_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                                d_rollout_states_, d_perturbed_, d_weights_,
                                planning_scenario_.params,
                                planning_scenario_.cost_params,
                                planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                                start_step, k_samples_, t_horizon_,
                                variant_.td_terminal_value_scale,
                                variant_.td_safe_margin,
                                variant_.td_discount_sigma,
                                variant_.td_discount_power,
                                variant_.td_failure_cost);
                            compute_weights_kernel<<<1, 1>>>(
                                d_weights_, d_weights_, k_samples_, variant_.sampling_lambda);
                        } else if (variant_.use_pa_perception_cost) {
                            compute_pa_perception_scores_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                planning_scenario_.params,
                                planning_scenario_.cost_params,
                                planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                                start_step, k_samples_, t_horizon_,
                                variant_.pa_safe_margin,
                                variant_.pa_poi_weight,
                                variant_.pa_occlusion_weight,
                                variant_.pa_frontier_reward,
                                variant_.pa_forward_occ_weight,
                                variant_.pa_goal_gate,
                                variant_.pa_activation,
                                variant_.pa_ray_length,
                                variant_.pa_score_cap);
                            compute_weights_kernel<<<1, 1>>>(
                                d_weights_, d_weights_, k_samples_, variant_.sampling_lambda);
                        } else if (variant_.use_c2u_chance_constraints) {
                            compute_c2u_chance_scores_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                planning_scenario_.params,
                                planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                                start_step, k_samples_, t_horizon_,
                                variant_.sampling_lambda,
                                variant_.c2u_safe_margin,
                                variant_.c2u_robot_sigma,
                                variant_.c2u_dyn_sigma0,
                                variant_.c2u_dyn_sigma_growth,
                                variant_.c2u_risk_z,
                                variant_.c2u_prob_sigma,
                                variant_.c2u_probability_power,
                                variant_.c2u_violation_weight,
                                variant_.c2u_min_probability);
                            compute_weights_kernel<<<1, 1>>>(
                                d_weights_, d_weights_, k_samples_, variant_.sampling_lambda);
                        } else if (variant_.use_ducct_risk) {
                            compute_ducct_risk_scores_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                planning_scenario_.params,
                                planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                                start_step, k_samples_, t_horizon_,
                                variant_.sampling_lambda,
                                variant_.ducct_loc_sigma0,
                                variant_.ducct_loc_sigma_growth,
                                variant_.ducct_pred_sigma0,
                                variant_.ducct_pred_sigma_growth,
                                variant_.ducct_static_sigma,
                                variant_.ducct_risk_weight,
                                variant_.ducct_hard_threshold,
                                variant_.ducct_reject_cost,
                                variant_.ducct_survival_power,
                                variant_.ducct_min_survival);
                            compute_weights_kernel<<<1, 1>>>(
                                d_weights_, d_weights_, k_samples_, variant_.sampling_lambda);
                        } else if (variant_.use_dra_risk) {
                            compute_dra_risk_scores_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                planning_scenario_.params,
                                planning_scenario_.n_dyn_obs,
                                start_step, k_samples_, t_horizon_,
                                variant_.sampling_lambda,
                                variant_.dra_mc_samples,
                                variant_.dra_robot_radius,
                                variant_.dra_pred_sigma0,
                                variant_.dra_pred_sigma_growth,
                                variant_.dra_mode_weight,
                                variant_.dra_mode_lateral_offset,
                                variant_.dra_soft_weight,
                                variant_.dra_hard_threshold,
                                variant_.dra_reject_cost,
                                variant_.dra_survival_power,
                                variant_.dra_min_survival);
                            compute_weights_kernel<<<1, 1>>>(
                                d_weights_, d_weights_, k_samples_, variant_.sampling_lambda);
                        } else if (variant_.use_bc_safety_layer) {
                            compute_bc_safety_weights_kernel<<<1, 1>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                planning_scenario_.params,
                                planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                                start_step, k_samples_, t_horizon_,
                                variant_.sampling_lambda,
                                variant_.bc_safe_margin,
                                variant_.bc_prob_sigma,
                                variant_.bc_probability_power,
                                variant_.bc_min_probability);
                        } else if (variant_.use_svg_mode_guidance) {
                            compute_svg_mode_weights_kernel<<<1, 1>>>(
                                d_costs_, d_rollout_states_, d_weights_,
                                k_samples_, t_horizon_, variant_.sampling_lambda,
                                variant_.svg_bandwidth, variant_.svg_mode_weight,
                                variant_.svg_stride);
                        } else {
                            compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        }
                        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
                        if (variant_.use_deterministic_sampling && variant_.ds_momentum > 0.0f && d_nominal_pre_bias_ != nullptr) {
                            blend_controls_with_previous_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                                d_nominal_, d_nominal_pre_bias_, t_horizon_, variant_.ds_momentum);
                        }
                        if (variant_.use_deterministic_sampling && d_ds_sigma_) {
                            update_deterministic_sigma_kernel<<<(t_horizon_ * 2 + block - 1) / block, block>>>(
                                d_ds_sigma_, d_perturbed_, d_weights_, d_nominal_,
                                k_samples_, t_horizon_,
                                variant_.ds_sigma_blend,
                                variant_.ds_min_accel_sigma,
                                variant_.ds_min_steer_sigma,
                                variant_.ds_max_accel_sigma,
                                variant_.ds_max_steer_sigma);
                        }
                        if (variant_.use_learned_sigma && d_step_sigma_) {
                            update_deterministic_sigma_kernel<<<(t_horizon_ * 2 + block - 1) / block, block>>>(
                                d_step_sigma_, d_perturbed_, d_weights_, d_nominal_,
                                k_samples_, t_horizon_,
                                clampf(1.0f - variant_.learned_sigma_lr, 0.0f, 1.0f),
                                variant_.learned_min_accel_sigma,
                                variant_.learned_min_steer_sigma,
                                variant_.learned_max_accel_sigma,
                                variant_.learned_max_steer_sigma);
                        }
                    }
                }
                if (variant_.use_projection_sampling) {
                    project_nominal_controls_kernel<<<1, 1>>>(
                        d_nominal_, t_horizon_, planning_scenario_.params.max_steer,
                        variant_.projection_passes,
                        variant_.projection_max_accel_delta,
                        variant_.projection_max_steer_delta,
                        variant_.projection_max_accel_ddelta,
                        variant_.projection_max_steer_ddelta);
                }
            }

            // Step-MPPI: update the learned bias from cost-weighted control deviations
            if (variant_.use_learned_sampling) {
                update_sampling_bias_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_sampling_bias_, d_nominal_, d_nominal_pre_bias_,
                    t_horizon_, variant_.mlp_lr, 0.995f);
            }

            if (variant_.use_feedback) {
                if (uses_feedback_local_action()) {
                    rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                    if (variant_.feedback_mode == 5) {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        compute_sensitivity_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                            variant_.sampling_lambda, k_samples_, t_horizon_);
                    } else if (variant_.feedback_mode == 6 || variant_.feedback_mode == 9) {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_covariance_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_states_, d_perturbed_, d_rollout_states_, d_weights_, d_feedback_gains_,
                            k_samples_, t_horizon_, variant_.feedback_cov_regularization);
                        compute_feedback_gains_kernel<<<1, 1>>>(
                            d_states_, d_nominal_, d_feedback_gains_aux_, planning_scenario_.params, planning_scenario_.cost_params, t_horizon_,
                            variant_.feedback_q_position, variant_.feedback_q_heading, variant_.feedback_q_speed,
                            variant_.feedback_r_accel, variant_.feedback_r_steer, variant_.feedback_terminal_scale);
                        blend_feedback_gains_kernel<<<(t_horizon_ * 8 + block - 1) / block, block>>>(
                            d_feedback_gains_, d_feedback_gains_aux_, t_horizon_,
                            variant_.feedback_cov_blend, variant_.feedback_lqr_blend);
                    } else if (variant_.feedback_mode == 10) {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        compute_reference_feedback_gain_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                            variant_.sampling_lambda, k_samples_, t_horizon_);
                        compute_covariance_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_states_, d_perturbed_, d_rollout_states_, d_weights_, d_feedback_gains_aux_,
                            k_samples_, t_horizon_, variant_.feedback_cov_regularization);
                        blend_feedback_gains_kernel<<<(t_horizon_ * 8 + block - 1) / block, block>>>(
                            d_feedback_gains_, d_feedback_gains_aux_, t_horizon_,
                            variant_.feedback_ref_blend, variant_.feedback_cov_blend);
                        compute_feedback_gains_kernel<<<1, 1>>>(
                            d_states_, d_nominal_, d_feedback_gains_aux_, planning_scenario_.params, planning_scenario_.cost_params, t_horizon_,
                            variant_.feedback_q_position, variant_.feedback_q_heading, variant_.feedback_q_speed,
                            variant_.feedback_r_accel, variant_.feedback_r_steer, variant_.feedback_terminal_scale);
                        blend_feedback_gains_kernel<<<(t_horizon_ * 8 + block - 1) / block, block>>>(
                            d_feedback_gains_, d_feedback_gains_aux_, t_horizon_,
                            1.0f, variant_.feedback_lqr_blend);
                    } else {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
                        compute_reference_feedback_gain_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                            variant_.sampling_lambda, k_samples_, t_horizon_);
                        rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                    }
                } else {
                for (int fb_pass = 0; fb_pass < max(1, variant_.feedback_passes); fb_pass++) {
                    rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                    if (variant_.feedback_mode == 2) {
                        compute_rollout_initial_gradients_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            d_rollout_states_, d_perturbed_, d_rollout_init_grads_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                            start_step, k_samples_, t_horizon_);
                        compute_sensitivity_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_perturbed_, d_weights_, d_rollout_init_grads_, d_feedback_gains_,
                            variant_.sampling_lambda, k_samples_, t_horizon_);
                    } else if (variant_.feedback_mode == 3 || variant_.feedback_mode == 4) {
                        rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                            sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                            planning_scenario_.params, planning_scenario_.cost_params,
                            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, k_samples_, t_horizon_);
                        compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                        compute_covariance_feedback_gains_kernel<<<1, 1>>>(
                            d_nominal_, d_states_, d_perturbed_, d_rollout_states_, d_weights_, d_feedback_gains_,
                            k_samples_, t_horizon_, variant_.feedback_cov_regularization);
                        if (variant_.feedback_mode == 4) {
                            compute_feedback_gains_kernel<<<1, 1>>>(
                                d_states_, d_nominal_, d_feedback_gains_aux_, planning_scenario_.params, planning_scenario_.cost_params, t_horizon_,
                                variant_.feedback_q_position, variant_.feedback_q_heading, variant_.feedback_q_speed,
                                variant_.feedback_r_accel, variant_.feedback_r_steer, variant_.feedback_terminal_scale);
                            blend_feedback_gains_kernel<<<(t_horizon_ * 8 + block - 1) / block, block>>>(
                                d_feedback_gains_, d_feedback_gains_aux_, t_horizon_,
                                variant_.feedback_cov_blend, variant_.feedback_lqr_blend);
                        }
                    } else {
                        compute_feedback_gains_kernel<<<1, 1>>>(
                            d_states_, d_nominal_, d_feedback_gains_, planning_scenario_.params, planning_scenario_.cost_params, t_horizon_,
                            variant_.feedback_q_position, variant_.feedback_q_heading, variant_.feedback_q_speed,
                            variant_.feedback_r_accel, variant_.feedback_r_steer, variant_.feedback_terminal_scale);
                    }
                    rollout_feedback_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                        sx, sy, stheta, sv, d_nominal_, d_states_, d_feedback_gains_, d_costs_, d_perturbed_, d_rng_,
                        planning_scenario_.params, planning_scenario_.cost_params,
                        planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                        start_step, k_samples_, t_horizon_,
                        variant_.feedback_gain_scale,
                        variant_.feedback_noise_accel,
                        variant_.feedback_noise_steer,
                        variant_.feedback_longitudinal_gain,
                        variant_.feedback_speed_gain,
                        variant_.feedback_lateral_gain,
                        variant_.feedback_heading_gain,
                        variant_.feedback_setpoint_blend);
                    compute_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
                    update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                        d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
                }
                }
            }
        }

        if (trace_rows_ != nullptr) {
            CUDA_CHECK(cudaMemcpy(h_sample_nominal_.data(), d_nominal_, h_sample_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            fill(h_grad_snapshot_.begin(), h_grad_snapshot_.end(), 0.0f);
        }

        if (variant_.use_gradient) {
            for (int gs = 0; gs < variant_.grad_steps; gs++) {
                rollout_nominal_kernel<<<1, 1>>>(sx, sy, stheta, sv, d_nominal_, d_states_, planning_scenario_.params, t_horizon_);
                precompute_nav_gradients_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_states_, d_nominal_, d_nav_stage_grads_, d_nav_jacobians_,
                    planning_scenario_.params, planning_scenario_.cost_params,
                    planning_scenario_.n_obs, planning_scenario_.n_dyn_obs, start_step, t_horizon_);
                backward_nav_adjoint_kernel<<<1, 1>>>(
                    d_states_, d_nav_stage_grads_, d_nav_jacobians_, d_grad_,
                    planning_scenario_.cost_params, t_horizon_);
                int grad_update_horizon = variant_.grad_update_horizon > 0
                    ? min(variant_.grad_update_horizon, t_horizon_)
                    : t_horizon_;
                // Adaptive skip: check gradient norm and skip if below threshold
                if (variant_.grad_skip_threshold > 0.0f) {
                    gradient_norm_kernel<<<1, 1>>>(d_grad_, d_grad_norm_, grad_update_horizon);
                    CUDA_CHECK(cudaMemcpy(&h_grad_norm_, d_grad_norm_, sizeof(float), cudaMemcpyDeviceToHost));
                    if (h_grad_norm_ < variant_.grad_skip_threshold) {
                        grad_steps_skipped_++;
                        continue;
                    }
                }
                gradient_step_kernel<<<(grad_update_horizon + block - 1) / block, block>>>(
                    d_nominal_, d_grad_, grad_update_horizon,
                    variant_.alpha * planning_scenario_.grad_alpha_scale, planning_scenario_.params.max_steer);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (trace_rows_ != nullptr) {
            CUDA_CHECK(cudaMemcpy(h_final_nominal_.data(), d_nominal_, h_final_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (variant_.use_gradient) {
                CUDA_CHECK(cudaMemcpy(h_grad_snapshot_.data(), d_grad_, h_grad_snapshot_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
    }

    void append_trajectory_row(int episode_step, float goal_distance) {
        TrajectoryRow row;
        row.scenario = eval_scenario_.name;
        row.planner = variant_.name;
        row.seed = seed_;
        row.k_samples = k_samples_;
        row.episode_step = episode_step;
        row.x = rx_;
        row.y = ry_;
        row.theta = rtheta_;
        row.v = rv_;
        row.goal_distance = goal_distance;
        trajectory_rows_->push_back(row);
    }

    void append_trace_rows(int episode_step, float goal_distance, float min_margin, float control_ms) {
        for (int t = 0; t < t_horizon_; t++) {
            TraceRow row;
            row.scenario = eval_scenario_.name;
            row.planner = variant_.name;
            row.seed = seed_;
            row.k_samples = k_samples_;
            row.grad_steps = variant_.grad_steps;
            row.alpha = variant_.alpha;
            row.episode_step = episode_step;
            row.horizon_step = t;
            row.goal_distance = goal_distance;
            row.min_obstacle_margin = min_margin;
            row.control_ms = control_ms;
            row.sampled_accel = h_sample_nominal_[t * 2 + 0];
            row.sampled_steer = h_sample_nominal_[t * 2 + 1];
            row.final_accel = h_final_nominal_[t * 2 + 0];
            row.final_steer = h_final_nominal_[t * 2 + 1];
            row.delta_accel = row.final_accel - row.sampled_accel;
            row.delta_steer = row.final_steer - row.sampled_steer;
            row.delta_norm = sqrtf(row.delta_accel * row.delta_accel + row.delta_steer * row.delta_steer);
            row.grad_accel = h_grad_snapshot_[t * 2 + 0];
            row.grad_steer = h_grad_snapshot_[t * 2 + 1];
            row.grad_norm = sqrtf(row.grad_accel * row.grad_accel + row.grad_steer * row.grad_steer);
            trace_rows_->push_back(row);
        }
    }

    void warmup_controller() {
        for (int iter = 0; iter < BENCH_WARMUP_ITERS; iter++) {
            controller_update(planning_scenario_.start_x, planning_scenario_.start_y,
                              planning_scenario_.start_theta, planning_scenario_.start_v, 0);
        }
    }

    void dwa_controller_update(float sx, float sy, float stheta, float sv, int start_step) {
        int block = 64;
        int total = dwa_grid_size_;
        int T_dwa = max(1, variant_.dwa_predict_steps);
        dwa_grid_kernel<<<(total + block - 1) / block, block>>>(
            sx, sy, stheta, sv,
            d_dwa_costs_, d_dwa_accels_, d_dwa_steers_,
            planning_scenario_.params, planning_scenario_.cost_params,
            planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
            start_step, T_dwa,
            variant_.dwa_n_accel, variant_.dwa_n_steer,
            variant_.dwa_accel_min, variant_.dwa_accel_max,
            variant_.dwa_w_goal, variant_.dwa_w_speed,
            variant_.dwa_w_obs, variant_.dwa_w_heading,
            variant_.dwa_w_terminal);
        CUDA_CHECK(cudaMemcpy(h_dwa_costs_.data(), d_dwa_costs_,
                              dwa_grid_size_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dwa_accels_.data(), d_dwa_accels_,
                              dwa_grid_size_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dwa_steers_.data(), d_dwa_steers_,
                              dwa_grid_size_ * sizeof(float), cudaMemcpyDeviceToHost));
        int best = 0;
        float best_cost = h_dwa_costs_[0];
        for (int i = 1; i < dwa_grid_size_; i++) {
            if (h_dwa_costs_[i] < best_cost) { best_cost = h_dwa_costs_[i]; best = i; }
        }
        // DWA writes only the first action; subsequent slots are unused (we replan
        // every step). Mirror the result into d_nominal_ as well so that the
        // sync_nominal_from_device() call in run() does not clobber the choice.
        fill(h_nominal_.begin(), h_nominal_.end(), 0.0f);
        h_nominal_[0] = h_dwa_accels_[best];
        h_nominal_[1] = h_dwa_steers_[best];
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(),
                              h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (trace_rows_ != nullptr) {
            h_sample_nominal_ = h_nominal_;
            h_final_nominal_ = h_nominal_;
            fill(h_grad_snapshot_.begin(), h_grad_snapshot_.end(), 0.0f);
        }
    }

    void stomp_controller_update(float sx, float sy, float stheta, float sv, int start_step) {
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(),
                              h_nominal_.size() * sizeof(float), cudaMemcpyHostToDevice));
        int block = 256;
        int iters = max(1, variant_.stomp_iterations);
        for (int it = 0; it < iters; it++) {
            // Snapshot u_old before this iteration's cost-weighted update,
            // so the smoothness projection knows the previously committed
            // nominal it should anchor δu around.
            copy_kernel<<<(t_horizon_ * 2 + block - 1) / block, block>>>(
                d_stomp_old_, d_nominal_, t_horizon_ * 2);
            rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
                sx, sy, stheta, sv, d_nominal_, d_costs_, d_perturbed_, d_rollout_states_, d_rng_,
                planning_scenario_.params, planning_scenario_.cost_params,
                planning_scenario_.n_obs, planning_scenario_.n_dyn_obs,
                start_step, k_samples_, t_horizon_);
            compute_stomp_weights_kernel<<<1, 1>>>(d_costs_, d_weights_, k_samples_, variant_.stomp_h);
            update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
            // Apply M = (R^T R)^-1 smoothness projection to δu = (d_nominal_
            // - d_stomp_old_) repeatedly. Multiple passes drive δu further
            // into the null-space of the high-frequency penalty matrix R^T R.
            int passes = max(0, variant_.stomp_smoothing_passes);
            for (int sp = 0; sp < passes; sp++) {
                stomp_delta_project_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
                    d_stomp_old_, d_nominal_, d_stomp_scratch_, d_stomp_M_,
                    t_horizon_, planning_scenario_.params.max_steer);
                copy_kernel<<<(t_horizon_ * 2 + block - 1) / block, block>>>(
                    d_nominal_, d_stomp_scratch_, t_horizon_ * 2);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (trace_rows_ != nullptr) {
            CUDA_CHECK(cudaMemcpy(h_sample_nominal_.data(), d_nominal_,
                                  h_sample_nominal_.size() * sizeof(float), cudaMemcpyDeviceToHost));
            h_final_nominal_ = h_sample_nominal_;
            fill(h_grad_snapshot_.begin(), h_grad_snapshot_.end(), 0.0f);
        }
    }

    void ensure_hap_path_planned(float sx, float sy, float stheta,
                                 bool include_dynamic = false) {
        // Plan lazily on the first call of the episode. The plan is held
        // for the rest of the episode -- planner variants that consume it
        // (hybrid_astar_pp, hybrid_astar_dwa, hybrid_astar_dyn_pp) decide
        // how to track / react around dynamic obstacles. By default the
        // search sees only the STATIC obstacles; the dyn variant flips
        // ``include_dynamic`` so the search inflates predicted positions
        // along each candidate's time stamp.
        if (!hap_path_.empty() || hap_planning_failed_) return;
        HybridAStarParams hp;
        hp.workspace = WORKSPACE;
        hp.wheelbase = planning_scenario_.params.L;
        hp.max_steer = planning_scenario_.params.max_steer;
        hp.n_steer = variant_.hap_n_steer;
        hp.sub_steps = variant_.hap_sub_steps;
        hp.dt = variant_.hap_dt;
        // When predicting against moving obstacles, use the simulator's
        // target speed so the search's t-stamps roughly match the time
        // the robot will actually arrive at each pose. The static search
        // continues to use the configured hap_v_search.
        hp.v_search = include_dynamic
            ? std::max(variant_.hap_v_search,
                       eval_scenario_.cost_params.target_speed)
            : variant_.hap_v_search;
        hp.steer_penalty = variant_.hap_steer_penalty;
        // Dyn-aware search adds an extra inflation buffer to the robot
        // radius because the linearised obstacle prediction is brittle
        // against (a) the simulator's accelerating-from-rest model that
        // arrives at each pose later than the constant-speed search
        // assumes, and (b) sub-cell timing rounding. ~2 m of buffer
        // covers ~1 s of error against a 2 m/s obstacle.
        hp.robot_radius = include_dynamic
            ? variant_.hap_robot_radius + variant_.hap_dyn_inflation
            : variant_.hap_robot_radius;
        hp.max_expansions = variant_.hap_max_expansions;
        std::vector<ObstacleCircle> obs;
        obs.reserve(planning_scenario_.n_obs);
        for (int i = 0; i < planning_scenario_.n_obs; i++) {
            ObstacleCircle o;
            o.x = planning_scenario_.obstacles[i].x;
            o.y = planning_scenario_.obstacles[i].y;
            o.r = planning_scenario_.obstacles[i].r;
            obs.push_back(o);
        }
        std::vector<DynamicObstacleSpec> dyn;
        if (include_dynamic) {
            // We pass the *evaluation* scenario's dynamic obstacles so the
            // search predicts the same trajectories the simulator will roll
            // out -- not the nominal planning_scenario_ ones, which can
            // diverge under use_dynamic_mismatch.
            dyn.reserve(eval_scenario_.n_dyn_obs);
            for (int i = 0; i < eval_scenario_.n_dyn_obs; i++) {
                DynamicObstacleSpec d;
                d.x0 = eval_scenario_.dynamic_obstacles[i].x;
                d.y0 = eval_scenario_.dynamic_obstacles[i].y;
                d.vx = eval_scenario_.dynamic_obstacles[i].vx;
                d.vy = eval_scenario_.dynamic_obstacles[i].vy;
                d.r  = eval_scenario_.dynamic_obstacles[i].r;
                dyn.push_back(d);
            }
        }
        Pose2D start_pose;
        start_pose.x = sx;
        start_pose.y = sy;
        start_pose.theta = stheta;
        Pose2D goal_pose;
        goal_pose.x = planning_scenario_.cost_params.goal_x;
        goal_pose.y = planning_scenario_.cost_params.goal_y;
        // Goal heading is left at zero: the cost only cares about
        // position, and the goal_theta tolerance in the search is
        // generous so this rarely matters.
        goal_pose.theta = 0.0f;
        hap_path_ = hybrid_astar_plan(
            start_pose, goal_pose, obs, hp, dyn, /*t_offset=*/0.0f);
        if (hap_path_.empty()) hap_planning_failed_ = true;
    }

    void hybrid_astar_mppi_controller_update(float sx, float sy, float stheta, float sv, int start_step) {
        ensure_hap_path_planned(sx, sy, stheta);
        if (!hap_path_.empty() && d_had_path_ == nullptr) {
            int n = static_cast<int>(hap_path_.size());
            std::vector<float> flat(static_cast<size_t>(n) * 3);
            for (int i = 0; i < n; i++) {
                flat[i * 3 + 0] = hap_path_[i].x;
                flat[i * 3 + 1] = hap_path_[i].y;
                flat[i * 3 + 2] = hap_path_[i].theta;
            }
            CUDA_CHECK(cudaMalloc(&d_had_path_, flat.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_had_path_, flat.data(),
                                  flat.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            had_path_n_ = n;
        }
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(),
                              h_nominal_.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        int block = 256;
        hybrid_astar_mppi_rollout_kernel<<<(k_samples_ + block - 1) / block, block>>>(
            sx, sy, stheta, sv,
            d_nominal_, d_costs_, d_perturbed_, d_rng_,
            eval_scenario_.params, eval_scenario_.cost_params,
            eval_scenario_.n_obs, eval_scenario_.n_dyn_obs,
            start_step, k_samples_, t_horizon_,
            variant_.ham_w_path, variant_.ham_w_speed,
            variant_.ham_w_obs, variant_.ham_w_heading,
            variant_.ham_w_terminal,
            d_had_path_, had_path_n_, variant_.ham_lookahead_idx);
        compute_weights_kernel<<<1, 1>>>(
            d_costs_, d_weights_, k_samples_, variant_.sampling_lambda);
        update_controls_kernel<<<(t_horizon_ + block - 1) / block, block>>>(
            d_nominal_, d_perturbed_, d_weights_, k_samples_, t_horizon_);
    }

    void hybrid_astar_dyn_pp_controller_update(float sx, float sy, float stheta, float sv, int start_step) {
        // Same as hybrid_astar_pp but the global search sees the dynamic
        // obstacles' linearised trajectories. The path then no longer
        // crosses where the obstacle will be when we arrive there.
        ensure_hap_path_planned(sx, sy, stheta, /*include_dynamic=*/true);
        hybrid_astar_pp_track(sx, sy, stheta, sv);
        (void)start_step;
    }

    void hybrid_astar_pp_track(float sx, float sy, float stheta, float sv) {
        PurePursuitParams pp;
        pp.lookahead = variant_.hap_lookahead;
        pp.wheelbase = planning_scenario_.params.L;
        pp.target_speed = variant_.hap_target_speed;
        pp.speed_gain = variant_.hap_speed_gain;
        pp.max_accel = 3.0f;
        pp.max_steer = planning_scenario_.params.max_steer;
        PurePursuitCommand cmd = pure_pursuit_step(
            sx, sy, stheta, sv, hap_path_, pp);
        h_nominal_[0] = cmd.accel;
        h_nominal_[1] = cmd.steer;
        for (int t = 1; t < t_horizon_; t++) {
            h_nominal_[t * 2 + 0] = 0.0f;
            h_nominal_[t * 2 + 1] = 0.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(),
                              h_nominal_.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void hybrid_astar_pp_controller_update(float sx, float sy, float stheta, float sv, int /*start_step*/) {
        ensure_hap_path_planned(sx, sy, stheta);
        PurePursuitParams pp;
        pp.lookahead = variant_.hap_lookahead;
        pp.wheelbase = planning_scenario_.params.L;
        pp.target_speed = variant_.hap_target_speed;
        pp.speed_gain = variant_.hap_speed_gain;
        pp.max_accel = 3.0f;
        pp.max_steer = planning_scenario_.params.max_steer;
        PurePursuitCommand cmd = pure_pursuit_step(
            sx, sy, stheta, sv, hap_path_, pp);
        h_nominal_[0] = cmd.accel;
        h_nominal_[1] = cmd.steer;
        for (int t = 1; t < t_horizon_; t++) {
            h_nominal_[t * 2 + 0] = 0.0f;
            h_nominal_[t * 2 + 1] = 0.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(),
                              h_nominal_.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void hybrid_astar_dwa_controller_update(float sx, float sy, float stheta, float sv, int start_step) {
        ensure_hap_path_planned(sx, sy, stheta);
        // Upload path to device on the first successful plan of the episode.
        if (!hap_path_.empty() && d_had_path_ == nullptr) {
            int n = static_cast<int>(hap_path_.size());
            std::vector<float> flat(static_cast<size_t>(n) * 3);
            for (int i = 0; i < n; i++) {
                flat[i * 3 + 0] = hap_path_[i].x;
                flat[i * 3 + 1] = hap_path_[i].y;
                flat[i * 3 + 2] = hap_path_[i].theta;
            }
            CUDA_CHECK(cudaMalloc(&d_had_path_, flat.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_had_path_, flat.data(),
                                  flat.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            had_path_n_ = n;
        }
        int total = dwa_grid_size_;
        int block = 64;
        int T_dwa = max(1, variant_.dwa_predict_steps);
        hybrid_astar_dwa_grid_kernel<<<(total + block - 1) / block, block>>>(
            sx, sy, stheta, sv,
            d_dwa_costs_, d_dwa_accels_, d_dwa_steers_,
            eval_scenario_.params, eval_scenario_.cost_params,
            eval_scenario_.n_obs, eval_scenario_.n_dyn_obs,
            start_step, T_dwa,
            variant_.dwa_n_accel, variant_.dwa_n_steer,
            variant_.dwa_accel_min, variant_.dwa_accel_max,
            variant_.had_w_path, variant_.had_w_speed,
            variant_.had_w_obs, variant_.had_w_heading,
            variant_.had_w_terminal,
            d_had_path_, had_path_n_, variant_.had_lookahead_idx);
        CUDA_CHECK(cudaMemcpy(h_dwa_costs_.data(), d_dwa_costs_,
                              total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dwa_accels_.data(), d_dwa_accels_,
                              total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dwa_steers_.data(), d_dwa_steers_,
                              total * sizeof(float), cudaMemcpyDeviceToHost));
        int best = 0;
        float best_cost = h_dwa_costs_[0];
        for (int i = 1; i < total; i++) {
            if (h_dwa_costs_[i] < best_cost) { best_cost = h_dwa_costs_[i]; best = i; }
        }
        h_nominal_[0] = h_dwa_accels_[best];
        h_nominal_[1] = h_dwa_steers_[best];
        for (int t = 1; t < t_horizon_; t++) {
            h_nominal_[t * 2 + 0] = 0.0f;
            h_nominal_[t * 2 + 1] = 0.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_nominal_, h_nominal_.data(),
                              h_nominal_.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void reset_state() {
        rx_ = eval_scenario_.start_x;
        ry_ = eval_scenario_.start_y;
        rtheta_ = eval_scenario_.start_theta;
        rv_ = eval_scenario_.start_v;
        steps_taken_ = 0;
        collisions_ = 0;
        reached_goal_ = false;
        cumulative_cost_ = 0.0f;
        min_goal_distance_ = sqrtf((rx_ - eval_scenario_.cost_params.goal_x) * (rx_ - eval_scenario_.cost_params.goal_x)
                                 + (ry_ - eval_scenario_.cost_params.goal_y) * (ry_ - eval_scenario_.cost_params.goal_y));
        hap_path_.clear();
        hap_planning_failed_ = false;
        if (d_had_path_) {
            CUDA_CHECK(cudaFree(d_had_path_));
            d_had_path_ = nullptr;
        }
        had_path_n_ = 0;
    }

    PlannerVariant variant_;
    Scenario planning_scenario_;
    Scenario eval_scenario_;
    int k_samples_;
    int t_horizon_;
    int seed_;

    float rx_ = 0.0f;
    float ry_ = 0.0f;
    float rtheta_ = 0.0f;
    float rv_ = 0.0f;
    int steps_taken_ = 0;
    int collisions_ = 0;
    bool reached_goal_ = false;
    float cumulative_cost_ = 0.0f;
    float min_goal_distance_ = 0.0f;

    vector<float> h_nominal_;
    vector<float> h_costs_;
    vector<float> h_grad_;
    vector<float> h_states_;
    vector<float> h_feedback_gains_host_;
    vector<float> h_sample_nominal_;
    vector<float> h_final_nominal_;
    vector<float> h_grad_snapshot_;

    float* d_nominal_ = nullptr;
    float* d_costs_ = nullptr;
    float* d_weights_ = nullptr;
    float* d_perturbed_ = nullptr;
    float* d_rollout_states_ = nullptr;
    float* d_rollout_init_grads_ = nullptr;
    float* d_states_ = nullptr;
    float* d_grad_ = nullptr;
    float* d_nav_stage_grads_ = nullptr;
    float* d_nav_jacobians_ = nullptr;
    float* d_grad_norm_ = nullptr;
    float h_grad_norm_ = 0.0f;
    int grad_steps_skipped_ = 0;
    float* d_feedback_gains_ = nullptr;
    float* d_feedback_gains_aux_ = nullptr;
    curandState* d_rng_ = nullptr;
    // Step-MPPI state
    float* d_sampling_bias_ = nullptr;
    float* d_step_sigma_ = nullptr;
    float* d_nominal_pre_bias_ = nullptr;
    float* d_ds_sigma_ = nullptr;
    float* d_soppi_scratch_ = nullptr;
    float* d_soppi_scores_ = nullptr;
    // DWA state: host-side argmin over a small grid, so we hold the grid on device
    // and a host mirror for argmin.
    float* d_dwa_costs_ = nullptr;
    float* d_dwa_accels_ = nullptr;
    float* d_dwa_steers_ = nullptr;
    vector<float> h_dwa_costs_;
    vector<float> h_dwa_accels_;
    vector<float> h_dwa_steers_;
    int dwa_grid_size_ = 0;
    // STOMP scratch for smoothness projection (same size as h_nominal_).
    float* d_stomp_scratch_ = nullptr;
    float* d_stomp_old_ = nullptr;
    float* d_stomp_M_ = nullptr;
    std::vector<Pose2D> hap_path_;
    bool hap_planning_failed_ = false;
    // Device-side Hybrid A* path (flat x,y,theta) used by hybrid_astar_dwa.
    float* d_had_path_ = nullptr;
    int had_path_n_ = 0;
    vector<TraceRow>* trace_rows_ = nullptr;
    int trace_max_steps_ = 0;
    vector<TrajectoryRow>* trajectory_rows_ = nullptr;
};

static Scenario instantiate_eval_scenario(const Scenario& nominal, int seed) {
    Scenario eval = nominal;
    if (nominal.use_model_mismatch) {
        eval.params.L *= nominal.eval_wheelbase_scale;
        eval.params.max_speed *= nominal.eval_max_speed_scale;
        eval.params.max_steer *= nominal.eval_max_steer_scale;
    }
    if (!nominal.use_dynamic_mismatch || nominal.n_dyn_obs <= 0) return eval;

    std::mt19937 rng(static_cast<uint32_t>(seed) * 747796405u + 2891336453u);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

    for (int i = 0; i < eval.n_dyn_obs; i++) {
        DynamicObstacle& obs = eval.dynamic_obstacles[i];
        float speed = sqrtf(obs.vx * obs.vx + obs.vy * obs.vy);
        float nx = 1.0f;
        float ny = 0.0f;
        if (speed > 1.0e-5f) {
            nx = -obs.vy / speed;
            ny = obs.vx / speed;
        }
        float time_offset = nominal.dyn_time_offset_max * unit(rng);
        float speed_scale = 1.0f + nominal.dyn_speed_scale_max * unit(rng);
        float lateral_jitter = nominal.dyn_lateral_jitter * unit(rng);
        obs.x += obs.vx * time_offset + nx * lateral_jitter;
        obs.y += obs.vy * time_offset + ny * lateral_jitter;
        obs.vx *= speed_scale;
        obs.vy *= speed_scale;
    }
    return eval;
}

static Scenario make_cluttered_scene() {
    Scenario s;
    s.name = "cluttered";
    s.start_x = 5.0f;
    s.start_y = 5.0f;
    s.cost_params.goal_x = 45.0f;
    s.cost_params.goal_y = 45.0f;
    s.cost_params.goal_weight = 5.0f;
    s.cost_params.control_weight = 0.1f;
    s.cost_params.speed_weight = 0.15f;
    s.cost_params.target_speed = 3.5f;
    s.cost_params.heading_weight = 0.35f;
    s.cost_params.obs_weight = 10.0f;
    s.cost_params.obs_influence = 5.0f;
    s.cost_params.terminal_weight = 8.0f;
    const Obstacle obs[] = {
        {12.0f, 15.0f, 3.0f}, {20.0f, 25.0f, 3.5f}, {30.0f, 10.0f, 3.0f},
        {15.0f, 35.0f, 2.5f}, {25.0f, 18.0f, 3.5f}, {35.0f, 30.0f, 2.5f},
        {22.0f, 40.0f, 3.0f}, {38.0f, 20.0f, 3.0f}, {10.0f, 30.0f, 2.5f},
        {32.0f, 38.0f, 2.5f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_narrow_passage_scene() {
    Scenario s;
    s.name = "narrow_passage";
    s.start_x = 4.0f;
    s.start_y = 8.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 42.0f;
    s.max_steps = 260;
    s.cost_params.target_speed = 3.0f;
    s.cost_params.obs_weight = 14.0f;
    s.cost_params.obs_influence = 5.5f;
    const Obstacle obs[] = {
        {22.0f, 6.0f, 2.3f}, {23.0f, 12.0f, 2.3f}, {24.0f, 18.0f, 2.3f},
        {26.0f, 32.0f, 2.3f}, {27.0f, 38.0f, 2.3f}, {28.0f, 44.0f, 2.3f},
        {36.0f, 24.0f, 2.8f}, {14.0f, 26.0f, 2.8f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_slalom_scene() {
    Scenario s;
    s.name = "slalom";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 240;
    s.cost_params.target_speed = 3.6f;
    s.cost_params.obs_weight = 11.0f;
    const Obstacle obs[] = {
        {10.0f, 14.0f, 2.7f}, {16.0f, 32.0f, 2.8f}, {22.0f, 14.0f, 2.8f},
        {28.0f, 33.0f, 2.8f}, {34.0f, 15.0f, 2.8f}, {40.0f, 33.0f, 2.8f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_corner_scene() {
    Scenario s;
    s.name = "corner_turn";
    s.start_x = 6.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 44.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 240;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.obs_weight = 13.0f;
    const Obstacle obs[] = {
        {18.0f, 12.0f, 3.0f}, {24.0f, 12.0f, 3.0f}, {30.0f, 12.0f, 3.0f},
        {30.0f, 18.0f, 3.0f}, {30.0f, 24.0f, 3.0f}, {30.0f, 30.0f, 3.0f},
        {18.0f, 30.0f, 2.6f}, {12.0f, 24.0f, 2.6f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    return s;
}

static Scenario make_dynamic_crossing_scene() {
    Scenario s;
    s.name = "dynamic_crossing";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 260;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.2f;
    s.cost_params.heading_weight = 0.40f;
    s.grad_alpha_scale = 0.20f;
    const Obstacle obs[] = {
        {16.0f, 16.0f, 2.8f}, {16.0f, 34.0f, 2.8f},
        {34.0f, 14.0f, 2.6f}, {34.0f, 36.0f, 2.6f}
    };
    const DynamicObstacle dyn[] = {
        {11.0f, 24.0f, 1.55f, 0.0f, 2.4f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

static Scenario make_dynamic_slalom_scene() {
    Scenario s;
    s.name = "dynamic_slalom";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    // Slalom geometry needs more steps than crossing; diff-MPPI / hybrid
    // planners still solve this cell, but the lightweight sampling zoo at
    // K<=128 typically stalls ~5 m from goal without gradient guidance.
    s.max_steps = 320;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.2f;
    s.cost_params.heading_weight = 0.40f;
    s.grad_alpha_scale = 0.20f;
    const Obstacle obs[] = {
        {10.0f, 14.0f, 2.4f}, {16.0f, 32.0f, 2.5f}, {22.0f, 14.0f, 2.5f},
        {28.0f, 33.0f, 2.5f}, {34.0f, 15.0f, 2.5f}, {40.0f, 33.0f, 2.5f}
    };
    const DynamicObstacle dyn[] = {
        // Keep a lateral mover for DRA-family scoring, but start it high and
        // slow so it does not dominate the static slalom timing budget.
        {24.0f, 44.0f, 0.0f, -0.85f, 2.0f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

static Scenario make_uncertain_crossing_scene() {
    Scenario s = make_dynamic_crossing_scene();
    s.name = "uncertain_crossing";
    s.use_dynamic_mismatch = true;
    s.dyn_time_offset_max = 1.15f;
    s.dyn_speed_scale_max = 0.18f;
    s.dyn_lateral_jitter = 0.85f;
    return s;
}

static Scenario make_dynamic_pincer_scene() {
    Scenario s;
    s.name = "dynamic_pincer";
    s.start_x = 4.0f;
    s.start_y = 6.0f;
    s.cost_params.goal_x = 46.0f;
    s.cost_params.goal_y = 44.0f;
    s.max_steps = 260;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.2f;
    s.cost_params.heading_weight = 0.40f;
    s.grad_alpha_scale = 0.20f;
    const Obstacle obs[] = {
        {16.0f, 16.0f, 2.8f}, {16.0f, 34.0f, 2.8f},
        {34.0f, 14.0f, 2.6f}, {34.0f, 36.0f, 2.6f}
    };
    // Three dyn obstacles whose trajectories converge near the agent's
    // diagonal midpoint (~25,25): one descends from upper-left, one
    // ascends from lower-right, one rises from below-centre. The agent
    // must time its passage rather than dodge a single obstacle.
    const DynamicObstacle dyn[] = {
        { 8.0f, 30.0f,  1.30f, -0.60f, 2.2f},
        {42.0f, 18.0f, -1.30f,  0.60f, 2.2f},
        {25.0f,  4.0f,  0.00f,  1.40f, 2.2f},
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

// ---- Topology-stress benchmark scenarios (Day 1 of advisor roadmap) ----
//
// Goal of these four scenes: surface where global path planning matters,
// by constructing layouts in which DWA's local goal-pull alone is
// expected to fail (local minima, long-horizon detours, or timing-
// dependent gate passage). The existing dynamic_* scenes are "open
// dynamic" -- local reactive controllers can handle them. These add
// "global topology" and "topology + dynamic" as orthogonal axes.

// Static U-trap with the mouth opening west (toward robot). Goal is east
// behind the trap; greedy east-pull traps the robot against the east wall.
// Hybrid A* should detour over the top or under the bottom.
static Scenario make_static_u_trap_scene() {
    Scenario s;
    s.name = "static_u_trap";
    s.start_x = 5.0f;
    s.start_y = 25.0f;
    s.cost_params.goal_x = 47.0f;
    s.cost_params.goal_y = 25.0f;
    s.max_steps = 320;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.2f;
    s.cost_params.heading_weight = 0.40f;
    s.grad_alpha_scale = 0.20f;
    const Obstacle obs[] = {
        // top wall y=32
        {23.0f, 32.0f, 2.5f}, {28.0f, 32.0f, 2.5f},
        {33.0f, 32.0f, 2.5f}, {38.0f, 32.0f, 2.5f},
        // bottom wall y=18
        {23.0f, 18.0f, 2.5f}, {28.0f, 18.0f, 2.5f},
        {33.0f, 18.0f, 2.5f}, {38.0f, 18.0f, 2.5f},
        // east wall x=41 closes the trap
        {41.0f, 20.0f, 2.5f}, {41.0f, 25.0f, 2.5f}, {41.0f, 30.0f, 2.5f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = 0;
    return s;
}

// Static S-shaped corridor. Two parallel barriers with gaps on opposite
// sides force a back-and-forth detour. DWA's diagonal goal-pull aims
// straight at the first barrier with no way to recover locally.
static Scenario make_static_s_corridor_scene() {
    Scenario s;
    s.name = "static_s_corridor";
    s.start_x = 5.0f;
    s.start_y = 5.0f;
    s.cost_params.goal_x = 45.0f;
    s.cost_params.goal_y = 45.0f;
    s.max_steps = 360;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.2f;
    s.cost_params.heading_weight = 0.40f;
    s.grad_alpha_scale = 0.20f;
    const Obstacle obs[] = {
        // first barrier y=20, opens at east (x >= 38)
        {5.0f, 20.0f, 3.0f},  {12.0f, 20.0f, 3.0f},
        {19.0f, 20.0f, 3.0f}, {26.0f, 20.0f, 3.0f}, {33.0f, 20.0f, 3.0f},
        // second barrier y=35, opens at west (x <= 12)
        {17.0f, 35.0f, 3.0f}, {24.0f, 35.0f, 3.0f},
        {31.0f, 35.0f, 3.0f}, {38.0f, 35.0f, 3.0f}, {45.0f, 35.0f, 3.0f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = 0;
    return s;
}

// Static narrow gate plus a dynamic obstacle crossing it. The trap is
// not topological -- the gate centered at (25, 25) is reachable -- but
// the dynamic obstacle blocks the gate around t=6-9 s, exactly when a
// constant-speed robot would arrive. Pure global planning ignores the
// timing; pure local reactive can detect it but cannot detour because
// the walls leave no alternative path. Tests global + dyn-aware local.
static Scenario make_dynamic_bottleneck_scene() {
    Scenario s;
    s.name = "dynamic_bottleneck";
    s.start_x = 5.0f;
    s.start_y = 25.0f;
    s.cost_params.goal_x = 45.0f;
    s.cost_params.goal_y = 25.0f;
    s.max_steps = 320;
    s.cost_params.target_speed = 3.2f;
    s.cost_params.goal_weight = 5.2f;
    s.cost_params.obs_weight = 11.5f;
    s.cost_params.obs_influence = 5.2f;
    s.cost_params.heading_weight = 0.40f;
    s.grad_alpha_scale = 0.20f;
    const Obstacle obs[] = {
        // bottom wall x=25, y=0..22
        {25.0f, 4.0f, 4.0f},  {25.0f, 12.0f, 4.0f}, {25.0f, 18.0f, 4.0f},
        // top wall x=25, y=28..50
        {25.0f, 32.0f, 4.0f}, {25.0f, 40.0f, 4.0f}, {25.0f, 46.0f, 4.0f}
        // gap y=22..28 (6 units)
    };
    const DynamicObstacle dyn[] = {
        // Moderate-speed obstacle that occupies the gate during the
        // window when a constant-3.2 m/s robot would arrive: enters the
        // gap top at t ~= 2s (y = 30 - 1.0t), centres at t ~= 5s, and
        // clears the gap bottom at t ~= 8s. A purely greedy controller
        // collides; a reactive controller must slow to ~80% speed so
        // it reaches the gate around t = 8.5s. The walls leave no
        // detour, so timing is the only option -- pp-style trackers
        // that ignore the dyn obs collide while DWA / MPPI slow down.
        {25.0f, 30.0f, 0.0f, -1.0f, 2.5f}
    };
    s.n_obs = static_cast<int>(sizeof(obs) / sizeof(obs[0]));
    for (int i = 0; i < s.n_obs; i++) s.obstacles[i] = obs[i];
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

// U-trap topology plus a dynamic obstacle on the detour exit. Combines
// global-topology need (must detour around the U) with dynamic-obstacle
// need (the detour path is timed-blocked). This is the "both axes"
// cell expected to separate global-only / local-only / hybrid planners.
static Scenario make_dynamic_crossing_with_topology_scene() {
    Scenario s = make_static_u_trap_scene();
    s.name = "dynamic_crossing_with_topology";
    s.max_steps = 360;
    const DynamicObstacle dyn[] = {
        // moves west along y=36, just above the top-wall detour path.
        // A naive Hybrid A* path that ignores it crosses this trajectory
        // around t = 8-12 s; a dyn-aware controller has to slow or
        // adjust the detour angle.
        {45.0f, 36.0f, -1.6f, 0.0f, 2.2f}
    };
    s.n_dyn_obs = static_cast<int>(sizeof(dyn) / sizeof(dyn[0]));
    for (int i = 0; i < s.n_dyn_obs; i++) s.dynamic_obstacles[i] = dyn[i];
    return s;
}

static Scenario make_uncertain_slalom_scene() {
    Scenario s = make_dynamic_slalom_scene();
    s.name = "uncertain_slalom";
    s.use_dynamic_mismatch = true;
    s.dyn_time_offset_max = 0.95f;
    s.dyn_speed_scale_max = 0.16f;
    s.dyn_lateral_jitter = 0.75f;
    return s;
}

static Scenario make_model_mismatch_slalom_scene() {
    Scenario s = make_slalom_scene();
    s.name = "model_mismatch_slalom";
    s.use_model_mismatch = true;
    s.eval_wheelbase_scale = 1.45f;
    s.eval_max_speed_scale = 0.82f;
    s.eval_max_steer_scale = 0.80f;
    s.max_steps = 280;
    s.cost_params.target_speed = 3.4f;
    s.cost_params.obs_weight = 12.0f;
    s.cost_params.obs_influence = 5.4f;
    return s;
}

static Scenario make_model_mismatch_crossing_scene() {
    Scenario s = make_dynamic_crossing_scene();
    s.name = "model_mismatch_crossing";
    s.use_model_mismatch = true;
    s.eval_wheelbase_scale = 1.45f;
    s.eval_max_speed_scale = 0.82f;
    s.eval_max_steer_scale = 0.80f;
    s.max_steps = 300;
    s.cost_params.target_speed = 3.1f;
    s.cost_params.obs_weight = 12.5f;
    s.cost_params.obs_influence = 5.5f;
    return s;
}

static void ensure_build_dir() {
    mkdir("build", 0755);
}

static vector<int> parse_int_list(const string& text) {
    vector<int> values;
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) {
        if (token.empty()) continue;
        values.push_back(std::max(1, atoi(token.c_str())));
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    return values;
}

static vector<string> parse_string_list(const string& text) {
    vector<string> values;
    string token;
    stringstream ss(text);
    while (getline(ss, token, ',')) {
        if (!token.empty()) values.push_back(token);
    }
    sort(values.begin(), values.end());
    values.erase(unique(values.begin(), values.end()), values.end());
    return values;
}

static void write_csv(const vector<EpisodeMetrics>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,t_horizon,grad_steps,alpha,reached_goal,collision_free,success,steps,final_distance,min_goal_distance,cumulative_cost,collisions,mean_control_delta,control_roughness,avg_control_ms,total_control_ms,episode_ms,sample_budget\n";
    for (const auto& r : rows) {
        out << r.scenario << ','
            << r.planner << ','
            << r.seed << ','
            << r.k_samples << ','
            << r.t_horizon << ','
            << r.grad_steps << ','
            << r.alpha << ','
            << r.reached_goal << ','
            << r.collision_free << ','
            << r.success << ','
            << r.steps << ','
            << r.final_distance << ','
            << r.min_goal_distance << ','
            << r.cumulative_cost << ','
            << r.collisions << ','
            << r.mean_control_delta << ','
            << r.control_roughness << ','
            << r.avg_control_ms << ','
            << r.total_control_ms << ','
            << r.episode_ms << ','
            << r.sample_budget << '\n';
    }
}

static void write_trajectory_csv(const vector<TrajectoryRow>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,episode_step,x,y,theta,v,goal_distance\n";
    for (const auto& r : rows) {
        out << r.scenario << ','
            << r.planner << ','
            << r.seed << ','
            << r.k_samples << ','
            << r.episode_step << ','
            << r.x << ','
            << r.y << ','
            << r.theta << ','
            << r.v << ','
            << r.goal_distance << '\n';
    }
}

static void write_trace_csv(const vector<TraceRow>& rows, const string& path) {
    ofstream out(path);
    out << "scenario,planner,seed,k_samples,grad_steps,alpha,episode_step,horizon_step,goal_distance,min_obstacle_margin,control_ms,"
           "sampled_accel,sampled_steer,final_accel,final_steer,delta_accel,delta_steer,delta_norm,grad_accel,grad_steer,grad_norm\n";
    for (const auto& r : rows) {
        out << r.scenario << ','
            << r.planner << ','
            << r.seed << ','
            << r.k_samples << ','
            << r.grad_steps << ','
            << r.alpha << ','
            << r.episode_step << ','
            << r.horizon_step << ','
            << r.goal_distance << ','
            << r.min_obstacle_margin << ','
            << r.control_ms << ','
            << r.sampled_accel << ','
            << r.sampled_steer << ','
            << r.final_accel << ','
            << r.final_steer << ','
            << r.delta_accel << ','
            << r.delta_steer << ','
            << r.delta_norm << ','
            << r.grad_accel << ','
            << r.grad_steer << ','
            << r.grad_norm << '\n';
    }
}

static void print_summary(const vector<EpisodeMetrics>& rows) {
    map<string, SummaryStats> stats;
    for (const auto& r : rows) {
        string key = r.scenario + " | " + r.planner + " | K=" + to_string(r.k_samples);
        auto& s = stats[key];
        s.episodes++;
        s.successes += r.success;
        s.sum_steps += r.steps;
        s.sum_final_distance += r.final_distance;
        s.sum_min_goal_distance += r.min_goal_distance;
        s.sum_cumulative_cost += r.cumulative_cost;
        s.sum_avg_control_ms += r.avg_control_ms;
        s.sum_total_control_ms += r.total_control_ms;
        s.sum_collisions += r.collisions;
        s.sum_mean_control_delta += r.mean_control_delta;
        s.sum_control_roughness += r.control_roughness;
    }

    cout << "=== benchmark_diff_mppi summary ===" << endl;
    for (const auto& kv : stats) {
        const SummaryStats& s = kv.second;
        float n = static_cast<float>(s.episodes);
        printf("%s : success=%.2f steps=%.1f final_dist=%.2f min_dist=%.2f cost=%.1f du=%.3f rough=%.3f avg_ms=%.2f collisions=%.2f\n",
               kv.first.c_str(),
               s.successes / n,
               s.sum_steps / n,
               s.sum_final_distance / n,
               s.sum_min_goal_distance / n,
               s.sum_cumulative_cost / n,
               s.sum_mean_control_delta / n,
               s.sum_control_roughness / n,
               s.sum_avg_control_ms / n,
               s.sum_collisions / n);
    }
}

int main(int argc, char** argv) {
    bool quick = false;
    string csv_path = "build/benchmark_diff_mppi.csv";
    string trace_csv_path;
    string trajectory_csv_path;
    vector<int> k_values;
    vector<string> scenario_names;
    vector<string> planner_names;
    int seed_count = -1;
    int trace_max_steps = 0;
    float override_feedback_gain_scale = -1.0f;
    float override_feedback_ref_blend = -1.0f;
    float override_feedback_cov_blend = -1.0f;
    float override_feedback_lqr_blend = -1.0f;
    float override_feedback_setpoint_blend = -1.0f;
    float override_feedback_cov_regularization = -1.0f;
    int override_grad_steps = -1;
    int override_grad_update_horizon = -1;
    float override_alpha = -1.0f;
    float override_sampling_lambda = -1.0f;
    float override_mlp_lr = -1.0f;
    float override_lp_alpha = -1.0f;
    int override_ds_iterations = -1;
    float override_ds_alpha = -1.0f;
    float override_ds_noise_scale = -1.0f;
    float override_ds_momentum = -1.0f;
    int override_ds_stride = -1;
    int override_projection_passes = -1;
    float override_projection_accel_delta = -1.0f;
    float override_projection_steer_delta = -1.0f;
    float override_projection_accel_ddelta = -1.0f;
    float override_projection_steer_ddelta = -1.0f;
    int override_soppi_iters = -1;
    int override_soppi_neighbor_count = -1;
    float override_soppi_step_size = -1.0f;
    float override_soppi_bandwidth = -1.0f;
    float override_dyn_speed_scale = -1.0f;
    float override_dyn_radius_scale = -1.0f;
    // DWA cost-weight overrides. Sentinel < 0 = unset (weights are non-negative).
    float override_dwa_w_goal = -1.0f;
    float override_dwa_w_speed = -1.0f;
    float override_dwa_w_obs = -1.0f;
    float override_dwa_w_heading = -1.0f;
    float override_dwa_w_terminal = -1.0f;
    int override_t_horizon = -1;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--quick") quick = true;
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--trace-csv" && i + 1 < argc) trace_csv_path = argv[++i];
        else if (arg == "--trajectory-csv" && i + 1 < argc) trajectory_csv_path = argv[++i];
        else if (arg == "--trace-max-steps" && i + 1 < argc) trace_max_steps = std::max(0, atoi(argv[++i]));
        else if (arg == "--k-values" && i + 1 < argc) k_values = parse_int_list(argv[++i]);
        else if (arg == "--seed-count" && i + 1 < argc) seed_count = std::max(1, atoi(argv[++i]));
        else if (arg == "--t-horizon" && i + 1 < argc) override_t_horizon = std::max(1, atoi(argv[++i]));
        else if (arg == "--scenarios" && i + 1 < argc) scenario_names = parse_string_list(argv[++i]);
        else if (arg == "--planners" && i + 1 < argc) planner_names = parse_string_list(argv[++i]);
        else if (arg == "--override-feedback-gain-scale" && i + 1 < argc) override_feedback_gain_scale = atof(argv[++i]);
        else if (arg == "--override-feedback-ref-blend" && i + 1 < argc) override_feedback_ref_blend = atof(argv[++i]);
        else if (arg == "--override-feedback-cov-blend" && i + 1 < argc) override_feedback_cov_blend = atof(argv[++i]);
        else if (arg == "--override-feedback-lqr-blend" && i + 1 < argc) override_feedback_lqr_blend = atof(argv[++i]);
        else if (arg == "--override-feedback-setpoint-blend" && i + 1 < argc) override_feedback_setpoint_blend = atof(argv[++i]);
        else if (arg == "--override-feedback-cov-regularization" && i + 1 < argc) override_feedback_cov_regularization = atof(argv[++i]);
        else if (arg == "--override-grad-steps" && i + 1 < argc) override_grad_steps = atoi(argv[++i]);
        else if (arg == "--override-grad-update-horizon" && i + 1 < argc) override_grad_update_horizon = atoi(argv[++i]);
        else if (arg == "--override-alpha" && i + 1 < argc) override_alpha = atof(argv[++i]);
        else if (arg == "--override-lambda" && i + 1 < argc) override_sampling_lambda = atof(argv[++i]);
        else if (arg == "--override-mlp-lr" && i + 1 < argc) override_mlp_lr = atof(argv[++i]);
        else if (arg == "--override-lp-alpha" && i + 1 < argc) override_lp_alpha = atof(argv[++i]);
        else if (arg == "--override-ds-iters" && i + 1 < argc) override_ds_iterations = atoi(argv[++i]);
        else if (arg == "--override-ds-alpha" && i + 1 < argc) override_ds_alpha = atof(argv[++i]);
        else if (arg == "--override-ds-noise-scale" && i + 1 < argc) override_ds_noise_scale = atof(argv[++i]);
        else if (arg == "--override-ds-momentum" && i + 1 < argc) override_ds_momentum = atof(argv[++i]);
        else if (arg == "--override-ds-stride" && i + 1 < argc) override_ds_stride = atoi(argv[++i]);
        else if (arg == "--override-pi-passes" && i + 1 < argc) override_projection_passes = atoi(argv[++i]);
        else if (arg == "--override-pi-accel-delta" && i + 1 < argc) override_projection_accel_delta = atof(argv[++i]);
        else if (arg == "--override-pi-steer-delta" && i + 1 < argc) override_projection_steer_delta = atof(argv[++i]);
        else if (arg == "--override-pi-accel-ddelta" && i + 1 < argc) override_projection_accel_ddelta = atof(argv[++i]);
        else if (arg == "--override-pi-steer-ddelta" && i + 1 < argc) override_projection_steer_ddelta = atof(argv[++i]);
        else if (arg == "--override-soppi-iters" && i + 1 < argc) override_soppi_iters = atoi(argv[++i]);
        else if (arg == "--override-soppi-neighbors" && i + 1 < argc) override_soppi_neighbor_count = std::max(0, atoi(argv[++i]));
        else if (arg == "--override-soppi-step-size" && i + 1 < argc) override_soppi_step_size = atof(argv[++i]);
        else if (arg == "--override-soppi-bandwidth" && i + 1 < argc) override_soppi_bandwidth = atof(argv[++i]);
        else if (arg == "--override-dyn-speed-scale" && i + 1 < argc) override_dyn_speed_scale = atof(argv[++i]);
        else if (arg == "--override-dyn-radius-scale" && i + 1 < argc) override_dyn_radius_scale = atof(argv[++i]);
        else if (arg == "--override-dwa-w-goal" && i + 1 < argc) override_dwa_w_goal = atof(argv[++i]);
        else if (arg == "--override-dwa-w-speed" && i + 1 < argc) override_dwa_w_speed = atof(argv[++i]);
        else if (arg == "--override-dwa-w-obs" && i + 1 < argc) override_dwa_w_obs = atof(argv[++i]);
        else if (arg == "--override-dwa-w-heading" && i + 1 < argc) override_dwa_w_heading = atof(argv[++i]);
        else if (arg == "--override-dwa-w-terminal" && i + 1 < argc) override_dwa_w_terminal = atof(argv[++i]);
    }

    ensure_build_dir();

    vector<Scenario> all_scenarios;
    all_scenarios.push_back(make_cluttered_scene());
    all_scenarios.push_back(make_narrow_passage_scene());
    all_scenarios.push_back(make_slalom_scene());
    all_scenarios.push_back(make_corner_scene());
    all_scenarios.push_back(make_dynamic_crossing_scene());
    all_scenarios.push_back(make_dynamic_slalom_scene());
    all_scenarios.push_back(make_uncertain_crossing_scene());
    all_scenarios.push_back(make_uncertain_slalom_scene());
    all_scenarios.push_back(make_dynamic_pincer_scene());
    all_scenarios.push_back(make_static_u_trap_scene());
    all_scenarios.push_back(make_static_s_corridor_scene());
    all_scenarios.push_back(make_dynamic_bottleneck_scene());
    all_scenarios.push_back(make_dynamic_crossing_with_topology_scene());
    all_scenarios.push_back(make_model_mismatch_slalom_scene());
    all_scenarios.push_back(make_model_mismatch_crossing_scene());

    vector<Scenario> scenarios;
    if (!scenario_names.empty()) {
        for (const auto& wanted : scenario_names) {
            auto it = find_if(all_scenarios.begin(), all_scenarios.end(),
                              [&](const Scenario& s) { return s.name == wanted; });
            if (it == all_scenarios.end()) {
                fprintf(stderr, "Unknown scenario: %s\n", wanted.c_str());
                return 1;
            }
            scenarios.push_back(*it);
        }
    } else if (quick) {
        scenarios.push_back(make_cluttered_scene());
        scenarios.push_back(make_narrow_passage_scene());
    } else {
        scenarios.push_back(make_cluttered_scene());
        scenarios.push_back(make_narrow_passage_scene());
        scenarios.push_back(make_slalom_scene());
        scenarios.push_back(make_corner_scene());
    }

    vector<PlannerVariant> variants;
    {
        PlannerVariant v;
        v.name = "mppi";
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "lp_mppi";
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "lp_mppi_smooth";
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "log_mppi";
        v.use_dbas_log_sampling = true;
        v.dbas_barrier_weight = 0.0f;
        v.dbas_speed_damping = 0.0f;
        v.dbas_mu = 1.00f;
        v.dbas_log_sigma = 0.45f;
        v.dbas_lognormal_clip = 3.50f;
        v.dbas_noise_scale = 1.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dbas_log_mppi";
        v.use_dbas_log_sampling = true;
        v.dbas_safe_margin = 0.45f;
        v.dbas_barrier_eps = 0.45f;
        v.dbas_barrier_cap = 24.0f;
        v.dbas_barrier_weight = 220.0f;
        v.dbas_gamma = 0.25f;
        v.dbas_mu = 0.70f;
        v.dbas_log_sigma = 0.45f;
        v.dbas_lognormal_clip = 3.50f;
        v.dbas_noise_scale = 1.0f;
        v.dbas_speed_damping = 0.10f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dbas_log_mppi_smooth";
        v.use_dbas_log_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.dbas_safe_margin = 0.55f;
        v.dbas_barrier_eps = 0.50f;
        v.dbas_barrier_cap = 28.0f;
        v.dbas_barrier_weight = 260.0f;
        v.dbas_gamma = 0.35f;
        v.dbas_mu = 0.62f;
        v.dbas_log_sigma = 0.38f;
        v.dbas_lognormal_clip = 3.00f;
        v.dbas_noise_scale = 0.95f;
        v.dbas_speed_damping = 0.14f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dbas_log_mppi_agile";
        v.use_dbas_log_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.18f;
        v.dbas_safe_margin = 0.20f;
        v.dbas_barrier_eps = 0.65f;
        v.dbas_barrier_cap = 18.0f;
        v.dbas_barrier_weight = 90.0f;
        v.dbas_gamma = 0.18f;
        v.dbas_mu = 1.15f;
        v.dbas_log_sigma = 0.65f;
        v.dbas_lognormal_clip = 5.50f;
        v.dbas_noise_scale = 1.25f;
        v.dbas_speed_damping = 0.02f;
        v.sampling_lambda = 6.5f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dbas_log_mppi_safe";
        v.use_dbas_log_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.24f;
        v.dbas_safe_margin = 0.75f;
        v.dbas_barrier_eps = 0.45f;
        v.dbas_barrier_cap = 36.0f;
        v.dbas_barrier_weight = 420.0f;
        v.dbas_gamma = 0.40f;
        v.dbas_mu = 0.78f;
        v.dbas_log_sigma = 0.55f;
        v.dbas_lognormal_clip = 4.50f;
        v.dbas_noise_scale = 1.10f;
        v.dbas_speed_damping = 0.22f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "sc_mppi";
        v.use_safety_controlled_sampling = true;
        v.sc_safe_margin = 1.0f;
        v.sc_avoid_gain = 0.55f;
        v.sc_speed_gain = 0.80f;
        v.sc_max_steer_delta = 0.28f;
        v.sc_max_accel_delta = 1.8f;
        v.sc_control_weight = 0.05f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "sc_mppi_smooth";
        v.use_safety_controlled_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.sc_safe_margin = 1.0f;
        v.sc_avoid_gain = 0.55f;
        v.sc_speed_gain = 0.80f;
        v.sc_max_steer_delta = 0.28f;
        v.sc_max_accel_delta = 1.8f;
        v.sc_control_weight = 0.05f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "sc_mppi_timing";
        v.use_safety_controlled_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.30f;
        v.t_horizon = 12;
        v.sc_safe_margin = 0.65f;
        v.sc_avoid_gain = 0.32f;
        v.sc_speed_gain = 0.30f;
        v.sc_max_steer_delta = 0.16f;
        v.sc_max_accel_delta = 0.9f;
        v.sc_control_weight = 0.03f;
        v.sampling_lambda = 5.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "csc_mppi";
        v.use_safety_controlled_sampling = true;
        v.use_cluster_representative_update = true;
        v.sc_safe_margin = 0.80f;
        v.sc_avoid_gain = 0.45f;
        v.sc_speed_gain = 0.55f;
        v.sc_max_steer_delta = 0.24f;
        v.sc_max_accel_delta = 1.3f;
        v.csc_cluster_count = 4;
        v.csc_safe_margin = 0.15f;
        v.csc_constraint_weight = 3500.0f;
        v.csc_update_blend = 0.75f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "csc_mppi_smooth";
        v.use_safety_controlled_sampling = true;
        v.use_low_pass_sampling = true;
        v.use_cluster_representative_update = true;
        v.lp_alpha = 0.22f;
        v.sc_safe_margin = 0.85f;
        v.sc_avoid_gain = 0.50f;
        v.sc_speed_gain = 0.60f;
        v.sc_max_steer_delta = 0.24f;
        v.sc_max_accel_delta = 1.4f;
        v.csc_cluster_count = 4;
        v.csc_safe_margin = 0.20f;
        v.csc_constraint_weight = 4000.0f;
        v.csc_update_blend = 0.65f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "csc_mppi_strict";
        v.use_safety_controlled_sampling = true;
        v.use_low_pass_sampling = true;
        v.use_cluster_representative_update = true;
        v.lp_alpha = 0.26f;
        v.sc_safe_margin = 1.10f;
        v.sc_avoid_gain = 0.62f;
        v.sc_speed_gain = 0.75f;
        v.sc_max_steer_delta = 0.30f;
        v.sc_max_accel_delta = 1.7f;
        v.csc_cluster_count = 5;
        v.csc_safe_margin = 0.45f;
        v.csc_constraint_weight = 6500.0f;
        v.csc_update_blend = 0.55f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dm_mppi";
        v.use_datamodel_influence_pruning = true;
        v.dm_keep_fraction = 0.35f;
        v.dm_cost_temperature = 8.0f;
        v.dm_safe_margin = 0.65f;
        v.dm_prob_sigma = 0.70f;
        v.dm_violation_weight = 2500.0f;
        v.dm_safety_power = 0.75f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dm_mppi_smooth";
        v.use_datamodel_influence_pruning = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.dm_keep_fraction = 0.35f;
        v.dm_cost_temperature = 8.0f;
        v.dm_safe_margin = 0.70f;
        v.dm_prob_sigma = 0.70f;
        v.dm_violation_weight = 3000.0f;
        v.dm_safety_power = 0.85f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dm_mppi_safe";
        v.use_datamodel_influence_pruning = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.24f;
        v.dm_keep_fraction = 0.25f;
        v.dm_cost_temperature = 10.0f;
        v.dm_safe_margin = 1.05f;
        v.dm_prob_sigma = 0.55f;
        v.dm_violation_weight = 6500.0f;
        v.dm_safety_power = 1.40f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "tsallis_mppi_q07";
        v.use_tsallis_weights = true;
        v.tsallis_q = 0.70f;
        v.tsallis_temperature = 8.0f;
        v.tsallis_min_weight = 0.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "tsallis_mppi_smooth";
        v.use_tsallis_weights = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.tsallis_q = 0.70f;
        v.tsallis_temperature = 8.0f;
        v.tsallis_min_weight = 0.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "tsallis_mppi_q13";
        v.use_tsallis_weights = true;
        v.tsallis_q = 1.30f;
        v.tsallis_temperature = 5.0f;
        v.tsallis_min_weight = 1.0e-8f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "cc_mppi";
        v.use_covariance_control_weights = true;
        v.cc_terminal_weight = 1.25f;
        v.cc_terminal_target_radius = 4.5f;
        v.cc_heading_weight = 0.35f;
        v.cc_speed_weight = 0.10f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "cc_mppi_smooth";
        v.use_covariance_control_weights = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.cc_terminal_weight = 1.00f;
        v.cc_terminal_target_radius = 4.0f;
        v.cc_heading_weight = 0.30f;
        v.cc_speed_weight = 0.08f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "cc_mppi_tight";
        v.use_covariance_control_weights = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.26f;
        v.cc_terminal_weight = 2.50f;
        v.cc_terminal_target_radius = 2.8f;
        v.cc_heading_weight = 0.45f;
        v.cc_speed_weight = 0.12f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "td_v_mppi_short";
        v.use_td_cd_weights = true;
        v.use_low_pass_sampling = true;
        v.t_horizon = 12;
        v.lp_alpha = 0.24f;
        v.td_terminal_value_scale = 5.0f;
        v.td_safe_margin = -5.0f;
        v.td_discount_sigma = 10.0f;
        v.td_discount_power = 0.0f;
        v.td_failure_cost = 0.0f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "td_cd_mppi_soft";
        v.use_td_cd_weights = true;
        v.use_low_pass_sampling = true;
        v.t_horizon = 12;
        v.lp_alpha = 0.24f;
        v.td_terminal_value_scale = 4.0f;
        v.td_safe_margin = -0.25f;
        v.td_discount_sigma = 2.0f;
        v.td_discount_power = 0.08f;
        v.td_failure_cost = 1200.0f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "td_cd_mppi_guarded";
        v.use_td_cd_weights = true;
        v.use_low_pass_sampling = true;
        v.t_horizon = 16;
        v.lp_alpha = 0.26f;
        v.td_terminal_value_scale = 3.0f;
        v.td_safe_margin = -0.10f;
        v.td_discount_sigma = 2.5f;
        v.td_discount_power = 0.12f;
        v.td_failure_cost = 2200.0f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "shield_mppi";
        v.use_shield_cost = true;
        v.use_shield_repair = true;
        v.t_horizon = 12;
        v.shield_safe_margin = 1.2f;
        v.shield_cbf_alpha = 0.40f;
        v.shield_cbf_weight = 90.0f;
        v.shield_repair_steps = 8;
        v.shield_repair_grid = 5;
        v.shield_repair_safety_weight = 250.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "shield_mppi_smooth";
        v.use_shield_cost = true;
        v.use_shield_repair = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.25f;
        v.t_horizon = 12;
        v.shield_safe_margin = 1.2f;
        v.shield_cbf_alpha = 0.40f;
        v.shield_cbf_weight = 90.0f;
        v.shield_repair_steps = 8;
        v.shield_repair_grid = 5;
        v.shield_repair_safety_weight = 250.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "shield_mppi_repair";
        v.use_shield_repair = true;
        v.t_horizon = 12;
        v.shield_safe_margin = 1.2f;
        v.shield_cbf_alpha = 0.40f;
        v.shield_repair_steps = 8;
        v.shield_repair_grid = 5;
        v.shield_repair_safety_weight = 250.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "c2u_mppi";
        v.use_c2u_chance_constraints = true;
        v.c2u_safe_margin = 0.0f;
        v.c2u_robot_sigma = 0.10f;
        v.c2u_dyn_sigma0 = 0.28f;
        v.c2u_dyn_sigma_growth = 0.05f;
        v.c2u_risk_z = 1.28f;
        v.c2u_prob_sigma = 0.90f;
        v.c2u_probability_power = 0.70f;
        v.c2u_violation_weight = 900.0f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "c2u_mppi_smooth";
        v.use_c2u_chance_constraints = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.c2u_safe_margin = 0.0f;
        v.c2u_robot_sigma = 0.12f;
        v.c2u_dyn_sigma0 = 0.35f;
        v.c2u_dyn_sigma_growth = 0.06f;
        v.c2u_risk_z = 1.28f;
        v.c2u_prob_sigma = 1.00f;
        v.c2u_probability_power = 0.80f;
        v.c2u_violation_weight = 1200.0f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "c2u_mppi_strict";
        v.use_c2u_chance_constraints = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.24f;
        v.c2u_safe_margin = 0.20f;
        v.c2u_robot_sigma = 0.18f;
        v.c2u_dyn_sigma0 = 0.45f;
        v.c2u_dyn_sigma_growth = 0.08f;
        v.c2u_risk_z = 1.64f;
        v.c2u_prob_sigma = 0.80f;
        v.c2u_probability_power = 1.20f;
        v.c2u_violation_weight = 2200.0f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ducct_mppi_smooth";
        v.use_ducct_risk = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.ducct_loc_sigma0 = 0.08f;
        v.ducct_loc_sigma_growth = 0.02f;
        v.ducct_pred_sigma0 = 0.24f;
        v.ducct_pred_sigma_growth = 0.035f;
        v.ducct_static_sigma = 0.08f;
        v.ducct_risk_weight = 650.0f;
        v.ducct_hard_threshold = 0.88f;
        v.ducct_reject_cost = 7000.0f;
        v.ducct_survival_power = 0.20f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ducct_mppi_cautious";
        v.use_ducct_risk = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.ducct_loc_sigma0 = 0.18f;
        v.ducct_loc_sigma_growth = 0.06f;
        v.ducct_pred_sigma0 = 0.50f;
        v.ducct_pred_sigma_growth = 0.08f;
        v.ducct_static_sigma = 0.18f;
        v.ducct_risk_weight = 2600.0f;
        v.ducct_hard_threshold = 0.55f;
        v.ducct_reject_cost = 38000.0f;
        v.ducct_survival_power = 0.75f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ducct_mppi_diluted";
        v.use_ducct_risk = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.ducct_loc_sigma0 = 0.55f;
        v.ducct_loc_sigma_growth = 0.12f;
        v.ducct_pred_sigma0 = 1.20f;
        v.ducct_pred_sigma_growth = 0.16f;
        v.ducct_static_sigma = 0.45f;
        v.ducct_risk_weight = 2200.0f;
        v.ducct_hard_threshold = 0.50f;
        v.ducct_reject_cost = 32000.0f;
        v.ducct_survival_power = 0.70f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dra_mppi_soft";
        v.use_dra_risk = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.dra_mc_samples = 12;
        v.dra_robot_radius = 0.55f;
        v.dra_pred_sigma0 = 0.30f;
        v.dra_pred_sigma_growth = 0.045f;
        v.dra_mode_weight = 0.0f;
        v.dra_soft_weight = 500.0f;
        v.dra_hard_threshold = 0.85f;
        v.dra_reject_cost = 5000.0f;
        v.dra_survival_power = 0.15f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dra_mppi_hard";
        v.use_dra_risk = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.dra_mc_samples = 16;
        v.dra_robot_radius = 0.60f;
        v.dra_pred_sigma0 = 0.35f;
        v.dra_pred_sigma_growth = 0.06f;
        v.dra_mode_weight = 0.0f;
        v.dra_soft_weight = 900.0f;
        v.dra_hard_threshold = 0.55f;
        v.dra_reject_cost = 18000.0f;
        v.dra_survival_power = 0.30f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dra_mppi_multimodal";
        v.use_dra_risk = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.dra_mc_samples = 16;
        v.dra_robot_radius = 0.60f;
        v.dra_pred_sigma0 = 0.34f;
        v.dra_pred_sigma_growth = 0.055f;
        v.dra_mode_weight = 0.14f;
        v.dra_mode_lateral_offset = 1.35f;
        v.dra_soft_weight = 850.0f;
        v.dra_hard_threshold = 0.60f;
        v.dra_reject_cost = 16000.0f;
        v.dra_survival_power = 0.30f;
        v.sampling_lambda = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "bc_mppi";
        v.use_bc_safety_layer = true;
        v.bc_safe_margin = 1.0f;
        v.bc_prob_sigma = 0.80f;
        v.bc_probability_power = 1.25f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "bc_mppi_smooth";
        v.use_bc_safety_layer = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.bc_safe_margin = 1.0f;
        v.bc_prob_sigma = 0.80f;
        v.bc_probability_power = 1.25f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "bc_mppi_strict";
        v.use_bc_safety_layer = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.25f;
        v.bc_safe_margin = 1.4f;
        v.bc_prob_sigma = 0.55f;
        v.bc_probability_power = 2.0f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pr_mppi";
        v.use_parameter_robust_sampling = true;
        v.pr_param_particles = 3;
        v.pr_wheelbase_span = 0.45f;
        v.pr_max_speed_span = 0.18f;
        v.pr_max_steer_span = 0.20f;
        v.pr_worst_blend = 0.45f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pr_mppi_smooth";
        v.use_parameter_robust_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.pr_param_particles = 3;
        v.pr_wheelbase_span = 0.45f;
        v.pr_max_speed_span = 0.18f;
        v.pr_max_steer_span = 0.20f;
        v.pr_worst_blend = 0.45f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pr_mppi_cautious";
        v.use_parameter_robust_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.25f;
        v.pr_param_particles = 5;
        v.pr_wheelbase_span = 0.50f;
        v.pr_max_speed_span = 0.22f;
        v.pr_max_steer_span = 0.22f;
        v.pr_worst_blend = 0.75f;
        v.sampling_lambda = 7.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "cdf_mppi";
        v.use_cdf_guidance = true;
        v.t_horizon = 16;
        v.cdf_seed_blend = 0.25f;
        v.cdf_safe_margin = 3.0f;
        v.cdf_obs_cost = 1.2f;
        v.cdf_dyn_pull = 1.0f;
        v.cdf_dyn_cost = 0.6f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "cdf_lp_mppi";
        v.use_cdf_guidance = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.25f;
        v.t_horizon = 16;
        v.cdf_seed_blend = 0.25f;
        v.cdf_safe_margin = 3.0f;
        v.cdf_obs_cost = 1.2f;
        v.cdf_dyn_pull = 1.0f;
        v.cdf_dyn_cost = 0.6f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "cdf_mppi_one_step";
        v.use_cdf_guidance = true;
        v.t_horizon = 1;
        v.cdf_seed_blend = 0.90f;
        v.cdf_goal_pull = 1.2f;
        v.cdf_obs_pull = 4.0f;
        v.cdf_dyn_pull = 1.2f;
        v.cdf_safe_margin = 3.0f;
        v.cdf_obs_cost = 1.5f;
        v.cdf_dyn_cost = 0.8f;
        v.sampling_lambda = 5.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pa_mppi";
        v.use_pa_perception_cost = true;
        v.pa_safe_margin = 0.40f;
        v.pa_poi_weight = 120.0f;
        v.pa_occlusion_weight = 850.0f;
        v.pa_frontier_reward = 420.0f;
        v.pa_forward_occ_weight = 160.0f;
        v.pa_goal_gate = 3.0f;
        v.pa_activation = 0.06f;
        v.pa_ray_length = 8.0f;
        v.pa_score_cap = 350.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pa_mppi_soft";
        v.use_pa_perception_cost = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.pa_safe_margin = 0.30f;
        v.pa_poi_weight = 35.0f;
        v.pa_occlusion_weight = 160.0f;
        v.pa_frontier_reward = 90.0f;
        v.pa_forward_occ_weight = 35.0f;
        v.pa_goal_gate = 3.0f;
        v.pa_activation = 0.12f;
        v.pa_ray_length = 8.0f;
        v.pa_score_cap = 120.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pa_mppi_smooth";
        v.use_pa_perception_cost = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.22f;
        v.pa_safe_margin = 0.45f;
        v.pa_poi_weight = 140.0f;
        v.pa_occlusion_weight = 950.0f;
        v.pa_frontier_reward = 520.0f;
        v.pa_forward_occ_weight = 180.0f;
        v.pa_goal_gate = 3.0f;
        v.pa_activation = 0.06f;
        v.pa_ray_length = 8.0f;
        v.pa_score_cap = 300.0f;
        v.sampling_lambda = 6.5f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pa_mppi_frontier";
        v.use_pa_perception_cost = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.18f;
        v.pa_safe_margin = 0.30f;
        v.pa_poi_weight = 90.0f;
        v.pa_occlusion_weight = 650.0f;
        v.pa_frontier_reward = 900.0f;
        v.pa_forward_occ_weight = 120.0f;
        v.pa_goal_gate = 3.5f;
        v.pa_activation = 0.04f;
        v.pa_ray_length = 10.0f;
        v.pa_score_cap = 400.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ds_mppi";
        v.use_deterministic_sampling = true;
        v.ds_iterations = 2;
        v.ds_alpha = 0.35f;
        v.ds_noise_scale = 2.0f;
        v.ds_momentum = 0.0f;
        v.sampling_lambda = 4.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ds_mppi_smooth";
        v.use_deterministic_sampling = true;
        v.ds_iterations = 2;
        v.ds_alpha = 0.35f;
        v.ds_noise_scale = 2.0f;
        v.ds_momentum = 0.0f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ds_mppi_cov";
        v.use_deterministic_sampling = true;
        v.ds_iterations = 2;
        v.ds_alpha = 0.35f;
        v.ds_noise_scale = 2.0f;
        v.ds_momentum = 0.0f;
        v.ds_adapt_sigma = true;
        v.ds_sigma_blend = 0.35f;
        v.sampling_lambda = 4.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ds_mppi_cov_smooth";
        v.use_deterministic_sampling = true;
        v.ds_iterations = 2;
        v.ds_alpha = 0.35f;
        v.ds_noise_scale = 2.0f;
        v.ds_momentum = 0.0f;
        v.ds_adapt_sigma = true;
        v.ds_sigma_blend = 0.50f;
        v.ds_min_accel_sigma = 0.15f;
        v.ds_min_steer_sigma = 0.015f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ds_mppi_elite";
        v.use_deterministic_sampling = true;
        v.ds_iterations = 2;
        v.ds_alpha = 0.35f;
        v.ds_noise_scale = 2.0f;
        v.ds_momentum = 0.0f;
        v.ds_elite_update = true;
        v.ds_elite_count = 16;
        v.ds_elite_sigma_blend = 0.20f;
        v.sampling_lambda = 4.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "ds_mppi_elite_smooth";
        v.use_deterministic_sampling = true;
        v.ds_iterations = 2;
        v.ds_alpha = 0.35f;
        v.ds_noise_scale = 2.0f;
        v.ds_momentum = 0.0f;
        v.ds_elite_update = true;
        v.ds_elite_count = 32;
        v.ds_elite_sigma_blend = 0.35f;
        v.ds_min_accel_sigma = 0.15f;
        v.ds_min_steer_sigma = 0.015f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pi_mppi";
        v.use_projection_sampling = true;
        v.projection_passes = 2;
        v.projection_max_accel_delta = 1.20f;
        v.projection_max_steer_delta = 0.10f;
        v.projection_max_accel_ddelta = 1.00f;
        v.projection_max_steer_ddelta = 0.08f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "pi_mppi_smooth";
        v.use_projection_sampling = true;
        v.projection_passes = 4;
        v.projection_max_accel_delta = 0.60f;
        v.projection_max_steer_delta = 0.045f;
        v.projection_max_accel_ddelta = 0.40f;
        v.projection_max_steer_ddelta = 0.030f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "svg_mppi";
        v.use_svg_mode_guidance = true;
        v.svg_bandwidth = 24.0f;
        v.svg_mode_weight = 3.0f;
        v.svg_stride = 2;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "svg_mppi_smooth";
        v.use_svg_mode_guidance = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.svg_bandwidth = 24.0f;
        v.svg_mode_weight = 3.0f;
        v.svg_stride = 2;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "svg_mppi_strong";
        v.use_svg_mode_guidance = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.25f;
        v.svg_bandwidth = 16.0f;
        v.svg_mode_weight = 8.0f;
        v.svg_stride = 2;
        v.sampling_lambda = 5.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi";
        v.use_feedback = true;
        v.feedback_mode = 1;
        v.feedback_gain_scale = 1.0f;
        v.feedback_noise_accel = 0.9f;
        v.feedback_noise_steer = 0.10f;
        v.feedback_longitudinal_gain = 0.20f;
        v.feedback_speed_gain = 0.30f;
        v.feedback_lateral_gain = 0.28f;
        v.feedback_heading_gain = 0.42f;
        v.feedback_setpoint_blend = 0.0f;
        v.feedback_q_position = 1.8f;
        v.feedback_q_heading = 1.2f;
        v.feedback_q_speed = 1.0f;
        v.feedback_r_accel = 1.4f;
        v.feedback_r_steer = 1.1f;
        v.feedback_terminal_scale = 4.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_sens";
        v.use_feedback = true;
        v.feedback_mode = 2;
        v.feedback_gain_scale = 0.60f;
        v.feedback_noise_accel = 0.80f;
        v.feedback_noise_steer = 0.09f;
        v.feedback_longitudinal_gain = 0.12f;
        v.feedback_speed_gain = 0.18f;
        v.feedback_lateral_gain = 0.16f;
        v.feedback_heading_gain = 0.24f;
        v.feedback_setpoint_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_cov";
        v.use_feedback = true;
        v.feedback_mode = 3;
        v.feedback_gain_scale = 0.70f;
        v.feedback_noise_accel = 0.65f;
        v.feedback_noise_steer = 0.07f;
        v.feedback_longitudinal_gain = 0.18f;
        v.feedback_speed_gain = 0.24f;
        v.feedback_lateral_gain = 0.28f;
        v.feedback_heading_gain = 0.38f;
        v.feedback_setpoint_blend = 0.10f;
        v.feedback_cov_regularization = 0.20f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_fused";
        v.use_feedback = true;
        v.feedback_mode = 4;
        v.feedback_passes = 2;
        v.feedback_gain_scale = 0.75f;
        v.feedback_noise_accel = 0.60f;
        v.feedback_noise_steer = 0.07f;
        v.feedback_longitudinal_gain = 0.16f;
        v.feedback_speed_gain = 0.22f;
        v.feedback_lateral_gain = 0.24f;
        v.feedback_heading_gain = 0.34f;
        v.feedback_setpoint_blend = 0.15f;
        v.feedback_q_position = 1.6f;
        v.feedback_q_heading = 1.1f;
        v.feedback_q_speed = 0.9f;
        v.feedback_r_accel = 1.3f;
        v.feedback_r_steer = 1.0f;
        v.feedback_terminal_scale = 3.5f;
        v.feedback_cov_regularization = 0.18f;
        v.feedback_cov_blend = 0.75f;
        v.feedback_lqr_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_hf";
        v.use_feedback = true;
        v.feedback_mode = 6;
        v.replan_stride = 2;
        v.feedback_gain_scale = 0.55f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.06f;
        v.feedback_speed_gain = 0.08f;
        v.feedback_lateral_gain = 0.10f;
        v.feedback_heading_gain = 0.14f;
        v.feedback_setpoint_blend = 0.30f;
        v.feedback_q_position = 1.6f;
        v.feedback_q_heading = 1.1f;
        v.feedback_q_speed = 0.9f;
        v.feedback_r_accel = 1.3f;
        v.feedback_r_steer = 1.0f;
        v.feedback_terminal_scale = 3.5f;
        v.feedback_cov_regularization = 0.18f;
        v.feedback_cov_blend = 0.75f;
        v.feedback_lqr_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_ref";
        v.use_feedback = true;
        v.feedback_mode = 7;
        v.feedback_gain_scale = 1.0f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.0f;
        v.feedback_speed_gain = 0.0f;
        v.feedback_lateral_gain = 0.0f;
        v.feedback_heading_gain = 0.0f;
        v.feedback_setpoint_blend = 0.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_release";
        v.use_feedback = true;
        v.feedback_mode = 7;
        v.sampling_lambda = 1.0f / 5.0f;
        v.feedback_gain_scale = 1.0f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.0f;
        v.feedback_speed_gain = 0.0f;
        v.feedback_lateral_gain = 0.0f;
        v.feedback_heading_gain = 0.0f;
        v.feedback_setpoint_blend = 0.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_faithful";
        v.use_feedback = true;
        v.feedback_mode = 8;
        v.replan_stride = 2;
        v.feedback_gain_scale = 1.0f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.0f;
        v.feedback_speed_gain = 0.0f;
        v.feedback_lateral_gain = 0.0f;
        v.feedback_heading_gain = 0.0f;
        v.feedback_setpoint_blend = 0.30f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_paper";
        v.use_feedback = true;
        v.feedback_mode = 9;
        v.replan_stride = 1;
        v.feedback_gain_scale = 0.80f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.0f;
        v.feedback_speed_gain = 0.0f;
        v.feedback_lateral_gain = 0.0f;
        v.feedback_heading_gain = 0.0f;
        v.feedback_setpoint_blend = 0.15f;
        v.feedback_q_position = 1.6f;
        v.feedback_q_heading = 1.1f;
        v.feedback_q_speed = 0.9f;
        v.feedback_r_accel = 1.3f;
        v.feedback_r_steer = 1.0f;
        v.feedback_terminal_scale = 3.5f;
        v.feedback_cov_regularization = 0.15f;
        v.feedback_cov_blend = 0.80f;
        v.feedback_lqr_blend = 0.35f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "feedback_mppi_strong";
        v.use_feedback = true;
        v.feedback_mode = 10;
        v.replan_stride = 1;
        v.feedback_gain_scale = 0.85f;
        v.feedback_noise_accel = 0.0f;
        v.feedback_noise_steer = 0.0f;
        v.feedback_longitudinal_gain = 0.0f;
        v.feedback_speed_gain = 0.0f;
        v.feedback_lateral_gain = 0.0f;
        v.feedback_heading_gain = 0.0f;
        v.feedback_setpoint_blend = 0.15f;
        v.feedback_q_position = 1.6f;
        v.feedback_q_heading = 1.1f;
        v.feedback_q_speed = 0.9f;
        v.feedback_r_accel = 1.3f;
        v.feedback_r_steer = 1.0f;
        v.feedback_terminal_scale = 3.5f;
        v.feedback_ref_blend = 0.75f;
        v.feedback_cov_regularization = 0.15f;
        v.feedback_cov_blend = 0.55f;
        v.feedback_lqr_blend = 0.30f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "grad_only_3";
        v.use_sampling = false;
        v.use_gradient = true;
        v.grad_steps = 3;
        v.alpha = 0.004f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_1";
        v.use_gradient = true;
        v.grad_steps = 1;
        v.alpha = 0.010f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.alpha = 0.006f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3_early1";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.grad_update_horizon = 1;
        v.alpha = 0.006f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3_early2";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.grad_update_horizon = 2;
        v.alpha = 0.006f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3_early4";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.grad_update_horizon = 4;
        v.alpha = 0.006f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3_early8";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.grad_update_horizon = 8;
        v.alpha = 0.006f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_3_early16";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.grad_update_horizon = 16;
        v.alpha = 0.006f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "diff_mppi_adaptive";
        v.use_gradient = true;
        v.grad_steps = 3;
        v.alpha = 0.006f;
        v.grad_skip_threshold = 8.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "step_mppi";
        v.use_learned_sampling = true;
        v.mlp_lr = 0.001f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "step_mppi_fast";
        v.use_learned_sampling = true;
        v.mlp_lr = 0.025f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "step_mppi_smooth";
        v.use_learned_sampling = true;
        v.use_low_pass_sampling = true;
        v.lp_alpha = 0.20f;
        v.mlp_lr = 0.020f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "step_mppi_adaptive";
        v.use_learned_sampling = true;
        v.use_learned_sigma = true;
        v.mlp_lr = 0.025f;
        v.learned_sigma_lr = 0.12f;
        v.learned_min_accel_sigma = 0.25f;
        v.learned_min_steer_sigma = 0.025f;
        v.learned_max_accel_sigma = 3.0f;
        v.learned_max_steer_sigma = 0.30f;
        v.sampling_lambda = 6.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "step_mppi_single";
        v.use_learned_sampling = true;
        v.use_learned_sigma = true;
        v.t_horizon = 1;
        v.mlp_lr = 0.10f;
        v.learned_sigma_lr = 0.20f;
        v.learned_min_accel_sigma = 0.40f;
        v.learned_min_steer_sigma = 0.050f;
        v.learned_max_accel_sigma = 3.5f;
        v.learned_max_steer_sigma = 0.35f;
        v.sampling_lambda = 5.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "soppi";
        v.use_soppi_sampling = true;
        v.soppi_svgd_iters = 1;
        v.soppi_step_size = 0.045f;
        v.soppi_bandwidth = 2.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "soppi_fast";
        v.use_soppi_sampling = true;
        v.soppi_svgd_iters = 1;
        v.soppi_step_size = 0.075f;
        v.soppi_bandwidth = 2.0f;
        v.soppi_neighbor_count = 32;
        variants.push_back(v);
    }
    // DWA variants: discrete dynamic-window search over (accel, steer). Costs are
    // tuned to roughly match MPPI's cost weights so the comparison stays apples
    // to apples. dwa_med is the headline variant; fast/fine bracket it on cost.
    {
        PlannerVariant v;
        v.name = "dwa_fast";
        v.use_sampling = false;
        v.planner_kind = 1;
        v.dwa_n_accel = 5;
        v.dwa_n_steer = 9;
        v.dwa_predict_steps = 12;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dwa_med";
        v.use_sampling = false;
        v.planner_kind = 1;
        v.dwa_n_accel = 9;
        v.dwa_n_steer = 13;
        v.dwa_predict_steps = 20;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "dwa_fine";
        v.use_sampling = false;
        v.planner_kind = 1;
        v.dwa_n_accel = 13;
        v.dwa_n_steer = 21;
        v.dwa_predict_steps = 25;
        variants.push_back(v);
    }
    // Long-horizon DWA variant. Added during Day 3 of the topology-bench
    // roadmap: dynamic_bottleneck's slow obstacle occupies the gate for
    // ~6 s, exceeding dwa_med/fine's 2-2.5 s lookahead. With T=60 (6 s)
    // DWA can see past the obstacle and find a slowdown plan. Otherwise
    // identical grid resolution to dwa_med.
    {
        PlannerVariant v;
        v.name = "dwa_long";
        v.use_sampling = false;
        v.planner_kind = 1;
        v.dwa_n_accel = 9;
        v.dwa_n_steer = 13;
        v.dwa_predict_steps = 60;
        variants.push_back(v);
    }
    // STOMP variants: same rollout kernel as MPPI but STOMP-style normalised
    // weights and a smoothness projection (3-tap moving average) on the
    // updated nominal. iters > 1 == multiple inner cost-weighted updates per
    // controller call, smoothing_passes controls projection strength.
    {
        PlannerVariant v;
        v.name = "stomp_1";
        v.use_sampling = false;
        v.planner_kind = 2;
        v.stomp_iterations = 1;
        v.stomp_smoothing_passes = 1;
        v.stomp_h = 10.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "stomp_2";
        v.use_sampling = false;
        v.planner_kind = 2;
        v.stomp_iterations = 2;
        v.stomp_smoothing_passes = 1;
        v.stomp_h = 10.0f;
        variants.push_back(v);
    }
    {
        PlannerVariant v;
        v.name = "stomp_3_smooth";
        v.use_sampling = false;
        v.planner_kind = 2;
        v.stomp_iterations = 3;
        v.stomp_smoothing_passes = 2;
        v.stomp_h = 10.0f;
        variants.push_back(v);
    }
    // Hybrid A* + Pure Pursuit: plan once against static obstacles, track
    // with pure pursuit. The "blind global planner" baseline -- dynamic
    // obstacles are ignored in the search by design so the paradigm gap
    // versus local replanners is visible.
    {
        PlannerVariant v;
        v.name = "hybrid_astar_pp";
        v.use_sampling = false;
        v.planner_kind = 3;
        variants.push_back(v);
    }
    // Hybrid A* + DWA hybrid: global path (static-only) shapes the local DWA
    // cost. Closes the paradigm gap of the pure_pursuit baseline because the
    // DWA local cost can still react to dynamic obstacles while following the
    // pre-planned static path.
    {
        PlannerVariant v;
        v.name = "hybrid_astar_dwa";
        v.use_sampling = false;
        v.planner_kind = 4;
        variants.push_back(v);
    }
    // Long-horizon Hybrid A* + DWA. Same as hybrid_astar_dwa but with
    // dwa_predict_steps=60 so the per-step DWA local sees ~6 s ahead
    // -- enough to clear the dynamic_bottleneck scene where the slow
    // obstacle occupies the gate over the short-horizon controller's
    // entire prediction window.
    {
        PlannerVariant v;
        v.name = "hybrid_astar_dwa_long";
        v.use_sampling = false;
        v.planner_kind = 4;
        v.dwa_predict_steps = 60;
        variants.push_back(v);
    }
    // Hybrid A* + Pure Pursuit with DYNAMIC OBSTACLES included in the
    // search. Same tracker as hybrid_astar_pp; the difference is whether
    // the global planner is blind to or aware of moving obstacles.
    {
        PlannerVariant v;
        v.name = "hybrid_astar_dyn_pp";
        v.use_sampling = false;
        v.planner_kind = 5;
        variants.push_back(v);
    }
    // Hybrid A* + MPPI hybrid: static-only global path + per-step MPPI
    // sampling pipeline whose cost replaces goal-distance/heading with
    // path-follow terms. Parallel to hybrid_astar_dwa; closes the same
    // paradigm gap via cost-weighted noise rather than discrete grid
    // search.
    {
        PlannerVariant v;
        v.name = "hybrid_astar_mppi";
        v.use_sampling = false;
        v.planner_kind = 6;
        variants.push_back(v);
    }
    // Long-horizon variant of hybrid_astar_mppi (T=60 = 6 s lookahead).
    // Targets long-occupancy timing scenes (dynamic_bottleneck) where the
    // default T=30 reactive horizon cannot see past the obstacle. Kept as
    // a separate variant because T=60 was empirically regressive on the
    // open-dynamic 30-cell suite for the averaging-style MPPI controller
    // (DWA's argmin selection is unaffected, so hybrid_astar_dwa_long is
    // safe at T=60). See docs/topology_bench_day4_findings.md.
    {
        PlannerVariant v;
        v.name = "hybrid_astar_mppi_long";
        v.use_sampling = false;
        v.planner_kind = 6;
        v.t_horizon = 60;
        variants.push_back(v);
    }

    if (!planner_names.empty()) {
        vector<PlannerVariant> filtered;
        for (const auto& wanted : planner_names) {
            auto it = find_if(variants.begin(), variants.end(),
                              [&](const PlannerVariant& v) { return v.name == wanted; });
            if (it == variants.end()) {
                fprintf(stderr, "Unknown planner: %s\n", wanted.c_str());
                return 1;
            }
            filtered.push_back(*it);
        }
        variants.swap(filtered);
    }

    // Apply parameter overrides (used by multi-param tuning script)
    for (auto& v : variants) {
        if (override_feedback_gain_scale >= 0.0f && v.use_feedback)
            v.feedback_gain_scale = override_feedback_gain_scale;
        if (override_feedback_ref_blend >= 0.0f && v.use_feedback)
            v.feedback_ref_blend = override_feedback_ref_blend;
        if (override_feedback_cov_blend >= 0.0f && v.use_feedback)
            v.feedback_cov_blend = override_feedback_cov_blend;
        if (override_feedback_lqr_blend >= 0.0f && v.use_feedback)
            v.feedback_lqr_blend = override_feedback_lqr_blend;
        if (override_feedback_setpoint_blend >= 0.0f && v.use_feedback)
            v.feedback_setpoint_blend = override_feedback_setpoint_blend;
        if (override_feedback_cov_regularization >= 0.0f && v.use_feedback)
            v.feedback_cov_regularization = override_feedback_cov_regularization;
        if (override_grad_steps >= 0 && v.use_gradient)
            v.grad_steps = override_grad_steps;
        if (override_grad_update_horizon >= 0 && v.use_gradient)
            v.grad_update_horizon = override_grad_update_horizon;
        if (override_alpha >= 0.0f && v.use_gradient)
            v.alpha = override_alpha;
        if (override_sampling_lambda >= 0.0f)
            v.sampling_lambda = override_sampling_lambda;
        if (override_mlp_lr >= 0.0f && v.use_learned_sampling)
            v.mlp_lr = override_mlp_lr;
        if (override_lp_alpha >= 0.0f && v.use_low_pass_sampling)
            v.lp_alpha = override_lp_alpha;
        if (override_ds_iterations > 0 && v.use_deterministic_sampling)
            v.ds_iterations = override_ds_iterations;
        if (override_ds_alpha >= 0.0f && v.use_deterministic_sampling)
            v.ds_alpha = override_ds_alpha;
        if (override_ds_noise_scale >= 0.0f && v.use_deterministic_sampling)
            v.ds_noise_scale = override_ds_noise_scale;
        if (override_ds_momentum >= 0.0f && v.use_deterministic_sampling)
            v.ds_momentum = override_ds_momentum;
        if (override_ds_stride > 0 && v.use_deterministic_sampling)
            v.ds_stride = override_ds_stride;
        if (override_projection_passes > 0 && v.use_projection_sampling)
            v.projection_passes = override_projection_passes;
        if (override_projection_accel_delta >= 0.0f && v.use_projection_sampling)
            v.projection_max_accel_delta = override_projection_accel_delta;
        if (override_projection_steer_delta >= 0.0f && v.use_projection_sampling)
            v.projection_max_steer_delta = override_projection_steer_delta;
        if (override_projection_accel_ddelta >= 0.0f && v.use_projection_sampling)
            v.projection_max_accel_ddelta = override_projection_accel_ddelta;
        if (override_projection_steer_ddelta >= 0.0f && v.use_projection_sampling)
            v.projection_max_steer_ddelta = override_projection_steer_ddelta;
        if (override_soppi_iters >= 0 && v.use_soppi_sampling)
            v.soppi_svgd_iters = override_soppi_iters;
        if (override_soppi_neighbor_count >= 0 && v.use_soppi_sampling)
            v.soppi_neighbor_count = override_soppi_neighbor_count;
        if (override_soppi_step_size >= 0.0f && v.use_soppi_sampling)
            v.soppi_step_size = override_soppi_step_size;
        if (override_soppi_bandwidth >= 0.0f && v.use_soppi_sampling)
            v.soppi_bandwidth = override_soppi_bandwidth;
        if (override_dwa_w_goal >= 0.0f && v.planner_kind == 1)
            v.dwa_w_goal = override_dwa_w_goal;
        if (override_dwa_w_speed >= 0.0f && v.planner_kind == 1)
            v.dwa_w_speed = override_dwa_w_speed;
        if (override_dwa_w_obs >= 0.0f && v.planner_kind == 1)
            v.dwa_w_obs = override_dwa_w_obs;
        if (override_dwa_w_heading >= 0.0f && v.planner_kind == 1)
            v.dwa_w_heading = override_dwa_w_heading;
        if (override_dwa_w_terminal >= 0.0f && v.planner_kind == 1)
            v.dwa_w_terminal = override_dwa_w_terminal;
    }

    if (override_dyn_speed_scale >= 0.0f || override_dyn_radius_scale >= 0.0f) {
        for (auto& sc : scenarios) {
            for (int i = 0; i < sc.n_dyn_obs; i++) {
                if (override_dyn_speed_scale >= 0.0f) {
                    sc.dynamic_obstacles[i].vx *= override_dyn_speed_scale;
                    sc.dynamic_obstacles[i].vy *= override_dyn_speed_scale;
                }
                if (override_dyn_radius_scale >= 0.0f) {
                    sc.dynamic_obstacles[i].r *= override_dyn_radius_scale;
                }
            }
        }
    }

    if (k_values.empty()) k_values = quick ? vector<int>{1024, 4096} : vector<int>{1024, 2048, 4096};
    if (seed_count <= 0) seed_count = quick ? 2 : 4;

    vector<EpisodeMetrics> rows;
    rows.reserve(scenarios.size() * variants.size() * k_values.size() * seed_count);
    vector<TraceRow> trace_rows;
    vector<TrajectoryRow> trajectory_rows;
    bool trace_enabled = !trace_csv_path.empty();
    bool trajectory_enabled = !trajectory_csv_path.empty();
    if (trace_enabled && trace_max_steps <= 0) trace_max_steps = 64;

    for (size_t si = 0; si < scenarios.size(); si++) {
        const Scenario& scenario = scenarios[si];
        CUDA_CHECK(cudaMemcpyToSymbol(d_obstacles_bench, scenario.obstacles, sizeof(Obstacle) * scenario.n_obs));
        if (scenario.n_dyn_obs > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_dynamic_obstacles_bench, scenario.dynamic_obstacles,
                                          sizeof(DynamicObstacle) * scenario.n_dyn_obs));
        }
        for (int k_samples : k_values) {
            for (size_t vi = 0; vi < variants.size(); vi++) {
                const PlannerVariant& variant = variants[vi];
                for (int seed = 0; seed < seed_count; seed++) {
                    int run_seed = static_cast<int>(1000 + si * 100 + vi * 20 + seed * 7 + k_samples);
                    Scenario eval_scenario = instantiate_eval_scenario(scenario, run_seed);
                    // Resolve t_horizon with precedence: CLI override >
                    // per-variant default > global DEFAULT_T_HORIZON. The
                    // CLI wins so users can sweep T explicitly; the
                    // per-variant value lets registrations pick a horizon
                    // appropriate to the planner without requiring the
                    // user to remember a flag.
                    int t_horizon_to_use = DEFAULT_T_HORIZON;
                    if (variant.t_horizon > 0) t_horizon_to_use = variant.t_horizon;
                    if (override_t_horizon > 0) t_horizon_to_use = override_t_horizon;
                    EpisodeRunner runner(
                        variant, scenario, eval_scenario, k_samples, t_horizon_to_use, run_seed,
                        trace_enabled ? &trace_rows : nullptr, trace_max_steps,
                        trajectory_enabled ? &trajectory_rows : nullptr);
                    EpisodeMetrics metrics = runner.run();
                    rows.push_back(metrics);
                    printf("[%s] %s K=%d seed=%d success=%d steps=%d final_dist=%.2f avg_ms=%.2f collisions=%d\n",
                           scenario.name.c_str(), variant.name.c_str(), k_samples, seed,
                           metrics.success, metrics.steps, metrics.final_distance,
                           metrics.avg_control_ms, metrics.collisions);
                }
            }
        }
    }

    write_csv(rows, csv_path);
    if (trace_enabled) write_trace_csv(trace_rows, trace_csv_path);
    if (trajectory_enabled) write_trajectory_csv(trajectory_rows, trajectory_csv_path);
    print_summary(rows);
    cout << "CSV saved to " << csv_path << endl;
    if (trace_enabled) cout << "Trace CSV saved to " << trace_csv_path << endl;
    if (trajectory_enabled) cout << "Trajectory CSV saved to " << trajectory_csv_path << endl;
    return 0;
}
