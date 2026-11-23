// GPU MPPI optimizer core. Pure C++ interface — no ROS, no CUDA headers here.
// The CUDA implementation lives in src/mppi_gpu.cu behind a PIMPL.
#ifndef CUDA_MPPI_CONTROLLER__MPPI_GPU_HPP_
#define CUDA_MPPI_CONTROLLER__MPPI_GPU_HPP_

#include <memory>

namespace cuda_mppi_controller
{

enum class MotionModel
{
  DiffDrive,
  Ackermann,   // curvature-limited: |wz| <= |vx| / min_turning_r
  Omni         // adds lateral velocity vy
};

// A predicted moving obstacle in the costmap frame: a disc of `radius` that
// moves with constant velocity (vx, vy) from (x, y) at the time of compute().
struct DynamicObstacle
{
  float x = 0.0f;
  float y = 0.0f;
  float vx = 0.0f;
  float vy = 0.0f;
  float radius = 0.0f;
};

constexpr int kMaxDynamicObstacles = 64;

struct MppiParams
{
  int batch_size = 2048;        // K: sampled trajectories (1 CUDA thread each)
  int time_steps = 56;          // T: horizon length
  float model_dt = 0.05f;       // [s] integration step
  int iteration_count = 1;      // optimizer iterations per control cycle

  MotionModel motion_model = MotionModel::DiffDrive;

  // control limits
  float v_max = 0.5f;           // [m/s] forward
  float v_min = -0.35f;         // [m/s]
  float vy_max = 0.5f;          // [m/s] lateral (Omni only)
  float w_max = 1.9f;           // [rad/s]
  float min_turning_r = 0.2f;   // [m] (Ackermann only)

  // sampling noise std
  float v_std = 0.2f;
  float vy_std = 0.2f;          // (Omni only)
  float w_std = 0.4f;

  float lambda = 0.12f;         // softmin temperature

  // cost weights
  float goal_weight = 20.0f;         // terminal distance to local goal (linear)
  float goal_yaw_weight = 3.0f;      // terminal yaw error (active near final goal)
  float path_weight = 10.0f;         // stage lateral distance² to reference path
  float path_follow_weight = 5.0f;   // stage distance to a point ahead on the path
  float path_angle_weight = 0.25f;    // stage yaw error to the local path tangent
  float curvature_speed_weight = 0.0f; // optional overspeed penalty near sharp path bends
  float curvature_speed_min = 0.18f;   // [m/s] floor for curvature-limited target speed
  float follow_lookahead = 1.0f;     // [m] how far ahead that point is
  float costmap_weight = 3.0f;       // stage costmap cost (normalized 0..1, squared)
  // Optional truncated distance-field obstacle critic. Disabled by default so
  // existing costmap-only tuning stays unchanged.
  float distance_field_weight = 0.0f;  // stage clearance cost from nearest obstacle
  float distance_field_cutoff = 0.75f; // [m] no clearance penalty beyond this distance
  // Optional predicted dynamic-obstacle critic (see setDynamicObstacles).
  // Disabled by default. At rollout time tau = (t + 1) * model_dt the
  // clearance to obstacle i is
  //   c = |q(tau) - (p_i + v_i tau)| - r_i - robot_radius - growth * tau;
  // c < 0 adds collision_cost, 0 <= c < margin adds weight * (margin - c)^2 * dt.
  float dynamic_obstacle_weight = 0.0f;
  float dynamic_obstacle_margin = 0.4f;        // [m]
  float dynamic_obstacle_robot_radius = 0.25f; // [m] disc approximation of the robot
  float dynamic_obstacle_growth = 0.1f;        // [m/s] prediction uncertainty growth
  // Tolerated current overlap for tracked (noisy) obstacles. With a value
  // a > 0, an existing penetration p of disc i at the start pose (up to a) is
  // allowed: a rollout collides only when gap < -min(p, a), where
  // gap = |q(tau) - (p_i + v_i tau)| - r_i - robot_radius, and the growth
  // term enters only the soft cost. This keeps a robot that a tracking error
  // places just inside a disc from seeing every rollout collide. 0 (default)
  // keeps the hard check above.
  float dynamic_obstacle_overlap_allowance = 0.0f;  // [m]
  // Yielding obstacles. With a distance d >= 0, an obstacle approaching the
  // robot is predicted to stop where it first comes within d of the robot's
  // disc at the start pose: its constant-velocity prediction, and its growth,
  // run only until
  //   t_stop = min { t >= 0 : |p_i + v_i t - start| <= r_i + robot_radius + d }
  // (0 if it is already that close). Obstacles that move away or pass by are
  // unchanged. This models people who stop or step aside rather than walk into
  // a standing robot. Without it, a person walking towards the robot at close
  // range makes every rollout collide, standing still included, and the robot
  // can only back out. Negative (default) disables it.
  float dynamic_obstacle_yield_distance = -1.0f;  // [m]
  float smoothness_weight = 0.2f;    // stage (du)^2 between consecutive steps
  float backward_weight = 0.5f;      // stage penalty on v < 0
  // stage penalty on (v_max - v): counters the softmin saturation bias
  // (clamped samples average below v_max) so the robot actually cruises
  // at v_max on open path segments, like nav2's PreferForwardCritic
  float speed_weight = 3.0f;
  // stage penalty on wz²: damps the heading random walk that the noisy
  // weighted average otherwise accumulates
  float angular_weight = 0.5f;
  float collision_cost = 1.0e6f;     // added per step inside lethal/inscribed cells

  float yaw_goal_activation_dist = 0.5f;  // [m] enable yaw cost within this range of final goal
  unsigned char lethal_threshold = 253;   // costmap value >= this counts as collision

  // footprint collision checking (instead of the point-robot threshold).
  // Requires an inflation layer: the polygon edge check only runs on cells
  // with non-zero inflated cost.
  bool consider_footprint = false;

  // Recovery command when every sampled trajectory collides: back out by
  // reversing the most recent valid sequence instead of immediately failing.
  bool enable_retreat = true;
  float retreat_scale = 0.5f;
  // Reverse speed [m/s] the back-out may use even when v_min >= 0; otherwise a
  // forward-only robot can only turn in place. 0 keeps the v_min clamp.
  float retreat_reverse_speed = 0.0f;
  // Check the back-out command before using it, by holding it for
  // retreat_check_horizon seconds (0 disables both checks). If it would
  // - bring the robot within dynamic_obstacle_margin of a predicted dynamic
  //   obstacle and closer to it than standing still (when the critic is
  //   active), or
  // - enter a lethal costmap cell costlier than the start cell,
  // its translation is dropped (turning in place is kept) and
  // MppiResult::retreat_blocked is set.
  bool retreat_avoid_dynamic_obstacles = true;
  bool retreat_avoid_lethal_cells = true;
  float retreat_check_horizon = 1.0f;  // [s]
  // When every rollout collides but at least one stays collision-free for
  // this long [s], follow the optimizer (steer away) instead of backing out;
  // MppiResult::evading is set. 0 (default) always backs out.
  float evade_min_collision_time = 0.0f;
};

struct MppiResult
{
  float v = 0.0f;
  float vy = 0.0f;
  float w = 0.0f;
  float best_cost = 0.0f;   // min sampled trajectory cost (collision diagnosis)
  float mean_cost = 0.0f;   // mean sampled trajectory cost from the final iteration
  int sampled_rollouts = 0;
  int valid_rollouts = 0;    // sampled trajectories with no collision-cost hit
  float valid_rollout_ratio = 0.0f;
  bool all_colliding = false;
  bool retreating = false;   // true when command is a recovery back-out action
  bool retreat_blocked = false;  // back-out translation dropped for a dynamic obstacle
  bool evading = false;  // all rollouts collide, but the command steers on (see evade_min_collision_time)
  float time_to_collision = 0.0f;  // [s] latest first collision over the rollouts (set when evading is checked)
};

class MppiGpu
{
public:
  explicit MppiGpu(const MppiParams & params);
  ~MppiGpu();

  MppiGpu(const MppiGpu &) = delete;
  MppiGpu & operator=(const MppiGpu &) = delete;

  // Clear the warm-started nominal control sequence.
  void reset();

  void setSpeedLimit(float v_max);

  // Replace the predicted dynamic obstacles used by the following compute
  // calls (count == 0 clears them). At most kMaxDynamicObstacles; non-finite
  // values or a negative radius throw std::invalid_argument. The obstacles
  // only affect the cost when MppiParams::dynamic_obstacle_weight > 0.
  void setDynamicObstacles(const DynamicObstacle * obstacles, int count);
  int dynamicObstacleCount() const;

  // All poses are in the costmap global frame.
  // costmap: row-major uchar grid (size_x * size_y), nullptr allowed (free space).
  // path_xy: [path_len * 2] reference path points; goal is the local goal
  //          (window end), goal_is_final enables the terminal yaw cost.
  // footprint_xy: [footprint_len * 2] polygon in base frame, used when
  //          consider_footprint is set (max 16 vertices).
  MppiResult compute(
    float robot_x, float robot_y, float robot_yaw,
    const unsigned char * costmap, int size_x, int size_y,
    float origin_x, float origin_y, float resolution,
    const float * path_xy, int path_len,
    float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
    const float * footprint_xy = nullptr, int footprint_len = 0);

  // Variant for callers that already own a CUDA device costmap buffer, for
  // example a PyTorch/CuPy tensor consumed through DLPack. The pointer must
  // remain valid until this synchronous call returns.
  MppiResult computeWithDeviceCostmap(
    float robot_x, float robot_y, float robot_yaw,
    const unsigned char * device_costmap, int size_x, int size_y,
    float origin_x, float origin_y, float resolution,
    const float * path_xy, int path_len,
    float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
    const float * footprint_xy = nullptr, int footprint_len = 0);

private:
  MppiResult computeInternal(
    float robot_x, float robot_y, float robot_yaw,
    const unsigned char * costmap, int size_x, int size_y, bool costmap_is_device,
    float origin_x, float origin_y, float resolution,
    const float * path_xy, int path_len,
    float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
    const float * footprint_xy = nullptr, int footprint_len = 0);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cuda_mppi_controller

#endif  // CUDA_MPPI_CONTROLLER__MPPI_GPU_HPP_
