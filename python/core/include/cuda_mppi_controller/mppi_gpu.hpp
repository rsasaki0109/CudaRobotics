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
  float follow_lookahead = 1.0f;     // [m] how far ahead that point is
  float costmap_weight = 3.0f;       // stage costmap cost (normalized 0..1, squared)
  // Optional truncated distance-field obstacle critic. Disabled by default so
  // existing costmap-only tuning stays unchanged.
  float distance_field_weight = 0.0f;  // stage clearance cost from nearest obstacle
  float distance_field_cutoff = 0.75f; // [m] no clearance penalty beyond this distance
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
};

struct MppiResult
{
  float v = 0.0f;
  float vy = 0.0f;
  float w = 0.0f;
  float best_cost = 0.0f;   // min sampled trajectory cost (collision diagnosis)
  bool all_colliding = false;
  bool retreating = false;   // true when command is a recovery back-out action
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
