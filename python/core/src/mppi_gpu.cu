// CUDA implementation of the MPPI optimizer.
// Parallel pattern: 1 thread = 1 sampled trajectory (rollout + cost),
// matching the cudabot convention used across this repository.
#include "cuda_mppi_controller/mppi_gpu.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace cuda_mppi_controller
{

namespace
{

constexpr int kCtrlDim = 3;       // (vx, vy, wz); vy stays 0 outside Omni
constexpr int kMaxPathPoints = 256;
constexpr int kMaxFootprint = 16;

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
      throw std::runtime_error( \
              std::string("CUDA error: ") + cudaGetErrorString(err__) + \
              " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
  } while (0)

struct DeviceParams
{
  int K;
  int T;
  float dt;
  int motion_model;   // 0 DiffDrive, 1 Ackermann, 2 Omni
  float v_max, v_min, vy_max, w_max;
  float min_turning_r;
  float v_std, vy_std, w_std;
  float goal_w, goal_yaw_w, path_w, follow_w, path_angle_w, curvature_speed_w, costmap_w;
  float smooth_w, backward_w;
  float curvature_speed_min;
  float distance_field_w, distance_field_cutoff;
  float speed_w, angular_w;
  int follow_offset;   // path index offset for the path-follow cost
  float collision_cost;
  float yaw_activation_dist;
  unsigned char lethal_threshold;

  // costmap (size_x == 0 means no costmap -> free space)
  int size_x, size_y;
  float origin_x, origin_y, resolution;

  int path_len;
  float goal_x, goal_y, goal_yaw;
  int goal_is_final;

  // footprint polygon, base frame (footprint_len == 0 -> point robot)
  int footprint_len;
  float fp[kMaxFootprint * 2];

  float start_x, start_y, start_yaw;
};

__device__ __forceinline__ float wrap_angle(float a)
{
  return atan2f(sinf(a), cosf(a));
}

__device__ __forceinline__ unsigned char cell_cost(
  const unsigned char * __restrict__ costmap, const DeviceParams & p,
  float x, float y)
{
  int mx = __float2int_rd((x - p.origin_x) / p.resolution);
  int my = __float2int_rd((y - p.origin_y) / p.resolution);
  if (mx < 0 || mx >= p.size_x || my < 0 || my >= p.size_y) {
    return 0;  // out of the local costmap -> treat as free
  }
  return costmap[my * p.size_x + mx];
}

__device__ __forceinline__ bool is_obstacle_distance_cell(
  unsigned char cost, unsigned char lethal_threshold)
{
  return cost != 255 && cost >= lethal_threshold;
}

__device__ __forceinline__ float distance_field_value(
  const float * __restrict__ distance_field, const DeviceParams & p,
  float x, float y)
{
  int mx = __float2int_rd((x - p.origin_x) / p.resolution);
  int my = __float2int_rd((y - p.origin_y) / p.resolution);
  if (mx < 0 || mx >= p.size_x || my < 0 || my >= p.size_y) {
    return p.distance_field_cutoff;
  }
  return distance_field[my * p.size_x + mx];
}

__global__ void build_distance_field_kernel(
  const unsigned char * __restrict__ costmap,
  float * __restrict__ distance_field,
  int size_x, int size_y,
  float resolution, float cutoff,
  unsigned char lethal_threshold)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cell_count = size_x * size_y;
  if (idx >= cell_count) {
    return;
  }

  const unsigned char center = costmap[idx];
  if (is_obstacle_distance_cell(center, lethal_threshold)) {
    distance_field[idx] = 0.0f;
    return;
  }

  const int mx = idx % size_x;
  const int my = idx / size_x;
  const int radius_cells = max(1, __float2int_ru(cutoff / resolution));
  const float cutoff_cells = cutoff / resolution;
  float best_cells2 = cutoff_cells * cutoff_cells;

  for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
    const int yy = my + dy;
    if (yy < 0 || yy >= size_y) {
      continue;
    }
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
      const int xx = mx + dx;
      if (xx < 0 || xx >= size_x) {
        continue;
      }
      if (!is_obstacle_distance_cell(costmap[yy * size_x + xx], lethal_threshold)) {
        continue;
      }
      const float d2 = static_cast<float>(dx * dx + dy * dy);
      best_cells2 = fminf(best_cells2, d2);
    }
  }

  distance_field[idx] = fminf(sqrtf(best_cells2) * resolution, cutoff);
}

// Sample the footprint polygon edges at costmap resolution and report
// whether any cell under them is lethal.
__device__ bool footprint_collides(
  const unsigned char * __restrict__ costmap, const DeviceParams & p,
  float x, float y, float yaw)
{
  const float c = cosf(yaw);
  const float s = sinf(yaw);
  for (int i = 0; i < p.footprint_len; ++i) {
    const int j = (i + 1) % p.footprint_len;
    const float ax = x + p.fp[2 * i + 0] * c - p.fp[2 * i + 1] * s;
    const float ay = y + p.fp[2 * i + 0] * s + p.fp[2 * i + 1] * c;
    const float bx = x + p.fp[2 * j + 0] * c - p.fp[2 * j + 1] * s;
    const float by = y + p.fp[2 * j + 0] * s + p.fp[2 * j + 1] * c;
    const float len = hypotf(bx - ax, by - ay);
    const int n = max(1, __float2int_ru(len / p.resolution));
    for (int t = 0; t <= n; ++t) {
      const float u = static_cast<float>(t) / n;
      const unsigned char cost =
        cell_cost(costmap, p, ax + u * (bx - ax), ay + u * (by - ay));
      if (cost == 254) {  // LETHAL_OBSTACLE
        return true;
      }
    }
  }
  return false;
}

// Check the swept SE(2) footprint between two rollout samples. This catches
// corner clips during in-place rotation of asymmetric footprints that an
// endpoint-only polygon test can miss.
__device__ bool footprint_sweep_collides(
  const unsigned char * __restrict__ costmap, const DeviceParams & p,
  float x0, float y0, float yaw0,
  float x1, float y1, float yaw1)
{
  float max_radius = 0.0f;
  for (int i = 0; i < p.footprint_len; ++i) {
    const float fx = p.fp[2 * i + 0];
    const float fy = p.fp[2 * i + 1];
    max_radius = fmaxf(max_radius, hypotf(fx, fy));
  }

  const float dtrans = hypotf(x1 - x0, y1 - y0);
  const float dyaw = wrap_angle(yaw1 - yaw0);
  const float sweep_len = dtrans + fabsf(dyaw) * max_radius;
  const int n = max(1, __float2int_ru(sweep_len / p.resolution));

  for (int t = 0; t <= n; ++t) {
    const float u = static_cast<float>(t) / n;
    const float x = x0 + u * (x1 - x0);
    const float y = y0 + u * (y1 - y0);
    const float yaw = wrap_angle(yaw0 + u * dyaw);
    if (footprint_collides(costmap, p, x, y, yaw)) {
      return true;
    }
  }
  return false;
}

__global__ void init_rng_kernel(curandState * states, unsigned long long seed, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    curand_init(seed, i, 0, &states[i]);
  }
}

// One thread rolls out one perturbed control sequence and accumulates its cost.
__global__ void rollout_kernel(
  DeviceParams p,
  const unsigned char * __restrict__ costmap,
  const float * __restrict__ distance_field,
  const float * __restrict__ path,        // [path_len * 2]
  const float * __restrict__ nominal,     // [T * kCtrlDim]
  float * __restrict__ perturbed,         // [K * T * kCtrlDim]
  float * __restrict__ costs,             // [K]
  curandState * rng_states)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= p.K) {
    return;
  }
  curandState rng = rng_states[k];

  float x = p.start_x;
  float y = p.start_y;
  float yaw = p.start_yaw;
  float cost = 0.0f;
  float prev_v = nominal[0];
  float prev_vy = nominal[1];
  float prev_w = nominal[2];

  for (int t = 0; t < p.T; ++t) {
    // Sample k==0 carries the unperturbed nominal so the previous solution
    // always stays in the candidate set.
    //
    // The stored (averaged) controls are UNCLAMPED; only the executed ones
    // are clamped. Zero-mean noise clamped at v_max averages ~v_max - 0.4σ,
    // so a clamped average can never cruise at the limit — letting the
    // nominal sit above v_max (bounded by the anti-windup clamp after the
    // update) keeps the executed control saturated.
    const bool perturb = (k != 0);
    float nv = perturb ? curand_normal(&rng) * p.v_std : 0.0f;
    float nw = perturb ? curand_normal(&rng) * p.w_std : 0.0f;
    const float v_raw = nominal[t * kCtrlDim + 0] + nv;
    float v = fminf(fmaxf(v_raw, p.v_min), p.v_max);
    float vy_raw = 0.0f;
    float vy = 0.0f;
    if (p.motion_model == 2) {  // Omni
      float nvy = perturb ? curand_normal(&rng) * p.vy_std : 0.0f;
      vy_raw = nominal[t * kCtrlDim + 1] + nvy;
      vy = fminf(fmaxf(vy_raw, -p.vy_max), p.vy_max);
    }
    const float w_raw = nominal[t * kCtrlDim + 2] + nw;
    float w = fminf(fmaxf(w_raw, -p.w_max), p.w_max);
    if (p.motion_model == 1) {  // Ackermann: curvature limit
      const float w_dyn = fabsf(v) / p.min_turning_r;
      w = fminf(fmaxf(w, -w_dyn), w_dyn);
    }
    perturbed[(k * p.T + t) * kCtrlDim + 0] = v_raw;
    perturbed[(k * p.T + t) * kCtrlDim + 1] = vy_raw;
    perturbed[(k * p.T + t) * kCtrlDim + 2] = w_raw;

    const float prev_x = x;
    const float prev_y = y;
    const float prev_yaw = yaw;
    const float cy = cosf(yaw);
    const float sy = sinf(yaw);
    x += p.dt * (v * cy - vy * sy);
    y += p.dt * (v * sy + vy * cy);
    yaw = wrap_angle(yaw + p.dt * w);

    // costmap cost (treat out-of-bounds as free: local costmap edge)
    if (p.size_x > 0) {
      const unsigned char c = cell_cost(costmap, p, x, y);
      if (c != 255) {  // 255 = NO_INFORMATION
        if (p.footprint_len > 0) {
          // footprint mode: lethal center always collides; otherwise run the
          // polygon edge check, gated on inflated cost so free space is cheap
          const unsigned char prev_c = cell_cost(costmap, p, prev_x, prev_y);
          if (c == 254 ||
            ((c > 0 || prev_c > 0) &&
            footprint_sweep_collides(costmap, p, prev_x, prev_y, prev_yaw, x, y, yaw)))
          {
            cost += p.collision_cost;
          } else {
            const float cn = static_cast<float>(c) / 252.0f;
            cost += p.costmap_w * cn * cn * p.dt;
          }
        } else if (c >= p.lethal_threshold) {
          cost += p.collision_cost;
        } else {
          const float cn = static_cast<float>(c) / 252.0f;
          cost += p.costmap_w * cn * cn * p.dt;
        }
      }
    }

    if (p.distance_field_w > 0.0f && distance_field != nullptr &&
      p.distance_field_cutoff > 1.0e-6f)
    {
      const float dist = distance_field_value(distance_field, p, x, y);
      if (dist < p.distance_field_cutoff) {
        const float q = (p.distance_field_cutoff - dist) / p.distance_field_cutoff;
        cost += p.distance_field_w * q * q * p.dt;
      }
    }

    // reference path costs (brute-force nearest point; path_len <= 256):
    // lateral deviation to the nearest point, plus distance to a point a bit
    // further along the path, which is what pulls the rollout forward
    // (analogous to nav2's PathAlign + PathFollow critics).
    if (p.path_len > 0) {
      float best = 1.0e18f;
      int best_i = 0;
      for (int i = 0; i < p.path_len; ++i) {
        float dx = x - path[i * 2 + 0];
        float dy = y - path[i * 2 + 1];
        float d2 = dx * dx + dy * dy;
        if (d2 < best) {
          best = d2;
          best_i = i;
        }
      }
      cost += p.path_w * best * p.dt;

      int fi = min(best_i + p.follow_offset, p.path_len - 1);
      float fdx = x - path[fi * 2 + 0];
      float fdy = y - path[fi * 2 + 1];
      cost += p.follow_w * sqrtf(fdx * fdx + fdy * fdy) * p.dt;

      if (p.path_angle_w > 0.0f && p.path_len > 1) {
        int prev_i = max(0, fi - 1);
        int next_i = min(p.path_len - 1, fi + 1);
        float tx = path[next_i * 2 + 0] - path[prev_i * 2 + 0];
        float ty = path[next_i * 2 + 1] - path[prev_i * 2 + 1];
        if (tx * tx + ty * ty < 1.0e-8f) {
          prev_i = max(0, best_i - 1);
          next_i = min(p.path_len - 1, best_i + 1);
          tx = path[next_i * 2 + 0] - path[prev_i * 2 + 0];
          ty = path[next_i * 2 + 1] - path[prev_i * 2 + 1];
        }
        if (tx * tx + ty * ty > 1.0e-8f) {
          float path_yaw = atan2f(ty, tx);
          if (v < -1.0e-3f) {
            path_yaw = wrap_angle(path_yaw + 3.14159265358979323846f);
          }
          const float yaw_err = wrap_angle(yaw - path_yaw);
          cost += p.path_angle_w * yaw_err * yaw_err * p.dt;
        }
      }

      if (p.curvature_speed_w > 0.0f && p.path_len > 2 && v > 0.0f) {
        const int span = max(1, p.follow_offset / 2);
        const int prev_i = max(0, fi - span);
        const int next_i = min(p.path_len - 1, fi + span);
        const float ax = path[fi * 2 + 0] - path[prev_i * 2 + 0];
        const float ay = path[fi * 2 + 1] - path[prev_i * 2 + 1];
        const float bx = path[next_i * 2 + 0] - path[fi * 2 + 0];
        const float by = path[next_i * 2 + 1] - path[fi * 2 + 1];
        const float alen = sqrtf(ax * ax + ay * ay);
        const float blen = sqrtf(bx * bx + by * by);
        if (alen > 1.0e-4f && blen > 1.0e-4f) {
          const float ayaw = atan2f(ay, ax);
          const float byaw = atan2f(by, bx);
          const float arc = fmaxf(0.5f * (alen + blen), 1.0e-3f);
          const float curvature = fabsf(wrap_angle(byaw - ayaw)) / arc;
          if (curvature > 1.0e-4f) {
            const float floor_v = fminf(fmaxf(p.curvature_speed_min, 0.0f), p.v_max);
            const float target_v = fmaxf(floor_v, p.v_max / (1.0f + curvature));
            const float overspeed = fmaxf(v - target_v, 0.0f);
            cost += p.curvature_speed_w * overspeed * overspeed * p.dt;
          }
        }
      }
    }

    // smoothness + backward motion penalties
    float dv = v - prev_v;
    float dvy = vy - prev_vy;
    float dw = w - prev_w;
    cost += p.smooth_w * (dv * dv + dvy * dvy + dw * dw);
    cost += p.backward_w * fmaxf(-v, 0.0f) * p.dt;
    cost += p.speed_w * (p.v_max - v) * p.dt;
    cost += p.angular_w * w * w * p.dt;
    prev_v = v;
    prev_vy = vy;
    prev_w = w;
  }

  // terminal cost: local goal distance (+ yaw when approaching the final goal).
  // Linear in distance so the pull toward the goal does not vanish as the
  // local goal gets close (squared distance stalls in front of obstacles).
  float gdx = x - p.goal_x;
  float gdy = y - p.goal_y;
  float gd2 = gdx * gdx + gdy * gdy;
  cost += p.goal_w * sqrtf(gd2);
  if (p.goal_is_final && gd2 < p.yaw_activation_dist * p.yaw_activation_dist) {
    float dyaw = wrap_angle(yaw - p.goal_yaw);
    cost += p.goal_yaw_w * dyaw * dyaw;
  }

  costs[k] = cost;
  rng_states[k] = rng;
}

// nominal[t][d] = sum_k weight[k] * perturbed[k][t][d]
__global__ void update_controls_kernel(
  const float * __restrict__ perturbed,
  const float * __restrict__ weights,
  float * __restrict__ nominal,
  int K, int T)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * kCtrlDim) {
    return;
  }
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += weights[k] * perturbed[k * T * kCtrlDim + idx];
  }
  nominal[idx] = acc;
}

}  // namespace

struct MppiGpu::Impl
{
  MppiParams params;
  float v_max_limit;

  curandState * d_rng = nullptr;
  float * d_nominal = nullptr;     // [T * kCtrlDim]
  float * d_perturbed = nullptr;   // [K * T * kCtrlDim]
  float * d_costs = nullptr;       // [K]
  float * d_weights = nullptr;     // [K]
  float * d_path = nullptr;        // [kMaxPathPoints * 2]
  unsigned char * d_costmap = nullptr;
  float * d_distance_field = nullptr;
  size_t costmap_capacity = 0;
  size_t distance_field_capacity = 0;

  std::vector<float> h_costs;
  std::vector<float> h_weights;
  std::vector<float> h_nominal;
  std::vector<float> h_last_valid_nominal;
  bool has_last_valid_nominal = false;
  int consecutive_all_colliding = 0;

  explicit Impl(const MppiParams & p)
  : params(p), v_max_limit(p.v_max)
  {
    const int K = params.batch_size;
    const int T = params.time_steps;
    CUDA_CHECK(cudaMalloc(&d_rng, K * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_nominal, T * kCtrlDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_perturbed, static_cast<size_t>(K) * T * kCtrlDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_costs, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_path, kMaxPathPoints * 2 * sizeof(float)));

    h_costs.resize(K);
    h_weights.resize(K);
    h_nominal.assign(T * kCtrlDim, 0.0f);
    h_last_valid_nominal.assign(T * kCtrlDim, 0.0f);

    const int threads = 256;
    init_rng_kernel<<<(K + threads - 1) / threads, threads>>>(d_rng, 42ULL, K);
    CUDA_CHECK(cudaGetLastError());
    reset();
  }

  ~Impl()
  {
    cudaFree(d_rng);
    cudaFree(d_nominal);
    cudaFree(d_perturbed);
    cudaFree(d_costs);
    cudaFree(d_weights);
    cudaFree(d_path);
    cudaFree(d_costmap);
    cudaFree(d_distance_field);
  }

  void reset()
  {
    std::fill(h_nominal.begin(), h_nominal.end(), 0.0f);
    std::fill(h_last_valid_nominal.begin(), h_last_valid_nominal.end(), 0.0f);
    has_last_valid_nominal = false;
    consecutive_all_colliding = 0;
    CUDA_CHECK(cudaMemcpy(
        d_nominal, h_nominal.data(),
        h_nominal.size() * sizeof(float), cudaMemcpyHostToDevice));
  }

  void uploadCostmap(const unsigned char * costmap, size_t bytes)
  {
    if (bytes > costmap_capacity) {
      cudaFree(d_costmap);
      CUDA_CHECK(cudaMalloc(&d_costmap, bytes));
      costmap_capacity = bytes;
    }
    CUDA_CHECK(cudaMemcpy(d_costmap, costmap, bytes, cudaMemcpyHostToDevice));
  }

  void ensureDistanceField(size_t cells)
  {
    const size_t bytes = cells * sizeof(float);
    if (bytes > distance_field_capacity) {
      cudaFree(d_distance_field);
      CUDA_CHECK(cudaMalloc(&d_distance_field, bytes));
      distance_field_capacity = bytes;
    }
  }
};

MppiGpu::MppiGpu(const MppiParams & params)
: impl_(new Impl(params))
{
}

MppiGpu::~MppiGpu() = default;

void MppiGpu::reset()
{
  impl_->reset();
}

void MppiGpu::setSpeedLimit(float v_max)
{
  impl_->v_max_limit = v_max;
}

MppiResult MppiGpu::computeInternal(
  float robot_x, float robot_y, float robot_yaw,
  const unsigned char * costmap, int size_x, int size_y,
  bool costmap_is_device,
  float origin_x, float origin_y, float resolution,
  const float * path_xy, int path_len,
  float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
  const float * footprint_xy, int footprint_len)
{
  Impl & im = *impl_;
  const MppiParams & mp = im.params;
  const int K = mp.batch_size;
  const int T = mp.time_steps;

  path_len = std::min(path_len, kMaxPathPoints);
  int follow_offset = 1;
  if (path_len > 0) {
    CUDA_CHECK(cudaMemcpy(
        im.d_path, path_xy, path_len * 2 * sizeof(float), cudaMemcpyHostToDevice));
    if (path_len > 1) {
      float arc = 0.0f;
      for (int i = 1; i < path_len; ++i) {
        arc += std::hypot(
          path_xy[i * 2 + 0] - path_xy[(i - 1) * 2 + 0],
          path_xy[i * 2 + 1] - path_xy[(i - 1) * 2 + 1]);
      }
      const float spacing = arc / static_cast<float>(path_len - 1);
      if (spacing > 1.0e-6f) {
        follow_offset = std::max(
          1, std::min(
            path_len - 1,
            static_cast<int>(std::lround(mp.follow_lookahead / spacing))));
      }
    }
  }
  const unsigned char * rollout_costmap = nullptr;
  const float * rollout_distance_field = nullptr;
  if (costmap != nullptr && size_x > 0 && size_y > 0) {
    if (costmap_is_device) {
      rollout_costmap = costmap;
    } else {
      im.uploadCostmap(costmap, static_cast<size_t>(size_x) * size_y);
      rollout_costmap = im.d_costmap;
    }
  } else {
    size_x = 0;
    size_y = 0;
  }

  if (rollout_costmap != nullptr && mp.distance_field_weight > 0.0f &&
    mp.distance_field_cutoff > 1.0e-6f)
  {
    const int threads = 256;
    const int cell_count = size_x * size_y;
    im.ensureDistanceField(static_cast<size_t>(cell_count));
    build_distance_field_kernel<<<(cell_count + threads - 1) / threads, threads>>>(
      rollout_costmap, im.d_distance_field, size_x, size_y,
      resolution, mp.distance_field_cutoff, mp.lethal_threshold);
    CUDA_CHECK(cudaGetLastError());
    rollout_distance_field = im.d_distance_field;
  }

  DeviceParams dp;
  dp.K = K;
  dp.T = T;
  dp.dt = mp.model_dt;
  dp.motion_model = static_cast<int>(mp.motion_model);
  dp.v_max = std::min(mp.v_max, im.v_max_limit);
  dp.v_min = mp.v_min;
  dp.vy_max = mp.vy_max;
  dp.w_max = mp.w_max;
  dp.min_turning_r = std::max(mp.min_turning_r, 1.0e-3f);
  dp.v_std = mp.v_std;
  dp.vy_std = mp.vy_std;
  dp.w_std = mp.w_std;
  dp.goal_w = mp.goal_weight;
  dp.goal_yaw_w = mp.goal_yaw_weight;
  dp.path_w = mp.path_weight;
  dp.follow_w = mp.path_follow_weight;
  dp.path_angle_w = mp.path_angle_weight;
  dp.curvature_speed_w = mp.curvature_speed_weight;
  dp.curvature_speed_min = mp.curvature_speed_min;
  dp.follow_offset = follow_offset;
  dp.costmap_w = mp.costmap_weight;
  dp.distance_field_w = mp.distance_field_weight;
  dp.distance_field_cutoff = mp.distance_field_cutoff;
  dp.smooth_w = mp.smoothness_weight;
  dp.backward_w = mp.backward_weight;
  dp.speed_w = mp.speed_weight;
  dp.angular_w = mp.angular_weight;
  dp.collision_cost = mp.collision_cost;
  dp.yaw_activation_dist = mp.yaw_goal_activation_dist;
  dp.lethal_threshold = mp.lethal_threshold;
  dp.size_x = size_x;
  dp.size_y = size_y;
  dp.origin_x = origin_x;
  dp.origin_y = origin_y;
  dp.resolution = resolution;
  dp.path_len = path_len;
  dp.goal_x = goal_x;
  dp.goal_y = goal_y;
  dp.goal_yaw = goal_yaw;
  dp.goal_is_final = goal_is_final ? 1 : 0;
  dp.footprint_len = 0;
  if (mp.consider_footprint && footprint_xy != nullptr && footprint_len >= 3) {
    dp.footprint_len = std::min(footprint_len, kMaxFootprint);
    for (int i = 0; i < dp.footprint_len * 2; ++i) {
      dp.fp[i] = footprint_xy[i];
    }
  }
  dp.start_x = robot_x;
  dp.start_y = robot_y;
  dp.start_yaw = robot_yaw;

  const int threads = 256;
  const int rollout_blocks = (K + threads - 1) / threads;
  const int ctrl_count = T * kCtrlDim;
  const int update_blocks = (ctrl_count + threads - 1) / threads;

  float min_cost = 0.0f;
  for (int iter = 0; iter < mp.iteration_count; ++iter) {
    rollout_kernel<<<rollout_blocks, threads>>>(
      dp, rollout_costmap, rollout_distance_field, im.d_path, im.d_nominal, im.d_perturbed,
      im.d_costs, im.d_rng);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(
        im.h_costs.data(), im.d_costs, K * sizeof(float), cudaMemcpyDeviceToHost));

    // softmin weights on host: K floats, negligible next to the rollout
    min_cost = *std::min_element(im.h_costs.begin(), im.h_costs.end());
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
      im.h_weights[k] = std::exp(-(im.h_costs[k] - min_cost) / mp.lambda);
      sum += im.h_weights[k];
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int k = 0; k < K; ++k) {
      im.h_weights[k] *= inv;
    }
    CUDA_CHECK(cudaMemcpy(
        im.d_weights, im.h_weights.data(), K * sizeof(float), cudaMemcpyHostToDevice));

    update_controls_kernel<<<update_blocks, threads>>>(
      im.d_perturbed, im.d_weights, im.d_nominal, K, T);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaMemcpy(
      im.h_nominal.data(), im.d_nominal,
      ctrl_count * sizeof(float), cudaMemcpyDeviceToHost));

  // anti-windup: only the translational speeds may exceed their limit, and
  // only by one noise std — that is what lets the executed control saturate
  // at v_max (a clamped zero-mean average sits ~0.4σ below the limit).
  // The angular rate is clamped hard: letting it wind up makes the robot
  // pirouette through disturbances instead of counter-steering.
  for (int t = 0; t < T; ++t) {
    float & nv = im.h_nominal[t * kCtrlDim + 0];
    float & nvy = im.h_nominal[t * kCtrlDim + 1];
    float & nw = im.h_nominal[t * kCtrlDim + 2];
    nv = std::min(std::max(nv, dp.v_min), dp.v_max + mp.v_std);
    nvy = std::min(std::max(nvy, -mp.vy_max - mp.vy_std), mp.vy_max + mp.vy_std);
    nw = std::min(std::max(nw, -mp.w_max), mp.w_max);
  }

  MppiResult res;
  res.best_cost = min_cost;
  res.all_colliding = min_cost >= mp.collision_cost;

  if (res.all_colliding) {
    ++im.consecutive_all_colliding;
    if (mp.enable_retreat && im.has_last_valid_nominal) {
      const int retreat_step = std::min(im.consecutive_all_colliding - 1, T - 1);
      const int offset = retreat_step * kCtrlDim;
      const float retreat_scale = std::max(0.0f, mp.retreat_scale);
      res.v = std::min(
        std::max(-retreat_scale * im.h_last_valid_nominal[offset + 0], dp.v_min),
        dp.v_max);
      res.vy = std::min(
        std::max(-retreat_scale * im.h_last_valid_nominal[offset + 1], -mp.vy_max),
        mp.vy_max);
      res.w = std::min(
        std::max(-retreat_scale * im.h_last_valid_nominal[offset + 2], -mp.w_max),
        mp.w_max);
      if (mp.motion_model == MotionModel::Ackermann) {
        const float w_dyn = std::fabs(res.v) / std::max(mp.min_turning_r, 1.0e-3f);
        res.w = std::min(std::max(res.w, -w_dyn), w_dyn);
      }
      res.retreating = true;

      CUDA_CHECK(cudaMemcpy(
          im.d_nominal, im.h_last_valid_nominal.data(),
          ctrl_count * sizeof(float), cudaMemcpyHostToDevice));
    } else {
      std::fill(im.h_nominal.begin(), im.h_nominal.end(), 0.0f);
      CUDA_CHECK(cudaMemcpy(
          im.d_nominal, im.h_nominal.data(),
          ctrl_count * sizeof(float), cudaMemcpyHostToDevice));
    }
    return res;
  }

  im.consecutive_all_colliding = 0;
  im.h_last_valid_nominal = im.h_nominal;
  im.has_last_valid_nominal = true;

  res.v = std::min(std::max(im.h_nominal[0], dp.v_min), dp.v_max);
  res.vy = std::min(std::max(im.h_nominal[1], -mp.vy_max), mp.vy_max);
  res.w = std::min(std::max(im.h_nominal[2], -mp.w_max), mp.w_max);
  if (mp.motion_model == MotionModel::Ackermann) {
    const float w_dyn = std::fabs(res.v) / std::max(mp.min_turning_r, 1.0e-3f);
    res.w = std::min(std::max(res.w, -w_dyn), w_dyn);
  }

  // warm start: shift the horizon one step, repeat the last control
  for (int t = 0; t < T - 1; ++t) {
    for (int d = 0; d < kCtrlDim; ++d) {
      im.h_nominal[t * kCtrlDim + d] = im.h_nominal[(t + 1) * kCtrlDim + d];
    }
  }
  CUDA_CHECK(cudaMemcpy(
      im.d_nominal, im.h_nominal.data(),
      ctrl_count * sizeof(float), cudaMemcpyHostToDevice));

  return res;
}

MppiResult MppiGpu::compute(
  float robot_x, float robot_y, float robot_yaw,
  const unsigned char * costmap, int size_x, int size_y,
  float origin_x, float origin_y, float resolution,
  const float * path_xy, int path_len,
  float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
  const float * footprint_xy, int footprint_len)
{
  return computeInternal(
    robot_x, robot_y, robot_yaw,
    costmap, size_x, size_y, false,
    origin_x, origin_y, resolution,
    path_xy, path_len,
    goal_x, goal_y, goal_yaw, goal_is_final,
    footprint_xy, footprint_len);
}

MppiResult MppiGpu::computeWithDeviceCostmap(
  float robot_x, float robot_y, float robot_yaw,
  const unsigned char * device_costmap, int size_x, int size_y,
  float origin_x, float origin_y, float resolution,
  const float * path_xy, int path_len,
  float goal_x, float goal_y, float goal_yaw, bool goal_is_final,
  const float * footprint_xy, int footprint_len)
{
  return computeInternal(
    robot_x, robot_y, robot_yaw,
    device_costmap, size_x, size_y, true,
    origin_x, origin_y, resolution,
    path_xy, path_len,
    goal_x, goal_y, goal_yaw, goal_is_final,
    footprint_xy, footprint_len);
}

}  // namespace cuda_mppi_controller
