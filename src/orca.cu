/*************************************************************************
    > File Name: orca.cu
    > CUDA-parallelized ORCA (Optimal Reciprocal Collision Avoidance)
    > Multi-agent navigation with guaranteed collision-free velocity selection.
    >
    > Algorithm per timestep for each agent:
    >   1. For each other agent, compute the ORCA half-plane (velocity constraint)
    >   2. For static obstacles, compute VO half-planes
    >   3. Solve 2D linear program: find velocity closest to preferred that
    >      satisfies ALL half-planes
    >   4. Apply velocity, move agent
    >
    > CUDA kernels:
    >   - compute_orca_lines_kernel: 1 thread per agent, compute half-planes
    >   - solve_linear_program_kernel: 1 thread per agent, solve 2D LP
    >   - update_positions_kernel: apply velocities, update positions
    >
    > Scenario: 200 agents on a circle, goals at antipodal positions
    > Obstacles: (20,20,3), (15,25,2), (25,15,2)
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
#define N_AGENTS     200
#define MAX_NEIGHBORS 200   // max half-planes per agent (N_AGENTS-1 + obstacles)
#define N_OBSTACLES  3

static const float AGENT_RADIUS    = 0.5f;
static const float MAX_SPEED       = 2.0f;
static const float TIME_HORIZON    = 5.0f;
static const float TIME_HORIZON_OBS = 5.0f;
static const float DT              = 0.05f;
static const float SIM_TIME        = 30.0f;
static const float GOAL_TOL        = 0.3f;
static const float CIRCLE_RADIUS   = 15.0f;
static const float CIRCLE_CX       = 20.0f;
static const float CIRCLE_CY       = 20.0f;
static const float PI_F            = 3.14159265f;
static const float EPSILON         = 1e-5f;

// -------------------------------------------------------------------------
// ORCA half-plane: point on boundary + outward normal (direction)
// The constraint is: (v - point) dot direction >= 0
// Equivalently: v dot direction >= point dot direction
// -------------------------------------------------------------------------
struct OrcaLine {
    float point_x, point_y;   // a point on the half-plane boundary
    float dir_x, dir_y;       // direction of the line (tangent, not normal)
};

struct Obstacle {
    float x, y, r;
};

// -------------------------------------------------------------------------
// Device helper: 2D cross product (determinant)
// -------------------------------------------------------------------------
__device__ __forceinline__ float det2(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}

__device__ __forceinline__ float dot2(float ax, float ay, float bx, float by) {
    return ax * bx + ay * by;
}

__device__ __forceinline__ float len2(float x, float y) {
    return sqrtf(x * x + y * y);
}

__device__ __forceinline__ float sqr(float x) { return x * x; }

// -------------------------------------------------------------------------
// Kernel 1: compute ORCA half-planes for each agent
// Each agent gets up to (N_AGENTS-1 + N_OBSTACLES) half-planes
// -------------------------------------------------------------------------
__global__ void compute_orca_lines_kernel(
    const float* px, const float* py,
    const float* vx, const float* vy,
    const float* pref_vx, const float* pref_vy,
    const float* obs_x, const float* obs_y, const float* obs_r,
    int n_agents, int n_obs,
    OrcaLine* all_lines,   // [n_agents * MAX_NEIGHBORS]
    int* line_counts,      // [n_agents]
    float agent_radius, float time_horizon, float time_horizon_obs, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_agents) return;

    int count = 0;
    OrcaLine* lines = all_lines + i * MAX_NEIGHBORS;

    float my_px = px[i], my_py = py[i];
    float my_vx = vx[i], my_vy = vy[i];

    // --- Obstacle ORCA lines ---
    for (int k = 0; k < n_obs; k++) {
        // Vector from agent to obstacle center
        float rel_x = obs_x[k] - my_px;
        float rel_y = obs_y[k] - my_py;
        float dist = len2(rel_x, rel_y);
        float combined_radius = obs_r[k] + agent_radius;

        if (dist <= combined_radius) {
            // Already inside obstacle - push out
            if (dist > EPSILON) {
                float nx = -rel_x / dist;
                float ny = -rel_y / dist;
                lines[count].point_x = my_vx + nx * (combined_radius - dist) / dt * 0.5f;
                lines[count].point_y = my_vy + ny * (combined_radius - dist) / dt * 0.5f;
                lines[count].dir_x = -ny;
                lines[count].dir_y = nx;
            } else {
                lines[count].point_x = 0.0f;
                lines[count].point_y = -1.0f / dt;
                lines[count].dir_x = 1.0f;
                lines[count].dir_y = 0.0f;
            }
            count++;
            if (count >= MAX_NEIGHBORS) break;
            continue;
        }

        // Cutoff circle center for obstacle VO (scaled by time_horizon_obs)
        float inv_th = 1.0f / time_horizon_obs;
        float cutoff_cx = rel_x * inv_th;
        float cutoff_cy = rel_y * inv_th;

        // Vector from cutoff center to velocity
        float w_x = my_vx - cutoff_cx;
        float w_y = my_vy - cutoff_cy;

        float w_len_sq = w_x * w_x + w_y * w_y;
        float dot_prod = w_x * rel_x + w_y * rel_y;

        float cutoff_r = combined_radius * inv_th;

        // Project onto the VO cone
        // Leg length
        float leg = sqrtf(dist * dist - combined_radius * combined_radius);

        if (dot_prod < 0.0f && dot_prod * dot_prod > cutoff_r * cutoff_r * w_len_sq) {
            // Project on cutoff circle
            float w_len = sqrtf(w_len_sq);
            if (w_len > EPSILON) {
                float unit_w_x = w_x / w_len;
                float unit_w_y = w_y / w_len;
                lines[count].dir_x = unit_w_y;
                lines[count].dir_y = -unit_w_x;
                float u_mag = cutoff_r - w_len;
                lines[count].point_x = my_vx + u_mag * unit_w_x;
                lines[count].point_y = my_vy + u_mag * unit_w_y;
            } else {
                lines[count].dir_x = 1.0f;
                lines[count].dir_y = 0.0f;
                lines[count].point_x = my_vx;
                lines[count].point_y = my_vy;
            }
        } else {
            // Project on legs
            float leg_val = leg / dist;
            float rdist = combined_radius / dist;

            if (det2(rel_x, rel_y, w_x, w_y) > 0.0f) {
                // Left leg
                lines[count].dir_x = rel_x * leg_val - rel_y * rdist;
                lines[count].dir_y = rel_x * rdist + rel_y * leg_val;
                float d_len = len2(lines[count].dir_x, lines[count].dir_y);
                if (d_len > EPSILON) {
                    lines[count].dir_x /= d_len;
                    lines[count].dir_y /= d_len;
                }
            } else {
                // Right leg
                lines[count].dir_x = rel_x * leg_val + rel_y * rdist;
                lines[count].dir_y = -rel_x * rdist + rel_y * leg_val;
                float d_len = len2(lines[count].dir_x, lines[count].dir_y);
                if (d_len > EPSILON) {
                    lines[count].dir_x /= d_len;
                    lines[count].dir_y /= d_len;
                }
                // Flip for right leg
                lines[count].dir_x = -lines[count].dir_x;
                lines[count].dir_y = -lines[count].dir_y;
            }

            float dot_dl = dot2(lines[count].dir_x, lines[count].dir_y, my_vx, my_vy);
            lines[count].point_x = dot_dl * lines[count].dir_x;
            lines[count].point_y = dot_dl * lines[count].dir_y;
            // Project velocity onto the line direction - the point should be on the boundary
            float proj_x = my_vx - lines[count].point_x;
            float proj_y = my_vy - lines[count].point_y;
            lines[count].point_x = my_vx - proj_x;
            lines[count].point_y = my_vy - proj_y;
        }

        count++;
        if (count >= MAX_NEIGHBORS) break;
    }

    // --- Agent-Agent ORCA lines ---
    for (int j = 0; j < n_agents; j++) {
        if (j == i) continue;
        if (count >= MAX_NEIGHBORS) break;

        float rel_pos_x = px[j] - my_px;
        float rel_pos_y = px[j + 0] - my_py;
        // Fix: need py[j]
        rel_pos_y = py[j] - my_py;
        float rel_vel_x = my_vx - vx[j];
        float rel_vel_y = my_vy - vy[j];

        float dist_sq = rel_pos_x * rel_pos_x + rel_pos_y * rel_pos_y;
        float combined_radius_val = 2.0f * agent_radius;
        float combined_radius_sq = combined_radius_val * combined_radius_val;

        float inv_th = 1.0f / time_horizon;

        // Cutoff center (relative position scaled by time horizon)
        float cutoff_cx = rel_pos_x * inv_th;
        float cutoff_cy = rel_pos_y * inv_th;

        // w = relative velocity - cutoff center
        float w_x = rel_vel_x - cutoff_cx;
        float w_y = rel_vel_y - cutoff_cy;
        float w_len_sq = w_x * w_x + w_y * w_y;

        float dot_rp_w = dot2(rel_pos_x, rel_pos_y, w_x, w_y);

        float cutoff_r = combined_radius_val * inv_th;

        float nx, ny, u_x, u_y;

        if (dist_sq > combined_radius_sq) {
            // No collision
            if (dot_rp_w < 0.0f && dot_rp_w * dot_rp_w > cutoff_r * cutoff_r * w_len_sq) {
                // Project on cutoff circle
                float w_len = sqrtf(w_len_sq);
                if (w_len < EPSILON) {
                    // agents at same relative velocity as cutoff
                    nx = 1.0f; ny = 0.0f;
                    u_x = 0.0f; u_y = 0.0f;
                } else {
                    float uw_x = w_x / w_len;
                    float uw_y = w_y / w_len;
                    nx = uw_x;
                    ny = uw_y;
                    float u_mag = cutoff_r - w_len;
                    u_x = u_mag * uw_x;
                    u_y = u_mag * uw_y;
                }
            } else {
                // Project on legs
                float dist_val = sqrtf(dist_sq);
                float leg = sqrtf(dist_sq - combined_radius_sq);

                if (det2(rel_pos_x, rel_pos_y, w_x, w_y) > 0.0f) {
                    // Left leg
                    float leg_dir_x = (rel_pos_x * leg - rel_pos_y * combined_radius_val) / dist_sq;
                    float leg_dir_y = (rel_pos_x * combined_radius_val + rel_pos_y * leg) / dist_sq;

                    float proj = dot2(rel_vel_x, rel_vel_y, leg_dir_x, leg_dir_y);
                    float proj_x = proj * leg_dir_x;
                    float proj_y = proj * leg_dir_y;

                    u_x = proj_x - rel_vel_x;
                    u_y = proj_y - rel_vel_y;
                    // Normal points to the left of the leg direction
                    nx = -leg_dir_y;
                    ny = leg_dir_x;
                } else {
                    // Right leg (negate direction)
                    float leg_dir_x = (rel_pos_x * leg + rel_pos_y * combined_radius_val) / dist_sq;
                    float leg_dir_y = (-rel_pos_x * combined_radius_val + rel_pos_y * leg) / dist_sq;

                    float proj = dot2(rel_vel_x, rel_vel_y, leg_dir_x, leg_dir_y);
                    float proj_x = proj * leg_dir_x;
                    float proj_y = proj * leg_dir_y;

                    u_x = proj_x - rel_vel_x;
                    u_y = proj_y - rel_vel_y;
                    // Normal points to the right of the leg direction
                    nx = leg_dir_y;
                    ny = -leg_dir_x;
                }
            }
        } else {
            // Collision - project on cutoff circle at time dt
            float inv_dt = 1.0f / dt;
            float w2_x = rel_vel_x - rel_pos_x * inv_dt;
            float w2_y = rel_vel_y - rel_pos_y * inv_dt;
            float w2_len = len2(w2_x, w2_y);

            if (w2_len > EPSILON) {
                nx = w2_x / w2_len;
                ny = w2_y / w2_len;
                float u_mag = combined_radius_val * inv_dt - w2_len;
                u_x = u_mag * nx;
                u_y = u_mag * ny;
            } else {
                nx = 1.0f; ny = 0.0f;
                u_x = combined_radius_val * inv_dt;
                u_y = 0.0f;
            }
        }

        // ORCA line: each agent takes half the responsibility
        lines[count].dir_x = -ny;  // tangent along the line
        lines[count].dir_y = nx;
        lines[count].point_x = my_vx + 0.5f * u_x;
        lines[count].point_y = my_vy + 0.5f * u_y;

        count++;
    }

    line_counts[i] = count;
}

// -------------------------------------------------------------------------
// Device function: 1D linear program on a single line
// Given constraints already established, find best point on line idx
// Returns true if feasible, and sets result_x, result_y
// -------------------------------------------------------------------------
__device__ bool linearProgram1(
    const OrcaLine* lines, int line_no,
    float radius, float opt_vx, float opt_vy,
    bool direction_opt,
    float& result_x, float& result_y)
{
    float dot_product = dot2(lines[line_no].point_x, lines[line_no].point_y,
                              lines[line_no].dir_x, lines[line_no].dir_y);
    float discriminant = sqr(dot_product) + sqr(radius) -
        (sqr(lines[line_no].point_x) + sqr(lines[line_no].point_y));

    if (discriminant < 0.0f) {
        return false;
    }

    float sqrt_disc = sqrtf(discriminant);
    float t_left = -dot_product - sqrt_disc;
    float t_right = -dot_product + sqrt_disc;

    // Tighten bounds using previous lines
    for (int i = 0; i < line_no; i++) {
        float denom = det2(lines[line_no].dir_x, lines[line_no].dir_y,
                           lines[i].dir_x, lines[i].dir_y);
        float numer = det2(lines[i].dir_x, lines[i].dir_y,
                           lines[line_no].point_x - lines[i].point_x,
                           lines[line_no].point_y - lines[i].point_y);

        if (fabsf(denom) <= EPSILON) {
            // Lines are parallel
            if (numer < 0.0f) {
                return false;
            }
            continue;
        }

        float t = numer / denom;

        if (denom > 0.0f) {
            t_right = fminf(t_right, t);
        } else {
            t_left = fmaxf(t_left, t);
        }

        if (t_left > t_right) {
            return false;
        }
    }

    // Optimize along the line
    if (direction_opt) {
        // Optimize in the direction of opt_v
        float t = dot2(lines[line_no].dir_x, lines[line_no].dir_y, opt_vx, opt_vy);
        if (t > t_right) {
            t = t_right;
        } else if (t < t_left) {
            t = t_left;
        }
        result_x = lines[line_no].point_x + t * lines[line_no].dir_x;
        result_y = lines[line_no].point_y + t * lines[line_no].dir_y;
    } else {
        // Optimize closest to opt_v
        float t = dot2(lines[line_no].dir_x, lines[line_no].dir_y,
                        opt_vx - lines[line_no].point_x,
                        opt_vy - lines[line_no].point_y);

        if (t < t_left) {
            t = t_left;
        } else if (t > t_right) {
            t = t_right;
        }

        result_x = lines[line_no].point_x + t * lines[line_no].dir_x;
        result_y = lines[line_no].point_y + t * lines[line_no].dir_y;
    }

    return true;
}

// -------------------------------------------------------------------------
// Device function: 2D linear program
// Incrementally add constraints; find velocity closest to preferred
// -------------------------------------------------------------------------
__device__ bool linearProgram2(
    const OrcaLine* lines, int n_lines,
    float radius, float opt_vx, float opt_vy,
    bool direction_opt,
    float& result_x, float& result_y)
{
    if (direction_opt) {
        // Optimize in direction opt_v, magnitude radius
        result_x = opt_vx * radius;
        result_y = opt_vy * radius;
    } else if (sqr(opt_vx) + sqr(opt_vy) > sqr(radius)) {
        // Preferred velocity outside circle - project onto boundary
        float len = len2(opt_vx, opt_vy);
        result_x = opt_vx / len * radius;
        result_y = opt_vy / len * radius;
    } else {
        result_x = opt_vx;
        result_y = opt_vy;
    }

    for (int i = 0; i < n_lines; i++) {
        // Check if current result satisfies constraint i
        // Constraint: det(dir, point - v) >= 0  (v is on the left side of the line)
        if (det2(lines[i].dir_x, lines[i].dir_y,
                 lines[i].point_x - result_x,
                 lines[i].point_y - result_y) > 0.0f) {
            // Current result violates this constraint
            float old_x = result_x;
            float old_y = result_y;

            if (!linearProgram1(lines, i, radius, opt_vx, opt_vy,
                                direction_opt, result_x, result_y)) {
                result_x = old_x;
                result_y = old_y;
                return false;
            }
        }
    }

    return true;
}

// -------------------------------------------------------------------------
// Device function: Handle infeasible LP - project to closest feasible
// -------------------------------------------------------------------------
__device__ void linearProgram3(
    const OrcaLine* lines, int n_lines, int begin_line,
    float radius,
    float& result_x, float& result_y)
{
    float distance = 0.0f;

    for (int i = begin_line; i < n_lines; i++) {
        if (det2(lines[i].dir_x, lines[i].dir_y,
                 lines[i].point_x - result_x,
                 lines[i].point_y - result_y) > distance) {
            // Line i is the most violated

            // Try to satisfy lines 0..i using the 2D LP with the constraint
            // that we minimize penetration into line i
            // Use direction optimization along the normal of line i
            float dir_x = -lines[i].dir_y;  // normal pointing inward
            float dir_y = lines[i].dir_x;

            // Project result onto line i
            float t = det2(lines[i].dir_x, lines[i].dir_y,
                           lines[i].point_x - result_x,
                           lines[i].point_y - result_y);

            float proj_x = result_x + t * (-lines[i].dir_y);
            float proj_y = result_y + t * lines[i].dir_x;

            result_x = proj_x;
            result_y = proj_y;

            // Clamp to max speed circle
            float spd = len2(result_x, result_y);
            if (spd > radius) {
                result_x *= radius / spd;
                result_y *= radius / spd;
            }

            distance = det2(lines[i].dir_x, lines[i].dir_y,
                             lines[i].point_x - result_x,
                             lines[i].point_y - result_y);
        }
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Solve linear program for each agent
// -------------------------------------------------------------------------
__global__ void solve_linear_program_kernel(
    const float* pref_vx, const float* pref_vy,
    const OrcaLine* all_lines,
    const int* line_counts,
    float* out_vx, float* out_vy,
    int n_agents, float max_speed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_agents) return;

    const OrcaLine* lines = all_lines + i * MAX_NEIGHBORS;
    int n_lines = line_counts[i];

    float result_x, result_y;

    bool feasible = linearProgram2(lines, n_lines, max_speed,
                                    pref_vx[i], pref_vy[i],
                                    false, result_x, result_y);

    if (!feasible) {
        linearProgram3(lines, n_lines, 0, max_speed, result_x, result_y);
    }

    out_vx[i] = result_x;
    out_vy[i] = result_y;
}

// -------------------------------------------------------------------------
// Kernel 3: Update positions
// -------------------------------------------------------------------------
__global__ void update_positions_kernel(
    float* px, float* py,
    float* vx, float* vy,
    const float* new_vx, const float* new_vy,
    int n_agents, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_agents) return;

    vx[i] = new_vx[i];
    vy[i] = new_vy[i];
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
}

// -------------------------------------------------------------------------
// Host: Compute preferred velocities (toward goal, capped)
// -------------------------------------------------------------------------
__global__ void compute_preferred_velocity_kernel(
    const float* px, const float* py,
    const float* gx, const float* gy,
    float* pref_vx, float* pref_vy,
    int n_agents, float max_speed, float goal_tol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_agents) return;

    float dx = gx[i] - px[i];
    float dy = gy[i] - py[i];
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < goal_tol) {
        // Already at goal - zero velocity
        pref_vx[i] = 0.0f;
        pref_vy[i] = 0.0f;
    } else if (dist < max_speed * 0.5f) {
        // Near goal - proportional slowdown
        pref_vx[i] = dx * 2.0f;
        pref_vy[i] = dy * 2.0f;
    } else {
        // Move toward goal at max speed
        pref_vx[i] = dx / dist * max_speed;
        pref_vy[i] = dy / dist * max_speed;
    }
}

// -------------------------------------------------------------------------
// Visualization helpers
// -------------------------------------------------------------------------
cv::Scalar agent_color(int idx, int total) {
    float hue = (float)idx / (float)total * 180.0f;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((int)hue, 255, 230));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "CUDA ORCA (Optimal Reciprocal Collision Avoidance)" << endl;
    cout << "Agents: " << N_AGENTS << ", Simulation time: " << SIM_TIME << "s" << endl;

    // --- Setup scenario: agents on circle, antipodal goals ---
    float h_px[N_AGENTS], h_py[N_AGENTS];
    float h_vx[N_AGENTS], h_vy[N_AGENTS];
    float h_gx[N_AGENTS], h_gy[N_AGENTS];

    for (int i = 0; i < N_AGENTS; i++) {
        float angle = 2.0f * PI_F * (float)i / (float)N_AGENTS;
        h_px[i] = CIRCLE_CX + CIRCLE_RADIUS * cosf(angle);
        h_py[i] = CIRCLE_CY + CIRCLE_RADIUS * sinf(angle);
        h_vx[i] = 0.0f;
        h_vy[i] = 0.0f;
        float goal_angle = angle + PI_F;
        h_gx[i] = CIRCLE_CX + CIRCLE_RADIUS * cosf(goal_angle);
        h_gy[i] = CIRCLE_CY + CIRCLE_RADIUS * sinf(goal_angle);
    }

    // Obstacles
    Obstacle obstacles[N_OBSTACLES] = {
        {20.0f, 20.0f, 3.0f},
        {15.0f, 25.0f, 2.0f},
        {25.0f, 15.0f, 2.0f}
    };

    float h_obs_x[N_OBSTACLES], h_obs_y[N_OBSTACLES], h_obs_r[N_OBSTACLES];
    for (int k = 0; k < N_OBSTACLES; k++) {
        h_obs_x[k] = obstacles[k].x;
        h_obs_y[k] = obstacles[k].y;
        h_obs_r[k] = obstacles[k].r;
    }

    // --- Allocate device memory ---
    float *d_px, *d_py, *d_vx, *d_vy, *d_gx, *d_gy;
    float *d_pref_vx, *d_pref_vy, *d_new_vx, *d_new_vy;
    float *d_obs_x, *d_obs_y, *d_obs_r;
    OrcaLine* d_lines;
    int* d_line_counts;

    size_t agent_bytes = N_AGENTS * sizeof(float);
    size_t obs_bytes = N_OBSTACLES * sizeof(float);
    size_t lines_bytes = N_AGENTS * MAX_NEIGHBORS * sizeof(OrcaLine);

    CUDA_CHECK(cudaMalloc(&d_px, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_py, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_gx, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_gy, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_pref_vx, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_pref_vy, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_new_vx, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_new_vy, agent_bytes));
    CUDA_CHECK(cudaMalloc(&d_obs_x, obs_bytes));
    CUDA_CHECK(cudaMalloc(&d_obs_y, obs_bytes));
    CUDA_CHECK(cudaMalloc(&d_obs_r, obs_bytes));
    CUDA_CHECK(cudaMalloc(&d_lines, lines_bytes));
    CUDA_CHECK(cudaMalloc(&d_line_counts, N_AGENTS * sizeof(int)));

    // Copy initial data
    CUDA_CHECK(cudaMemcpy(d_px, h_px, agent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py, agent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, agent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, agent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx, h_gx, agent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, h_gy, agent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_x, h_obs_x, obs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_y, h_obs_y, obs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_r, h_obs_r, obs_bytes, cudaMemcpyHostToDevice));

    // --- Trail storage ---
    vector<vector<cv::Point2f>> trails(N_AGENTS);

    // --- Visualization setup ---
    const float WORLD_MIN = -2.0f;
    const float WORLD_MAX = 42.0f;
    const int IMG_SIZE = 800;
    float scale = (float)IMG_SIZE / (WORLD_MAX - WORLD_MIN);

    auto to_pixel = [&](float wx, float wy) -> cv::Point {
        int px = (int)((wx - WORLD_MIN) * scale);
        int py = IMG_SIZE - (int)((wy - WORLD_MIN) * scale);
        return cv::Point(px, py);
    };

    // Precompute colors
    vector<cv::Scalar> colors(N_AGENTS);
    for (int i = 0; i < N_AGENTS; i++) {
        colors[i] = agent_color(i, N_AGENTS);
    }

    cv::VideoWriter video("gif/orca.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(IMG_SIZE, IMG_SIZE));

    // CUDA launch config
    int blockSize = 128;
    int gridSize = (N_AGENTS + blockSize - 1) / blockSize;

    // --- Simulation loop ---
    int total_steps = (int)(SIM_TIME / DT);
    int vis_skip = 2;

    for (int step = 0; step < total_steps; step++) {
        // 1. Compute preferred velocities
        compute_preferred_velocity_kernel<<<gridSize, blockSize>>>(
            d_px, d_py, d_gx, d_gy,
            d_pref_vx, d_pref_vy,
            N_AGENTS, MAX_SPEED, GOAL_TOL);
        CUDA_CHECK(cudaGetLastError());

        // 2. Compute ORCA half-planes
        compute_orca_lines_kernel<<<gridSize, blockSize>>>(
            d_px, d_py, d_vx, d_vy,
            d_pref_vx, d_pref_vy,
            d_obs_x, d_obs_y, d_obs_r,
            N_AGENTS, N_OBSTACLES,
            d_lines, d_line_counts,
            AGENT_RADIUS, TIME_HORIZON, TIME_HORIZON_OBS, DT);
        CUDA_CHECK(cudaGetLastError());

        // 3. Solve linear programs
        solve_linear_program_kernel<<<gridSize, blockSize>>>(
            d_pref_vx, d_pref_vy,
            d_lines, d_line_counts,
            d_new_vx, d_new_vy,
            N_AGENTS, MAX_SPEED);
        CUDA_CHECK(cudaGetLastError());

        // 4. Update positions
        update_positions_kernel<<<gridSize, blockSize>>>(
            d_px, d_py, d_vx, d_vy,
            d_new_vx, d_new_vy,
            N_AGENTS, DT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy positions and velocities back for visualization
        CUDA_CHECK(cudaMemcpy(h_px, d_px, agent_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py, d_py, agent_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vx, d_vx, agent_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vy, d_vy, agent_bytes, cudaMemcpyDeviceToHost));

        // Record trails
        for (int i = 0; i < N_AGENTS; i++) {
            trails[i].push_back(cv::Point2f(h_px[i], h_py[i]));
        }

        // --- Visualization ---
        if (step % vis_skip != 0) continue;

        cv::Mat img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));

        // Grid
        for (int g = 0; g <= 40; g += 5) {
            cv::Point p1 = to_pixel((float)g, WORLD_MIN);
            cv::Point p2 = to_pixel((float)g, WORLD_MAX);
            cv::line(img, p1, p2, cv::Scalar(230, 230, 230), 1);
            p1 = to_pixel(WORLD_MIN, (float)g);
            p2 = to_pixel(WORLD_MAX, (float)g);
            cv::line(img, p1, p2, cv::Scalar(230, 230, 230), 1);
        }

        // Obstacles
        for (int k = 0; k < N_OBSTACLES; k++) {
            cv::Point center = to_pixel(obstacles[k].x, obstacles[k].y);
            int r_px = (int)(obstacles[k].r * scale);
            cv::circle(img, center, r_px, cv::Scalar(0, 0, 0), -1);
        }

        // Trails
        for (int i = 0; i < N_AGENTS; i++) {
            for (int t = 1; t < (int)trails[i].size(); t++) {
                cv::Point p1 = to_pixel(trails[i][t - 1].x, trails[i][t - 1].y);
                cv::Point p2 = to_pixel(trails[i][t].x, trails[i][t].y);
                cv::line(img, p1, p2, colors[i], 1);
            }
        }

        // Goals
        for (int i = 0; i < N_AGENTS; i++) {
            cv::Point gp = to_pixel(h_gx[i], h_gy[i]);
            cv::circle(img, gp, 4, colors[i], 1);
        }

        // Agents with velocity vectors
        for (int i = 0; i < N_AGENTS; i++) {
            cv::Point rp = to_pixel(h_px[i], h_py[i]);
            int r_px = max(3, (int)(AGENT_RADIUS * scale));
            cv::circle(img, rp, r_px, colors[i], -1);
            cv::circle(img, rp, r_px, cv::Scalar(0, 0, 0), 1);

            // Velocity vector
            float vel_scale = 2.0f;  // scale for visibility
            cv::Point vp = to_pixel(h_px[i] + h_vx[i] * vel_scale,
                                     h_py[i] + h_vy[i] * vel_scale);
            cv::line(img, rp, vp, colors[i], 1);
        }

        // Status
        char buf[128];
        snprintf(buf, sizeof(buf), "ORCA  t=%.1fs  step=%d/%d", step * DT, step, total_steps);
        cv::putText(img, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 0, 0), 1);

        int reached = 0;
        for (int i = 0; i < N_AGENTS; i++) {
            float dx = h_px[i] - h_gx[i];
            float dy = h_py[i] - h_gy[i];
            if (sqrtf(dx * dx + dy * dy) < GOAL_TOL) reached++;
        }
        snprintf(buf, sizeof(buf), "Reached: %d/%d", reached, N_AGENTS);
        cv::putText(img, buf, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 128, 0), 1);

        cv::imshow("orca", img);
        video.write(img);
        int key = cv::waitKey(1);
        if (key == 27) break;

        if (reached == N_AGENTS) {
            cout << "All agents reached their goals at t=" << step * DT << "s" << endl;
            cv::waitKey(0);
            break;
        }
    }

    video.release();
    system("ffmpeg -y -i gif/orca.avi "
           "-vf 'fps=15,scale=400:-1:flags=lanczos' -loop 0 "
           "gif/orca.gif 2>/dev/null");
    cout << "GIF saved to gif/orca.gif" << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_gx));
    CUDA_CHECK(cudaFree(d_gy));
    CUDA_CHECK(cudaFree(d_pref_vx));
    CUDA_CHECK(cudaFree(d_pref_vy));
    CUDA_CHECK(cudaFree(d_new_vx));
    CUDA_CHECK(cudaFree(d_new_vy));
    CUDA_CHECK(cudaFree(d_obs_x));
    CUDA_CHECK(cudaFree(d_obs_y));
    CUDA_CHECK(cudaFree(d_obs_r));
    CUDA_CHECK(cudaFree(d_lines));
    CUDA_CHECK(cudaFree(d_line_counts));

    cout << "Simulation complete." << endl;
    return 0;
}
