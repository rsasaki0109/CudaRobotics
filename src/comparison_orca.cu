/*************************************************************************
    Comparison: Potential Field vs ORCA side-by-side
    Left:  Potential Field (can get stuck, oscillations)
    Right: ORCA (smooth, guaranteed collision-free)
    200 agents on a circle, goals at antipodal positions
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#define PI_F 3.14159265f
#define N_AGENTS 200
#define N_OBS 3
#define MAX_NEIGHBORS 210
#define AGENT_RADIUS 0.5f
#define MAX_SPEED 2.0f
#define TIME_HORIZON 5.0f
#define TIME_HORIZON_OBS 5.0f
#define DT_SIM 0.05f
#define SIM_TIME_VAL 30.0f
#define GOAL_TOL 0.3f
#define IMG_W 400
#define EPSILON 1e-5f

// Potential field params
#define KP_ATT 5.0f
#define KP_REP 100.0f
#define KP_ROBOT 50.0f
#define OBS_INFLUENCE 5.0f
#define ROBOT_INFLUENCE 3.0f
#define DAMPING 0.8f

struct Obstacle { float x, y, r; };
static Obstacle h_obs[N_OBS] = {{20,20,3},{15,25,2},{25,15,2}};

struct OrcaLine {
    float point_x, point_y;
    float dir_x, dir_y;
};

// =====================================================================
// Device helpers
// =====================================================================
__device__ __forceinline__ float det2d(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}
__device__ __forceinline__ float dot2d(float ax, float ay, float bx, float by) {
    return ax * bx + ay * by;
}
__device__ __forceinline__ float len2d(float x, float y) {
    return sqrtf(x * x + y * y);
}
__device__ __forceinline__ float sqr_f(float x) { return x * x; }

// =====================================================================
// Potential Field kernel (same approach as multi_robot_planner)
// =====================================================================
__global__ void pf_compute_and_update(
    float* px, float* py, float* vx, float* vy,
    const float* gx, const float* gy,
    const float* obs_x, const float* obs_y, const float* obs_r,
    int n_agents, int n_obs, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_agents) return;

    float fx = 0.0f, fy = 0.0f;
    float rx = px[i], ry = py[i];

    // Attractive force toward goal
    float dx = gx[i] - rx, dy = gy[i] - ry;
    float dg = sqrtf(dx*dx + dy*dy);
    if (dg > 0.01f) {
        float f_att = KP_ATT * fminf(dg, 5.0f);
        fx += f_att * dx / dg;
        fy += f_att * dy / dg;
    }

    // Repulsive from obstacles
    for (int k = 0; k < n_obs; k++) {
        float odx = rx - obs_x[k], ody = ry - obs_y[k];
        float od = sqrtf(odx*odx + ody*ody);
        float clearance = od - obs_r[k];
        if (clearance < 0.01f) clearance = 0.01f;
        if (clearance < OBS_INFLUENCE) {
            float inv_diff = 1.0f / clearance - 1.0f / OBS_INFLUENCE;
            float mag = KP_REP * inv_diff / (clearance * clearance);
            if (od > 0.001f) {
                fx += mag * odx / od;
                fy += mag * ody / od;
            }
        }
    }

    // Repulsive from other agents
    for (int j = 0; j < n_agents; j++) {
        if (j == i) continue;
        float rdx = rx - px[j], rdy = ry - py[j];
        float rd = sqrtf(rdx*rdx + rdy*rdy);
        float clearance = rd - 2.0f * AGENT_RADIUS;
        if (clearance < 0.01f) clearance = 0.01f;
        if (clearance < ROBOT_INFLUENCE) {
            float inv_diff = 1.0f / clearance - 1.0f / ROBOT_INFLUENCE;
            float mag = KP_ROBOT * inv_diff / (clearance * clearance);
            if (rd > 0.001f) {
                fx += mag * rdx / rd;
                fy += mag * rdy / rd;
            }
        }
    }

    vx[i] = DAMPING * vx[i] + fx * dt;
    vy[i] = DAMPING * vy[i] + fy * dt;
    float spd = sqrtf(vx[i]*vx[i] + vy[i]*vy[i]);
    if (spd > MAX_SPEED) { vx[i] *= MAX_SPEED/spd; vy[i] *= MAX_SPEED/spd; }
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
}

// =====================================================================
// ORCA kernels
// =====================================================================
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
        pref_vx[i] = 0.0f;
        pref_vy[i] = 0.0f;
    } else if (dist < max_speed * 0.5f) {
        pref_vx[i] = dx * 2.0f;
        pref_vy[i] = dy * 2.0f;
    } else {
        pref_vx[i] = dx / dist * max_speed;
        pref_vy[i] = dy / dist * max_speed;
    }
}

__global__ void compute_orca_lines_kernel(
    const float* px, const float* py,
    const float* vx, const float* vy,
    const float* pref_vx, const float* pref_vy,
    const float* obs_x, const float* obs_y, const float* obs_r,
    int n_agents, int n_obs,
    OrcaLine* all_lines, int* line_counts,
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
        float rel_x = obs_x[k] - my_px;
        float rel_y = obs_y[k] - my_py;
        float dist = len2d(rel_x, rel_y);
        float combined_radius = obs_r[k] + agent_radius;

        if (dist <= combined_radius) {
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

        float inv_th = 1.0f / time_horizon_obs;
        float cutoff_cx = rel_x * inv_th;
        float cutoff_cy = rel_y * inv_th;
        float w_x = my_vx - cutoff_cx;
        float w_y = my_vy - cutoff_cy;
        float w_len_sq = w_x * w_x + w_y * w_y;
        float dot_prod = w_x * rel_x + w_y * rel_y;
        float cutoff_r = combined_radius * inv_th;
        float leg = sqrtf(dist * dist - combined_radius * combined_radius);

        if (dot_prod < 0.0f && dot_prod * dot_prod > cutoff_r * cutoff_r * w_len_sq) {
            float w_len = sqrtf(w_len_sq);
            if (w_len > EPSILON) {
                float uw_x = w_x / w_len, uw_y = w_y / w_len;
                lines[count].dir_x = uw_y;
                lines[count].dir_y = -uw_x;
                lines[count].point_x = my_vx + (cutoff_r - w_len) * uw_x;
                lines[count].point_y = my_vy + (cutoff_r - w_len) * uw_y;
            } else {
                lines[count].dir_x = 1.0f;
                lines[count].dir_y = 0.0f;
                lines[count].point_x = my_vx;
                lines[count].point_y = my_vy;
            }
        } else {
            float leg_val = leg / dist;
            float rdist = combined_radius / dist;
            if (det2d(rel_x, rel_y, w_x, w_y) > 0.0f) {
                lines[count].dir_x = rel_x * leg_val - rel_y * rdist;
                lines[count].dir_y = rel_x * rdist + rel_y * leg_val;
            } else {
                lines[count].dir_x = -(rel_x * leg_val + rel_y * rdist);
                lines[count].dir_y = -(- rel_x * rdist + rel_y * leg_val);
            }
            float d_len = len2d(lines[count].dir_x, lines[count].dir_y);
            if (d_len > EPSILON) {
                lines[count].dir_x /= d_len;
                lines[count].dir_y /= d_len;
            }
            float dot_dl = dot2d(lines[count].dir_x, lines[count].dir_y, my_vx, my_vy);
            lines[count].point_x = dot_dl * lines[count].dir_x;
            lines[count].point_y = dot_dl * lines[count].dir_y;
        }
        count++;
        if (count >= MAX_NEIGHBORS) break;
    }

    // --- Agent-Agent ORCA lines ---
    for (int j = 0; j < n_agents; j++) {
        if (j == i) continue;
        if (count >= MAX_NEIGHBORS) break;

        float rel_pos_x = px[j] - my_px;
        float rel_pos_y = py[j] - my_py;
        float rel_vel_x = my_vx - vx[j];
        float rel_vel_y = my_vy - vy[j];

        float dist_sq = rel_pos_x * rel_pos_x + rel_pos_y * rel_pos_y;
        float combined_r = 2.0f * agent_radius;
        float combined_r_sq = combined_r * combined_r;
        float inv_th = 1.0f / time_horizon;

        float cutoff_cx = rel_pos_x * inv_th;
        float cutoff_cy = rel_pos_y * inv_th;
        float w_x = rel_vel_x - cutoff_cx;
        float w_y = rel_vel_y - cutoff_cy;
        float w_len_sq = w_x * w_x + w_y * w_y;
        float dot_rp_w = dot2d(rel_pos_x, rel_pos_y, w_x, w_y);
        float cutoff_r = combined_r * inv_th;

        float nx, ny, u_x, u_y;

        if (dist_sq > combined_r_sq) {
            if (dot_rp_w < 0.0f && dot_rp_w * dot_rp_w > cutoff_r * cutoff_r * w_len_sq) {
                float w_len = sqrtf(w_len_sq);
                if (w_len < EPSILON) { nx = 1.0f; ny = 0.0f; u_x = 0.0f; u_y = 0.0f; }
                else {
                    float uw_x = w_x / w_len, uw_y = w_y / w_len;
                    nx = uw_x; ny = uw_y;
                    u_x = (cutoff_r - w_len) * uw_x;
                    u_y = (cutoff_r - w_len) * uw_y;
                }
            } else {
                float dist_val = sqrtf(dist_sq);
                float leg = sqrtf(dist_sq - combined_r_sq);
                if (det2d(rel_pos_x, rel_pos_y, w_x, w_y) > 0.0f) {
                    float ld_x = (rel_pos_x * leg - rel_pos_y * combined_r) / dist_sq;
                    float ld_y = (rel_pos_x * combined_r + rel_pos_y * leg) / dist_sq;
                    float proj = dot2d(rel_vel_x, rel_vel_y, ld_x, ld_y);
                    u_x = proj * ld_x - rel_vel_x;
                    u_y = proj * ld_y - rel_vel_y;
                    nx = -ld_y; ny = ld_x;
                } else {
                    float ld_x = (rel_pos_x * leg + rel_pos_y * combined_r) / dist_sq;
                    float ld_y = (-rel_pos_x * combined_r + rel_pos_y * leg) / dist_sq;
                    float proj = dot2d(rel_vel_x, rel_vel_y, ld_x, ld_y);
                    u_x = proj * ld_x - rel_vel_x;
                    u_y = proj * ld_y - rel_vel_y;
                    nx = ld_y; ny = -ld_x;
                }
            }
        } else {
            float inv_dt = 1.0f / dt;
            float w2_x = rel_vel_x - rel_pos_x * inv_dt;
            float w2_y = rel_vel_y - rel_pos_y * inv_dt;
            float w2_len = len2d(w2_x, w2_y);
            if (w2_len > EPSILON) {
                nx = w2_x / w2_len; ny = w2_y / w2_len;
                u_x = (combined_r * inv_dt - w2_len) * nx;
                u_y = (combined_r * inv_dt - w2_len) * ny;
            } else {
                nx = 1.0f; ny = 0.0f;
                u_x = combined_r * inv_dt; u_y = 0.0f;
            }
        }

        lines[count].dir_x = -ny;
        lines[count].dir_y = nx;
        lines[count].point_x = my_vx + 0.5f * u_x;
        lines[count].point_y = my_vy + 0.5f * u_y;
        count++;
    }

    line_counts[i] = count;
}

// =====================================================================
// LP solver (device functions)
// =====================================================================
__device__ bool linearProgram1(
    const OrcaLine* lines, int line_no, float radius,
    float opt_vx, float opt_vy, bool dir_opt,
    float& rx, float& ry)
{
    float dp = dot2d(lines[line_no].point_x, lines[line_no].point_y,
                      lines[line_no].dir_x, lines[line_no].dir_y);
    float disc = sqr_f(dp) + sqr_f(radius) -
        (sqr_f(lines[line_no].point_x) + sqr_f(lines[line_no].point_y));
    if (disc < 0.0f) return false;

    float sq = sqrtf(disc);
    float tl = -dp - sq, tr = -dp + sq;

    for (int i = 0; i < line_no; i++) {
        float den = det2d(lines[line_no].dir_x, lines[line_no].dir_y,
                          lines[i].dir_x, lines[i].dir_y);
        float num = det2d(lines[i].dir_x, lines[i].dir_y,
                          lines[line_no].point_x - lines[i].point_x,
                          lines[line_no].point_y - lines[i].point_y);
        if (fabsf(den) <= EPSILON) {
            if (num < 0.0f) return false;
            continue;
        }
        float t = num / den;
        if (den > 0.0f) tr = fminf(tr, t);
        else tl = fmaxf(tl, t);
        if (tl > tr) return false;
    }

    float t;
    if (dir_opt) {
        t = dot2d(lines[line_no].dir_x, lines[line_no].dir_y, opt_vx, opt_vy);
        t = fmaxf(tl, fminf(tr, t));
    } else {
        t = dot2d(lines[line_no].dir_x, lines[line_no].dir_y,
                   opt_vx - lines[line_no].point_x, opt_vy - lines[line_no].point_y);
        t = fmaxf(tl, fminf(tr, t));
    }
    rx = lines[line_no].point_x + t * lines[line_no].dir_x;
    ry = lines[line_no].point_y + t * lines[line_no].dir_y;
    return true;
}

__device__ bool linearProgram2(
    const OrcaLine* lines, int n, float radius,
    float opt_vx, float opt_vy, bool dir_opt,
    float& rx, float& ry)
{
    if (dir_opt) {
        rx = opt_vx * radius; ry = opt_vy * radius;
    } else if (sqr_f(opt_vx) + sqr_f(opt_vy) > sqr_f(radius)) {
        float l = len2d(opt_vx, opt_vy);
        rx = opt_vx / l * radius; ry = opt_vy / l * radius;
    } else {
        rx = opt_vx; ry = opt_vy;
    }

    for (int i = 0; i < n; i++) {
        if (det2d(lines[i].dir_x, lines[i].dir_y,
                  lines[i].point_x - rx, lines[i].point_y - ry) > 0.0f) {
            float ox = rx, oy = ry;
            if (!linearProgram1(lines, i, radius, opt_vx, opt_vy, dir_opt, rx, ry)) {
                rx = ox; ry = oy;
                return false;
            }
        }
    }
    return true;
}

__device__ void linearProgram3(
    const OrcaLine* lines, int n, int begin,
    float radius, float& rx, float& ry)
{
    float distance = 0.0f;
    for (int i = begin; i < n; i++) {
        if (det2d(lines[i].dir_x, lines[i].dir_y,
                  lines[i].point_x - rx, lines[i].point_y - ry) > distance) {
            float t = det2d(lines[i].dir_x, lines[i].dir_y,
                            lines[i].point_x - rx, lines[i].point_y - ry);
            rx += t * (-lines[i].dir_y);
            ry += t * lines[i].dir_x;
            float spd = len2d(rx, ry);
            if (spd > radius) { rx *= radius / spd; ry *= radius / spd; }
            distance = det2d(lines[i].dir_x, lines[i].dir_y,
                             lines[i].point_x - rx, lines[i].point_y - ry);
        }
    }
}

__global__ void solve_lp_kernel(
    const float* pref_vx, const float* pref_vy,
    const OrcaLine* all_lines, const int* line_counts,
    float* out_vx, float* out_vy, int n, float max_speed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const OrcaLine* lines = all_lines + i * MAX_NEIGHBORS;
    int nl = line_counts[i];
    float rx, ry;
    if (!linearProgram2(lines, nl, max_speed, pref_vx[i], pref_vy[i], false, rx, ry))
        linearProgram3(lines, nl, 0, max_speed, rx, ry);
    out_vx[i] = rx;
    out_vy[i] = ry;
}

__global__ void update_pos_kernel(
    float* px, float* py, float* vx, float* vy,
    const float* nvx, const float* nvy, int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] = nvx[i]; vy[i] = nvy[i];
    px[i] += vx[i] * dt; py[i] += vy[i] * dt;
}

// =====================================================================
// Visualization
// =====================================================================
cv::Point2i to_px(float x, float y, int sz) {
    float scale = sz / 44.0f;
    return cv::Point2i((int)((x + 2) * scale), (int)((44.0f - (y + 2)) * scale));
}

void draw_scene(cv::Mat& img, float* px, float* py, float* gx, float* gy,
                float* vx, float* vy,
                std::vector<std::vector<cv::Point>>& trails, int n,
                const char* label, float sim_t, bool draw_vel) {
    // obstacles
    for (int i = 0; i < N_OBS; i++)
        cv::circle(img, to_px(h_obs[i].x, h_obs[i].y, img.cols),
                   (int)(h_obs[i].r * img.cols / 44.0f), cv::Scalar(50, 50, 50), -1);
    // trails + robots
    for (int i = 0; i < n; i++) {
        float hue = i * 180.0f / n;
        cv::Mat hsv(1,1,CV_8UC3, cv::Scalar((int)hue, 255, 230));
        cv::Mat rgb; cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar col(rgb.at<cv::Vec3b>(0,0)[0], rgb.at<cv::Vec3b>(0,0)[1], rgb.at<cv::Vec3b>(0,0)[2]);

        trails[i].push_back(to_px(px[i], py[i], img.cols));
        for (size_t j = 1; j < trails[i].size(); j++)
            cv::line(img, trails[i][j-1], trails[i][j], col, 1);

        int rpx = (int)(AGENT_RADIUS * img.cols / 44.0f) + 1;
        cv::circle(img, to_px(px[i], py[i], img.cols), rpx, col, -1);
        cv::circle(img, to_px(px[i], py[i], img.cols), rpx, cv::Scalar(0,0,0), 1);

        // velocity vectors
        if (draw_vel) {
            cv::Point p1 = to_px(px[i], py[i], img.cols);
            cv::Point p2 = to_px(px[i] + vx[i] * 1.5f, py[i] + vy[i] * 1.5f, img.cols);
            cv::line(img, p1, p2, col, 1);
        }

        // goal
        cv::drawMarker(img, to_px(gx[i], gy[i], img.cols), col, cv::MARKER_SQUARE, 4, 1);
    }

    // Count reached
    int reached = 0;
    for (int i = 0; i < n; i++) {
        float dx = px[i] - gx[i], dy = py[i] - gy[i];
        if (sqrtf(dx*dx+dy*dy) < GOAL_TOL) reached++;
    }

    cv::putText(img, label, cv::Point(5, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2);
    char buf[64];
    snprintf(buf, sizeof(buf), "t=%.1fs", sim_t);
    cv::putText(img, buf, cv::Point(img.cols - 90, 22), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    snprintf(buf, sizeof(buf), "Reached: %d/%d", reached, n);
    cv::putText(img, buf, cv::Point(5, 44), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,128,0), 1);
}

// =====================================================================
// Main
// =====================================================================
int main() {
    printf("Comparison: Potential Field vs ORCA (%d agents)\n", N_AGENTS);

    float cx = 20.0f, cy = 20.0f, cr = 15.0f;

    // Host arrays for PF side
    float pf_px[N_AGENTS], pf_py[N_AGENTS], pf_vx[N_AGENTS], pf_vy[N_AGENTS];
    float pf_gx[N_AGENTS], pf_gy[N_AGENTS];
    // Host arrays for ORCA side
    float or_px[N_AGENTS], or_py[N_AGENTS], or_vx[N_AGENTS], or_vy[N_AGENTS];
    float or_gx[N_AGENTS], or_gy[N_AGENTS];

    for (int i = 0; i < N_AGENTS; i++) {
        float a = 2 * PI_F * i / N_AGENTS;
        pf_px[i] = or_px[i] = cx + cr * cosf(a);
        pf_py[i] = or_py[i] = cy + cr * sinf(a);
        pf_vx[i] = or_vx[i] = 0.0f;
        pf_vy[i] = or_vy[i] = 0.0f;
        pf_gx[i] = or_gx[i] = cx + cr * cosf(a + PI_F);
        pf_gy[i] = or_gy[i] = cy + cr * sinf(a + PI_F);
    }

    float h_ox[N_OBS], h_oy[N_OBS], h_or[N_OBS];
    for (int i = 0; i < N_OBS; i++) {
        h_ox[i] = h_obs[i].x; h_oy[i] = h_obs[i].y; h_or[i] = h_obs[i].r;
    }

    size_t ab = N_AGENTS * sizeof(float);
    size_t ob = N_OBS * sizeof(float);

    // --- PF device ---
    float *dpf_px, *dpf_py, *dpf_vx, *dpf_vy, *dpf_gx, *dpf_gy;
    float *dpf_ox, *dpf_oy, *dpf_or;
    CUDA_CHECK(cudaMalloc(&dpf_px, ab)); CUDA_CHECK(cudaMalloc(&dpf_py, ab));
    CUDA_CHECK(cudaMalloc(&dpf_vx, ab)); CUDA_CHECK(cudaMalloc(&dpf_vy, ab));
    CUDA_CHECK(cudaMalloc(&dpf_gx, ab)); CUDA_CHECK(cudaMalloc(&dpf_gy, ab));
    CUDA_CHECK(cudaMalloc(&dpf_ox, ob)); CUDA_CHECK(cudaMalloc(&dpf_oy, ob));
    CUDA_CHECK(cudaMalloc(&dpf_or, ob));

    CUDA_CHECK(cudaMemcpy(dpf_px, pf_px, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_py, pf_py, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_vx, pf_vx, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_vy, pf_vy, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_gx, pf_gx, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_gy, pf_gy, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_ox, h_ox, ob, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_oy, h_oy, ob, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpf_or, h_or, ob, cudaMemcpyHostToDevice));

    // --- ORCA device ---
    float *dor_px, *dor_py, *dor_vx, *dor_vy, *dor_gx, *dor_gy;
    float *dor_ox, *dor_oy, *dor_or;
    float *dor_pvx, *dor_pvy, *dor_nvx, *dor_nvy;
    OrcaLine* dor_lines;
    int* dor_lcounts;

    CUDA_CHECK(cudaMalloc(&dor_px, ab)); CUDA_CHECK(cudaMalloc(&dor_py, ab));
    CUDA_CHECK(cudaMalloc(&dor_vx, ab)); CUDA_CHECK(cudaMalloc(&dor_vy, ab));
    CUDA_CHECK(cudaMalloc(&dor_gx, ab)); CUDA_CHECK(cudaMalloc(&dor_gy, ab));
    CUDA_CHECK(cudaMalloc(&dor_ox, ob)); CUDA_CHECK(cudaMalloc(&dor_oy, ob));
    CUDA_CHECK(cudaMalloc(&dor_or, ob));
    CUDA_CHECK(cudaMalloc(&dor_pvx, ab)); CUDA_CHECK(cudaMalloc(&dor_pvy, ab));
    CUDA_CHECK(cudaMalloc(&dor_nvx, ab)); CUDA_CHECK(cudaMalloc(&dor_nvy, ab));
    CUDA_CHECK(cudaMalloc(&dor_lines, N_AGENTS * MAX_NEIGHBORS * sizeof(OrcaLine)));
    CUDA_CHECK(cudaMalloc(&dor_lcounts, N_AGENTS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dor_px, or_px, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_py, or_py, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_vx, or_vx, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_vy, or_vy, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_gx, or_gx, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_gy, or_gy, ab, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_ox, h_ox, ob, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_oy, h_oy, ob, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dor_or, h_or, ob, cudaMemcpyHostToDevice));

    int bs = 128;
    int gs = (N_AGENTS + bs - 1) / bs;

    cv::VideoWriter video("gif/comparison_orca.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(IMG_W * 2, IMG_W));

    std::vector<std::vector<cv::Point>> trails_pf(N_AGENTS), trails_orca(N_AGENTS);
    int steps = 0;

    for (float t = 0; t < SIM_TIME_VAL; t += DT_SIM) {
        // --- PF step ---
        pf_compute_and_update<<<gs, bs>>>(
            dpf_px, dpf_py, dpf_vx, dpf_vy,
            dpf_gx, dpf_gy, dpf_ox, dpf_oy, dpf_or,
            N_AGENTS, N_OBS, DT_SIM);
        CUDA_CHECK(cudaGetLastError());

        // --- ORCA step ---
        compute_preferred_velocity_kernel<<<gs, bs>>>(
            dor_px, dor_py, dor_gx, dor_gy,
            dor_pvx, dor_pvy, N_AGENTS, MAX_SPEED, GOAL_TOL);
        compute_orca_lines_kernel<<<gs, bs>>>(
            dor_px, dor_py, dor_vx, dor_vy,
            dor_pvx, dor_pvy,
            dor_ox, dor_oy, dor_or,
            N_AGENTS, N_OBS,
            dor_lines, dor_lcounts,
            AGENT_RADIUS, TIME_HORIZON, TIME_HORIZON_OBS, DT_SIM);
        solve_lp_kernel<<<gs, bs>>>(
            dor_pvx, dor_pvy,
            dor_lines, dor_lcounts,
            dor_nvx, dor_nvy, N_AGENTS, MAX_SPEED);
        update_pos_kernel<<<gs, bs>>>(
            dor_px, dor_py, dor_vx, dor_vy,
            dor_nvx, dor_nvy, N_AGENTS, DT_SIM);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back
        CUDA_CHECK(cudaMemcpy(pf_px, dpf_px, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pf_py, dpf_py, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pf_vx, dpf_vx, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pf_vy, dpf_vy, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(or_px, dor_px, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(or_py, dor_py, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(or_vx, dor_vx, ab, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(or_vy, dor_vy, ab, cudaMemcpyDeviceToHost));

        steps++;

        // Draw every 4th frame
        if (steps % 4 != 0) continue;

        cv::Mat left(IMG_W, IMG_W, CV_8UC3, cv::Scalar(245, 245, 245));
        cv::Mat right(IMG_W, IMG_W, CV_8UC3, cv::Scalar(245, 245, 245));

        draw_scene(left, pf_px, pf_py, pf_gx, pf_gy, pf_vx, pf_vy,
                   trails_pf, N_AGENTS, "Potential Field", t, true);
        draw_scene(right, or_px, or_py, or_gx, or_gy, or_vx, or_vy,
                   trails_orca, N_AGENTS, "ORCA", t, true);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);

        cv::imshow("comparison_orca", combined);
        if (cv::waitKey(1) == 27) break;
    }

    video.release();
    system("ffmpeg -y -i gif/comparison_orca.avi "
           "-vf 'fps=20,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_orca.gif 2>/dev/null");

    printf("GIF saved to gif/comparison_orca.gif\n");

    // Cleanup PF
    cudaFree(dpf_px); cudaFree(dpf_py); cudaFree(dpf_vx); cudaFree(dpf_vy);
    cudaFree(dpf_gx); cudaFree(dpf_gy); cudaFree(dpf_ox); cudaFree(dpf_oy); cudaFree(dpf_or);
    // Cleanup ORCA
    cudaFree(dor_px); cudaFree(dor_py); cudaFree(dor_vx); cudaFree(dor_vy);
    cudaFree(dor_gx); cudaFree(dor_gy); cudaFree(dor_ox); cudaFree(dor_oy); cudaFree(dor_or);
    cudaFree(dor_pvx); cudaFree(dor_pvy); cudaFree(dor_nvx); cudaFree(dor_nvy);
    cudaFree(dor_lines); cudaFree(dor_lcounts);

    printf("Done.\n");
    return 0;
}
