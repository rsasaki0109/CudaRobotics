/*************************************************************************
    DWA Visual Comparison: CPU (~50 samples) vs CUDA (~50,000 samples)
    Shows the VISUAL difference in trajectory quality and smoothness.
    Left panel: CPU with coarse sampling (sparse candidate trajectories)
    Right panel: CUDA with fine sampling (dense candidate trajectories)
 ************************************************************************/

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#define PI 3.141592653f

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ===================== Types =====================
using State   = std::array<float, 5>;  // x, y, yaw, v, omega
using Control = std::array<float, 2>;
using Point   = std::array<float, 2>;
using Obstacle = std::vector<std::array<float, 2>>;
using Traj    = std::vector<std::array<float, 5>>;

// ===================== Config =====================
struct DWAConfig {
    float max_speed      = 1.0f;
    float min_speed      = -0.5f;
    float max_yawrate    = 40.0f * PI / 180.0f;
    float max_accel      = 0.2f;
    float robot_radius   = 1.0f;
    float max_dyawrate   = 40.0f * PI / 180.0f;
    float v_reso;
    float yawrate_reso;
    float dt             = 0.1f;
    float predict_time   = 3.0f;
    float to_goal_cost_gain = 1.0f;
    float speed_cost_gain   = 1.0f;
};

// ===================== Motion Model =====================
State motion(State x, Control u, float dt) {
    x[2] += u[1] * dt;
    x[0] += u[0] * cosf(x[2]) * dt;
    x[1] += u[0] * sinf(x[2]) * dt;
    x[3] = u[0];
    x[4] = u[1];
    return x;
}

// ===================== CPU DWA (coarse ~50 samples) =====================
Traj cpu_calc_trajectory(State x, float v, float yr, const DWAConfig& c) {
    Traj traj;
    traj.push_back(x);
    float t = 0.0f;
    while (t <= c.predict_time) {
        x = motion(x, {{v, yr}}, c.dt);
        traj.push_back(x);
        t += c.dt;
    }
    return traj;
}

float cpu_obstacle_cost(Traj& traj, Obstacle& ob, float rr) {
    float minr = FLT_MAX;
    for (unsigned int i = 0; i < traj.size(); i += 2)
        for (auto& o : ob) {
            float r = sqrtf((traj[i][0]-o[0])*(traj[i][0]-o[0]) +
                            (traj[i][1]-o[1])*(traj[i][1]-o[1]));
            if (r <= rr) return FLT_MAX;
            if (r < minr) minr = r;
        }
    return 1.0f / minr;
}

// Returns best trajectory AND populates all_trajs with every candidate
Traj cpu_dwa(State x, Control& u, const DWAConfig& c, Point goal,
             Obstacle& ob, std::vector<Traj>& all_trajs) {
    float dw[4] = {
        fmaxf(x[3] - c.max_accel * c.dt, c.min_speed),
        fminf(x[3] + c.max_accel * c.dt, c.max_speed),
        fmaxf(x[4] - c.max_dyawrate * c.dt, -c.max_yawrate),
        fminf(x[4] + c.max_dyawrate * c.dt, c.max_yawrate)
    };
    float min_cost = FLT_MAX;
    Traj best;
    all_trajs.clear();

    for (float v = dw[0]; v <= dw[1]; v += c.v_reso) {
        for (float yr = dw[2]; yr <= dw[3]; yr += c.yawrate_reso) {
            Traj traj = cpu_calc_trajectory(x, v, yr, c);

            float gm = sqrtf(goal[0]*goal[0] + goal[1]*goal[1]);
            float tm = sqrtf(traj.back()[0]*traj.back()[0] +
                             traj.back()[1]*traj.back()[1]);
            float dot = goal[0]*traj.back()[0] + goal[1]*traj.back()[1];
            float ca = fminf(fmaxf(dot / (gm * tm + 1e-10f), -1.0f), 1.0f);
            float gc = c.to_goal_cost_gain * acosf(ca);
            float sc = c.speed_cost_gain * (c.max_speed - v);
            float oc = cpu_obstacle_cost(traj, ob, c.robot_radius);
            float fc = gc + sc + oc;

            if (oc < FLT_MAX) {
                all_trajs.push_back(traj);
            }

            if (fc < min_cost) {
                min_cost = fc;
                u = {{v, yr}};
                best = traj;
            }
        }
    }
    return best;
}

// ===================== CUDA DWA Kernel (~50K samples) =====================
__global__ void dwa_eval_kernel(
    float sx, float sy, float syaw, float sv, float so,
    float v_min, float v_max, float yr_min, float yr_max,
    float v_reso, float yr_reso, float dt, float predict_time,
    float max_speed, float to_goal_cost_gain, float speed_cost_gain,
    float robot_radius,
    float gx, float gy,
    const float* ob, int n_ob,
    int n_v, int n_yr,
    float* costs, float* ctrl_v, float* ctrl_yr,
    // Store full trajectory for visualization (predict_time/dt + 1 steps)
    float* traj_x, float* traj_y, int max_steps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_v * n_yr) return;

    float v  = v_min + (idx / n_yr) * v_reso;
    float yr = yr_min + (idx % n_yr) * yr_reso;
    if (v > v_max) v = v_max;
    if (yr > yr_max) yr = yr_max;
    ctrl_v[idx] = v;
    ctrl_yr[idx] = yr;

    float px = sx, py = sy, pyaw = syaw;
    float t = 0.0f;
    float minr = FLT_MAX;
    bool coll = false;
    int step = 0;

    // Store initial position
    traj_x[idx * max_steps] = px;
    traj_y[idx * max_steps] = py;
    step = 1;

    while (t <= predict_time && step < max_steps) {
        pyaw += yr * dt;
        px += v * cosf(pyaw) * dt;
        py += v * sinf(pyaw) * dt;
        t += dt;

        traj_x[idx * max_steps + step] = px;
        traj_y[idx * max_steps + step] = py;
        step++;

        for (int i = 0; i < n_ob; i++) {
            float dx = px - ob[i * 2];
            float dy = py - ob[i * 2 + 1];
            float r = sqrtf(dx * dx + dy * dy);
            if (r <= robot_radius) coll = true;
            if (r < minr) minr = r;
        }
    }

    // Fill remaining steps with last position
    for (int s = step; s < max_steps; s++) {
        traj_x[idx * max_steps + s] = px;
        traj_y[idx * max_steps + s] = py;
    }

    if (coll) { costs[idx] = FLT_MAX; return; }

    float gm = sqrtf(gx * gx + gy * gy);
    float tm = sqrtf(px * px + py * py);
    float ca = fminf(fmaxf((gx * px + gy * py) / (gm * tm + 1e-10f), -1.0f), 1.0f);
    costs[idx] = to_goal_cost_gain * acosf(ca) +
                 speed_cost_gain * (max_speed - v) +
                 1.0f / minr;
}

__global__ void find_min_kernel(const float* costs, int* min_idx, int n) {
    extern __shared__ char sm[];
    float* sv = (float*)sm;
    int* si = (int*)(sm + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    float bv = FLT_MAX;
    int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (costs[i] < bv) { bv = costs[i]; bi = i; }
    }
    sv[tid] = bv;
    si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] < sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) *min_idx = si[0];
}

// ===================== Visualization =====================
static const int PANEL_W = 800;
static const int PANEL_H = 800;

cv::Point2i cv_offset(float x, float y, int w, int h) {
    return cv::Point2i(int(x * 55) + w / 2 - 100, h - int(y * 55) - h / 4);
}

void draw_scene(cv::Mat& img, State& x, Traj& best_traj, Traj& history,
                Point& goal, Obstacle& ob, bool terminal,
                std::vector<Traj>* candidate_trajs,
                const char* title, int sample_count) {
    int w = img.cols, h = img.rows;

    // Draw candidate trajectories (thin light green)
    if (candidate_trajs) {
        for (auto& ct : *candidate_trajs) {
            for (unsigned int j = 1; j < ct.size(); j++) {
                cv::line(img,
                         cv_offset(ct[j-1][0], ct[j-1][1], w, h),
                         cv_offset(ct[j][0], ct[j][1], w, h),
                         cv::Scalar(144, 238, 144), 1, cv::LINE_AA);
            }
        }
    }

    // Draw obstacles (black filled circles)
    for (auto& o : ob) {
        cv::circle(img, cv_offset(o[0], o[1], w, h), 15, cv::Scalar(0, 0, 0), -1);
    }

    // Draw goal (blue circle)
    cv::circle(img, cv_offset(goal[0], goal[1], w, h), 20,
               cv::Scalar(255, 0, 0), 3);
    cv::putText(img, "Goal", cv::Point(cv_offset(goal[0], goal[1], w, h).x - 20,
                cv_offset(goal[0], goal[1], w, h).y - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    // Draw history trajectory (thin blue line)
    for (unsigned int j = 1; j < history.size(); j++) {
        cv::line(img,
                 cv_offset(history[j-1][0], history[j-1][1], w, h),
                 cv_offset(history[j][0], history[j][1], w, h),
                 cv::Scalar(200, 100, 50), 1, cv::LINE_AA);
    }

    // Draw best predicted trajectory (thick dark green)
    for (unsigned int j = 1; j < best_traj.size(); j++) {
        cv::line(img,
                 cv_offset(best_traj[j-1][0], best_traj[j-1][1], w, h),
                 cv_offset(best_traj[j][0], best_traj[j][1], w, h),
                 cv::Scalar(0, 160, 0), 3, cv::LINE_AA);
    }

    // Draw robot (red circle with direction arrow)
    cv::circle(img, cv_offset(x[0], x[1], w, h), 12, cv::Scalar(0, 0, 255), -1);
    cv::arrowedLine(img,
                    cv_offset(x[0], x[1], w, h),
                    cv_offset(x[0] + 0.8f * cosf(x[2]),
                              x[1] + 0.8f * sinf(x[2]), w, h),
                    cv::Scalar(0, 0, 200), 2, cv::LINE_AA);

    // Terminal: draw full path in red
    if (terminal) {
        for (unsigned int j = 1; j < history.size(); j++) {
            cv::line(img,
                     cv_offset(history[j-1][0], history[j-1][1], w, h),
                     cv_offset(history[j][0], history[j][1], w, h),
                     cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }

    // Title and sample count
    cv::putText(img, title, cv::Point(15, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);
    char buf[128];
    snprintf(buf, sizeof(buf), "Samples: ~%d", sample_count);
    cv::putText(img, buf, cv::Point(15, 65),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(80, 80, 80), 1);
}

// ===================== Main =====================
int main() {
    // Same start, goal, obstacles for both
    State x_cpu  = {{0.0f, 0.0f, PI / 8.0f, 0.0f, 0.0f}};
    State x_cuda = {{0.0f, 0.0f, PI / 8.0f, 0.0f, 0.0f}};
    Point goal = {{10.0f, 10.0f}};
    Obstacle ob = {
        {{-1.0f, -1.0f}}, {{ 0.0f,  2.0f}}, {{ 4.0f,  2.0f}},
        {{ 5.0f,  4.0f}}, {{ 5.0f,  5.0f}}, {{ 5.0f,  6.0f}},
        {{ 5.0f,  9.0f}}, {{ 8.0f,  9.0f}}, {{ 7.0f,  9.0f}},
        {{12.0f, 12.0f}}
    };

    Control u_cpu = {{0.0f, 0.0f}}, u_cuda = {{0.0f, 0.0f}};

    // CPU config: coarse resolution -> ~50 samples
    DWAConfig cfg_cpu;
    cfg_cpu.v_reso       = 0.1f;
    cfg_cpu.yawrate_reso = 2.0f * PI / 180.0f;

    // CUDA config: fine resolution -> ~50K samples
    DWAConfig cfg_cuda;
    cfg_cuda.v_reso       = 0.005f;
    cfg_cuda.yawrate_reso = 0.05f * PI / 180.0f;

    Traj hist_cpu, hist_cuda;
    hist_cpu.push_back(x_cpu);
    hist_cuda.push_back(x_cuda);

    int n_ob = (int)ob.size();

    // Trajectory step count for CUDA storage
    int max_steps = (int)(cfg_cuda.predict_time / cfg_cuda.dt) + 2;

    // GPU allocations for CUDA DWA
    int max_nv  = (int)((cfg_cuda.max_speed - cfg_cuda.min_speed) / cfg_cuda.v_reso) + 2;
    int max_nyr = (int)((2.0f * cfg_cuda.max_yawrate) / cfg_cuda.yawrate_reso) + 2;
    int max_samples = max_nv * max_nyr;

    std::cout << "CUDA max grid: " << max_nv << " x " << max_nyr
              << " = " << max_samples << " samples" << std::endl;

    float *d_ob;
    CUDA_CHECK(cudaMalloc(&d_ob, n_ob * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ob, ob.data(), n_ob * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_costs, *d_cv, *d_cyr;
    int *d_mi;
    CUDA_CHECK(cudaMalloc(&d_costs, max_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cv, max_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cyr, max_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mi, sizeof(int)));

    // Trajectory storage on GPU (for visualization subsampling)
    float *d_traj_x, *d_traj_y;
    CUDA_CHECK(cudaMalloc(&d_traj_x, (size_t)max_samples * max_steps * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_traj_y, (size_t)max_samples * max_steps * sizeof(float)));

    // Host buffers for trajectory readback (subsample ~200 trajectories)
    const int VIS_TRAJ_COUNT = 200;
    std::vector<float> h_costs(max_samples);

    cv::VideoWriter video(
        "gif/comparison_dwa_visual.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
        cv::Size(PANEL_W * 2, PANEL_H));

    if (!video.isOpened()) {
        std::cerr << "Failed to open video writer" << std::endl;
        return 1;
    }

    std::cout << "Running visual comparison..." << std::endl;

    for (int iter = 0; iter < 1000; iter++) {
        // ========== CPU DWA (coarse) ==========
        std::vector<Traj> cpu_candidates;
        Traj lt_cpu = cpu_dwa(x_cpu, u_cpu, cfg_cpu, goal, ob, cpu_candidates);
        int cpu_sample_count = (int)cpu_candidates.size();

        x_cpu = motion(x_cpu, u_cpu, cfg_cpu.dt);
        hist_cpu.push_back(x_cpu);

        // ========== CUDA DWA (fine) ==========
        float dw[4] = {
            fmaxf(x_cuda[3] - cfg_cuda.max_accel * cfg_cuda.dt, cfg_cuda.min_speed),
            fminf(x_cuda[3] + cfg_cuda.max_accel * cfg_cuda.dt, cfg_cuda.max_speed),
            fmaxf(x_cuda[4] - cfg_cuda.max_dyawrate * cfg_cuda.dt, -cfg_cuda.max_yawrate),
            fminf(x_cuda[4] + cfg_cuda.max_dyawrate * cfg_cuda.dt, cfg_cuda.max_yawrate)
        };
        int nv  = (int)((dw[1] - dw[0]) / cfg_cuda.v_reso) + 1;
        int nyr = (int)((dw[3] - dw[2]) / cfg_cuda.yawrate_reso) + 1;
        int ns  = nv * nyr;
        int blk = (ns + 255) / 256;

        dwa_eval_kernel<<<blk, 256>>>(
            x_cuda[0], x_cuda[1], x_cuda[2], x_cuda[3], x_cuda[4],
            dw[0], dw[1], dw[2], dw[3],
            cfg_cuda.v_reso, cfg_cuda.yawrate_reso,
            cfg_cuda.dt, cfg_cuda.predict_time,
            cfg_cuda.max_speed, cfg_cuda.to_goal_cost_gain,
            cfg_cuda.speed_cost_gain, cfg_cuda.robot_radius,
            goal[0], goal[1], d_ob, n_ob,
            nv, nyr, d_costs, d_cv, d_cyr,
            d_traj_x, d_traj_y, max_steps);

        find_min_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int))>>>(
            d_costs, d_mi, ns);

        int hmi;
        float bv, byr;
        CUDA_CHECK(cudaMemcpy(&hmi, d_mi, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&bv, d_cv + hmi, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&byr, d_cyr + hmi, sizeof(float), cudaMemcpyDeviceToHost));

        // Get costs for subsampling valid trajectories
        CUDA_CHECK(cudaMemcpy(h_costs.data(), d_costs, ns * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Collect indices of valid (non-collision) trajectories
        std::vector<int> valid_indices;
        for (int j = 0; j < ns; j++) {
            if (h_costs[j] < FLT_MAX) valid_indices.push_back(j);
        }

        // Subsample ~VIS_TRAJ_COUNT trajectories evenly from valid set
        std::vector<Traj> cuda_candidates;
        if (!valid_indices.empty()) {
            int step_vis = std::max(1, (int)valid_indices.size() / VIS_TRAJ_COUNT);
            std::vector<float> h_tx(max_steps), h_ty(max_steps);
            for (int j = 0; j < (int)valid_indices.size(); j += step_vis) {
                int tidx = valid_indices[j];
                CUDA_CHECK(cudaMemcpy(h_tx.data(),
                    d_traj_x + (size_t)tidx * max_steps,
                    max_steps * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_ty.data(),
                    d_traj_y + (size_t)tidx * max_steps,
                    max_steps * sizeof(float), cudaMemcpyDeviceToHost));
                Traj ct;
                for (int s = 0; s < max_steps; s++) {
                    ct.push_back({{h_tx[s], h_ty[s], 0.0f, 0.0f, 0.0f}});
                }
                cuda_candidates.push_back(ct);
            }
        }

        // Best trajectory for CUDA (on host)
        Traj lt_cuda = cpu_calc_trajectory(x_cuda, bv, byr, cfg_cuda);

        u_cuda = {{bv, byr}};
        x_cuda = motion(x_cuda, u_cuda, cfg_cuda.dt);
        hist_cuda.push_back(x_cuda);

        int cuda_sample_count = ns;

        // ========== Draw ==========
        cv::Mat left(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat right(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(255, 255, 255));

        bool term_cpu  = sqrtf((x_cpu[0] - goal[0]) * (x_cpu[0] - goal[0]) +
                               (x_cpu[1] - goal[1]) * (x_cpu[1] - goal[1]))
                         <= cfg_cpu.robot_radius;
        bool term_cuda = sqrtf((x_cuda[0] - goal[0]) * (x_cuda[0] - goal[0]) +
                               (x_cuda[1] - goal[1]) * (x_cuda[1] - goal[1]))
                         <= cfg_cuda.robot_radius;

        draw_scene(left, x_cpu, lt_cpu, hist_cpu, goal, ob, term_cpu,
                   &cpu_candidates, "CPU: ~50 samples", cpu_sample_count);
        draw_scene(right, x_cuda, lt_cuda, hist_cuda, goal, ob, term_cuda,
                   &cuda_candidates, "CUDA: ~50,000 samples", cuda_sample_count);

        // Separator line
        cv::Mat combined;
        cv::hconcat(left, right, combined);
        cv::line(combined, cv::Point(PANEL_W, 0), cv::Point(PANEL_W, PANEL_H),
                 cv::Scalar(100, 100, 100), 2);

        video.write(combined);

        if (iter % 50 == 0) {
            std::cout << "  iter " << iter
                      << "  CPU samples: " << cpu_sample_count
                      << "  CUDA samples: " << cuda_sample_count << std::endl;
        }

        if (term_cpu && term_cuda) {
            std::cout << "Both reached goal at iter " << iter << std::endl;
            // Write a few extra frames at the end
            for (int k = 0; k < 30; k++) video.write(combined);
            break;
        }
    }

    video.release();
    std::cout << "Video saved to gif/comparison_dwa_visual.avi" << std::endl;

    // Convert to GIF
    system("ffmpeg -y -i gif/comparison_dwa_visual.avi "
           "-vf 'fps=15,scale=1200:-1:flags=lanczos' -loop 0 "
           "gif/comparison_dwa_visual.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_dwa_visual.gif" << std::endl;

    // Cleanup
    cudaFree(d_ob);
    cudaFree(d_costs);
    cudaFree(d_cv);
    cudaFree(d_cyr);
    cudaFree(d_mi);
    cudaFree(d_traj_x);
    cudaFree(d_traj_y);

    return 0;
}
