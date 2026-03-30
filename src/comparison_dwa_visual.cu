/*************************************************************************
    DWA Visual Comparison: CPU ~50 samples vs CUDA ~50,000 samples
    Left: coarse resolution (v_reso=0.1, yr_reso=2deg), all candidate trajectories shown
    Right: fine resolution (v_reso=0.005, yr_reso=0.05deg), subsample of trajectories shown
    800x800 per side
 ************************************************************************/

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#define PI 3.141592653f

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

struct DWAConfig {
    float max_speed = 1.0f, min_speed = -0.5f;
    float max_yawrate = 40.0f * PI / 180.0f;
    float max_accel = 0.2f, robot_radius = 1.0f;
    float max_dyawrate = 40.0f * PI / 180.0f;
    float dt = 0.1f, predict_time = 3.0f;
    float to_goal_cost_gain = 1.0f, speed_cost_gain = 1.0f;
};

using State = std::array<float, 5>;
using Control = std::array<float, 2>;
using Point = std::array<float, 2>;
using Obstacle = std::vector<std::array<float, 2>>;
using Traj = std::vector<std::array<float, 5>>;

State motion(State x, Control u, float dt) {
    x[2] += u[1] * dt;
    x[0] += u[0] * cosf(x[2]) * dt;
    x[1] += u[0] * sinf(x[2]) * dt;
    x[3] = u[0]; x[4] = u[1];
    return x;
}

Traj calc_traj(State x, float v, float yr, const DWAConfig& c) {
    Traj t; t.push_back(x); float tm = 0;
    while (tm <= c.predict_time) { x = motion(x, {{v, yr}}, c.dt); t.push_back(x); tm += c.dt; }
    return t;
}

// CPU DWA with all trajectories output
Traj cpu_dwa_visual(State x, Control& u, const DWAConfig& c, Point goal, Obstacle& ob,
    float v_reso, float yr_reso, std::vector<Traj>& all_trajs) {
    float dw[4] = {
        fmaxf(x[3] - c.max_accel * c.dt, c.min_speed),
        fminf(x[3] + c.max_accel * c.dt, c.max_speed),
        fmaxf(x[4] - c.max_dyawrate * c.dt, -c.max_yawrate),
        fminf(x[4] + c.max_dyawrate * c.dt, c.max_yawrate)
    };
    float mc = FLT_MAX; Traj best; all_trajs.clear();

    for (float v = dw[0]; v <= dw[1]; v += v_reso) {
        for (float yr = dw[2]; yr <= dw[3]; yr += yr_reso) {
            Traj traj = calc_traj(x, v, yr, c);
            all_trajs.push_back(traj);

            // Obstacle cost
            float minr = FLT_MAX; bool coll = false;
            for (unsigned i = 0; i < traj.size(); i += 2) {
                for (auto& o : ob) {
                    float r = sqrtf((traj[i][0] - o[0]) * (traj[i][0] - o[0]) + (traj[i][1] - o[1]) * (traj[i][1] - o[1]));
                    if (r <= c.robot_radius) coll = true;
                    if (r < minr) minr = r;
                }
            }
            if (coll) continue;

            float gm = sqrtf(goal[0] * goal[0] + goal[1] * goal[1]);
            float tm = sqrtf(traj.back()[0] * traj.back()[0] + traj.back()[1] * traj.back()[1]);
            float ca = fminf(fmaxf((goal[0] * traj.back()[0] + goal[1] * traj.back()[1]) / (gm * tm + 1e-10f), -1.0f), 1.0f);
            float fc = c.to_goal_cost_gain * acosf(ca) + c.speed_cost_gain * (c.max_speed - v) + 1.0f / minr;
            if (fc < mc) { mc = fc; u = {{v, yr}}; best = traj; }
        }
    }
    return best;
}

// CUDA kernel
__global__ void dwa_eval_kernel(
    float sx, float sy, float syaw, float sv, float so,
    float v_min, float v_max, float yr_min, float yr_max,
    float v_reso, float yr_reso, float dt, float predict_time,
    float max_speed, float tg_gain, float sp_gain, float rr,
    float gx, float gy, const float* ob, int n_ob,
    int n_v, int n_yr,
    float* costs, float* ctrl_v, float* ctrl_yr,
    float* traj_x, float* traj_y, int traj_steps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_v * n_yr) return;
    float v = v_min + (idx / n_yr) * v_reso;
    float yr = yr_min + (idx % n_yr) * yr_reso;
    if (v > v_max) v = v_max; if (yr > yr_max) yr = yr_max;
    ctrl_v[idx] = v; ctrl_yr[idx] = yr;

    float px = sx, py = sy, pyaw = syaw, t = 0;
    float minr = FLT_MAX; bool coll = false;
    int step = 0;
    while (t <= predict_time) {
        pyaw += yr * dt; px += v * cosf(pyaw) * dt; py += v * sinf(pyaw) * dt; t += dt;
        // Store trajectory points
        if (step < traj_steps) {
            traj_x[idx * traj_steps + step] = px;
            traj_y[idx * traj_steps + step] = py;
        }
        step++;
        for (int i = 0; i < n_ob; i++) {
            float r = sqrtf((px - ob[i * 2]) * (px - ob[i * 2]) + (py - ob[i * 2 + 1]) * (py - ob[i * 2 + 1]));
            if (r <= rr) coll = true;
            if (r < minr) minr = r;
        }
    }
    if (coll) { costs[idx] = FLT_MAX; return; }
    float gm = sqrtf(gx * gx + gy * gy), tm = sqrtf(px * px + py * py);
    float ca = fminf(fmaxf((gx * px + gy * py) / (gm * tm + 1e-10f), -1.0f), 1.0f);
    costs[idx] = tg_gain * acosf(ca) + sp_gain * (max_speed - v) + 1.0f / minr;
}

__global__ void find_min_kernel(const float* costs, int* min_idx, int n) {
    extern __shared__ char sm[];
    float* sv = (float*)sm; int* si = (int*)(sm + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    float bv = FLT_MAX; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) if (costs[i] < bv) { bv = costs[i]; bi = i; }
    sv[tid] = bv; si[tid] = bi; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] < sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *min_idx = si[0];
}

// Visualization
cv::Point2i cv_offset(float x, float y, int w, int h) {
    return cv::Point2i(int(x * 80) + w / 2, h - int(y * 80) - h / 3);
}

void draw_scene(cv::Mat& img, State& x, Traj& best, Traj& hist, Point& goal, Obstacle& ob,
    const char* label, int n_samples) {
    int W = img.cols, H = img.rows;
    cv::circle(img, cv_offset(goal[0], goal[1], W, H), 20, cv::Scalar(255, 0, 0), 3);
    for (auto& o : ob) cv::circle(img, cv_offset(o[0], o[1], W, H), 15, cv::Scalar(0, 0, 0), -1);
    for (auto& p : best) cv::circle(img, cv_offset(p[0], p[1], W, H), 4, cv::Scalar(0, 200, 0), -1);
    cv::circle(img, cv_offset(x[0], x[1], W, H), 20, cv::Scalar(0, 0, 255), 3);
    cv::arrowedLine(img, cv_offset(x[0], x[1], W, H),
        cv_offset(x[0] + cosf(x[2]), x[1] + sinf(x[2]), W, H), cv::Scalar(255, 0, 255), 4);
    cv::putText(img, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    char buf[64]; snprintf(buf, sizeof(buf), "%d samples", n_samples);
    cv::putText(img, buf, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 200), 2);
}

int main() {
    State x_cpu = {{0, 0, PI / 8, 0, 0}}, x_cuda = {{0, 0, PI / 8, 0, 0}};
    Point goal = {{10, 10}};
    Obstacle ob = {{{-1, -1}}, {{0, 2}}, {{4, 2}}, {{5, 4}}, {{5, 5}}, {{5, 6}}, {{5, 9}}, {{8, 9}}, {{7, 9}}, {{12, 12}}};
    Control u_cpu = {{0, 0}}, u_cuda = {{0, 0}};
    DWAConfig config;
    Traj traj_cpu, traj_cuda;
    traj_cpu.push_back(x_cpu); traj_cuda.push_back(x_cuda);

    // CPU: coarse resolution ~50 samples
    float cpu_v_reso = 0.1f;
    float cpu_yr_reso = 2.0f * PI / 180.0f;

    // CUDA: fine resolution ~50K samples
    float cuda_v_reso = 0.005f;
    float cuda_yr_reso = 0.05f * PI / 180.0f;

    int n_ob = ob.size();
    int max_nv = (int)((config.max_speed - config.min_speed) / cuda_v_reso) + 2;
    int max_nyr = (int)(2.0f * config.max_yawrate / cuda_yr_reso) + 2;
    int max_s = max_nv * max_nyr;
    int traj_steps = (int)(config.predict_time / config.dt) + 2;

    float *d_ob, *d_costs, *d_cv, *d_cyr, *d_tx, *d_ty;
    int *d_mi;
    CUDA_CHECK(cudaMalloc(&d_ob, n_ob * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ob, ob.data(), n_ob * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_costs, max_s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cv, max_s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cyr, max_s * sizeof(float)));
    // Limit trajectory storage to avoid OOM: store for subsample only
    int traj_store = std::min(max_s, 200000);
    CUDA_CHECK(cudaMalloc(&d_tx, traj_store * traj_steps * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ty, traj_store * traj_steps * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mi, sizeof(int)));

    int W = 800, H = 800;
    cv::VideoWriter video("gif/comparison_dwa_visual.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(W * 2, H));

    for (int i = 0; i < 1000; i++) {
        // CPU DWA with all trajectories
        std::vector<Traj> cpu_all_trajs;
        Traj lt_cpu = cpu_dwa_visual(x_cpu, u_cpu, config, goal, ob, cpu_v_reso, cpu_yr_reso, cpu_all_trajs);
        x_cpu = motion(x_cpu, u_cpu, config.dt);
        traj_cpu.push_back(x_cpu);
        int cpu_samples = cpu_all_trajs.size();

        // CUDA DWA
        float dw[4] = {
            fmaxf(x_cuda[3] - config.max_accel * config.dt, config.min_speed),
            fminf(x_cuda[3] + config.max_accel * config.dt, config.max_speed),
            fmaxf(x_cuda[4] - config.max_dyawrate * config.dt, -config.max_yawrate),
            fminf(x_cuda[4] + config.max_dyawrate * config.dt, config.max_yawrate)
        };
        int nv = (int)((dw[1] - dw[0]) / cuda_v_reso) + 1;
        int nyr = (int)((dw[3] - dw[2]) / cuda_yr_reso) + 1;
        int ns = nv * nyr;
        int actual_store = std::min(ns, traj_store);

        dwa_eval_kernel<<<(ns + 255) / 256, 256>>>(
            x_cuda[0], x_cuda[1], x_cuda[2], x_cuda[3], x_cuda[4],
            dw[0], dw[1], dw[2], dw[3],
            cuda_v_reso, cuda_yr_reso, config.dt, config.predict_time,
            config.max_speed, config.to_goal_cost_gain, config.speed_cost_gain, config.robot_radius,
            goal[0], goal[1], d_ob, n_ob, nv, nyr,
            d_costs, d_cv, d_cyr, d_tx, d_ty, traj_steps);

        find_min_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int))>>>(d_costs, d_mi, ns);
        int hmi; float bv, byr;
        CUDA_CHECK(cudaMemcpy(&hmi, d_mi, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&bv, d_cv + hmi, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&byr, d_cyr + hmi, sizeof(float), cudaMemcpyDeviceToHost));
        u_cuda = {{bv, byr}};
        Traj lt_cuda = calc_traj(x_cuda, bv, byr, config);
        x_cuda = motion(x_cuda, u_cuda, config.dt);
        traj_cuda.push_back(x_cuda);

        // Read back subsample of trajectories for visualization
        int vis_count = std::min(actual_store, 500);
        std::vector<float> h_tx(vis_count * traj_steps), h_ty(vis_count * traj_steps);
        // Sample evenly
        int step_size = std::max(1, actual_store / vis_count);
        // Just read first vis_count trajectories (they span the space)
        CUDA_CHECK(cudaMemcpy(h_tx.data(), d_tx, vis_count * traj_steps * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ty.data(), d_ty, vis_count * traj_steps * sizeof(float), cudaMemcpyDeviceToHost));

        // Draw LEFT: CPU with all trajectories
        cv::Mat left(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        // Draw all candidate trajectories (thin gray)
        for (auto& tr : cpu_all_trajs) {
            for (unsigned j = 1; j < tr.size(); j++) {
                cv::line(left, cv_offset(tr[j - 1][0], tr[j - 1][1], W, H),
                    cv_offset(tr[j][0], tr[j][1], W, H), cv::Scalar(200, 200, 200), 1);
            }
        }
        draw_scene(left, x_cpu, lt_cpu, traj_cpu, goal, ob, "CPU (coarse)", cpu_samples);

        // Draw RIGHT: CUDA with subsample of trajectories
        cv::Mat right(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        for (int t = 0; t < vis_count; t++) {
            for (int j = 1; j < traj_steps; j++) {
                float x1 = h_tx[t * traj_steps + j - 1], y1 = h_ty[t * traj_steps + j - 1];
                float x2 = h_tx[t * traj_steps + j], y2 = h_ty[t * traj_steps + j];
                if (fabsf(x1) < 50 && fabsf(y1) < 50 && fabsf(x2) < 50 && fabsf(y2) < 50)
                    cv::line(right, cv_offset(x1, y1, W, H), cv_offset(x2, y2, W, H),
                        cv::Scalar(220, 220, 220), 1);
            }
        }
        draw_scene(right, x_cuda, lt_cuda, traj_cuda, goal, ob, "CUDA (fine)", ns);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);

        bool term_cpu = sqrtf((x_cpu[0] - goal[0]) * (x_cpu[0] - goal[0]) + (x_cpu[1] - goal[1]) * (x_cpu[1] - goal[1])) <= config.robot_radius;
        bool term_cuda = sqrtf((x_cuda[0] - goal[0]) * (x_cuda[0] - goal[0]) + (x_cuda[1] - goal[1]) * (x_cuda[1] - goal[1])) <= config.robot_radius;
        if (term_cpu && term_cuda) break;
    }

    video.release();
    system("ffmpeg -y -i gif/comparison_dwa_visual.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_dwa_visual.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_dwa_visual.gif" << std::endl;

    cudaFree(d_ob); cudaFree(d_costs); cudaFree(d_cv); cudaFree(d_cyr);
    cudaFree(d_tx); cudaFree(d_ty); cudaFree(d_mi);
    return 0;
}
