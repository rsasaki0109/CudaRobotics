/*************************************************************************
    DWA: CPU vs CUDA side-by-side comparison GIF generator
    Runs both versions and creates a combined visualization
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

// Config
struct DWAConfig {
    float max_speed = 1.0f, min_speed = -0.5f;
    float max_yawrate = 40.0f * PI / 180.0f;
    float max_accel = 0.2f, robot_radius = 1.0f;
    float max_dyawrate = 40.0f * PI / 180.0f;
    float v_reso = 0.01f, yawrate_reso = 0.1f * PI / 180.0f;
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

// ===================== CPU DWA =====================
Traj cpu_calc_trajectory(State x, float v, float y, const DWAConfig& c) {
    Traj traj; traj.push_back(x);
    float t = 0;
    while (t <= c.predict_time) { x = motion(x, {{v, y}}, c.dt); traj.push_back(x); t += c.dt; }
    return traj;
}

float cpu_obstacle_cost(Traj& traj, Obstacle& ob, float rr) {
    float minr = FLT_MAX;
    for (unsigned int i = 0; i < traj.size(); i += 2)
        for (auto& o : ob) {
            float r = sqrtf((traj[i][0]-o[0])*(traj[i][0]-o[0]) + (traj[i][1]-o[1])*(traj[i][1]-o[1]));
            if (r <= rr) return FLT_MAX;
            if (r < minr) minr = r;
        }
    return 1.0f / minr;
}

Traj cpu_dwa(State x, Control& u, const DWAConfig& c, Point goal, Obstacle& ob) {
    float dw[4] = {
        fmaxf(x[3]-c.max_accel*c.dt, c.min_speed), fminf(x[3]+c.max_accel*c.dt, c.max_speed),
        fmaxf(x[4]-c.max_dyawrate*c.dt, -c.max_yawrate), fminf(x[4]+c.max_dyawrate*c.dt, c.max_yawrate)
    };
    float min_cost = FLT_MAX; Traj best;
    for (float v = dw[0]; v <= dw[1]; v += c.v_reso) {
        for (float yr = dw[2]; yr <= dw[3]; yr += c.yawrate_reso) {
            Traj traj = cpu_calc_trajectory(x, v, yr, c);
            float gm = sqrtf(goal[0]*goal[0]+goal[1]*goal[1]);
            float tm = sqrtf(traj.back()[0]*traj.back()[0]+traj.back()[1]*traj.back()[1]);
            float dot = goal[0]*traj.back()[0]+goal[1]*traj.back()[1];
            float ca = fminf(fmaxf(dot/(gm*tm+1e-10f),-1.0f),1.0f);
            float gc = c.to_goal_cost_gain * acosf(ca);
            float sc = c.speed_cost_gain * (c.max_speed - v);
            float oc = cpu_obstacle_cost(traj, ob, c.robot_radius);
            float fc = gc + sc + oc;
            if (fc < min_cost) { min_cost = fc; u = {{v, yr}}; best = traj; }
        }
    }
    return best;
}

// ===================== CUDA DWA =====================
__global__ void dwa_eval_kernel(
    float sx, float sy, float syaw, float sv, float so,
    float v_min, float v_max, float yr_min, float yr_max,
    float v_reso, float yr_reso, float dt, float predict_time,
    float max_speed, float to_goal_cost_gain, float speed_cost_gain, float robot_radius,
    float gx, float gy, const float* ob, int n_ob,
    int n_v, int n_yr, float* costs, float* ctrl_v, float* ctrl_yr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_v * n_yr) return;
    float v = v_min + (idx / n_yr) * v_reso;
    float yr = yr_min + (idx % n_yr) * yr_reso;
    if (v > v_max) v = v_max; if (yr > yr_max) yr = yr_max;
    ctrl_v[idx] = v; ctrl_yr[idx] = yr;

    float px = sx, py = sy, pyaw = syaw, t = 0;
    float minr = FLT_MAX; bool coll = false;
    while (t <= predict_time) {
        pyaw += yr * dt; px += v * cosf(pyaw) * dt; py += v * sinf(pyaw) * dt; t += dt;
        for (int i = 0; i < n_ob; i++) {
            float r = sqrtf((px-ob[i*2])*(px-ob[i*2])+(py-ob[i*2+1])*(py-ob[i*2+1]));
            if (r <= robot_radius) coll = true;
            if (r < minr) minr = r;
        }
    }
    if (coll) { costs[idx] = FLT_MAX; return; }
    float gm = sqrtf(gx*gx+gy*gy), tm = sqrtf(px*px+py*py);
    float ca = fminf(fmaxf((gx*px+gy*py)/(gm*tm+1e-10f),-1.0f),1.0f);
    costs[idx] = to_goal_cost_gain*acosf(ca) + speed_cost_gain*(max_speed-v) + 1.0f/minr;
}

__global__ void find_min_kernel(const float* costs, int* min_idx, int n) {
    extern __shared__ char sm[];
    float* sv = (float*)sm; int* si = (int*)(sm + blockDim.x*sizeof(float));
    int tid = threadIdx.x;
    float bv = FLT_MAX; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) if (costs[i] < bv) { bv = costs[i]; bi = i; }
    sv[tid] = bv; si[tid] = bi; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s && sv[tid+s] < sv[tid]) { sv[tid] = sv[tid+s]; si[tid] = si[tid+s]; } __syncthreads(); }
    if (tid == 0) *min_idx = si[0];
}

// ===================== Visualization =====================
cv::Point2i cv_offset(float x, float y, int w, int h) {
    return cv::Point2i(int(x*100)+w/2, h-int(y*100)-h/3);
}

void draw_scene(cv::Mat& img, State& x, Traj& ltraj, Traj& traj, Point& goal, Obstacle& ob, bool terminal, const char* label, double ms) {
    cv::circle(img, cv_offset(goal[0], goal[1], img.cols, img.rows), 30, cv::Scalar(255,0,0), 5);
    for (auto& o : ob) cv::circle(img, cv_offset(o[0], o[1], img.cols, img.rows), 20, cv::Scalar(0,0,0), -1);
    for (auto& p : ltraj) cv::circle(img, cv_offset(p[0], p[1], img.cols, img.rows), 7, cv::Scalar(0,255,0), -1);
    cv::circle(img, cv_offset(x[0], x[1], img.cols, img.rows), 30, cv::Scalar(0,0,255), 5);
    cv::arrowedLine(img, cv_offset(x[0], x[1], img.cols, img.rows),
        cv_offset(x[0]+cosf(x[2]), x[1]+sinf(x[2]), img.cols, img.rows), cv::Scalar(255,0,255), 7);
    if (terminal) for (auto& p : traj) cv::circle(img, cv_offset(p[0], p[1], img.cols, img.rows), 7, cv::Scalar(0,0,255), -1);
    cv::putText(img, label, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0,0,0), 3);
    char buf[64]; snprintf(buf, sizeof(buf), "%.1f ms/step", ms);
    cv::putText(img, buf, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,200), 2);
}

int main() {
    State x_cpu = {{0,0,PI/8,0,0}}, x_cuda = {{0,0,PI/8,0,0}};
    Point goal = {{10,10}};
    Obstacle ob = {{{-1,-1}},{{0,2}},{{4,2}},{{5,4}},{{5,5}},{{5,6}},{{5,9}},{{8,9}},{{7,9}},{{12,12}}};
    Control u_cpu = {{0,0}}, u_cuda = {{0,0}};
    DWAConfig config;
    Traj traj_cpu, traj_cuda;
    traj_cpu.push_back(x_cpu); traj_cuda.push_back(x_cuda);

    int n_ob = ob.size();
    int max_nv = (int)((config.max_speed-config.min_speed)/config.v_reso)+2;
    int max_nyr = (int)(2*config.max_yawrate/config.yawrate_reso)+2;
    int max_s = max_nv * max_nyr;

    float *d_ob, *d_costs, *d_cv, *d_cyr; int *d_mi;
    CUDA_CHECK(cudaMalloc(&d_ob, n_ob*2*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ob, ob.data(), n_ob*2*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_costs, max_s*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cv, max_s*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cyr, max_s*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mi, sizeof(int)));

    int W = 1750, H = 1750;
    cv::VideoWriter video("gif/comparison_dwa.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(W*2, H));

    for (int i = 0; i < 1000; i++) {
        // CPU
        auto t0 = std::chrono::high_resolution_clock::now();
        Traj lt_cpu = cpu_dwa(x_cpu, u_cpu, config, goal, ob);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        x_cpu = motion(x_cpu, u_cpu, config.dt);
        traj_cpu.push_back(x_cpu);

        // CUDA
        float dw[4] = {
            fmaxf(x_cuda[3]-config.max_accel*config.dt, config.min_speed),
            fminf(x_cuda[3]+config.max_accel*config.dt, config.max_speed),
            fmaxf(x_cuda[4]-config.max_dyawrate*config.dt, -config.max_yawrate),
            fminf(x_cuda[4]+config.max_dyawrate*config.dt, config.max_yawrate)
        };
        int nv = (int)((dw[1]-dw[0])/config.v_reso)+1;
        int nyr = (int)((dw[3]-dw[2])/config.yawrate_reso)+1;
        int ns = nv*nyr, blk = (ns+255)/256;

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        dwa_eval_kernel<<<blk, 256>>>(x_cuda[0],x_cuda[1],x_cuda[2],x_cuda[3],x_cuda[4],
            dw[0],dw[1],dw[2],dw[3],config.v_reso,config.yawrate_reso,config.dt,config.predict_time,
            config.max_speed,config.to_goal_cost_gain,config.speed_cost_gain,config.robot_radius,
            goal[0],goal[1],d_ob,n_ob,nv,nyr,d_costs,d_cv,d_cyr);
        find_min_kernel<<<1,256,256*(sizeof(float)+sizeof(int))>>>(d_costs,d_mi,ns);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float cuda_ms; cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);

        int hmi; float bv, byr;
        CUDA_CHECK(cudaMemcpy(&hmi, d_mi, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&bv, d_cv+hmi, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&byr, d_cyr+hmi, sizeof(float), cudaMemcpyDeviceToHost));
        u_cuda = {{bv, byr}};
        Traj lt_cuda = cpu_calc_trajectory(x_cuda, bv, byr, config);
        x_cuda = motion(x_cuda, u_cuda, config.dt);
        traj_cuda.push_back(x_cuda);

        // Draw
        cv::Mat left(H, W, CV_8UC3, cv::Scalar(255,255,255));
        cv::Mat right(H, W, CV_8UC3, cv::Scalar(255,255,255));
        bool term_cpu = sqrtf((x_cpu[0]-goal[0])*(x_cpu[0]-goal[0])+(x_cpu[1]-goal[1])*(x_cpu[1]-goal[1])) <= config.robot_radius;
        bool term_cuda = sqrtf((x_cuda[0]-goal[0])*(x_cuda[0]-goal[0])+(x_cuda[1]-goal[1])*(x_cuda[1]-goal[1])) <= config.robot_radius;

        draw_scene(left, x_cpu, lt_cpu, traj_cpu, goal, ob, term_cpu, "CPU (C++)", cpu_ms);
        draw_scene(right, x_cuda, lt_cuda, traj_cuda, goal, ob, term_cuda, "CUDA (GPU)", cuda_ms);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);

        if (term_cpu && term_cuda) break;
    }

    video.release();
    std::cout << "Comparison video saved to gif/comparison_dwa.avi" << std::endl;

    // Convert to gif
    system("ffmpeg -y -i gif/comparison_dwa.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_dwa.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_dwa.gif" << std::endl;

    cudaFree(d_ob); cudaFree(d_costs); cudaFree(d_cv); cudaFree(d_cyr); cudaFree(d_mi);
    return 0;
}
