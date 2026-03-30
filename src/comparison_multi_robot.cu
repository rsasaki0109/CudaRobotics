/*************************************************************************
    Multi-Robot Planner: CPU vs CUDA side-by-side comparison GIF
    20 robots swapping positions through obstacles
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define PI 3.14159265f
#define N_ROBOTS 500
#define N_OBS 3
#define KP_ATT 5.0f
#define KP_REP 100.0f
#define KP_ROBOT 50.0f
#define ROBOT_RADIUS 0.5f
#define OBS_INFLUENCE 5.0f
#define ROBOT_INFLUENCE 3.0f
#define MAX_SPEED 1.0f
#define DT_SIM 0.05f
#define SIM_TIME 30.0f
#define IMG_SIZE 400

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

struct Obstacle { float x, y, r; };

static Obstacle h_obs[N_OBS] = {{20,20,3},{15,25,2},{25,15,2}};

// ===================== CPU =====================
void cpu_compute_forces(float* px, float* py, float* vx, float* vy,
                        float* gx, float* gy, int n) {
    for (int i = 0; i < n; i++) {
        float fx = 0, fy = 0;
        // attractive
        float dx = gx[i]-px[i], dy = gy[i]-py[i];
        float dg = sqrtf(dx*dx+dy*dy);
        if (dg > 0.01f) { fx += KP_ATT*dx/dg; fy += KP_ATT*dy/dg; }
        // repulsive from obstacles
        for (int j = 0; j < N_OBS; j++) {
            float odx = px[i]-h_obs[j].x, ody = py[i]-h_obs[j].y;
            float od = sqrtf(odx*odx+ody*ody) - h_obs[j].r - ROBOT_RADIUS;
            if (od < OBS_INFLUENCE && od > 0.01f) {
                float f = KP_REP * (1.0f/od - 1.0f/OBS_INFLUENCE) / (od*od);
                fx += f*odx/(od+h_obs[j].r+ROBOT_RADIUS);
                fy += f*ody/(od+h_obs[j].r+ROBOT_RADIUS);
            }
        }
        // repulsive from other robots
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            float rdx = px[i]-px[j], rdy = py[i]-py[j];
            float rd = sqrtf(rdx*rdx+rdy*rdy) - 2*ROBOT_RADIUS;
            if (rd < ROBOT_INFLUENCE && rd > 0.01f) {
                float f = KP_ROBOT * (1.0f/rd - 1.0f/ROBOT_INFLUENCE) / (rd*rd);
                fx += f*rdx/(rd+2*ROBOT_RADIUS);
                fy += f*rdy/(rd+2*ROBOT_RADIUS);
            }
        }
        vx[i] = vx[i]*0.8f + fx*DT_SIM;
        vy[i] = vy[i]*0.8f + fy*DT_SIM;
        float spd = sqrtf(vx[i]*vx[i]+vy[i]*vy[i]);
        if (spd > MAX_SPEED) { vx[i] *= MAX_SPEED/spd; vy[i] *= MAX_SPEED/spd; }
        px[i] += vx[i]*DT_SIM;
        py[i] += vy[i]*DT_SIM;
    }
}

// ===================== CUDA =====================
__constant__ float d_obs_x[N_OBS], d_obs_y[N_OBS], d_obs_r[N_OBS];

__global__ void compute_forces_kernel(float* px, float* py, float* vx, float* vy,
                                      float* gx, float* gy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fx = 0, fy = 0;
    float dx = gx[i]-px[i], dy = gy[i]-py[i];
    float dg = sqrtf(dx*dx+dy*dy);
    if (dg > 0.01f) { fx += KP_ATT*dx/dg; fy += KP_ATT*dy/dg; }
    for (int j = 0; j < N_OBS; j++) {
        float odx = px[i]-d_obs_x[j], ody = py[i]-d_obs_y[j];
        float od = sqrtf(odx*odx+ody*ody) - d_obs_r[j] - ROBOT_RADIUS;
        if (od < OBS_INFLUENCE && od > 0.01f) {
            float f = KP_REP*(1.0f/od-1.0f/OBS_INFLUENCE)/(od*od);
            fx += f*odx/(od+d_obs_r[j]+ROBOT_RADIUS);
            fy += f*ody/(od+d_obs_r[j]+ROBOT_RADIUS);
        }
    }
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        float rdx = px[i]-px[j], rdy = py[i]-py[j];
        float rd = sqrtf(rdx*rdx+rdy*rdy) - 2*ROBOT_RADIUS;
        if (rd < ROBOT_INFLUENCE && rd > 0.01f) {
            float f = KP_ROBOT*(1.0f/rd-1.0f/ROBOT_INFLUENCE)/(rd*rd);
            fx += f*rdx/(rd+2*ROBOT_RADIUS);
            fy += f*rdy/(rd+2*ROBOT_RADIUS);
        }
    }
    vx[i] = vx[i]*0.8f + fx*DT_SIM;
    vy[i] = vy[i]*0.8f + fy*DT_SIM;
    float spd = sqrtf(vx[i]*vx[i]+vy[i]*vy[i]);
    if (spd > MAX_SPEED) { vx[i] *= MAX_SPEED/spd; vy[i] *= MAX_SPEED/spd; }
    px[i] += vx[i]*DT_SIM;
    py[i] += vy[i]*DT_SIM;
}

// ===================== Visualization =====================
cv::Point2i to_px(float x, float y, int sz) {
    float scale = sz / 40.0f;
    return cv::Point2i((int)((x) * scale), (int)((40.0f - y) * scale));
}

void draw(cv::Mat& img, float* px, float* py, float* gx, float* gy,
          std::vector<std::vector<cv::Point>>& trails, int n,
          const char* label, double ms, float sim_t) {
    // obstacles
    for (int i = 0; i < N_OBS; i++)
        cv::circle(img, to_px(h_obs[i].x, h_obs[i].y, img.cols),
                   (int)(h_obs[i].r * img.cols / 40.0f), cv::Scalar(50,50,50), -1);
    // trails + robots
    for (int i = 0; i < n; i++) {
        float hue = i * 180.0f / n;
        cv::Mat hsv(1,1,CV_8UC3, cv::Scalar((int)hue, 255, 255));
        cv::Mat rgb; cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar col(rgb.at<cv::Vec3b>(0,0)[0], rgb.at<cv::Vec3b>(0,0)[1], rgb.at<cv::Vec3b>(0,0)[2]);
        // trail
        trails[i].push_back(to_px(px[i], py[i], img.cols));
        for (size_t j = 1; j < trails[i].size(); j++)
            cv::line(img, trails[i][j-1], trails[i][j], col, 1);
        // robot
        cv::circle(img, to_px(px[i], py[i], img.cols), (int)(ROBOT_RADIUS*img.cols/40.0f)+2, col, -1);
        // goal
        cv::drawMarker(img, to_px(gx[i], gy[i], img.cols), col, cv::MARKER_SQUARE, 6, 1);
    }
    cv::putText(img, label, cv::Point(5, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f ms/step", ms);
    cv::putText(img, buf, cv::Point(5, 48), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,200), 2);
    snprintf(buf, sizeof(buf), "t=%.1fs", sim_t);
    cv::putText(img, buf, cv::Point(img.cols-100, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 1);
}

int main() {
    // Init positions on circle
    float cx = 20, cy = 20, cr = 15;
    float cpu_px[N_ROBOTS], cpu_py[N_ROBOTS], cpu_vx[N_ROBOTS], cpu_vy[N_ROBOTS];
    float cpu_gx[N_ROBOTS], cpu_gy[N_ROBOTS];
    float gpu_px[N_ROBOTS], gpu_py[N_ROBOTS], gpu_vx[N_ROBOTS], gpu_vy[N_ROBOTS];
    float gpu_gx[N_ROBOTS], gpu_gy[N_ROBOTS];

    for (int i = 0; i < N_ROBOTS; i++) {
        float a = 2*PI*i/N_ROBOTS;
        cpu_px[i] = gpu_px[i] = cx + cr*cosf(a);
        cpu_py[i] = gpu_py[i] = cy + cr*sinf(a);
        cpu_vx[i] = gpu_vx[i] = 0;
        cpu_vy[i] = gpu_vy[i] = 0;
        cpu_gx[i] = gpu_gx[i] = cx + cr*cosf(a + PI);
        cpu_gy[i] = gpu_gy[i] = cy + cr*sinf(a + PI);
    }

    // GPU setup
    float *d_px, *d_py, *d_vx, *d_vy, *d_gx, *d_gy;
    CUDA_CHECK(cudaMalloc(&d_px, N_ROBOTS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, N_ROBOTS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vx, N_ROBOTS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy, N_ROBOTS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, N_ROBOTS*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, N_ROBOTS*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_px, gpu_px, N_ROBOTS*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, gpu_py, N_ROBOTS*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, gpu_vx, N_ROBOTS*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, gpu_vy, N_ROBOTS*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx, gpu_gx, N_ROBOTS*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, gpu_gy, N_ROBOTS*sizeof(float), cudaMemcpyHostToDevice));

    float ox[N_OBS], oy[N_OBS], or_[N_OBS];
    for (int i = 0; i < N_OBS; i++) { ox[i]=h_obs[i].x; oy[i]=h_obs[i].y; or_[i]=h_obs[i].r; }
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_x, ox, N_OBS*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_y, oy, N_OBS*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_r, or_, N_OBS*sizeof(float)));

    int W = IMG_SIZE;
    cv::VideoWriter video("gif/comparison_multi_robot.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(W*2, W));

    std::vector<std::vector<cv::Point>> trails_cpu(N_ROBOTS), trails_gpu(N_ROBOTS);
    double cpu_total = 0, cuda_total = 0;
    int steps = 0;

    for (float t = 0; t < SIM_TIME; t += DT_SIM) {
        // CPU step
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_compute_forces(cpu_px, cpu_py, cpu_vx, cpu_vy, cpu_gx, cpu_gy, N_ROBOTS);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        cpu_total += cpu_ms;

        // CUDA step
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        compute_forces_kernel<<<1, 256>>>(d_px, d_py, d_vx, d_vy, d_gx, d_gy, N_ROBOTS);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float cuda_ms; cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cuda_total += cuda_ms;

        CUDA_CHECK(cudaMemcpy(gpu_px, d_px, N_ROBOTS*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_py, d_py, N_ROBOTS*sizeof(float), cudaMemcpyDeviceToHost));

        steps++;

        // Draw every 4th frame
        if (steps % 4 != 0) continue;

        cv::Mat left(W, W, CV_8UC3, cv::Scalar(245,245,245));
        cv::Mat right(W, W, CV_8UC3, cv::Scalar(245,245,245));
        draw(left, cpu_px, cpu_py, cpu_gx, cpu_gy, trails_cpu, N_ROBOTS,
             "CPU (C++)", cpu_total/steps, t);
        draw(right, gpu_px, gpu_py, gpu_gx, gpu_gy, trails_gpu, N_ROBOTS,
             "CUDA (GPU)", cuda_total/steps, t);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    system("ffmpeg -y -i gif/comparison_multi_robot.avi "
           "-vf 'fps=20,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_multi_robot.gif 2>/dev/null");

    printf("CPU avg: %.4f ms/step, CUDA avg: %.4f ms/step\n", cpu_total/steps, cuda_total/steps);
    printf("GIF saved to gif/comparison_multi_robot.gif\n");

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_gx); cudaFree(d_gy);
    return 0;
}
