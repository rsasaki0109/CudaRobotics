/*************************************************************************
    Multi-Robot Visual Comparison: CPU (5 robots) vs CUDA (500 robots)
    Visually demonstrates the scalability advantage of GPU parallelism
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define PI 3.14159265f
#define N_CPU 5
#define N_CUDA 500
#define N_OBS 3
#define KP_ATT 5.0f
#define KP_REP 100.0f
#define KP_ROBOT 50.0f
#define ROBOT_RADIUS 0.5f
#define OBS_INFLUENCE 5.0f
#define ROBOT_INFLUENCE 3.0f
#define MAX_SPEED 1.0f
#define DT_SIM 0.05f
#define SIM_TIME 20.0f
#define IMG_SIZE 400

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

struct Obstacle { float x, y, r; };

static Obstacle h_obs[N_OBS] = {{20,20,3},{15,25,2},{25,15,2}};

// ===================== CPU (5 robots, sequential) =====================
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

// ===================== CUDA (500 robots, parallel) =====================
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
    return cv::Point2i((int)(x * scale), (int)((40.0f - y) * scale));
}

cv::Scalar hsv_color(int i, int n) {
    float hue = i * 180.0f / n;
    cv::Mat hsv(1,1,CV_8UC3, cv::Scalar((int)hue, 255, 255));
    cv::Mat rgb; cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.at<cv::Vec3b>(0,0)[0], rgb.at<cv::Vec3b>(0,0)[1], rgb.at<cv::Vec3b>(0,0)[2]);
}

// 5 distinct colors for CPU robots
static cv::Scalar cpu_colors[N_CPU] = {
    cv::Scalar(0,0,255),     // red
    cv::Scalar(0,180,0),     // green
    cv::Scalar(255,0,0),     // blue
    cv::Scalar(0,165,255),   // orange
    cv::Scalar(180,0,180)    // purple
};

void draw_cpu(cv::Mat& img, float* px, float* py, float* gx, float* gy,
              std::vector<std::vector<cv::Point>>& trails, int n,
              double ms, float sim_t) {
    // obstacles
    for (int i = 0; i < N_OBS; i++)
        cv::circle(img, to_px(h_obs[i].x, h_obs[i].y, img.cols),
                   (int)(h_obs[i].r * img.cols / 40.0f), cv::Scalar(50,50,50), -1);
    // trails + robots
    for (int i = 0; i < n; i++) {
        cv::Scalar col = cpu_colors[i];
        trails[i].push_back(to_px(px[i], py[i], img.cols));
        for (size_t j = 1; j < trails[i].size(); j++)
            cv::line(img, trails[i][j-1], trails[i][j], col, 2);
        // large circle (radius 8px)
        cv::circle(img, to_px(px[i], py[i], img.cols), 8, col, -1);
        cv::circle(img, to_px(px[i], py[i], img.cols), 8, cv::Scalar(0,0,0), 1);
        // goal marker
        cv::drawMarker(img, to_px(gx[i], gy[i], img.cols), col, cv::MARKER_CROSS, 12, 2);
    }
    cv::putText(img, "CPU - 5 robots", cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.3f ms/step", ms);
    cv::putText(img, buf, cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,200), 2);
    snprintf(buf, sizeof(buf), "t=%.1fs", sim_t);
    cv::putText(img, buf, cv::Point(img.cols-100, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,0), 1);
}

void draw_cuda(cv::Mat& img, float* px, float* py, float* gx, float* gy,
               std::vector<std::vector<cv::Point>>& trails, int n,
               double ms, float sim_t) {
    // obstacles
    for (int i = 0; i < N_OBS; i++)
        cv::circle(img, to_px(h_obs[i].x, h_obs[i].y, img.cols),
                   (int)(h_obs[i].r * img.cols / 40.0f), cv::Scalar(50,50,50), -1);
    // trails + robots (small circles with HSV colormap)
    for (int i = 0; i < n; i++) {
        cv::Scalar col = hsv_color(i, n);
        trails[i].push_back(to_px(px[i], py[i], img.cols));
        for (size_t j = 1; j < trails[i].size(); j++)
            cv::line(img, trails[i][j-1], trails[i][j], col, 1);
        // small circle (radius 3px)
        cv::circle(img, to_px(px[i], py[i], img.cols), 3, col, -1);
    }
    cv::putText(img, "CUDA - 500 robots", cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 2);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.3f ms/step", ms);
    cv::putText(img, buf, cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,200), 2);
    snprintf(buf, sizeof(buf), "t=%.1fs", sim_t);
    cv::putText(img, buf, cv::Point(img.cols-100, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,0), 1);
}

int main() {
    float cx = 20, cy = 20, cr = 15;

    // CPU: 5 robots
    float cpu_px[N_CPU], cpu_py[N_CPU], cpu_vx[N_CPU], cpu_vy[N_CPU];
    float cpu_gx[N_CPU], cpu_gy[N_CPU];
    for (int i = 0; i < N_CPU; i++) {
        float a = 2*PI*i/N_CPU;
        cpu_px[i] = cx + cr*cosf(a);
        cpu_py[i] = cy + cr*sinf(a);
        cpu_vx[i] = cpu_vy[i] = 0;
        cpu_gx[i] = cx + cr*cosf(a + PI);
        cpu_gy[i] = cy + cr*sinf(a + PI);
    }

    // CUDA: 500 robots
    float *gpu_px = new float[N_CUDA], *gpu_py = new float[N_CUDA];
    float *gpu_vx = new float[N_CUDA], *gpu_vy = new float[N_CUDA];
    float *gpu_gx = new float[N_CUDA], *gpu_gy = new float[N_CUDA];
    for (int i = 0; i < N_CUDA; i++) {
        float a = 2*PI*i/N_CUDA;
        gpu_px[i] = cx + cr*cosf(a);
        gpu_py[i] = cy + cr*sinf(a);
        gpu_vx[i] = gpu_vy[i] = 0;
        gpu_gx[i] = cx + cr*cosf(a + PI);
        gpu_gy[i] = cy + cr*sinf(a + PI);
    }

    // GPU memory allocation
    float *d_px, *d_py, *d_vx, *d_vy, *d_gx, *d_gy;
    CUDA_CHECK(cudaMalloc(&d_px, N_CUDA*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, N_CUDA*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vx, N_CUDA*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy, N_CUDA*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, N_CUDA*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gy, N_CUDA*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_px, gpu_px, N_CUDA*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, gpu_py, N_CUDA*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, gpu_vx, N_CUDA*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, gpu_vy, N_CUDA*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx, gpu_gx, N_CUDA*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, gpu_gy, N_CUDA*sizeof(float), cudaMemcpyHostToDevice));

    float ox[N_OBS], oy[N_OBS], or_[N_OBS];
    for (int i = 0; i < N_OBS; i++) { ox[i]=h_obs[i].x; oy[i]=h_obs[i].y; or_[i]=h_obs[i].r; }
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_x, ox, N_OBS*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_y, oy, N_OBS*sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_obs_r, or_, N_OBS*sizeof(float)));

    int W = IMG_SIZE;
    cv::VideoWriter video("gif/comparison_multi_robot_visual.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(W*2, W));

    std::vector<std::vector<cv::Point>> trails_cpu(N_CPU), trails_gpu(N_CUDA);
    double cpu_total = 0, cuda_total = 0;
    int steps = 0;
    int blocks = (N_CUDA + 255) / 256;

    for (float t = 0; t < SIM_TIME; t += DT_SIM) {
        // CPU step (5 robots)
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_compute_forces(cpu_px, cpu_py, cpu_vx, cpu_vy, cpu_gx, cpu_gy, N_CPU);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        cpu_total += cpu_ms;

        // CUDA step (500 robots)
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        compute_forces_kernel<<<blocks, 256>>>(d_px, d_py, d_vx, d_vy, d_gx, d_gy, N_CUDA);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float cuda_ms; cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cuda_total += cuda_ms;

        CUDA_CHECK(cudaMemcpy(gpu_px, d_px, N_CUDA*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpu_py, d_py, N_CUDA*sizeof(float), cudaMemcpyDeviceToHost));

        steps++;

        // Draw every 4th frame
        if (steps % 4 != 0) continue;

        cv::Mat left(W, W, CV_8UC3, cv::Scalar(230,230,230));
        cv::Mat right(W, W, CV_8UC3, cv::Scalar(230,230,230));

        draw_cpu(left, cpu_px, cpu_py, cpu_gx, cpu_gy, trails_cpu, N_CPU,
                 cpu_total/steps, t);
        draw_cuda(right, gpu_px, gpu_py, gpu_gx, gpu_gy, trails_gpu, N_CUDA,
                  cuda_total/steps, t);

        // separator line
        cv::Mat combined;
        cv::hconcat(left, right, combined);
        cv::line(combined, cv::Point(W, 0), cv::Point(W, W), cv::Scalar(100,100,100), 2);
        video.write(combined);
    }

    video.release();
    system("ffmpeg -y -i gif/comparison_multi_robot_visual.avi "
           "-vf 'fps=20,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_multi_robot_visual.gif 2>/dev/null");

    printf("CPU (5 robots) avg: %.4f ms/step\n", cpu_total/steps);
    printf("CUDA (500 robots) avg: %.4f ms/step\n", cuda_total/steps);
    printf("Speedup: %.1fx (with 100x more robots!)\n", (cpu_total/steps * 100.0) / (cuda_total/steps));
    printf("GIF saved to gif/comparison_multi_robot_visual.gif\n");

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_gx); cudaFree(d_gy);
    delete[] gpu_px; delete[] gpu_py; delete[] gpu_vx; delete[] gpu_vy; delete[] gpu_gx; delete[] gpu_gy;
    return 0;
}
