/*************************************************************************
    Social Force Model (Helbing & Molnar 1995) - CUDA parallelized
    300 pedestrians crossing flow simulation
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#define N_PED 300
#define DT 0.02f
#define SIM_TIME 30.0f
#define DESIRED_SPEED 1.3f
#define TAU 0.5f
#define A_SOC 2.1f
#define B_SOC 0.3f
#define A_WALL 10.0f
#define B_WALL 0.2f
#define PED_RADIUS 0.3f
#define CORRIDOR_LEN 50.0f
#define CORRIDOR_W 10.0f

__global__ void compute_forces_kernel(
    const float* px, const float* py,
    const float* vx, const float* vy,
    const float* gx, const float* gy,
    float* fx, float* fy, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = px[i], yi = py[i];
    float vxi = vx[i], vyi = vy[i];
    float gxi = gx[i], gyi = gy[i];

    // Desired force
    float dx = gxi - xi, dy = gyi - yi;
    float dist_g = sqrtf(dx * dx + dy * dy) + 1e-6f;
    float desired_vx = DESIRED_SPEED * dx / dist_g;
    float desired_vy = DESIRED_SPEED * dy / dist_g;
    float f_dx = (desired_vx - vxi) / TAU;
    float f_dy = (desired_vy - vyi) / TAU;

    float f_sx = 0, f_sy = 0;

    // Social force from other pedestrians
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float djx = xi - px[j], djy = yi - py[j];
        float dj = sqrtf(djx * djx + djy * djy) + 1e-6f;
        float overlap = 2.0f * PED_RADIUS - dj;
        float nxij = djx / dj, nyij = djy / dj;
        float f_rep = A_SOC * expf(-dj / B_SOC);
        if (overlap > 0) f_rep += 120.0f * overlap;
        f_sx += f_rep * nxij;
        f_sy += f_rep * nyij;
    }

    // Wall forces (top y=CORRIDOR_W, bottom y=0)
    float dist_bot = yi + 1e-6f;
    float dist_top = CORRIDOR_W - yi + 1e-6f;
    f_sy += A_WALL * expf(-dist_bot / B_WALL);
    f_sy -= A_WALL * expf(-dist_top / B_WALL);
    if (dist_bot < PED_RADIUS) f_sy += 120.0f * (PED_RADIUS - dist_bot);
    if (dist_top < PED_RADIUS) f_sy -= 120.0f * (PED_RADIUS - dist_top);

    fx[i] = f_dx + f_sx;
    fy[i] = f_dy + f_sy;
}

__global__ void update_kernel(
    float* px, float* py, float* vx, float* vy,
    const float* fx, const float* fy, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vx[i] += fx[i] * DT;
    vy[i] += fy[i] * DT;
    float spd = sqrtf(vx[i] * vx[i] + vy[i] * vy[i]);
    float max_spd = 2.0f * DESIRED_SPEED;
    if (spd > max_spd) { vx[i] *= max_spd / spd; vy[i] *= max_spd / spd; }
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    // clamp y
    if (py[i] < PED_RADIUS) { py[i] = PED_RADIUS; vy[i] = 0; }
    if (py[i] > CORRIDOR_W - PED_RADIUS) { py[i] = CORRIDOR_W - PED_RADIUS; vy[i] = 0; }
}

int main() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> uni_y(1.0f, CORRIDOR_W - 1.0f);
    std::uniform_real_distribution<float> uni_x_l(0.0f, 5.0f);
    std::uniform_real_distribution<float> uni_x_r(45.0f, 50.0f);

    std::vector<float> h_px(N_PED), h_py(N_PED), h_vx(N_PED, 0), h_vy(N_PED, 0);
    std::vector<float> h_gx(N_PED), h_gy(N_PED);
    std::vector<int> direction(N_PED); // 0=left→right, 1=right→left

    for (int i = 0; i < N_PED; i++) {
        if (i < N_PED / 2) {
            h_px[i] = uni_x_l(gen); h_py[i] = uni_y(gen);
            h_gx[i] = CORRIDOR_LEN + 5.0f; h_gy[i] = h_py[i];
            h_vx[i] = DESIRED_SPEED;
            direction[i] = 0;
        } else {
            h_px[i] = uni_x_r(gen); h_py[i] = uni_y(gen);
            h_gx[i] = -5.0f; h_gy[i] = h_py[i];
            h_vx[i] = -DESIRED_SPEED;
            direction[i] = 1;
        }
    }

    float *d_px, *d_py, *d_vx, *d_vy, *d_gx, *d_gy, *d_fx, *d_fy;
    size_t sz = N_PED * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_px, sz)); CUDA_CHECK(cudaMalloc(&d_py, sz));
    CUDA_CHECK(cudaMalloc(&d_vx, sz)); CUDA_CHECK(cudaMalloc(&d_vy, sz));
    CUDA_CHECK(cudaMalloc(&d_gx, sz)); CUDA_CHECK(cudaMalloc(&d_gy, sz));
    CUDA_CHECK(cudaMalloc(&d_fx, sz)); CUDA_CHECK(cudaMalloc(&d_fy, sz));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx, h_gx.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, h_gy.data(), sz, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N_PED + threads - 1) / threads;

    cv::namedWindow("social_force", cv::WINDOW_NORMAL);
    int img_w = 1000, img_h = 200;
    float scale = img_w / CORRIDOR_LEN;
    cv::VideoWriter video("gif/social_force.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(img_w, img_h));

    std::cout << "Social Force Model with CUDA (" << N_PED << " pedestrians)" << std::endl;

    float t = 0;
    int frame = 0;
    while (t < SIM_TIME) {
        compute_forces_kernel<<<blocks, threads>>>(d_px, d_py, d_vx, d_vy, d_gx, d_gy, d_fx, d_fy, N_PED);
        update_kernel<<<blocks, threads>>>(d_px, d_py, d_vx, d_vy, d_fx, d_fy, N_PED);
        CUDA_CHECK(cudaDeviceSynchronize());
        t += DT;

        if (frame % 5 == 0) { // draw every 5th frame
            CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, sz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_py.data(), d_py, sz, cudaMemcpyDeviceToHost));

            cv::Mat bg(img_h, img_w, CV_8UC3, cv::Scalar(240, 240, 240));
            // walls
            cv::line(bg, cv::Point(0, 0), cv::Point(img_w, 0), cv::Scalar(0,0,0), 3);
            cv::line(bg, cv::Point(0, img_h-1), cv::Point(img_w, img_h-1), cv::Scalar(0,0,0), 3);

            for (int i = 0; i < N_PED; i++) {
                int cx = (int)(h_px[i] * scale);
                int cy = (int)(h_py[i] / CORRIDOR_W * img_h);
                if (cx < 0 || cx >= img_w || cy < 0 || cy >= img_h) continue;
                cv::Scalar color = direction[i] == 0 ? cv::Scalar(255, 100, 50) : cv::Scalar(50, 50, 255);
                cv::circle(bg, cv::Point(cx, cy), 3, color, -1);
            }

            cv::putText(bg, "Social Force Model - " + std::to_string(N_PED) + " pedestrians  t=" +
                        std::to_string(t).substr(0,4) + "s",
                        cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);

            video.write(bg);
            cv::imshow("social_force", bg);
            cv::waitKey(1);
        }
        frame++;
    }

    video.release();
    system("ffmpeg -y -i gif/social_force.avi "
           "-vf 'fps=15,scale=800:-1' -loop 0 "
           "gif/social_force.gif 2>/dev/null");
    std::cout << "GIF saved to gif/social_force.gif" << std::endl;

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_vx); cudaFree(d_vy);
    cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_fx); cudaFree(d_fy);
    return 0;
}
