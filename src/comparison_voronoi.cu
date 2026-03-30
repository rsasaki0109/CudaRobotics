/*************************************************************************
    Voronoi Diagram: CPU vs CUDA side-by-side comparison GIF generator
    Left panel:  CPU (brute-force nearest seed for each cell, O(cells*seeds))
    Right panel: CUDA (Jump Flooding Algorithm, O(cells*log(N)) fully parallel)
    Same map as voronoi_road_map.cu. Shows colored Voronoi regions.
    Measures Voronoi construction time only.
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// JFA Init Kernel: clear grid
// -------------------------------------------------------------------------
__global__ void jfa_init_clear_kernel(int* grid, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grid[idx] = -1;
    }
}

// -------------------------------------------------------------------------
// JFA Init Kernel: seed cells
// -------------------------------------------------------------------------
__global__ void jfa_init_seed_kernel(int* grid, int grid_w, int grid_h,
                                     const float* obs_x, const float* obs_y,
                                     int num_obs, int min_x, int min_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_obs) return;

    int gx = (int)roundf(obs_x[idx]) - min_x;
    int gy = (int)roundf(obs_y[idx]) - min_y;
    if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h) {
        grid[gy * grid_w + gx] = idx;
    }
}

// -------------------------------------------------------------------------
// JFA Step Kernel
// -------------------------------------------------------------------------
__global__ void jfa_step_kernel(int* grid_out, const int* grid_in,
                                int grid_w, int grid_h,
                                const float* seed_x, const float* seed_y,
                                int num_seeds, int min_x, int min_y,
                                int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_w * grid_h;
    if (idx >= total) return;

    int cx = idx % grid_w;
    int cy = idx / grid_w;

    float px = (float)(cx + min_x);
    float py = (float)(cy + min_y);

    int best_seed = grid_in[idx];
    float best_dist = 1e30f;

    if (best_seed >= 0 && best_seed < num_seeds) {
        float dx = px - seed_x[best_seed];
        float dy = py - seed_y[best_seed];
        best_dist = dx * dx + dy * dy;
    }

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx * step;
            int ny = cy + dy * step;
            if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;

            int neighbor_seed = grid_in[ny * grid_w + nx];
            if (neighbor_seed < 0 || neighbor_seed >= num_seeds) continue;

            float sx = px - seed_x[neighbor_seed];
            float sy = py - seed_y[neighbor_seed];
            float d = sx * sx + sy * sy;

            if (d < best_dist) {
                best_dist = d;
                best_seed = neighbor_seed;
            }
        }
    }

    grid_out[idx] = best_seed;
}

// -------------------------------------------------------------------------
// CPU: brute-force Voronoi (O(cells * seeds))
// -------------------------------------------------------------------------
void cpu_voronoi_bruteforce(vector<int>& voronoi_grid,
                            int grid_w, int grid_h,
                            const vector<float>& ox, const vector<float>& oy,
                            int min_x, int min_y)
{
    int num_obs = (int)ox.size();
    int total = grid_w * grid_h;
    voronoi_grid.resize(total, -1);

    for (int cy = 0; cy < grid_h; cy++) {
        for (int cx = 0; cx < grid_w; cx++) {
            float px = (float)(cx + min_x);
            float py = (float)(cy + min_y);

            float best_dist = 1e30f;
            int best_seed = -1;

            for (int s = 0; s < num_obs; s++) {
                float dx = px - ox[s];
                float dy = py - oy[s];
                float d = dx * dx + dy * dy;
                if (d < best_dist) {
                    best_dist = d;
                    best_seed = s;
                }
            }

            voronoi_grid[cy * grid_w + cx] = best_seed;
        }
    }
}

// -------------------------------------------------------------------------
// CUDA: JFA Voronoi construction
// -------------------------------------------------------------------------
void cuda_voronoi_jfa(vector<int>& voronoi_grid,
                      int grid_w, int grid_h,
                      const vector<float>& ox, const vector<float>& oy,
                      int min_x, int min_y)
{
    int num_obs = (int)ox.size();
    int total = grid_w * grid_h;
    voronoi_grid.resize(total, -1);

    float *d_ox, *d_oy;
    int *d_grid_a, *d_grid_b;

    CUDA_CHECK(cudaMalloc(&d_ox, num_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy, num_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_a, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_b, total * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), num_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), num_obs * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int clearGridSize = (total + blockSize - 1) / blockSize;
    int seedGridSize = (num_obs + blockSize - 1) / blockSize;

    jfa_init_clear_kernel<<<clearGridSize, blockSize>>>(d_grid_a, total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    jfa_init_seed_kernel<<<seedGridSize, blockSize>>>(d_grid_a, grid_w, grid_h,
                                                       d_ox, d_oy, num_obs,
                                                       min_x, min_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // JFA passes
    int max_dim = max(grid_w, grid_h);
    int step = 1;
    while (step < max_dim) step <<= 1;
    step >>= 1;

    int gridSizeJFA = (total + blockSize - 1) / blockSize;
    int* src = d_grid_a;
    int* dst = d_grid_b;

    while (step >= 1) {
        jfa_step_kernel<<<gridSizeJFA, blockSize>>>(dst, src,
                                                     grid_w, grid_h,
                                                     d_ox, d_oy, num_obs,
                                                     min_x, min_y, step);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int* tmp = src;
        src = dst;
        dst = tmp;

        step >>= 1;
    }

    CUDA_CHECK(cudaMemcpy(voronoi_grid.data(), src, total * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));
    CUDA_CHECK(cudaFree(d_grid_a));
    CUDA_CHECK(cudaFree(d_grid_b));
}

// -------------------------------------------------------------------------
// Render Voronoi diagram to image
// -------------------------------------------------------------------------
void render_voronoi(cv::Mat& img, const vector<int>& voronoi_grid,
                    int grid_w, int grid_h, int num_obs,
                    const vector<float>& ox, const vector<float>& oy,
                    int min_x, int min_y,
                    const vector<cv::Vec3b>& seed_colors,
                    int img_size)
{
    float scale = (float)img_size / (float)max(grid_w, grid_h);

    img = cv::Mat(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int cy = 0; cy < grid_h; cy++) {
        for (int cx = 0; cx < grid_w; cx++) {
            int seed = voronoi_grid[cy * grid_w + cx];
            if (seed >= 0 && seed < num_obs) {
                int px0 = (int)(cx * scale);
                int py0 = (int)(cy * scale);
                int px1 = (int)((cx + 1) * scale);
                int py1 = (int)((cy + 1) * scale);
                if (px1 > img_size) px1 = img_size;
                if (py1 > img_size) py1 = img_size;
                cv::rectangle(img, cv::Point(px0, py0), cv::Point(px1, py1),
                              cv::Scalar(seed_colors[seed][0], seed_colors[seed][1], seed_colors[seed][2]),
                              -1);
            }
        }
    }

    // Draw obstacles as black dots
    for (int i = 0; i < num_obs; i++) {
        int gx = (int)roundf(ox[i]) - min_x;
        int gy = (int)roundf(oy[i]) - min_y;
        if (gx >= 0 && gx < grid_w && gy >= 0 && gy < grid_h) {
            int px = (int)((gx + 0.5f) * scale);
            int py = (int)((gy + 0.5f) * scale);
            cv::circle(img, cv::Point(px, py), 1, cv::Scalar(0, 0, 0), -1);
        }
    }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    // Same map as voronoi_road_map.cu
    vector<float> ox, oy;

    // Boundary obstacles (rectangular area 0-60 x 0-60)
    for (float i = 0; i <= 60; i++) {
        ox.push_back(i); oy.push_back(0.0f);
    }
    for (float i = 0; i <= 60; i++) {
        ox.push_back(i); oy.push_back(60.0f);
    }
    for (float i = 0; i <= 60; i++) {
        ox.push_back(0.0f); oy.push_back(i);
    }
    for (float i = 0; i <= 60; i++) {
        ox.push_back(60.0f); oy.push_back(i);
    }

    // Internal obstacle: rectangle at (20,10)-(25,35)
    for (float x = 20; x <= 25; x++) {
        for (float y = 10; y <= 35; y++) {
            ox.push_back(x);
            oy.push_back(y);
        }
    }

    // Internal obstacle: rectangle at (35,25)-(40,50)
    for (float x = 35; x <= 40; x++) {
        for (float y = 25; y <= 50; y++) {
            ox.push_back(x);
            oy.push_back(y);
        }
    }

    int num_obs = (int)ox.size();

    // Compute grid bounds
    float fmin_x = *min_element(ox.begin(), ox.end());
    float fmax_x = *max_element(ox.begin(), ox.end());
    float fmin_y = *min_element(oy.begin(), oy.end());
    float fmax_y = *max_element(oy.begin(), oy.end());

    int min_x = (int)floorf(fmin_x) - 2;
    int max_x = (int)ceilf(fmax_x) + 2;
    int min_y = (int)floorf(fmin_y) - 2;
    int max_y = (int)ceilf(fmax_y) + 2;

    int grid_w = max_x - min_x + 1;
    int grid_h = max_y - min_y + 1;

    printf("Grid: %d x %d (%d cells), obstacles: %d\n", grid_w, grid_h, grid_w * grid_h, num_obs);

    // Generate seed colors (deterministic)
    vector<cv::Vec3b> seed_colors(num_obs);
    srand(42);
    for (int i = 0; i < num_obs; i++) {
        seed_colors[i] = cv::Vec3b(100 + rand() % 130, 100 + rand() % 130, 100 + rand() % 130);
    }

    // ======== CPU: brute-force Voronoi ========
    printf("Running CPU brute-force Voronoi...\n");
    vector<int> cpu_voronoi;
    auto cpu_t0 = chrono::high_resolution_clock::now();
    cpu_voronoi_bruteforce(cpu_voronoi, grid_w, grid_h, ox, oy, min_x, min_y);
    auto cpu_t1 = chrono::high_resolution_clock::now();
    double cpu_ms = chrono::duration<double, milli>(cpu_t1 - cpu_t0).count();
    printf("CPU Voronoi: %.2f ms\n", cpu_ms);

    // ======== CUDA: JFA Voronoi ========
    printf("Running CUDA JFA Voronoi...\n");
    vector<int> cuda_voronoi;

    // Warm up GPU
    {
        vector<int> dummy;
        cuda_voronoi_jfa(dummy, grid_w, grid_h, ox, oy, min_x, min_y);
    }

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    cuda_voronoi_jfa(cuda_voronoi, grid_w, grid_h, ox, oy, min_x, min_y);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float cuda_ms;
    cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    printf("CUDA Voronoi (JFA): %.2f ms\n", cuda_ms);

    // ======== Render both sides ========
    int S = 400;
    cv::Mat cpu_img, cuda_img;
    render_voronoi(cpu_img, cpu_voronoi, grid_w, grid_h, num_obs, ox, oy,
                   min_x, min_y, seed_colors, S);
    render_voronoi(cuda_img, cuda_voronoi, grid_w, grid_h, num_obs, ox, oy,
                   min_x, min_y, seed_colors, S);

    // Add timing labels
    char buf[128];

    cv::putText(cpu_img, "CPU (brute-force)", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    snprintf(buf, sizeof(buf), "CPU: %.2f ms", cpu_ms);
    cv::putText(cpu_img, buf, cv::Point(10, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 200), 2);
    snprintf(buf, sizeof(buf), "O(cells * seeds)");
    cv::putText(cpu_img, buf, cv::Point(10, 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

    cv::putText(cuda_img, "CUDA (JFA)", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    snprintf(buf, sizeof(buf), "CUDA: %.2f ms", cuda_ms);
    cv::putText(cuda_img, buf, cv::Point(10, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 200), 2);
    snprintf(buf, sizeof(buf), "O(cells * log(N)) parallel");
    cv::putText(cuda_img, buf, cv::Point(10, 72),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

    // Speedup label
    if (cuda_ms > 0.0f) {
        snprintf(buf, sizeof(buf), "Speedup: %.1fx", cpu_ms / cuda_ms);
        cv::putText(cuda_img, buf, cv::Point(10, 95),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 150, 0), 2);
    }

    cv::Mat combined;
    cv::hconcat(cpu_img, cuda_img, combined);

    // ======== Write video/GIF ========
    string avi_path = "gif/comparison_voronoi.avi";
    string gif_path = "gif/comparison_voronoi.gif";

    cv::VideoWriter video(avi_path,
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                          30, cv::Size(S * 2, S));
    if (!video.isOpened()) {
        cerr << "Failed to open video writer at " << avi_path << endl;
        return 1;
    }

    // Write the construction process as an animation:
    // CPU side builds up progressively (row by row), CUDA side appears quickly

    // Phase 1: CPU building up row by row
    int row_step = max(1, grid_h / 60);  // ~60 frames for build-up
    float scale = (float)S / (float)max(grid_w, grid_h);

    cv::Mat cpu_progressive(S, S, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::Mat cuda_done = cuda_img.clone();

    // CUDA side: show "computing..." then snap to result
    cv::Mat cuda_computing(S, S, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::putText(cuda_computing, "CUDA (JFA)", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    cv::putText(cuda_computing, "Computing...", cv::Point(S / 2 - 60, S / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 200), 2);

    // First few frames: both computing
    for (int f = 0; f < 5; f++) {
        cv::Mat left(S, S, CV_8UC3, cv::Scalar(200, 200, 200));
        cv::putText(left, "CPU (brute-force)", cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(left, "Computing...", cv::Point(S / 2 - 60, S / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 200), 2);
        cv::Mat frame;
        cv::hconcat(left, cuda_computing, frame);
        video.write(frame);
    }

    // CUDA finishes quickly (frame 6) - show result, CPU still building
    for (int row = 0; row < grid_h; row += row_step) {
        int end_row = min(row + row_step, grid_h);

        // Draw rows on cpu_progressive
        for (int cy = row; cy < end_row; cy++) {
            for (int cx = 0; cx < grid_w; cx++) {
                int seed = cpu_voronoi[cy * grid_w + cx];
                if (seed >= 0 && seed < num_obs) {
                    int px0 = (int)(cx * scale);
                    int py0 = (int)(cy * scale);
                    int px1 = (int)((cx + 1) * scale);
                    int py1 = (int)((cy + 1) * scale);
                    if (px1 > S) px1 = S;
                    if (py1 > S) py1 = S;
                    cv::rectangle(cpu_progressive, cv::Point(px0, py0), cv::Point(px1, py1),
                                  cv::Scalar(seed_colors[seed][0], seed_colors[seed][1], seed_colors[seed][2]),
                                  -1);
                }
            }
        }

        cv::Mat left = cpu_progressive.clone();
        cv::putText(left, "CPU (brute-force)", cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Row %d / %d", end_row, grid_h);
        cv::putText(left, buf, cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);

        cv::Mat frame;
        cv::hconcat(left, cuda_done, frame);
        video.write(frame);
    }

    // Hold final combined frame
    for (int f = 0; f < 90; f++) {
        video.write(combined);
    }

    video.release();
    cout << "Video saved to " << avi_path << endl;

    // Convert to GIF
    string cmd = "ffmpeg -y -i " + avi_path +
        " -vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 " + gif_path + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        cout << "GIF saved to " << gif_path << endl;
    } else {
        cout << "ffmpeg conversion failed (ffmpeg may not be installed)" << endl;
    }

    printf("\n=== Results ===\n");
    printf("CPU  (brute-force):  %.2f ms\n", cpu_ms);
    printf("CUDA (JFA):          %.2f ms\n", cuda_ms);
    if (cuda_ms > 0.0f)
        printf("Speedup:             %.1fx\n", cpu_ms / cuda_ms);

    return 0;
}
