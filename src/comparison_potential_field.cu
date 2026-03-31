/*************************************************************************
    Potential Field: CPU vs CUDA side-by-side comparison
    Left panel:  CPU double loop over grid cells
    Right panel: CUDA kernel (1 thread per grid cell)
    Output: gif/comparison_potential_field.gif
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <set>
#include <chrono>
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const float KP         = 5.0f;
const float ETA        = 100.0f;
const float AREA_WIDTH = 30.0f;
const float grid_reso  = 0.5f;
const float rr         = 5.0f;

// ---------------------------------------------------------------------------
// CPU: compute potential field (double loop over grid)
// ---------------------------------------------------------------------------
void cpu_calc_potential_field(
    float gx, float gy,
    const vector<float>& ox, const vector<float>& oy,
    vector<float>& h_pmap,
    float min_x, float min_y,
    int xwidth, int ywidth)
{
    int n_obs = (int)ox.size();
    int total = xwidth * ywidth;
    h_pmap.resize(total, 0.0f);

    for (int ix = 0; ix < xwidth; ix++) {
        for (int iy = 0; iy < ywidth; iy++) {
            float wx = (float)ix * grid_reso + min_x;
            float wy = (float)iy * grid_reso + min_y;

            // Attractive potential
            float dx = wx - gx;
            float dy = wy - gy;
            float dist_goal = sqrtf(dx * dx + dy * dy);
            float u_att = 0.5f * KP * dist_goal;

            // Repulsive potential
            float u_rep = 0.0f;
            for (int k = 0; k < n_obs; k++) {
                float odx = wx - ox[k];
                float ody = wy - oy[k];
                float d = sqrtf(odx * odx + ody * ody);
                if (d <= 0.001f) {
                    u_rep = 1.0e6f;
                    break;
                }
                if (d <= rr) {
                    float inv_diff = 1.0f / d - 1.0f / rr;
                    u_rep += 0.5f * ETA * inv_diff * inv_diff;
                }
            }

            h_pmap[ix * ywidth + iy] = u_att + u_rep;
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: compute potential field (1 thread per grid cell)
// ---------------------------------------------------------------------------
__global__ void calc_potential_field_kernel(
    float* d_pmap,
    int xwidth, int ywidth,
    float min_x, float min_y,
    float gx, float gy,
    const float* d_ox, const float* d_oy,
    int n_obs,
    float reso, float kp, float eta, float robot_radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = xwidth * ywidth;
    if (idx >= total) return;

    int ix = idx / ywidth;
    int iy = idx % ywidth;

    float wx = (float)ix * reso + min_x;
    float wy = (float)iy * reso + min_y;

    // Attractive potential
    float dx = wx - gx;
    float dy = wy - gy;
    float dist_goal = sqrtf(dx * dx + dy * dy);
    float u_att = 0.5f * kp * dist_goal;

    // Repulsive potential
    float u_rep = 0.0f;
    for (int k = 0; k < n_obs; k++) {
        float odx = wx - d_ox[k];
        float ody = wy - d_oy[k];
        float d = sqrtf(odx * odx + ody * ody);
        if (d <= 0.001f) {
            u_rep = 1.0e6f;
            break;
        }
        if (d <= robot_radius) {
            float inv_diff = 1.0f / d - 1.0f / robot_radius;
            u_rep += 0.5f * eta * inv_diff * inv_diff;
        }
    }

    d_pmap[idx] = u_att + u_rep;
}

// ---------------------------------------------------------------------------
// Gradient descent path following (shared, operates on flat pmap)
// ---------------------------------------------------------------------------
bool gradient_descent_path(
    const vector<float>& pmap,
    int xw, int yw,
    float min_x, float min_y,
    float sx, float sy, float gx, float gy,
    vector<float>& rx, vector<float>& ry)
{
    int ix = (int)roundf((sx - min_x) / grid_reso);
    int iy = (int)roundf((sy - min_y) / grid_reso);
    int gix = (int)roundf((gx - min_x) / grid_reso);
    int giy = (int)roundf((gy - min_y) / grid_reso);

    rx.clear(); ry.clear();
    rx.push_back(sx); ry.push_back(sy);

    int motion_x[] = {1, 0, -1, 0, -1, -1, 1, 1};
    int motion_y[] = {0, 1, 0, -1, -1, 1, -1, 1};

    set<pair<int,int>> visited;
    const int oscillation_window = 3;
    vector<pair<int,int>> recent_positions;

    while (true) {
        float dist_to_goal = sqrtf((float)(ix - gix) * (ix - gix) +
                                   (float)(iy - giy) * (iy - giy)) * grid_reso;
        if (dist_to_goal < grid_reso) {
            rx.push_back(gx); ry.push_back(gy);
            return true;
        }

        float min_potential = numeric_limits<float>::max();
        int min_ix = ix, min_iy = iy;
        for (int i = 0; i < 8; i++) {
            int nx = ix + motion_x[i];
            int ny = iy + motion_y[i];
            if (nx >= 0 && nx < xw && ny >= 0 && ny < yw) {
                if (pmap[nx * yw + ny] < min_potential) {
                    min_potential = pmap[nx * yw + ny];
                    min_ix = nx; min_iy = ny;
                }
            }
        }

        ix = min_ix; iy = min_iy;
        float wx = (float)ix * grid_reso + min_x;
        float wy = (float)iy * grid_reso + min_y;
        rx.push_back(wx); ry.push_back(wy);

        pair<int,int> pos = {ix, iy};
        if (visited.count(pos)) break;
        recent_positions.push_back(pos);
        if ((int)recent_positions.size() > oscillation_window)
            visited.erase(recent_positions[recent_positions.size() - oscillation_window - 1]);
        visited.insert(pos);

        if ((int)rx.size() > xw * yw) break;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Draw heatmap + path on an image
// ---------------------------------------------------------------------------
void draw_potential_field(
    cv::Mat& img,
    const vector<float>& pmap,
    int xw, int yw,
    float min_x, float min_y,
    float sx, float sy, float gx, float gy,
    const vector<float>& ox, const vector<float>& oy,
    const vector<float>& rx, const vector<float>& ry,
    const char* label, double ms)
{
    // Find min/max potential for normalization
    float p_min = numeric_limits<float>::max();
    float p_max = -numeric_limits<float>::max();
    for (int i = 0; i < xw; i++) {
        for (int j = 0; j < yw; j++) {
            float v = pmap[i * yw + j];
            if (v < 1.0e5f) {
                p_min = fminf(p_min, v);
                p_max = fmaxf(p_max, v);
            }
        }
    }
    if (p_max <= p_min) p_max = p_min + 1.0f;

    // Scale to fit target image size
    int cell_size = max(1, min(img.cols / yw, img.rows / xw));
    int img_w = yw * cell_size;
    int img_h = xw * cell_size;

    // Create heatmap
    cv::Mat gray(img_h, img_w, CV_8UC1);
    for (int i = 0; i < xw; i++) {
        for (int j = 0; j < yw; j++) {
            float v = pmap[i * yw + j];
            if (v > 1.0e5f) v = p_max;
            float norm = (v - p_min) / (p_max - p_min);
            norm = fminf(fmaxf(norm, 0.0f), 1.0f);
            unsigned char pixel = (unsigned char)(255.0f * (1.0f - norm));
            for (int di = 0; di < cell_size; di++)
                for (int dj = 0; dj < cell_size; dj++)
                    gray.at<uchar>(i * cell_size + di, j * cell_size + dj) = pixel;
        }
    }

    cv::Mat colored;
    cv::applyColorMap(gray, colored, cv::COLORMAP_JET);

    // Resize to target image
    cv::resize(colored, img, img.size());

    // Helper: world coords -> image pixel (scaled)
    float sx_scale = (float)img.cols / (float)img_w;
    float sy_scale = (float)img.rows / (float)img_h;
    auto to_pixel = [&](float wx, float wy) -> cv::Point {
        int iix = (int)roundf((wx - min_x) / grid_reso);
        int iiy = (int)roundf((wy - min_y) / grid_reso);
        return cv::Point((int)((iiy * cell_size + cell_size / 2) * sx_scale),
                         (int)((iix * cell_size + cell_size / 2) * sy_scale));
    };

    // Draw obstacles (black)
    for (int k = 0; k < (int)ox.size(); k++)
        cv::circle(img, to_pixel(ox[k], oy[k]), 6, cv::Scalar(0, 0, 0), -1);

    // Draw start (green) and goal (blue)
    cv::circle(img, to_pixel(sx, sy), 8, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, to_pixel(gx, gy), 8, cv::Scalar(255, 0, 0), -1);

    // Draw path (red line)
    for (int i = 0; i < (int)rx.size() - 1; i++)
        cv::line(img, to_pixel(rx[i], ry[i]), to_pixel(rx[i + 1], ry[i + 1]),
                 cv::Scalar(0, 0, 255), 2);

    // Label and timing
    cv::putText(img, label, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    char buf[64];
    snprintf(buf, sizeof(buf), "Field: %.2f ms", ms);
    cv::putText(img, buf, cv::Point(10, 65),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    cout << "Potential Field: CPU vs CUDA Comparison" << endl;

    float sx = 0.0f,  sy = 10.0f;
    float gx = 30.0f, gy = 30.0f;

    vector<float> ox = {15.0f, 5.0f, 20.0f, 25.0f};
    vector<float> oy = {25.0f, 15.0f, 26.0f, 25.0f};
    int n_obs = (int)ox.size();

    // Compute grid bounds
    float min_x = *min_element(ox.begin(), ox.end()) - AREA_WIDTH / 2.0f;
    float min_y = *min_element(oy.begin(), oy.end()) - AREA_WIDTH / 2.0f;
    float max_x = *max_element(ox.begin(), ox.end()) + AREA_WIDTH / 2.0f;
    float max_y = *max_element(oy.begin(), oy.end()) + AREA_WIDTH / 2.0f;
    min_x = fminf(min_x, gx - AREA_WIDTH / 2.0f);
    min_y = fminf(min_y, gy - AREA_WIDTH / 2.0f);
    max_x = fmaxf(max_x, gx + AREA_WIDTH / 2.0f);
    max_y = fmaxf(max_y, gy + AREA_WIDTH / 2.0f);

    int xwidth = (int)((max_x - min_x) / grid_reso);
    int ywidth = (int)((max_y - min_y) / grid_reso);
    int total = xwidth * ywidth;

    cout << "Grid: " << xwidth << " x " << ywidth << " = " << total << " cells" << endl;

    // ===================== CPU =====================
    vector<float> cpu_pmap;
    auto t0 = chrono::high_resolution_clock::now();
    cpu_calc_potential_field(gx, gy, ox, oy, cpu_pmap, min_x, min_y, xwidth, ywidth);
    auto t1 = chrono::high_resolution_clock::now();
    double cpu_ms = chrono::duration<double, milli>(t1 - t0).count();
    printf("CPU field computation: %.2f ms\n", cpu_ms);

    // CPU path
    vector<float> cpu_rx, cpu_ry;
    gradient_descent_path(cpu_pmap, xwidth, ywidth, min_x, min_y,
                          sx, sy, gx, gy, cpu_rx, cpu_ry);

    // ===================== CUDA =====================
    float *d_pmap, *d_ox, *d_oy;
    CUDA_CHECK(cudaMalloc(&d_pmap, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ox, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    calc_potential_field_kernel<<<gridSize, blockSize>>>(
        d_pmap, xwidth, ywidth,
        min_x, min_y, gx, gy,
        d_ox, d_oy, n_obs,
        grid_reso, KP, ETA, rr);
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cuda_ms;
    cudaEventElapsedTime(&cuda_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("CUDA field computation: %.2f ms\n", cuda_ms);

    // Copy result back
    vector<float> cuda_pmap(total);
    CUDA_CHECK(cudaMemcpy(cuda_pmap.data(), d_pmap, total * sizeof(float), cudaMemcpyDeviceToHost));

    // CUDA path (gradient descent on the same field)
    vector<float> cuda_rx, cuda_ry;
    gradient_descent_path(cuda_pmap, xwidth, ywidth, min_x, min_y,
                          sx, sy, gx, gy, cuda_rx, cuda_ry);

    // ===================== Visualization =====================
    int W = 500, H = 500;
    cv::Mat left(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat right(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    draw_potential_field(left, cpu_pmap, xwidth, ywidth, min_x, min_y,
                         sx, sy, gx, gy, ox, oy, cpu_rx, cpu_ry,
                         "CPU (C++)", cpu_ms);
    draw_potential_field(right, cuda_pmap, xwidth, ywidth, min_x, min_y,
                         sx, sy, gx, gy, ox, oy, cuda_rx, cuda_ry,
                         "CUDA (GPU)", (double)cuda_ms);

    cv::Mat combined;
    cv::hconcat(left, right, combined);

    // Write as video (path animation)
    cv::VideoWriter video(
        "gif/comparison_potential_field.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(W * 2, H));

    if (!video.isOpened()) {
        cerr << "Failed to open video writer" << endl;
        return 1;
    }

    // Animate: draw paths incrementally
    int max_path_len = max((int)cpu_rx.size(), (int)cuda_rx.size());
    for (int step = 1; step <= max_path_len; step++) {
        cv::Mat frame_left(H, W, CV_8UC3);
        cv::Mat frame_right(H, W, CV_8UC3);

        // Partial paths up to current step
        vector<float> cpu_rx_partial(cpu_rx.begin(), cpu_rx.begin() + min(step, (int)cpu_rx.size()));
        vector<float> cpu_ry_partial(cpu_ry.begin(), cpu_ry.begin() + min(step, (int)cpu_ry.size()));
        vector<float> cuda_rx_partial(cuda_rx.begin(), cuda_rx.begin() + min(step, (int)cuda_rx.size()));
        vector<float> cuda_ry_partial(cuda_ry.begin(), cuda_ry.begin() + min(step, (int)cuda_ry.size()));

        draw_potential_field(frame_left, cpu_pmap, xwidth, ywidth, min_x, min_y,
                             sx, sy, gx, gy, ox, oy, cpu_rx_partial, cpu_ry_partial,
                             "CPU (C++)", cpu_ms);
        draw_potential_field(frame_right, cuda_pmap, xwidth, ywidth, min_x, min_y,
                             sx, sy, gx, gy, ox, oy, cuda_rx_partial, cuda_ry_partial,
                             "CUDA (GPU)", (double)cuda_ms);

        cv::Mat frame;
        cv::hconcat(frame_left, frame_right, frame);
        video.write(frame);
    }

    // Hold final frame
    for (int i = 0; i < 30; i++)
        video.write(combined);

    video.release();
    cout << "Video saved to gif/comparison_potential_field.avi" << endl;

    // Convert to gif
    system("ffmpeg -y -i gif/comparison_potential_field.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_potential_field.gif 2>/dev/null");
    cout << "GIF saved to gif/comparison_potential_field.gif" << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_pmap));
    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));

    return 0;
}
