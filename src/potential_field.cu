/*************************************************************************
    > File Name: potential_field.cu
    > CUDA-parallelized Potential Field Planning
    > Ported from PythonRobotics by Atsushi Sakai
    > GPU kernel parallelizes potential field computation:
    >   each thread computes one grid cell's total potential
    >     (attractive toward goal + repulsive from obstacles)
    > Gradient descent path following remains on CPU (sequential)
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <set>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Constants (matching PythonRobotics defaults)
// -------------------------------------------------------------------------
const float KP         = 5.0f;     // attractive potential gain
const float ETA        = 100.0f;   // repulsive potential gain
const float AREA_WIDTH = 30.0f;    // area margin around obstacles
const float grid_reso  = 0.5f;     // grid resolution [m]
const float rr         = 5.0f;     // robot radius (repulsive area) [m]

// -------------------------------------------------------------------------
// CUDA kernel: compute potential field in parallel
// Each thread handles one grid cell (ix, iy).
// For that cell it computes:
//   attractive potential = 0.5 * KP * dist_to_goal
//   repulsive potential  = sum over obstacles within rr of
//                          0.5 * ETA * (1/d - 1/rr)^2
//   total = attractive + repulsive
// -------------------------------------------------------------------------
__global__ void calc_potential_field_kernel(
    float* d_pmap,
    int xwidth,
    int ywidth,
    float min_x,
    float min_y,
    float gx,
    float gy,
    const float* d_ox,
    const float* d_oy,
    int n_obs,
    float reso,
    float kp,
    float eta,
    float robot_radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = xwidth * ywidth;
    if (idx >= total) return;

    int ix = idx / ywidth;
    int iy = idx % ywidth;

    // World coordinate of this grid cell
    float wx = (float)ix * reso + min_x;
    float wy = (float)iy * reso + min_y;

    // Attractive potential: 0.5 * KP * dist_to_goal
    float dx = wx - gx;
    float dy = wy - gy;
    float dist_goal = sqrtf(dx * dx + dy * dy);
    float u_att = 0.5f * kp * dist_goal;

    // Repulsive potential: sum over all obstacles
    float u_rep = 0.0f;
    for (int k = 0; k < n_obs; k++) {
        float odx = wx - d_ox[k];
        float ody = wy - d_oy[k];
        float d = sqrtf(odx * odx + ody * ody);

        if (d <= 0.001f) {
            // On top of obstacle -> large value
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

// -------------------------------------------------------------------------
// calc_potential_field: launches CUDA kernel, returns 2D potential field
// Also returns grid origin (min_x, min_y) for coordinate mapping
// -------------------------------------------------------------------------
void calc_potential_field(
    float gx, float gy,
    const vector<float>& ox, const vector<float>& oy,
    vector<vector<float>>& pmap,
    float& out_min_x, float& out_min_y,
    int& out_xw, int& out_yw)
{
    // Compute grid bounds
    float min_x = *min_element(ox.begin(), ox.end()) - AREA_WIDTH / 2.0f;
    float min_y = *min_element(oy.begin(), oy.end()) - AREA_WIDTH / 2.0f;
    float max_x = *max_element(ox.begin(), ox.end()) + AREA_WIDTH / 2.0f;
    float max_y = *max_element(oy.begin(), oy.end()) + AREA_WIDTH / 2.0f;

    // Also include goal in bounds
    min_x = fminf(min_x, gx - AREA_WIDTH / 2.0f);
    min_y = fminf(min_y, gy - AREA_WIDTH / 2.0f);
    max_x = fmaxf(max_x, gx + AREA_WIDTH / 2.0f);
    max_y = fmaxf(max_y, gy + AREA_WIDTH / 2.0f);

    int xwidth = (int)((max_x - min_x) / grid_reso);
    int ywidth = (int)((max_y - min_y) / grid_reso);
    int total = xwidth * ywidth;
    int n_obs = (int)ox.size();

    out_min_x = min_x;
    out_min_y = min_y;
    out_xw = xwidth;
    out_yw = ywidth;

    // Host flat potential map
    vector<float> h_pmap(total, 0.0f);

    // Allocate device memory
    float *d_pmap, *d_ox, *d_oy;
    CUDA_CHECK(cudaMalloc(&d_pmap, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ox, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy, n_obs * sizeof(float)));

    // Copy obstacle coordinates to device
    CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    calc_potential_field_kernel<<<gridSize, blockSize>>>(
        d_pmap, xwidth, ywidth,
        min_x, min_y, gx, gy,
        d_ox, d_oy, n_obs,
        grid_reso, KP, ETA, rr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_pmap.data(), d_pmap, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_pmap));
    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));

    // Convert flat array to 2D vector
    pmap.resize(xwidth, vector<float>(ywidth, 0.0f));
    for (int i = 0; i < xwidth; i++) {
        for (int j = 0; j < ywidth; j++) {
            pmap[i][j] = h_pmap[i * ywidth + j];
        }
    }
}

// -------------------------------------------------------------------------
// Gradient descent path following (CPU, sequential)
// At each step, check 8 neighbors, pick the one with lowest potential.
// Stop when close to goal or oscillation detected.
// -------------------------------------------------------------------------
bool potential_field_planning(
    float sx, float sy,
    float gx, float gy,
    const vector<float>& ox, const vector<float>& oy,
    vector<float>& rx, vector<float>& ry,
    vector<vector<float>>& pmap,
    float& out_min_x, float& out_min_y,
    int& out_xw, int& out_yw)
{
    // Compute potential field on GPU
    calc_potential_field(gx, gy, ox, oy, pmap, out_min_x, out_min_y, out_xw, out_yw);

    // Convert start position to grid index
    int ix = (int)roundf((sx - out_min_x) / grid_reso);
    int iy = (int)roundf((sy - out_min_y) / grid_reso);

    // Goal grid index
    int gix = (int)roundf((gx - out_min_x) / grid_reso);
    int giy = (int)roundf((gy - out_min_y) / grid_reso);

    rx.clear();
    ry.clear();
    rx.push_back(sx);
    ry.push_back(sy);

    // 8 directions: 4 cardinal + 4 diagonal
    int motion_x[] = {1, 0, -1, 0, -1, -1, 1, 1};
    int motion_y[] = {0, 1, 0, -1, -1, 1, -1, 1};

    // Oscillation detection: track recent positions
    set<pair<int,int>> visited;
    const int oscillation_window = 3;
    vector<pair<int,int>> recent_positions;

    while (true) {
        // Check if reached goal
        float dist_to_goal = sqrtf((float)(ix - gix) * (ix - gix) +
                                   (float)(iy - giy) * (iy - giy)) * grid_reso;
        if (dist_to_goal < grid_reso) {
            cout << "Goal reached!" << endl;
            rx.push_back(gx);
            ry.push_back(gy);
            return true;
        }

        // Find neighbor with minimum potential
        float min_potential = numeric_limits<float>::max();
        int min_ix = ix;
        int min_iy = iy;

        for (int i = 0; i < 8; i++) {
            int nx = ix + motion_x[i];
            int ny = iy + motion_y[i];

            if (nx >= 0 && nx < out_xw && ny >= 0 && ny < out_yw) {
                if (pmap[nx][ny] < min_potential) {
                    min_potential = pmap[nx][ny];
                    min_ix = nx;
                    min_iy = ny;
                }
            }
        }

        // Move to best neighbor
        ix = min_ix;
        iy = min_iy;

        float wx = (float)ix * grid_reso + out_min_x;
        float wy = (float)iy * grid_reso + out_min_y;
        rx.push_back(wx);
        ry.push_back(wy);

        // Oscillation detection
        pair<int,int> pos = {ix, iy};
        if (visited.count(pos)) {
            cout << "Oscillation detected at (" << wx << ", " << wy << ")" << endl;
            break;
        }

        recent_positions.push_back(pos);
        if ((int)recent_positions.size() > oscillation_window) {
            // Remove oldest from visited set
            visited.erase(recent_positions[recent_positions.size() - oscillation_window - 1]);
        }
        visited.insert(pos);

        // Safety: max iterations
        if ((int)rx.size() > out_xw * out_yw) {
            cout << "Path too long, aborting." << endl;
            break;
        }
    }

    return false;
}

// -------------------------------------------------------------------------
// Visualization
// -------------------------------------------------------------------------
void draw_result(
    const vector<vector<float>>& pmap,
    int xw, int yw,
    float min_x, float min_y,
    float sx, float sy,
    float gx, float gy,
    const vector<float>& ox, const vector<float>& oy,
    const vector<float>& rx, const vector<float>& ry)
{
    // Find min/max potential for normalization (clamp extremes)
    float p_min = numeric_limits<float>::max();
    float p_max = -numeric_limits<float>::max();
    for (int i = 0; i < xw; i++) {
        for (int j = 0; j < yw; j++) {
            float v = pmap[i][j];
            if (v < 1.0e5f) {  // ignore extreme values for normalization
                p_min = fminf(p_min, v);
                p_max = fmaxf(p_max, v);
            }
        }
    }
    if (p_max <= p_min) p_max = p_min + 1.0f;

    // Scale factor: pixels per grid cell
    int cell_size = 4;
    int img_w = yw * cell_size;
    int img_h = xw * cell_size;

    // Create heatmap image
    cv::Mat gray(img_h, img_w, CV_8UC1);
    for (int i = 0; i < xw; i++) {
        for (int j = 0; j < yw; j++) {
            float v = pmap[i][j];
            // Clamp
            if (v > 1.0e5f) v = p_max;
            // Normalize to 0-255
            float norm = (v - p_min) / (p_max - p_min);
            norm = fminf(fmaxf(norm, 0.0f), 1.0f);
            unsigned char pixel = (unsigned char)(255.0f * (1.0f - norm));
            // Fill cell block (note: image y = j, image x = i for proper orientation)
            for (int di = 0; di < cell_size; di++) {
                for (int dj = 0; dj < cell_size; dj++) {
                    gray.at<uchar>(i * cell_size + di, j * cell_size + dj) = pixel;
                }
            }
        }
    }

    // Apply colormap
    cv::Mat img;
    cv::applyColorMap(gray, img, cv::COLORMAP_JET);

    // Helper lambda: world coords -> image pixel
    auto to_pixel = [&](float wx, float wy) -> cv::Point {
        int ix = (int)roundf((wx - min_x) / grid_reso);
        int iy = (int)roundf((wy - min_y) / grid_reso);
        return cv::Point(iy * cell_size + cell_size / 2,
                         ix * cell_size + cell_size / 2);
    };

    // Draw obstacles as black circles
    for (int k = 0; k < (int)ox.size(); k++) {
        cv::Point pt = to_pixel(ox[k], oy[k]);
        cv::circle(img, pt, cell_size * 2, cv::Scalar(0, 0, 0), -1);
    }

    // Draw start (green) and goal (blue)
    cv::Point sp = to_pixel(sx, sy);
    cv::Point gp = to_pixel(gx, gy);
    cv::circle(img, sp, cell_size * 3, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, gp, cell_size * 3, cv::Scalar(255, 0, 0), -1);

    // Draw path as red line
    cv::VideoWriter video("gif/potential_field.avi", cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(img.cols, img.rows));
    for (int i = 0; i < (int)rx.size() - 1; i++) {
        cv::Point p1 = to_pixel(rx[i], ry[i]);
        cv::Point p2 = to_pixel(rx[i + 1], ry[i + 1]);
        cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2);
        video.write(img);
    }

    cv::namedWindow("potential_field", cv::WINDOW_AUTOSIZE);
    cv::imshow("potential_field", img);
    video.write(img);
    video.release();
    std::cout << "Video saved to videos/potential_field.avi" << std::endl;
    cv::waitKey(0);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "CUDA Potential Field Planning" << endl;

    // Test case (same as PythonRobotics)
    float sx = 0.0f,  sy = 10.0f;   // start
    float gx = 30.0f, gy = 30.0f;   // goal

    vector<float> ox = {15.0f, 5.0f, 20.0f, 25.0f};
    vector<float> oy = {25.0f, 15.0f, 26.0f, 25.0f};

    // Plan path
    vector<float> rx, ry;
    vector<vector<float>> pmap;
    float min_x, min_y;
    int xw, yw;

    bool success = potential_field_planning(
        sx, sy, gx, gy, ox, oy,
        rx, ry, pmap, min_x, min_y, xw, yw);

    cout << "Path length: " << rx.size() << " points" << endl;
    if (success) {
        cout << "Planning succeeded." << endl;
    } else {
        cout << "Planning finished (oscillation or limit reached)." << endl;
    }

    // Visualize
    draw_result(pmap, xw, yw, min_x, min_y,
                sx, sy, gx, gy, ox, oy, rx, ry);

    return 0;
}
