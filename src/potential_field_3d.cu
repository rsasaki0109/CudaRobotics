/*************************************************************************
    > File Name: potential_field_3d.cu
    > CUDA-parallelized 3D Potential Field Planning
    > Extends 2D potential field to 3D for drone/UAV navigation
    > GPU kernel parallelizes 3D potential field computation:
    >   each thread computes one 3D grid cell's total potential
    >     (attractive toward goal + repulsive from spherical obstacles)
    > Gradient descent path following remains on CPU (sequential, 26-neighbors)
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <set>
#include <tuple>
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
// Constants
// -------------------------------------------------------------------------
const float KP         = 5.0f;     // attractive potential gain
const float ETA        = 100.0f;   // repulsive potential gain
const float AREA_WIDTH = 10.0f;    // area margin around obstacles
const float grid_reso  = 1.0f;     // grid resolution [m]
const float rr         = 5.0f;     // repulsive influence radius [m]

// Obstacle: (x, y, z, radius)
struct Obstacle {
    float x, y, z, r;
};

// -------------------------------------------------------------------------
// CUDA kernel: compute 3D potential field in parallel
// Each thread handles one grid cell (ix, iy, iz).
// For that cell it computes:
//   attractive potential = 0.5 * KP * dist_to_goal
//   repulsive potential  = sum over obstacles within rr of
//                          0.5 * ETA * (1/d_surface - 1/rr)^2
//   total = attractive + repulsive
// -------------------------------------------------------------------------
__global__ void calc_potential_field_3d_kernel(
    float* d_pmap,
    int xwidth,
    int ywidth,
    int zwidth,
    float min_x,
    float min_y,
    float min_z,
    float gx,
    float gy,
    float gz,
    const float* d_obs_x,
    const float* d_obs_y,
    const float* d_obs_z,
    const float* d_obs_r,
    int n_obs,
    float reso,
    float kp,
    float eta,
    float robot_radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = xwidth * ywidth * zwidth;
    if (idx >= total) return;

    // Decompose flat index into (ix, iy, iz)
    int iz = idx % zwidth;
    int iy = (idx / zwidth) % ywidth;
    int ix = idx / (ywidth * zwidth);

    // World coordinate of this grid cell
    float wx = (float)ix * reso + min_x;
    float wy = (float)iy * reso + min_y;
    float wz = (float)iz * reso + min_z;

    // Attractive potential: 0.5 * KP * dist_to_goal
    float dx = wx - gx;
    float dy = wy - gy;
    float dz = wz - gz;
    float dist_goal = sqrtf(dx * dx + dy * dy + dz * dz);
    float u_att = 0.5f * kp * dist_goal;

    // Repulsive potential: sum over all spherical obstacles
    float u_rep = 0.0f;
    for (int k = 0; k < n_obs; k++) {
        float odx = wx - d_obs_x[k];
        float ody = wy - d_obs_y[k];
        float odz = wz - d_obs_z[k];
        float dist_center = sqrtf(odx * odx + ody * ody + odz * odz);

        // Distance to obstacle surface
        float d = dist_center - d_obs_r[k];

        if (d <= 0.001f) {
            // Inside obstacle or on surface -> large value
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
// calc_potential_field_3d: launches CUDA kernel, returns 3D potential field
// -------------------------------------------------------------------------
void calc_potential_field_3d(
    float gx, float gy, float gz,
    const vector<Obstacle>& obstacles,
    vector<float>& pmap_flat,
    float& out_min_x, float& out_min_y, float& out_min_z,
    int& out_xw, int& out_yw, int& out_zw)
{
    int n_obs = (int)obstacles.size();

    // Separate obstacle data
    vector<float> obs_x(n_obs), obs_y(n_obs), obs_z(n_obs), obs_r(n_obs);
    for (int i = 0; i < n_obs; i++) {
        obs_x[i] = obstacles[i].x;
        obs_y[i] = obstacles[i].y;
        obs_z[i] = obstacles[i].z;
        obs_r[i] = obstacles[i].r;
    }

    // Compute grid bounds from obstacles and goal
    float min_x = gx, max_x = gx;
    float min_y = gy, max_y = gy;
    float min_z = gz, max_z = gz;
    for (int i = 0; i < n_obs; i++) {
        min_x = fminf(min_x, obs_x[i] - obs_r[i]);
        max_x = fmaxf(max_x, obs_x[i] + obs_r[i]);
        min_y = fminf(min_y, obs_y[i] - obs_r[i]);
        max_y = fmaxf(max_y, obs_y[i] + obs_r[i]);
        min_z = fminf(min_z, obs_z[i] - obs_r[i]);
        max_z = fmaxf(max_z, obs_z[i] + obs_r[i]);
    }
    min_x -= AREA_WIDTH;  min_y -= AREA_WIDTH;  min_z -= AREA_WIDTH;
    max_x += AREA_WIDTH;  max_y += AREA_WIDTH;  max_z += AREA_WIDTH;

    int xwidth = (int)((max_x - min_x) / grid_reso);
    int ywidth = (int)((max_y - min_y) / grid_reso);
    int zwidth = (int)((max_z - min_z) / grid_reso);
    int total = xwidth * ywidth * zwidth;

    out_min_x = min_x;  out_min_y = min_y;  out_min_z = min_z;
    out_xw = xwidth;    out_yw = ywidth;    out_zw = zwidth;

    printf("Grid size: %d x %d x %d = %d cells\n", xwidth, ywidth, zwidth, total);

    // Allocate host flat potential map
    pmap_flat.resize(total, 0.0f);

    // Allocate device memory
    float *d_pmap, *d_ox, *d_oy, *d_oz, *d_or;
    CUDA_CHECK(cudaMalloc(&d_pmap, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ox, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oy, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oz, n_obs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_or, n_obs * sizeof(float)));

    // Copy obstacle data to device
    CUDA_CHECK(cudaMemcpy(d_ox, obs_x.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, obs_y.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oz, obs_z.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_or, obs_r.data(), n_obs * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    calc_potential_field_3d_kernel<<<gridSize, blockSize>>>(
        d_pmap, xwidth, ywidth, zwidth,
        min_x, min_y, min_z,
        gx, gy, gz,
        d_ox, d_oy, d_oz, d_or,
        n_obs, grid_reso, KP, ETA, rr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(pmap_flat.data(), d_pmap, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_pmap));
    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));
    CUDA_CHECK(cudaFree(d_oz));
    CUDA_CHECK(cudaFree(d_or));
}

// -------------------------------------------------------------------------
// Helper: access flat 3D array
// -------------------------------------------------------------------------
static inline int idx3d(int ix, int iy, int iz, int yw, int zw) {
    return ix * (yw * zw) + iy * zw + iz;
}

// -------------------------------------------------------------------------
// Gradient descent path following in 3D (CPU, sequential)
// At each step, check 26 neighbors (3^3 - 1), pick the one with lowest potential.
// Stop when close to goal.
// -------------------------------------------------------------------------
bool potential_field_planning_3d(
    float sx, float sy, float sz,
    float gx, float gy, float gz,
    const vector<Obstacle>& obstacles,
    vector<float>& rx, vector<float>& ry, vector<float>& rz,
    vector<float>& pmap_flat,
    float& out_min_x, float& out_min_y, float& out_min_z,
    int& out_xw, int& out_yw, int& out_zw)
{
    // Compute potential field on GPU
    calc_potential_field_3d(gx, gy, gz, obstacles, pmap_flat,
                           out_min_x, out_min_y, out_min_z,
                           out_xw, out_yw, out_zw);

    // Convert start position to grid index
    int ix = (int)roundf((sx - out_min_x) / grid_reso);
    int iy = (int)roundf((sy - out_min_y) / grid_reso);
    int iz = (int)roundf((sz - out_min_z) / grid_reso);

    // Goal grid index
    int gix = (int)roundf((gx - out_min_x) / grid_reso);
    int giy = (int)roundf((gy - out_min_y) / grid_reso);
    int giz = (int)roundf((gz - out_min_z) / grid_reso);

    rx.clear(); ry.clear(); rz.clear();
    rx.push_back(sx); ry.push_back(sy); rz.push_back(sz);

    // Oscillation detection
    set<tuple<int,int,int>> visited;
    const int oscillation_window = 3;
    vector<tuple<int,int,int>> recent_positions;

    int max_iter = out_xw * out_yw * out_zw;

    while (true) {
        // Check if reached goal
        float dist_to_goal = sqrtf(
            (float)(ix - gix) * (ix - gix) +
            (float)(iy - giy) * (iy - giy) +
            (float)(iz - giz) * (iz - giz)) * grid_reso;
        if (dist_to_goal < grid_reso) {
            cout << "Goal reached!" << endl;
            rx.push_back(gx); ry.push_back(gy); rz.push_back(gz);
            return true;
        }

        // Find neighbor with minimum potential among 26 neighbors
        float min_potential = numeric_limits<float>::max();
        int min_ix = ix, min_iy = iy, min_iz = iz;

        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                for (int dk = -1; dk <= 1; dk++) {
                    if (di == 0 && dj == 0 && dk == 0) continue;
                    int nx = ix + di;
                    int ny = iy + dj;
                    int nz = iz + dk;
                    if (nx >= 0 && nx < out_xw &&
                        ny >= 0 && ny < out_yw &&
                        nz >= 0 && nz < out_zw) {
                        float p = pmap_flat[idx3d(nx, ny, nz, out_yw, out_zw)];
                        if (p < min_potential) {
                            min_potential = p;
                            min_ix = nx;
                            min_iy = ny;
                            min_iz = nz;
                        }
                    }
                }
            }
        }

        // Move to best neighbor
        ix = min_ix; iy = min_iy; iz = min_iz;

        float wx = (float)ix * grid_reso + out_min_x;
        float wy = (float)iy * grid_reso + out_min_y;
        float wz = (float)iz * grid_reso + out_min_z;
        rx.push_back(wx); ry.push_back(wy); rz.push_back(wz);

        // Oscillation detection
        auto pos = make_tuple(ix, iy, iz);
        if (visited.count(pos)) {
            cout << "Oscillation detected at (" << wx << ", " << wy << ", " << wz << ")" << endl;
            break;
        }

        recent_positions.push_back(pos);
        if ((int)recent_positions.size() > oscillation_window) {
            visited.erase(recent_positions[recent_positions.size() - oscillation_window - 1]);
        }
        visited.insert(pos);

        // Safety: max iterations
        if ((int)rx.size() > max_iter) {
            cout << "Path too long, aborting." << endl;
            break;
        }
    }

    return false;
}

// -------------------------------------------------------------------------
// Visualization: XY slice and XZ slice side by side
// -------------------------------------------------------------------------
void draw_result_3d(
    const vector<float>& pmap_flat,
    int xw, int yw, int zw,
    float min_x, float min_y, float min_z,
    float sx, float sy, float sz,
    float gx, float gy, float gz,
    const vector<Obstacle>& obstacles,
    const vector<float>& rx, const vector<float>& ry, const vector<float>& rz)
{
    // Scale factor: pixels per grid cell
    int cell_size = 6;

    // We will draw two views side by side:
    //   Left:  XY slice at current z-level (z of last path point)
    //   Right: XZ slice at current y-level (y of last path point)

    // Use the midpoint of the path for slice levels
    int path_mid = (int)rx.size() / 2;
    float slice_z = rz[path_mid];
    float slice_y = ry[path_mid];
    int iz_slice = (int)roundf((slice_z - min_z) / grid_reso);
    int iy_slice = (int)roundf((slice_y - min_y) / grid_reso);
    iz_slice = max(0, min(iz_slice, zw - 1));
    iy_slice = max(0, min(iy_slice, yw - 1));

    // --- Extract XY slice (fixed z = iz_slice) ---
    vector<float> xy_slice(xw * yw);
    for (int i = 0; i < xw; i++)
        for (int j = 0; j < yw; j++)
            xy_slice[i * yw + j] = pmap_flat[idx3d(i, j, iz_slice, yw, zw)];

    // --- Extract XZ slice (fixed y = iy_slice) ---
    vector<float> xz_slice(xw * zw);
    for (int i = 0; i < xw; i++)
        for (int k = 0; k < zw; k++)
            xz_slice[i * zw + k] = pmap_flat[idx3d(i, iy_slice, k, yw, zw)];

    // Helper: normalize and create heatmap for a 2D slice
    auto make_heatmap = [&](const vector<float>& data, int rows, int cols, int cs) -> cv::Mat {
        float p_min = numeric_limits<float>::max();
        float p_max = -numeric_limits<float>::max();
        for (int i = 0; i < rows * cols; i++) {
            float v = data[i];
            if (v < 1.0e5f) {
                p_min = fminf(p_min, v);
                p_max = fmaxf(p_max, v);
            }
        }
        if (p_max <= p_min) p_max = p_min + 1.0f;

        cv::Mat gray(rows * cs, cols * cs, CV_8UC1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float v = data[i * cols + j];
                if (v > 1.0e5f) v = p_max;
                float norm = (v - p_min) / (p_max - p_min);
                norm = fminf(fmaxf(norm, 0.0f), 1.0f);
                unsigned char pixel = (unsigned char)(255.0f * (1.0f - norm));
                for (int di = 0; di < cs; di++)
                    for (int dj = 0; dj < cs; dj++)
                        gray.at<uchar>(i * cs + di, j * cs + dj) = pixel;
            }
        }
        cv::Mat colored;
        cv::applyColorMap(gray, colored, cv::COLORMAP_JET);
        return colored;
    };

    // Create heatmaps
    cv::Mat xy_img = make_heatmap(xy_slice, xw, yw, cell_size);
    cv::Mat xz_img = make_heatmap(xz_slice, xw, zw, cell_size);

    // --- Draw on XY view ---
    {
        auto to_pixel = [&](float wx, float wy) -> cv::Point {
            int pi = (int)roundf((wx - min_x) / grid_reso);
            int pj = (int)roundf((wy - min_y) / grid_reso);
            return cv::Point(pj * cell_size + cell_size / 2,
                             pi * cell_size + cell_size / 2);
        };

        // Draw obstacle cross-sections (circles where sphere intersects the z-plane)
        for (int k = 0; k < (int)obstacles.size(); k++) {
            float dz = slice_z - obstacles[k].z;
            float r2 = obstacles[k].r * obstacles[k].r - dz * dz;
            if (r2 > 0.0f) {
                float cr = sqrtf(r2);
                cv::Point center = to_pixel(obstacles[k].x, obstacles[k].y);
                int pr = (int)roundf(cr / grid_reso) * cell_size;
                cv::circle(xy_img, center, max(pr, 1), cv::Scalar(0, 0, 0), -1);
            }
        }

        // Draw path projected onto XY
        for (int i = 0; i < (int)rx.size() - 1; i++) {
            cv::Point p1 = to_pixel(rx[i], ry[i]);
            cv::Point p2 = to_pixel(rx[i + 1], ry[i + 1]);
            cv::line(xy_img, p1, p2, cv::Scalar(0, 0, 255), 2);
        }

        // Start (green), goal (blue)
        cv::circle(xy_img, to_pixel(sx, sy), cell_size * 2, cv::Scalar(0, 255, 0), -1);
        cv::circle(xy_img, to_pixel(gx, gy), cell_size * 2, cv::Scalar(255, 0, 0), -1);

        // Label
        cv::putText(xy_img, "XY slice (z=" + to_string((int)slice_z) + ")",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // --- Draw on XZ view ---
    {
        auto to_pixel = [&](float wx, float wz) -> cv::Point {
            int pi = (int)roundf((wx - min_x) / grid_reso);
            int pk = (int)roundf((wz - min_z) / grid_reso);
            return cv::Point(pk * cell_size + cell_size / 2,
                             pi * cell_size + cell_size / 2);
        };

        // Draw obstacle cross-sections (circles where sphere intersects the y-plane)
        for (int k = 0; k < (int)obstacles.size(); k++) {
            float dy = slice_y - obstacles[k].y;
            float r2 = obstacles[k].r * obstacles[k].r - dy * dy;
            if (r2 > 0.0f) {
                float cr = sqrtf(r2);
                cv::Point center = to_pixel(obstacles[k].x, obstacles[k].z);
                int pr = (int)roundf(cr / grid_reso) * cell_size;
                cv::circle(xz_img, center, max(pr, 1), cv::Scalar(0, 0, 0), -1);
            }
        }

        // Draw path projected onto XZ
        for (int i = 0; i < (int)rx.size() - 1; i++) {
            cv::Point p1 = to_pixel(rx[i], rz[i]);
            cv::Point p2 = to_pixel(rx[i + 1], rz[i + 1]);
            cv::line(xz_img, p1, p2, cv::Scalar(0, 0, 255), 2);
        }

        // Start (green), goal (blue)
        cv::circle(xz_img, to_pixel(sx, sz), cell_size * 2, cv::Scalar(0, 255, 0), -1);
        cv::circle(xz_img, to_pixel(gx, gz), cell_size * 2, cv::Scalar(255, 0, 0), -1);

        // Label
        cv::putText(xz_img, "XZ slice (y=" + to_string((int)slice_y) + ")",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // --- Combine side by side ---
    // Resize to same height if needed
    int target_h = max(xy_img.rows, xz_img.rows);
    if (xy_img.rows != target_h) {
        cv::resize(xy_img, xy_img, cv::Size(xy_img.cols * target_h / xy_img.rows, target_h));
    }
    if (xz_img.rows != target_h) {
        cv::resize(xz_img, xz_img, cv::Size(xz_img.cols * target_h / xz_img.rows, target_h));
    }

    cv::Mat combined;
    cv::hconcat(xy_img, xz_img, combined);

    cv::namedWindow("potential_field_3d", cv::WINDOW_AUTOSIZE);
    cv::imshow("potential_field_3d", combined);
    cv::waitKey(0);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "CUDA 3D Potential Field Planning" << endl;

    // Test case
    float sx = 0.0f,  sy = 0.0f,  sz = 0.0f;   // start
    float gx = 30.0f, gy = 30.0f, gz = 30.0f;   // goal

    // Spherical obstacles: (x, y, z, radius)
    vector<Obstacle> obstacles = {
        {15.0f, 15.0f, 15.0f, 3.0f},
        {10.0f, 25.0f, 10.0f, 2.0f},
        {25.0f, 10.0f, 20.0f, 2.0f},
        {20.0f, 20.0f, 25.0f, 2.0f}
    };

    // Plan path
    vector<float> rx, ry, rz;
    vector<float> pmap_flat;
    float min_x, min_y, min_z;
    int xw, yw, zw;

    bool success = potential_field_planning_3d(
        sx, sy, sz, gx, gy, gz, obstacles,
        rx, ry, rz, pmap_flat,
        min_x, min_y, min_z, xw, yw, zw);

    cout << "Path length: " << rx.size() << " points" << endl;
    if (success) {
        cout << "Planning succeeded." << endl;
    } else {
        cout << "Planning finished (oscillation or limit reached)." << endl;
    }

    // Visualize
    draw_result_3d(pmap_flat, xw, yw, zw,
                   min_x, min_y, min_z,
                   sx, sy, sz, gx, gy, gz,
                   obstacles, rx, ry, rz);

    return 0;
}
