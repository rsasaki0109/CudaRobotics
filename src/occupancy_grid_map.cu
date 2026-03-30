/*************************************************************************
    > File Name: occupancy_grid_map.cu
    > CUDA-parallelized Occupancy Grid Mapping
    > Reference: PythonRobotics gaussian_grid_map by Atsushi Sakai
    > GPU kernel parallelizes lidar ray processing:
    >   each thread handles one lidar ray, walks grid cells via DDA,
    >   updates log-odds for free/occupied cells
    > Simulation: robot drives rectangular path with simulated lidar
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
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
const int   GRID_W       = 100;         // grid width in cells
const int   GRID_H       = 100;         // grid height in cells
const float RESOLUTION   = 0.5f;        // meters per cell
const int   NUM_RAYS     = 360;         // lidar rays per scan
const float MAX_RANGE    = 20.0f;       // lidar max range [m]
const float ANGULAR_RES  = 1.0f;        // degrees between rays
const float L_OCC        = 0.85f;       // log-odds increment for occupied
const float L_FREE       = -0.4f;       // log-odds increment for free
const float L_PRIOR      = 0.0f;        // initial log-odds
const float L_MIN        = -5.0f;       // clamp min
const float L_MAX        = 5.0f;        // clamp max

// Grid origin in world coordinates (bottom-left corner)
const float GRID_ORIGIN_X = 0.0f;
const float GRID_ORIGIN_Y = 0.0f;

// -------------------------------------------------------------------------
// Obstacle definition for simulation
// -------------------------------------------------------------------------
struct Segment {
    float x0, y0, x1, y1;
};

// -------------------------------------------------------------------------
// CPU: simulate lidar by raycasting against line segments
// Returns range for each ray (MAX_RANGE if no hit)
// -------------------------------------------------------------------------
void simulate_lidar(
    float rx, float ry, float rtheta,
    const vector<Segment>& walls,
    vector<float>& ranges)
{
    ranges.resize(NUM_RAYS);
    for (int i = 0; i < NUM_RAYS; i++) {
        float angle = rtheta + (float)i * ANGULAR_RES * M_PI / 180.0f;
        float dx = cosf(angle);
        float dy = sinf(angle);

        float best_t = MAX_RANGE;

        for (int w = 0; w < (int)walls.size(); w++) {
            // Ray: P = (rx, ry) + t * (dx, dy)
            // Segment: Q = (x0, y0) + s * (x1-x0, y1-y0), s in [0,1]
            float ex = walls[w].x1 - walls[w].x0;
            float ey = walls[w].y1 - walls[w].y0;

            float denom = dx * ey - dy * ex;
            if (fabsf(denom) < 1e-8f) continue;

            float qpx = walls[w].x0 - rx;
            float qpy = walls[w].y0 - ry;

            float t = (qpx * ey - qpy * ex) / denom;
            float s = (qpx * dy - qpy * dx) / denom;

            if (t > 0.0f && t < best_t && s >= 0.0f && s <= 1.0f) {
                best_t = t;
            }
        }

        ranges[i] = best_t;
    }
}

// -------------------------------------------------------------------------
// CUDA kernel: update_grid_kernel
// Each thread handles one lidar ray. Walks grid cells using DDA,
// applies L_FREE to cells along the ray, L_OCC to the endpoint cell.
// Uses atomicAdd for thread-safe log-odds updates.
// -------------------------------------------------------------------------
__global__ void update_grid_kernel(
    float* d_logodds,
    int grid_w,
    int grid_h,
    float origin_x,
    float origin_y,
    float resolution,
    float robot_x,
    float robot_y,
    float robot_theta,
    const float* d_ranges,
    int num_rays,
    float angular_res_rad,
    float max_range,
    float l_occ,
    float l_free,
    float l_min,
    float l_max)
{
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    float range = d_ranges[ray_idx];
    float angle = robot_theta + (float)ray_idx * angular_res_rad;

    // Endpoint in world coordinates
    float ex = robot_x + range * cosf(angle);
    float ey = robot_y + range * sinf(angle);

    // Convert robot and endpoint to grid coordinates (continuous)
    float gx0 = (robot_x - origin_x) / resolution;
    float gy0 = (robot_y - origin_y) / resolution;
    float gx1 = (ex - origin_x) / resolution;
    float gy1 = (ey - origin_y) / resolution;

    // DDA line algorithm to walk through grid cells
    float ddx = gx1 - gx0;
    float ddy = gy1 - gy0;
    float dist = sqrtf(ddx * ddx + ddy * ddy);
    if (dist < 1e-6f) return;

    int steps = (int)(dist * 2.0f) + 1;  // oversample to not miss cells
    float step_x = ddx / (float)steps;
    float step_y = ddy / (float)steps;

    int prev_cx = -1, prev_cy = -1;

    for (int s = 0; s <= steps; s++) {
        float px = gx0 + step_x * (float)s;
        float py = gy0 + step_y * (float)s;
        int cx = (int)floorf(px);
        int cy = (int)floorf(py);

        // Skip duplicate cells
        if (cx == prev_cx && cy == prev_cy) continue;
        prev_cx = cx;
        prev_cy = cy;

        // Bounds check
        if (cx < 0 || cx >= grid_w || cy < 0 || cy >= grid_h) continue;

        int cell_idx = cy * grid_w + cx;

        // Last step or close to endpoint: occupied
        // All earlier cells: free
        float fdx = px - gx1;
        float fdy = py - gy1;
        float dist_to_end = sqrtf(fdx * fdx + fdy * fdy);

        float update;
        if (dist_to_end < 1.0f && range < max_range) {
            update = l_occ;
        } else {
            update = l_free;
        }

        // Atomic update with clamping
        float old_val = atomicAdd(&d_logodds[cell_idx], update);
        float new_val = old_val + update;
        if (new_val > l_max) {
            atomicAdd(&d_logodds[cell_idx], l_max - new_val);
        } else if (new_val < l_min) {
            atomicAdd(&d_logodds[cell_idx], l_min - new_val);
        }
    }
}

// -------------------------------------------------------------------------
// Visualization helper: log-odds to grayscale
// -------------------------------------------------------------------------
void logodds_to_image(
    const vector<float>& logodds,
    cv::Mat& img,
    int grid_w, int grid_h,
    int cell_px)
{
    int img_w = grid_w * cell_px;
    int img_h = grid_h * cell_px;
    img = cv::Mat(img_h, img_w, CV_8UC3);

    for (int cy = 0; cy < grid_h; cy++) {
        for (int cx = 0; cx < grid_w; cx++) {
            float lo = logodds[cy * grid_w + cx];
            // Convert log-odds to probability: p = 1 / (1 + exp(-lo))
            float p = 1.0f / (1.0f + expf(-lo));
            // White = free (p~0), Black = occupied (p~1), Gray = unknown (p~0.5)
            unsigned char val = (unsigned char)((1.0f - p) * 255.0f);
            cv::Vec3b color(val, val, val);

            for (int dy = 0; dy < cell_px; dy++) {
                for (int dx = 0; dx < cell_px; dx++) {
                    int px_x = cx * cell_px + dx;
                    int px_y = (grid_h - 1 - cy) * cell_px + dy;  // flip y
                    if (px_x < img_w && px_y < img_h) {
                        img.at<cv::Vec3b>(px_y, px_x) = color;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Draw robot position and lidar rays on image
// -------------------------------------------------------------------------
void draw_robot_and_lidar(
    cv::Mat& img,
    float robot_x, float robot_y, float robot_theta,
    const vector<float>& ranges,
    int grid_w, int grid_h,
    float origin_x, float origin_y,
    float resolution, int cell_px)
{
    // World to pixel
    auto to_pixel = [&](float wx, float wy) -> cv::Point {
        float gx = (wx - origin_x) / resolution;
        float gy = (wy - origin_y) / resolution;
        int px = (int)(gx * cell_px + cell_px / 2);
        int py = (int)((grid_h - 1 - gy) * cell_px + cell_px / 2);
        return cv::Point(px, py);
    };

    // Draw lidar rays as thin green lines
    for (int i = 0; i < NUM_RAYS; i++) {
        float angle = robot_theta + (float)i * ANGULAR_RES * M_PI / 180.0f;
        float ex = robot_x + ranges[i] * cosf(angle);
        float ey = robot_y + ranges[i] * sinf(angle);
        cv::Point p1 = to_pixel(robot_x, robot_y);
        cv::Point p2 = to_pixel(ex, ey);
        cv::line(img, p1, p2, cv::Scalar(0, 180, 0), 1);
    }

    // Draw robot as red circle
    cv::Point rp = to_pixel(robot_x, robot_y);
    cv::circle(img, rp, cell_px * 2, cv::Scalar(0, 0, 255), -1);

    // Direction arrow
    float arrow_len = 2.0f;  // meters
    float ax = robot_x + arrow_len * cosf(robot_theta);
    float ay = robot_y + arrow_len * sinf(robot_theta);
    cv::Point ap = to_pixel(ax, ay);
    cv::arrowedLine(img, rp, ap, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "CUDA Occupancy Grid Mapping" << endl;

    // --- Define walls and obstacles ---
    vector<Segment> walls;
    float world_w = GRID_W * RESOLUTION;  // 50m
    float world_h = GRID_H * RESOLUTION;  // 50m
    float margin = 2.0f;

    // Perimeter walls
    walls.push_back({margin, margin, world_w - margin, margin});               // bottom
    walls.push_back({world_w - margin, margin, world_w - margin, world_h - margin}); // right
    walls.push_back({world_w - margin, world_h - margin, margin, world_h - margin}); // top
    walls.push_back({margin, world_h - margin, margin, margin});               // left

    // Internal obstacles
    // Horizontal wall segment
    walls.push_back({15.0f, 20.0f, 25.0f, 20.0f});
    // Vertical wall segment
    walls.push_back({30.0f, 10.0f, 30.0f, 25.0f});
    // L-shaped obstacle
    walls.push_back({10.0f, 35.0f, 10.0f, 42.0f});
    walls.push_back({10.0f, 42.0f, 18.0f, 42.0f});
    // Small box
    walls.push_back({38.0f, 35.0f, 44.0f, 35.0f});
    walls.push_back({44.0f, 35.0f, 44.0f, 42.0f});
    walls.push_back({44.0f, 42.0f, 38.0f, 42.0f});
    walls.push_back({38.0f, 42.0f, 38.0f, 35.0f});
    // Diagonal wall
    walls.push_back({20.0f, 30.0f, 28.0f, 38.0f});

    // --- Allocate log-odds grid on GPU ---
    int total_cells = GRID_W * GRID_H;
    float* d_logodds;
    CUDA_CHECK(cudaMalloc(&d_logodds, total_cells * sizeof(float)));

    // Initialize to L_PRIOR (0)
    vector<float> h_logodds(total_cells, L_PRIOR);
    CUDA_CHECK(cudaMemcpy(d_logodds, h_logodds.data(), total_cells * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Lidar ranges on GPU
    float* d_ranges;
    CUDA_CHECK(cudaMalloc(&d_ranges, NUM_RAYS * sizeof(float)));

    // --- Robot path: rectangular loop ---
    vector<float> path_x, path_y, path_theta;
    float speed = 0.8f;   // m per step
    float cx = 25.0f, cy = 25.0f;  // center of rectangle
    float half_w = 15.0f, half_h = 15.0f;

    // Generate waypoints along rectangle
    // Bottom edge: left to right
    for (float x = cx - half_w; x <= cx + half_w; x += speed) {
        path_x.push_back(x);
        path_y.push_back(cy - half_h);
        path_theta.push_back(0.0f);
    }
    // Right edge: bottom to top
    for (float y = cy - half_h; y <= cy + half_h; y += speed) {
        path_x.push_back(cx + half_w);
        path_y.push_back(y);
        path_theta.push_back(M_PI / 2.0f);
    }
    // Top edge: right to left
    for (float x = cx + half_w; x >= cx - half_w; x -= speed) {
        path_x.push_back(x);
        path_y.push_back(cy + half_h);
        path_theta.push_back(M_PI);
    }
    // Left edge: top to bottom
    for (float y = cy + half_h; y >= cy - half_h; y -= speed) {
        path_x.push_back(cx - half_w);
        path_y.push_back(y);
        path_theta.push_back(-M_PI / 2.0f);
    }

    int num_steps = (int)path_x.size();
    cout << "Path steps: " << num_steps << endl;

    // --- Visualization setup ---
    int cell_px = 5;  // pixels per grid cell
    int img_w = GRID_W * cell_px;
    int img_h = GRID_H * cell_px;

    string avi_path = "gif/occupancy_grid.avi";
    string gif_path = "gif/occupancy_grid.gif";

    cv::VideoWriter video(avi_path,
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                          30, cv::Size(img_w, img_h));
    if (!video.isOpened()) {
        cerr << "Failed to open video writer at " << avi_path << endl;
        return 1;
    }

    float angular_res_rad = ANGULAR_RES * M_PI / 180.0f;

    // CUDA kernel config
    int blockSize = 128;
    int gridSize = (NUM_RAYS + blockSize - 1) / blockSize;

    // --- Main loop ---
    cv::namedWindow("occupancy_grid", cv::WINDOW_AUTOSIZE);

    for (int step = 0; step < num_steps; step++) {
        float rx = path_x[step];
        float ry = path_y[step];
        float rtheta = path_theta[step];

        // 1. Simulate lidar
        vector<float> ranges;
        simulate_lidar(rx, ry, rtheta, walls, ranges);

        // 2. Copy ranges to GPU
        CUDA_CHECK(cudaMemcpy(d_ranges, ranges.data(), NUM_RAYS * sizeof(float),
                               cudaMemcpyHostToDevice));

        // 3. Launch kernel to update grid
        update_grid_kernel<<<gridSize, blockSize>>>(
            d_logodds,
            GRID_W, GRID_H,
            GRID_ORIGIN_X, GRID_ORIGIN_Y,
            RESOLUTION,
            rx, ry, rtheta,
            d_ranges, NUM_RAYS,
            angular_res_rad, MAX_RANGE,
            L_OCC, L_FREE, L_MIN, L_MAX);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Copy grid back for visualization (every 2 steps to save time)
        if (step % 2 == 0 || step == num_steps - 1) {
            CUDA_CHECK(cudaMemcpy(h_logodds.data(), d_logodds,
                                   total_cells * sizeof(float),
                                   cudaMemcpyDeviceToHost));

            cv::Mat img;
            logodds_to_image(h_logodds, img, GRID_W, GRID_H, cell_px);
            draw_robot_and_lidar(img, rx, ry, rtheta, ranges,
                                 GRID_W, GRID_H,
                                 GRID_ORIGIN_X, GRID_ORIGIN_Y,
                                 RESOLUTION, cell_px);

            video.write(img);
            cv::imshow("occupancy_grid", img);
            int key = cv::waitKey(10);
            if (key == 27) break;  // ESC to quit
        }
    }

    video.release();
    cout << "Video saved to " << avi_path << endl;

    // Convert to GIF
    string cmd = "ffmpeg -y -i " + avi_path + " -vf \"fps=15,scale=500:-1:flags=lanczos\" "
                 + gif_path + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        cout << "GIF saved to " << gif_path << endl;
    } else {
        cout << "ffmpeg conversion failed (ffmpeg may not be installed)" << endl;
    }

    // Show final map
    CUDA_CHECK(cudaMemcpy(h_logodds.data(), d_logodds,
                           total_cells * sizeof(float),
                           cudaMemcpyDeviceToHost));
    cv::Mat final_img;
    logodds_to_image(h_logodds, final_img, GRID_W, GRID_H, cell_px);

    cv::imshow("occupancy_grid", final_img);
    cout << "Press any key to exit..." << endl;
    cv::waitKey(0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_logodds));
    CUDA_CHECK(cudaFree(d_ranges));
    cv::destroyAllWindows();

    return 0;
}
