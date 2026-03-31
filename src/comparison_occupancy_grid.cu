/*************************************************************************
    Occupancy Grid: CPU vs CUDA side-by-side comparison GIF generator
    Left panel:  CPU (for each ray, walk cells sequentially, update log-odds)
    Right panel: CUDA (update_grid_kernel, 1 thread per ray, 360 rays parallel)
    Same map/robot path as occupancy_grid_map.cu. Shows grid building up.
    Displays timing per scan.
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
// Constants (same as occupancy_grid_map.cu)
// -------------------------------------------------------------------------
const int   GRID_W       = 100;
const int   GRID_H       = 100;
const float RESOLUTION   = 0.5f;
const int   NUM_RAYS     = 360;
const float MAX_RANGE    = 20.0f;
const float ANGULAR_RES  = 1.0f;
const float L_OCC        = 0.85f;
const float L_FREE       = -0.4f;
const float L_PRIOR      = 0.0f;
const float L_MIN        = -5.0f;
const float L_MAX        = 5.0f;

const float GRID_ORIGIN_X = 0.0f;
const float GRID_ORIGIN_Y = 0.0f;

// -------------------------------------------------------------------------
// Obstacle segment
// -------------------------------------------------------------------------
struct Segment {
    float x0, y0, x1, y1;
};

// -------------------------------------------------------------------------
// CPU: simulate lidar
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
// CPU: update grid with one scan (sequential ray walking)
// -------------------------------------------------------------------------
void cpu_update_grid(
    vector<float>& logodds,
    int grid_w, int grid_h,
    float origin_x, float origin_y, float resolution,
    float robot_x, float robot_y, float robot_theta,
    const vector<float>& ranges,
    float angular_res_rad, float max_range,
    float l_occ, float l_free, float l_min, float l_max)
{
    for (int ray_idx = 0; ray_idx < NUM_RAYS; ray_idx++) {
        float range = ranges[ray_idx];
        float angle = robot_theta + (float)ray_idx * angular_res_rad;

        float ex = robot_x + range * cosf(angle);
        float ey = robot_y + range * sinf(angle);

        float gx0 = (robot_x - origin_x) / resolution;
        float gy0 = (robot_y - origin_y) / resolution;
        float gx1 = (ex - origin_x) / resolution;
        float gy1 = (ey - origin_y) / resolution;

        float ddx = gx1 - gx0;
        float ddy = gy1 - gy0;
        float dist = sqrtf(ddx * ddx + ddy * ddy);
        if (dist < 1e-6f) continue;

        int steps = (int)(dist * 2.0f) + 1;
        float step_x = ddx / (float)steps;
        float step_y = ddy / (float)steps;

        int prev_cx = -1, prev_cy = -1;

        for (int s = 0; s <= steps; s++) {
            float px = gx0 + step_x * (float)s;
            float py = gy0 + step_y * (float)s;
            int cx = (int)floorf(px);
            int cy = (int)floorf(py);

            if (cx == prev_cx && cy == prev_cy) continue;
            prev_cx = cx;
            prev_cy = cy;

            if (cx < 0 || cx >= grid_w || cy < 0 || cy >= grid_h) continue;

            int cell_idx = cy * grid_w + cx;

            float fdx = px - gx1;
            float fdy = py - gy1;
            float dist_to_end = sqrtf(fdx * fdx + fdy * fdy);

            float update;
            if (dist_to_end < 1.0f && range < max_range) {
                update = l_occ;
            } else {
                update = l_free;
            }

            logodds[cell_idx] += update;
            if (logodds[cell_idx] > l_max) logodds[cell_idx] = l_max;
            if (logodds[cell_idx] < l_min) logodds[cell_idx] = l_min;
        }
    }
}

// -------------------------------------------------------------------------
// CUDA kernel: update_grid_kernel (1 thread per ray)
// -------------------------------------------------------------------------
__global__ void update_grid_kernel(
    float* d_logodds,
    int grid_w, int grid_h,
    float origin_x, float origin_y, float resolution,
    float robot_x, float robot_y, float robot_theta,
    const float* d_ranges,
    int num_rays, float angular_res_rad,
    float max_range, float l_occ, float l_free,
    float l_min, float l_max)
{
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    float range = d_ranges[ray_idx];
    float angle = robot_theta + (float)ray_idx * angular_res_rad;

    float ex = robot_x + range * cosf(angle);
    float ey = robot_y + range * sinf(angle);

    float gx0 = (robot_x - origin_x) / resolution;
    float gy0 = (robot_y - origin_y) / resolution;
    float gx1 = (ex - origin_x) / resolution;
    float gy1 = (ey - origin_y) / resolution;

    float ddx = gx1 - gx0;
    float ddy = gy1 - gy0;
    float dist = sqrtf(ddx * ddx + ddy * ddy);
    if (dist < 1e-6f) return;

    int steps = (int)(dist * 2.0f) + 1;
    float step_x = ddx / (float)steps;
    float step_y = ddy / (float)steps;

    int prev_cx = -1, prev_cy = -1;

    for (int s = 0; s <= steps; s++) {
        float px = gx0 + step_x * (float)s;
        float py = gy0 + step_y * (float)s;
        int cx = (int)floorf(px);
        int cy = (int)floorf(py);

        if (cx == prev_cx && cy == prev_cy) continue;
        prev_cx = cx;
        prev_cy = cy;

        if (cx < 0 || cx >= grid_w || cy < 0 || cy >= grid_h) continue;

        int cell_idx = cy * grid_w + cx;

        float fdx = px - gx1;
        float fdy = py - gy1;
        float dist_to_end = sqrtf(fdx * fdx + fdy * fdy);

        float update;
        if (dist_to_end < 1.0f && range < max_range) {
            update = l_occ;
        } else {
            update = l_free;
        }

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
// Visualization: log-odds to image
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
            float p = 1.0f / (1.0f + expf(-lo));
            unsigned char val = (unsigned char)((1.0f - p) * 255.0f);
            cv::Vec3b color(val, val, val);

            for (int dy = 0; dy < cell_px; dy++) {
                for (int dx = 0; dx < cell_px; dx++) {
                    int px_x = cx * cell_px + dx;
                    int px_y = (grid_h - 1 - cy) * cell_px + dy;
                    if (px_x < img_w && px_y < img_h) {
                        img.at<cv::Vec3b>(px_y, px_x) = color;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Draw robot and lidar on image
// -------------------------------------------------------------------------
void draw_robot_and_lidar(
    cv::Mat& img,
    float robot_x, float robot_y, float robot_theta,
    const vector<float>& ranges,
    int grid_w, int grid_h,
    float origin_x, float origin_y,
    float resolution, int cell_px)
{
    auto to_pixel = [&](float wx, float wy) -> cv::Point {
        float gx = (wx - origin_x) / resolution;
        float gy = (wy - origin_y) / resolution;
        int px = (int)(gx * cell_px + cell_px / 2);
        int py = (int)((grid_h - 1 - gy) * cell_px + cell_px / 2);
        return cv::Point(px, py);
    };

    for (int i = 0; i < NUM_RAYS; i += 10) {
        float angle = robot_theta + (float)i * ANGULAR_RES * M_PI / 180.0f;
        float ex = robot_x + ranges[i] * cosf(angle);
        float ey = robot_y + ranges[i] * sinf(angle);
        cv::line(img, to_pixel(robot_x, robot_y), to_pixel(ex, ey),
                 cv::Scalar(0, 180, 0), 1);
    }

    cv::circle(img, to_pixel(robot_x, robot_y), cell_px * 2,
               cv::Scalar(0, 0, 255), -1);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "Occupancy Grid comparison: CPU vs CUDA" << endl;

    // Same walls as occupancy_grid_map.cu
    vector<Segment> walls;
    float world_w = GRID_W * RESOLUTION;
    float world_h = GRID_H * RESOLUTION;
    float margin = 2.0f;

    walls.push_back({margin, margin, world_w - margin, margin});
    walls.push_back({world_w - margin, margin, world_w - margin, world_h - margin});
    walls.push_back({world_w - margin, world_h - margin, margin, world_h - margin});
    walls.push_back({margin, world_h - margin, margin, margin});

    walls.push_back({15.0f, 20.0f, 25.0f, 20.0f});
    walls.push_back({30.0f, 10.0f, 30.0f, 25.0f});
    walls.push_back({10.0f, 35.0f, 10.0f, 42.0f});
    walls.push_back({10.0f, 42.0f, 18.0f, 42.0f});
    walls.push_back({38.0f, 35.0f, 44.0f, 35.0f});
    walls.push_back({44.0f, 35.0f, 44.0f, 42.0f});
    walls.push_back({44.0f, 42.0f, 38.0f, 42.0f});
    walls.push_back({38.0f, 42.0f, 38.0f, 35.0f});
    walls.push_back({20.0f, 30.0f, 28.0f, 38.0f});

    // CPU and CUDA grids
    int total_cells = GRID_W * GRID_H;
    vector<float> cpu_logodds(total_cells, L_PRIOR);
    vector<float> cuda_logodds_host(total_cells, L_PRIOR);

    // CUDA device memory
    float* d_logodds;
    float* d_ranges;
    CUDA_CHECK(cudaMalloc(&d_logodds, total_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ranges, NUM_RAYS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_logodds, cuda_logodds_host.data(),
                           total_cells * sizeof(float), cudaMemcpyHostToDevice));

    // Robot path (same as occupancy_grid_map.cu)
    vector<float> path_x, path_y, path_theta;
    float speed = 0.8f;
    float cx = 25.0f, cy = 25.0f;
    float half_w = 15.0f, half_h = 15.0f;

    for (float x = cx - half_w; x <= cx + half_w; x += speed) {
        path_x.push_back(x); path_y.push_back(cy - half_h); path_theta.push_back(0.0f);
    }
    for (float y = cy - half_h; y <= cy + half_h; y += speed) {
        path_x.push_back(cx + half_w); path_y.push_back(y); path_theta.push_back(M_PI / 2.0f);
    }
    for (float x = cx + half_w; x >= cx - half_w; x -= speed) {
        path_x.push_back(x); path_y.push_back(cy + half_h); path_theta.push_back(M_PI);
    }
    for (float y = cy + half_h; y >= cy - half_h; y -= speed) {
        path_x.push_back(cx - half_w); path_y.push_back(y); path_theta.push_back(-M_PI / 2.0f);
    }

    int num_steps = (int)path_x.size();
    cout << "Path steps: " << num_steps << endl;

    // Visualization: 500x500 per side = 1000x500
    int cell_px = 5;
    int img_w = GRID_W * cell_px;  // 500
    int img_h = GRID_H * cell_px;  // 500

    string avi_path = "gif/comparison_occupancy_grid.avi";
    string gif_path = "gif/comparison_occupancy_grid.gif";

    cv::VideoWriter video(avi_path,
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                          30, cv::Size(img_w * 2, img_h));
    if (!video.isOpened()) {
        cerr << "Failed to open video writer at " << avi_path << endl;
        return 1;
    }

    float angular_res_rad = ANGULAR_RES * M_PI / 180.0f;
    int blockSize = 128;
    int gridSize = (NUM_RAYS + blockSize - 1) / blockSize;

    double cpu_total_ms = 0.0, cuda_total_ms = 0.0;
    int scan_count = 0;

    for (int step = 0; step < num_steps; step++) {
        float rx = path_x[step];
        float ry = path_y[step];
        float rtheta = path_theta[step];

        // Simulate lidar
        vector<float> ranges;
        simulate_lidar(rx, ry, rtheta, walls, ranges);

        // ======== CPU update ========
        auto cpu_t0 = chrono::high_resolution_clock::now();
        cpu_update_grid(cpu_logodds, GRID_W, GRID_H,
                        GRID_ORIGIN_X, GRID_ORIGIN_Y, RESOLUTION,
                        rx, ry, rtheta, ranges,
                        angular_res_rad, MAX_RANGE,
                        L_OCC, L_FREE, L_MIN, L_MAX);
        auto cpu_t1 = chrono::high_resolution_clock::now();
        double cpu_ms = chrono::duration<double, milli>(cpu_t1 - cpu_t0).count();
        cpu_total_ms += cpu_ms;

        // ======== CUDA update ========
        CUDA_CHECK(cudaMemcpy(d_ranges, ranges.data(), NUM_RAYS * sizeof(float),
                               cudaMemcpyHostToDevice));

        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start);

        update_grid_kernel<<<gridSize, blockSize>>>(
            d_logodds, GRID_W, GRID_H,
            GRID_ORIGIN_X, GRID_ORIGIN_Y, RESOLUTION,
            rx, ry, rtheta,
            d_ranges, NUM_RAYS, angular_res_rad, MAX_RANGE,
            L_OCC, L_FREE, L_MIN, L_MAX);
        CUDA_CHECK(cudaGetLastError());

        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, ev_start, ev_stop);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cuda_total_ms += cuda_ms;

        scan_count++;

        // Visualize every 2 steps
        if (step % 2 == 0 || step == num_steps - 1) {
            // Copy CUDA grid back
            CUDA_CHECK(cudaMemcpy(cuda_logodds_host.data(), d_logodds,
                                   total_cells * sizeof(float), cudaMemcpyDeviceToHost));

            cv::Mat cpu_img, cuda_img;
            logodds_to_image(cpu_logodds, cpu_img, GRID_W, GRID_H, cell_px);
            logodds_to_image(cuda_logodds_host, cuda_img, GRID_W, GRID_H, cell_px);

            draw_robot_and_lidar(cpu_img, rx, ry, rtheta, ranges,
                                 GRID_W, GRID_H,
                                 GRID_ORIGIN_X, GRID_ORIGIN_Y,
                                 RESOLUTION, cell_px);
            draw_robot_and_lidar(cuda_img, rx, ry, rtheta, ranges,
                                 GRID_W, GRID_H,
                                 GRID_ORIGIN_X, GRID_ORIGIN_Y,
                                 RESOLUTION, cell_px);

            // Labels
            char buf[128];

            cv::putText(cpu_img, "CPU (sequential rays)", cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            cv::putText(cpu_img, "CPU (sequential rays)", cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            snprintf(buf, sizeof(buf), "CPU: %.3f ms/scan (avg)", cpu_total_ms / scan_count);
            cv::putText(cpu_img, buf, cv::Point(10, 42),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 2);
            cv::putText(cpu_img, buf, cv::Point(10, 42),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 200), 1);
            snprintf(buf, sizeof(buf), "Step: %d / %d", step + 1, num_steps);
            cv::putText(cpu_img, buf, cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 2);
            cv::putText(cpu_img, buf, cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

            cv::putText(cuda_img, "CUDA (360 rays parallel)", cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            cv::putText(cuda_img, "CUDA (360 rays parallel)", cv::Point(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            snprintf(buf, sizeof(buf), "CUDA: %.3f ms/scan (avg)", cuda_total_ms / scan_count);
            cv::putText(cuda_img, buf, cv::Point(10, 42),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 2);
            cv::putText(cuda_img, buf, cv::Point(10, 42),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 200), 1);
            if (cuda_total_ms > 0.0) {
                snprintf(buf, sizeof(buf), "Speedup: %.1fx",
                         cpu_total_ms / cuda_total_ms);
                cv::putText(cuda_img, buf, cv::Point(10, 60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 2);
                cv::putText(cuda_img, buf, cv::Point(10, 60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 150, 0), 1);
            }

            cv::Mat combined;
            cv::hconcat(cpu_img, cuda_img, combined);
            video.write(combined);
        }
    }

    video.release();
    cout << "Video saved to " << avi_path << endl;

    // Convert to GIF
    string cmd = "ffmpeg -y -i " + avi_path +
        " -vf 'fps=15,scale=1000:-1:flags=lanczos' -loop 0 " + gif_path + " 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        cout << "GIF saved to " << gif_path << endl;
    } else {
        cout << "ffmpeg conversion failed (ffmpeg may not be installed)" << endl;
    }

    printf("\n=== Results ===\n");
    printf("CPU  avg per scan: %.3f ms (%d scans)\n", cpu_total_ms / scan_count, scan_count);
    printf("CUDA avg per scan: %.3f ms (%d scans)\n", cuda_total_ms / scan_count, scan_count);
    if (cuda_total_ms > 0.0)
        printf("Speedup:           %.1fx\n", cpu_total_ms / cuda_total_ms);

    // Cleanup
    CUDA_CHECK(cudaFree(d_logodds));
    CUDA_CHECK(cudaFree(d_ranges));

    return 0;
}
