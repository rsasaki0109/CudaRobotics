/*************************************************************************
    > File Name: multi_robot_planner.cu
    > CUDA-parallelized Multi-Robot Potential Field Path Planner
    > N robots navigate from start to goal using potential fields with
    >   inter-robot collision avoidance.
    > GPU kernel: compute_forces_kernel (1 thread per robot)
    >   - attractive force toward goal
    >   - repulsive force from obstacles
    >   - repulsive force from other robots
    > GPU kernel: update_positions_kernel
    >   - apply forces, update velocity and position
    > Scenario: 20 robots on a circle swap to diametrically opposite goals
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
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
const int   N_ROBOTS     = 20;
const float KP_ATT       = 5.0f;      // attractive potential gain
const float KP_REP       = 100.0f;    // repulsive potential gain (obstacles)
const float KP_ROBOT     = 50.0f;     // repulsive potential gain (other robots)
const float ROBOT_RADIUS = 0.5f;      // robot body radius [m]
const float MAX_SPEED    = 1.0f;      // max velocity magnitude [m/s]
const float DT           = 0.05f;     // time step [s]
const float SIM_TIME     = 30.0f;     // total simulation time [s]

const float OBS_INFLUENCE = 5.0f;     // obstacle repulsive influence range [m]
const float ROBOT_INFLUENCE = 3.0f;   // inter-robot repulsive influence range [m]
const float GOAL_TOL     = 0.3f;      // goal reached tolerance [m]
const float DAMPING      = 0.8f;      // velocity damping factor

// Circle scenario
const float CIRCLE_RADIUS = 15.0f;
const float CIRCLE_CX     = 20.0f;
const float CIRCLE_CY     = 20.0f;

// Obstacles: (cx, cy, radius)
const int N_OBSTACLES = 3;

struct Obstacle {
    float x, y, r;
};

// -------------------------------------------------------------------------
// Robot state: position, velocity, goal
// Stored as arrays of floats for easy GPU transfer
// -------------------------------------------------------------------------
struct RobotState {
    float px[N_ROBOTS];
    float py[N_ROBOTS];
    float vx[N_ROBOTS];
    float vy[N_ROBOTS];
    float gx[N_ROBOTS];
    float gy[N_ROBOTS];
};

// -------------------------------------------------------------------------
// GPU kernel: compute forces for each robot (1 thread per robot)
// Attractive force toward goal + repulsive from obstacles + repulsive from
// other robots
// -------------------------------------------------------------------------
__global__ void compute_forces_kernel(
    const float* px, const float* py,
    const float* gx, const float* gy,
    const float* obs_x, const float* obs_y, const float* obs_r,
    int n_obs, int n_robots,
    float* fx, float* fy,
    float kp_att, float kp_rep, float kp_robot,
    float obs_influence, float robot_influence, float robot_radius)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_robots) return;

    float force_x = 0.0f;
    float force_y = 0.0f;

    float rx = px[i];
    float ry = py[i];

    // --- Attractive force toward goal ---
    float dx_goal = gx[i] - rx;
    float dy_goal = gy[i] - ry;
    float dist_goal = sqrtf(dx_goal * dx_goal + dy_goal * dy_goal);

    if (dist_goal > 0.001f) {
        // Linear attractive force (capped for stability)
        float f_att = kp_att * fminf(dist_goal, 5.0f);
        force_x += f_att * (dx_goal / dist_goal);
        force_y += f_att * (dy_goal / dist_goal);
    }

    // --- Repulsive force from obstacles ---
    for (int k = 0; k < n_obs; k++) {
        float dx_obs = rx - obs_x[k];
        float dy_obs = ry - obs_y[k];
        float dist_obs = sqrtf(dx_obs * dx_obs + dy_obs * dy_obs);
        float clearance = dist_obs - obs_r[k];

        if (clearance < 0.01f) clearance = 0.01f;

        if (clearance < obs_influence) {
            // Repulsive force: magnitude = kp_rep * (1/clearance - 1/influence) / clearance^2
            float inv_diff = 1.0f / clearance - 1.0f / obs_influence;
            float magnitude = kp_rep * inv_diff / (clearance * clearance);

            // Direction: away from obstacle center
            if (dist_obs > 0.001f) {
                force_x += magnitude * (dx_obs / dist_obs);
                force_y += magnitude * (dy_obs / dist_obs);
            }
        }
    }

    // --- Repulsive force from other robots ---
    for (int j = 0; j < n_robots; j++) {
        if (j == i) continue;

        float dx_r = rx - px[j];
        float dy_r = ry - py[j];
        float dist_r = sqrtf(dx_r * dx_r + dy_r * dy_r);
        float clearance = dist_r - 2.0f * robot_radius;

        if (clearance < 0.01f) clearance = 0.01f;

        if (clearance < robot_influence) {
            float inv_diff = 1.0f / clearance - 1.0f / robot_influence;
            float magnitude = kp_robot * inv_diff / (clearance * clearance);

            if (dist_r > 0.001f) {
                force_x += magnitude * (dx_r / dist_r);
                force_y += magnitude * (dy_r / dist_r);
            }
        }
    }

    fx[i] = force_x;
    fy[i] = force_y;
}

// -------------------------------------------------------------------------
// GPU kernel: update velocity and position
// -------------------------------------------------------------------------
__global__ void update_positions_kernel(
    float* px, float* py,
    float* vx, float* vy,
    const float* fx, const float* fy,
    int n_robots,
    float dt, float max_speed, float damping)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_robots) return;

    // Update velocity: v = damping * v + force * dt
    vx[i] = damping * vx[i] + fx[i] * dt;
    vy[i] = damping * vy[i] + fy[i] * dt;

    // Clamp speed
    float speed = sqrtf(vx[i] * vx[i] + vy[i] * vy[i]);
    if (speed > max_speed) {
        vx[i] = vx[i] * max_speed / speed;
        vy[i] = vy[i] * max_speed / speed;
    }

    // Update position
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
}

// -------------------------------------------------------------------------
// Visualization helpers
// -------------------------------------------------------------------------
cv::Scalar robot_color(int idx, int total) {
    float hue = (float)idx / (float)total * 180.0f;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((int)hue, 255, 230));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    cout << "CUDA Multi-Robot Potential Field Planner" << endl;
    cout << "Robots: " << N_ROBOTS << ", Simulation time: " << SIM_TIME << "s" << endl;

    // --- Setup scenario ---
    RobotState state;
    memset(&state, 0, sizeof(state));

    // Place robots on a circle, goals are diametrically opposite
    for (int i = 0; i < N_ROBOTS; i++) {
        float angle = 2.0f * M_PI * (float)i / (float)N_ROBOTS;
        state.px[i] = CIRCLE_CX + CIRCLE_RADIUS * cosf(angle);
        state.py[i] = CIRCLE_CY + CIRCLE_RADIUS * sinf(angle);
        state.vx[i] = 0.0f;
        state.vy[i] = 0.0f;
        // Diametrically opposite goal
        float goal_angle = angle + M_PI;
        state.gx[i] = CIRCLE_CX + CIRCLE_RADIUS * cosf(goal_angle);
        state.gy[i] = CIRCLE_CY + CIRCLE_RADIUS * sinf(goal_angle);
    }

    // Obstacles
    Obstacle obstacles[N_OBSTACLES] = {
        {20.0f, 20.0f, 3.0f},
        {15.0f, 25.0f, 2.0f},
        {25.0f, 15.0f, 2.0f}
    };

    float h_obs_x[N_OBSTACLES], h_obs_y[N_OBSTACLES], h_obs_r[N_OBSTACLES];
    for (int k = 0; k < N_OBSTACLES; k++) {
        h_obs_x[k] = obstacles[k].x;
        h_obs_y[k] = obstacles[k].y;
        h_obs_r[k] = obstacles[k].r;
    }

    // --- Allocate device memory ---
    float *d_px, *d_py, *d_vx, *d_vy, *d_gx, *d_gy;
    float *d_fx, *d_fy;
    float *d_obs_x, *d_obs_y, *d_obs_r;

    size_t robot_bytes = N_ROBOTS * sizeof(float);
    size_t obs_bytes = N_OBSTACLES * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_px, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_py, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_gx, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_gy, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_fx, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_fy, robot_bytes));
    CUDA_CHECK(cudaMalloc(&d_obs_x, obs_bytes));
    CUDA_CHECK(cudaMalloc(&d_obs_y, obs_bytes));
    CUDA_CHECK(cudaMalloc(&d_obs_r, obs_bytes));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_px, state.px, robot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, state.py, robot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, state.vx, robot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, state.vy, robot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx, state.gx, robot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, state.gy, robot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_x, h_obs_x, obs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_y, h_obs_y, obs_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs_r, h_obs_r, obs_bytes, cudaMemcpyHostToDevice));

    // --- Trail storage (host) ---
    // Store position history for each robot
    vector<vector<cv::Point2f>> trails(N_ROBOTS);

    // --- Visualization setup ---
    const float WORLD_MIN = -2.0f;
    const float WORLD_MAX = 42.0f;
    const int IMG_SIZE = 800;
    float scale = (float)IMG_SIZE / (WORLD_MAX - WORLD_MIN);

    auto to_pixel = [&](float wx, float wy) -> cv::Point {
        int px = (int)((wx - WORLD_MIN) * scale);
        int py = IMG_SIZE - (int)((wy - WORLD_MIN) * scale);
        return cv::Point(px, py);
    };

    // Precompute robot colors
    vector<cv::Scalar> colors(N_ROBOTS);
    for (int i = 0; i < N_ROBOTS; i++) {
        colors[i] = robot_color(i, N_ROBOTS);
    }

    cv::VideoWriter video("gif/multi_robot.avi",
                          cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(IMG_SIZE, IMG_SIZE));

    // CUDA launch config
    int blockSize = 32;
    int gridSize = (N_ROBOTS + blockSize - 1) / blockSize;

    // --- Simulation loop ---
    int total_steps = (int)(SIM_TIME / DT);
    int vis_skip = 2;  // visualize every N steps

    for (int step = 0; step < total_steps; step++) {
        // Launch force computation kernel
        compute_forces_kernel<<<gridSize, blockSize>>>(
            d_px, d_py, d_gx, d_gy,
            d_obs_x, d_obs_y, d_obs_r,
            N_OBSTACLES, N_ROBOTS,
            d_fx, d_fy,
            KP_ATT, KP_REP, KP_ROBOT,
            OBS_INFLUENCE, ROBOT_INFLUENCE, ROBOT_RADIUS);
        CUDA_CHECK(cudaGetLastError());

        // Launch position update kernel
        update_positions_kernel<<<gridSize, blockSize>>>(
            d_px, d_py, d_vx, d_vy,
            d_fx, d_fy,
            N_ROBOTS,
            DT, MAX_SPEED, DAMPING);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy positions back for visualization
        CUDA_CHECK(cudaMemcpy(state.px, d_px, robot_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.py, d_py, robot_bytes, cudaMemcpyDeviceToHost));

        // Record trails
        for (int i = 0; i < N_ROBOTS; i++) {
            trails[i].push_back(cv::Point2f(state.px[i], state.py[i]));
        }

        // --- Visualization ---
        if (step % vis_skip != 0) continue;

        cv::Mat img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw grid lines (light gray)
        for (int g = 0; g <= 40; g += 5) {
            cv::Point p1 = to_pixel((float)g, WORLD_MIN);
            cv::Point p2 = to_pixel((float)g, WORLD_MAX);
            cv::line(img, p1, p2, cv::Scalar(230, 230, 230), 1);
            p1 = to_pixel(WORLD_MIN, (float)g);
            p2 = to_pixel(WORLD_MAX, (float)g);
            cv::line(img, p1, p2, cv::Scalar(230, 230, 230), 1);
        }

        // Draw obstacles as filled black circles
        for (int k = 0; k < N_OBSTACLES; k++) {
            cv::Point center = to_pixel(obstacles[k].x, obstacles[k].y);
            int r_px = (int)(obstacles[k].r * scale);
            cv::circle(img, center, r_px, cv::Scalar(0, 0, 0), -1);
        }

        // Draw trails
        for (int i = 0; i < N_ROBOTS; i++) {
            for (int t = 1; t < (int)trails[i].size(); t++) {
                cv::Point p1 = to_pixel(trails[i][t - 1].x, trails[i][t - 1].y);
                cv::Point p2 = to_pixel(trails[i][t].x, trails[i][t].y);
                cv::line(img, p1, p2, colors[i], 1);
            }
        }

        // Draw goals as small rings
        for (int i = 0; i < N_ROBOTS; i++) {
            cv::Point gp = to_pixel(state.gx[i], state.gy[i]);
            cv::circle(img, gp, 5, colors[i], 1);
        }

        // Draw robots as filled colored circles
        for (int i = 0; i < N_ROBOTS; i++) {
            cv::Point rp = to_pixel(state.px[i], state.py[i]);
            int r_px = max(3, (int)(ROBOT_RADIUS * scale));
            cv::circle(img, rp, r_px, colors[i], -1);
            cv::circle(img, rp, r_px, cv::Scalar(0, 0, 0), 1);
        }

        // Status text
        char buf[128];
        snprintf(buf, sizeof(buf), "t=%.1fs  step=%d/%d", step * DT, step, total_steps);
        cv::putText(img, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 0, 0), 1);

        // Count robots that reached goal
        int reached = 0;
        for (int i = 0; i < N_ROBOTS; i++) {
            float dx = state.px[i] - state.gx[i];
            float dy = state.py[i] - state.gy[i];
            if (sqrtf(dx * dx + dy * dy) < GOAL_TOL) reached++;
        }
        snprintf(buf, sizeof(buf), "Reached: %d/%d", reached, N_ROBOTS);
        cv::putText(img, buf, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 128, 0), 1);

        cv::imshow("multi_robot", img);
        video.write(img);
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC to quit

        // Early termination if all robots reached goals
        if (reached == N_ROBOTS) {
            cout << "All robots reached their goals at t=" << step * DT << "s" << endl;
            cv::waitKey(0);
            break;
        }
    }

    video.release();
    system("ffmpeg -y -i gif/multi_robot.avi "
           "-vf 'fps=15,scale=400:-1:flags=lanczos' -loop 0 "
           "gif/multi_robot.gif 2>/dev/null");
    std::cout << "GIF saved to gif/multi_robot.gif" << std::endl;

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_gx));
    CUDA_CHECK(cudaFree(d_gy));
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_obs_x));
    CUDA_CHECK(cudaFree(d_obs_y));
    CUDA_CHECK(cudaFree(d_obs_r));

    cout << "Simulation complete." << endl;
    return 0;
}
