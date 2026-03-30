/*************************************************************************
 * dwa_node.cu
 * ROS2 node wrapping the CUDA-parallelized Dynamic Window Approach
 *
 * Subscribes:
 *   /odom       (nav_msgs/Odometry)           - robot odometry
 *   /goal       (geometry_msgs/PoseStamped)   - goal position
 *   /obstacles  (geometry_msgs/PoseArray)     - obstacle positions
 *
 * Publishes:
 *   /cmd_vel        (geometry_msgs/Twist)     - velocity command
 *   /dwa/trajectory (geometry_msgs/PoseArray) - predicted best trajectory
 *
 * Timer: 20 Hz
 ************************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <cmath>
#include <vector>
#include <array>
#include <cfloat>
#include <mutex>
#include <algorithm>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define PI 3.141592653f
#define THREADS 256

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// DWA config
// ---------------------------------------------------------------------------
struct DWAConfig {
    float max_speed         = 1.0f;
    float min_speed         = -0.5f;
    float max_yawrate       = 40.0f * PI / 180.0f;
    float max_accel         = 0.2f;
    float robot_radius      = 1.0f;
    float max_dyawrate      = 40.0f * PI / 180.0f;
    float v_reso            = 0.01f;
    float yawrate_reso      = 0.1f * PI / 180.0f;
    float dt                = 0.1f;
    float predict_time      = 3.0f;
    float to_goal_cost_gain = 1.0f;
    float speed_cost_gain   = 1.0f;
};

// ===================================================================
// CUDA Kernels (inline from CudaRobotics dynamic_window_approach.cu)
// ===================================================================

__global__ void dwa_eval_kernel(
    float sx, float sy, float syaw, float sv, float somega,
    float v_min, float v_max, float yr_min, float yr_max,
    float v_reso, float yr_reso, float dt, float predict_time,
    float max_speed, float to_goal_cost_gain, float speed_cost_gain,
    float robot_radius,
    float gx, float gy,
    const float* ob, int n_ob,
    int n_v, int n_yr,
    float* costs,
    float* ctrl_v,
    float* ctrl_yr,
    float* traj_end_x,
    float* traj_end_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_v * n_yr;
    if (idx >= total) return;

    int iv  = idx / n_yr;
    int iyr = idx % n_yr;

    float v  = v_min + iv  * v_reso;
    float yr = yr_min + iyr * yr_reso;

    if (v > v_max) v = v_max;
    if (yr > yr_max) yr = yr_max;

    ctrl_v[idx]  = v;
    ctrl_yr[idx] = yr;

    // Simulate trajectory
    float px = sx, py = sy, pyaw = syaw;
    float time_sim = 0.0f;
    float minr = FLT_MAX;
    bool collision = false;
    int skip_n = 2;
    int step = 0;

    while (time_sim <= predict_time) {
        pyaw += yr * dt;
        px   += v * cosf(pyaw) * dt;
        py   += v * sinf(pyaw) * dt;
        time_sim += dt;
        step++;

        if (step % skip_n == 0) {
            for (int i = 0; i < n_ob; i++) {
                float dx = px - ob[i * 2 + 0];
                float dy = py - ob[i * 2 + 1];
                float r  = sqrtf(dx * dx + dy * dy);
                if (r <= robot_radius) collision = true;
                if (r < minr) minr = r;
            }
        }
    }

    // Check last point
    for (int i = 0; i < n_ob; i++) {
        float dx = px - ob[i * 2 + 0];
        float dy = py - ob[i * 2 + 1];
        float r  = sqrtf(dx * dx + dy * dy);
        if (r <= robot_radius) collision = true;
        if (r < minr) minr = r;
    }

    traj_end_x[idx] = px;
    traj_end_y[idx] = py;

    if (collision) {
        costs[idx] = FLT_MAX;
        return;
    }

    // to_goal_cost
    float goal_mag = sqrtf(gx * gx + gy * gy);
    float traj_mag = sqrtf(px * px + py * py);
    float dot = gx * px + gy * py;
    float cos_angle = dot / (goal_mag * traj_mag + 1e-10f);
    if (cos_angle > 1.0f) cos_angle = 1.0f;
    if (cos_angle < -1.0f) cos_angle = -1.0f;
    float to_goal_cost = to_goal_cost_gain * acosf(cos_angle);

    // speed_cost
    float speed_cost = speed_cost_gain * (max_speed - v);

    // obstacle_cost
    float ob_cost = 1.0f / minr;

    costs[idx] = to_goal_cost + speed_cost + ob_cost;
}

__global__ void find_min_kernel(const float* costs, int* min_idx, int n) {
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int*   sidx = (int*)(smem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;

    float best_val = FLT_MAX;
    int   best_idx = 0;

    for (int i = tid; i < n; i += blockDim.x) {
        if (costs[i] < best_val) {
            best_val = costs[i];
            best_idx = i;
        }
    }
    sval[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] < sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *min_idx = sidx[0];
    }
}

// ===================================================================
// ROS2 Node
// ===================================================================

class DWANode : public rclcpp::Node {
public:
    DWANode() : Node("dwa_node") {
        // -----------------------------------------------------------
        // Parameters
        // -----------------------------------------------------------
        this->declare_parameter("max_speed", 1.0);
        this->declare_parameter("min_speed", -0.5);
        this->declare_parameter("max_yawrate", 40.0);
        this->declare_parameter("max_accel", 0.2);
        this->declare_parameter("max_dyawrate", 40.0);
        this->declare_parameter("v_reso", 0.01);
        this->declare_parameter("yawrate_reso", 0.1);
        this->declare_parameter("dt", 0.1);
        this->declare_parameter("predict_time", 3.0);
        this->declare_parameter("to_goal_cost_gain", 1.0);
        this->declare_parameter("speed_cost_gain", 1.0);
        this->declare_parameter("robot_radius", 1.0);
        this->declare_parameter("goal_tolerance", 0.5);

        cfg_.max_speed         = static_cast<float>(this->get_parameter("max_speed").as_double());
        cfg_.min_speed         = static_cast<float>(this->get_parameter("min_speed").as_double());
        cfg_.max_yawrate       = static_cast<float>(this->get_parameter("max_yawrate").as_double()) * PI / 180.0f;
        cfg_.max_accel         = static_cast<float>(this->get_parameter("max_accel").as_double());
        cfg_.max_dyawrate      = static_cast<float>(this->get_parameter("max_dyawrate").as_double()) * PI / 180.0f;
        cfg_.v_reso            = static_cast<float>(this->get_parameter("v_reso").as_double());
        cfg_.yawrate_reso      = static_cast<float>(this->get_parameter("yawrate_reso").as_double()) * PI / 180.0f;
        cfg_.dt                = static_cast<float>(this->get_parameter("dt").as_double());
        cfg_.predict_time      = static_cast<float>(this->get_parameter("predict_time").as_double());
        cfg_.to_goal_cost_gain = static_cast<float>(this->get_parameter("to_goal_cost_gain").as_double());
        cfg_.speed_cost_gain   = static_cast<float>(this->get_parameter("speed_cost_gain").as_double());
        cfg_.robot_radius      = static_cast<float>(this->get_parameter("robot_radius").as_double());
        goal_tolerance_        = static_cast<float>(this->get_parameter("goal_tolerance").as_double());

        // -----------------------------------------------------------
        // CUDA allocation (max possible grid)
        // -----------------------------------------------------------
        max_nv_  = static_cast<int>((cfg_.max_speed - cfg_.min_speed) / cfg_.v_reso) + 2;
        max_nyr_ = static_cast<int>((2.0f * cfg_.max_yawrate) / cfg_.yawrate_reso) + 2;
        max_samples_ = max_nv_ * max_nyr_;

        // Obstacle buffer (allocate for up to 256 obstacles)
        max_obstacles_ = 256;
        CUDA_CHECK(cudaMalloc(&d_ob_, max_obstacles_ * 2 * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_costs_,      max_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ctrl_v_,     max_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ctrl_yr_,    max_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_traj_end_x_, max_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_traj_end_y_, max_samples_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min_idx_,    sizeof(int)));

        // -----------------------------------------------------------
        // ROS2 subscribers
        // -----------------------------------------------------------
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&DWANode::odom_callback, this, std::placeholders::_1));

        sub_goal_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal", 10,
            std::bind(&DWANode::goal_callback, this, std::placeholders::_1));

        sub_obstacles_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/obstacles", 10,
            std::bind(&DWANode::obstacles_callback, this, std::placeholders::_1));

        // -----------------------------------------------------------
        // ROS2 publishers
        // -----------------------------------------------------------
        pub_cmd_vel_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        pub_traj_    = this->create_publisher<geometry_msgs::msg::PoseArray>("/dwa/trajectory", 10);

        // -----------------------------------------------------------
        // Timer at 20 Hz
        // -----------------------------------------------------------
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&DWANode::timer_callback, this));

        RCLCPP_INFO(this->get_logger(),
                    "DWANode started: max_samples=%d, predict_time=%.1f",
                    max_samples_, cfg_.predict_time);
    }

    ~DWANode() override {
        cudaFree(d_ob_);
        cudaFree(d_costs_);
        cudaFree(d_ctrl_v_);
        cudaFree(d_ctrl_yr_);
        cudaFree(d_traj_end_x_);
        cudaFree(d_traj_end_y_);
        cudaFree(d_min_idx_);
    }

private:
    // -----------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        state_[0] = static_cast<float>(msg->pose.pose.position.x);
        state_[1] = static_cast<float>(msg->pose.pose.position.y);
        // Extract yaw from quaternion
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        state_[2] = static_cast<float>(2.0 * std::atan2(qz, qw));
        state_[3] = static_cast<float>(msg->twist.twist.linear.x);
        state_[4] = static_cast<float>(msg->twist.twist.angular.z);
        odom_received_ = true;
    }

    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        goal_[0] = static_cast<float>(msg->pose.position.x);
        goal_[1] = static_cast<float>(msg->pose.position.y);
        goal_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Goal received: (%.2f, %.2f)",
                    goal_[0], goal_[1]);
    }

    void obstacles_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        obstacles_.clear();
        obstacles_.reserve(msg->poses.size() * 2);
        for (const auto& pose : msg->poses) {
            obstacles_.push_back(static_cast<float>(pose.position.x));
            obstacles_.push_back(static_cast<float>(pose.position.y));
        }
    }

    // -----------------------------------------------------------
    // Host-side trajectory simulation for visualization
    // -----------------------------------------------------------
    struct TrajPoint { float x, y, yaw; };

    std::vector<TrajPoint> simulate_trajectory(float v, float yr) const {
        std::vector<TrajPoint> traj;
        float px = state_[0], py = state_[1], pyaw = state_[2];
        float t = 0.0f;
        while (t <= cfg_.predict_time) {
            pyaw += yr * cfg_.dt;
            px   += v * std::cos(pyaw) * cfg_.dt;
            py   += v * std::sin(pyaw) * cfg_.dt;
            traj.push_back({px, py, pyaw});
            t += cfg_.dt;
        }
        return traj;
    }

    // -----------------------------------------------------------
    // Main DWA step (20 Hz timer)
    // -----------------------------------------------------------
    void timer_callback() {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!odom_received_ || !goal_received_) return;

        // Check if goal reached
        float dx = state_[0] - goal_[0];
        float dy = state_[1] - goal_[1];
        if (std::sqrt(dx * dx + dy * dy) <= goal_tolerance_) {
            // Publish zero velocity
            auto cmd = geometry_msgs::msg::Twist();
            pub_cmd_vel_->publish(cmd);
            return;
        }

        // Upload obstacles
        int n_ob = static_cast<int>(obstacles_.size()) / 2;
        if (n_ob > max_obstacles_) n_ob = max_obstacles_;
        if (n_ob > 0) {
            CUDA_CHECK(cudaMemcpy(d_ob_, obstacles_.data(),
                                  n_ob * 2 * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        // Dynamic window
        float dw_vmin  = std::max(state_[3] - cfg_.max_accel * cfg_.dt, cfg_.min_speed);
        float dw_vmax  = std::min(state_[3] + cfg_.max_accel * cfg_.dt, cfg_.max_speed);
        float dw_yrmin = std::max(state_[4] - cfg_.max_dyawrate * cfg_.dt, -cfg_.max_yawrate);
        float dw_yrmax = std::min(state_[4] + cfg_.max_dyawrate * cfg_.dt, cfg_.max_yawrate);

        int n_v  = static_cast<int>((dw_vmax - dw_vmin) / cfg_.v_reso) + 1;
        int n_yr = static_cast<int>((dw_yrmax - dw_yrmin) / cfg_.yawrate_reso) + 1;
        int n_samples = n_v * n_yr;

        if (n_samples <= 0) {
            auto cmd = geometry_msgs::msg::Twist();
            pub_cmd_vel_->publish(cmd);
            return;
        }

        int blocks = (n_samples + THREADS - 1) / THREADS;

        // GPU: evaluate all trajectory samples
        dwa_eval_kernel<<<blocks, THREADS>>>(
            state_[0], state_[1], state_[2], state_[3], state_[4],
            dw_vmin, dw_vmax, dw_yrmin, dw_yrmax,
            cfg_.v_reso, cfg_.yawrate_reso,
            cfg_.dt, cfg_.predict_time,
            cfg_.max_speed,
            cfg_.to_goal_cost_gain, cfg_.speed_cost_gain,
            cfg_.robot_radius,
            goal_[0], goal_[1],
            d_ob_, n_ob,
            n_v, n_yr,
            d_costs_, d_ctrl_v_, d_ctrl_yr_,
            d_traj_end_x_, d_traj_end_y_);

        // GPU: find minimum cost
        int red_threads = 256;
        size_t smem_size = red_threads * (sizeof(float) + sizeof(int));
        find_min_kernel<<<1, red_threads, smem_size>>>(d_costs_, d_min_idx_, n_samples);

        // Read back best control
        int h_min_idx;
        CUDA_CHECK(cudaMemcpy(&h_min_idx, d_min_idx_, sizeof(int),
                              cudaMemcpyDeviceToHost));

        float best_v, best_yr;
        CUDA_CHECK(cudaMemcpy(&best_v,  d_ctrl_v_  + h_min_idx, sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&best_yr, d_ctrl_yr_ + h_min_idx, sizeof(float),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaDeviceSynchronize());

        // -----------------------------------------------------------
        // Publish cmd_vel
        // -----------------------------------------------------------
        auto cmd = geometry_msgs::msg::Twist();
        cmd.linear.x  = best_v;
        cmd.angular.z = best_yr;
        pub_cmd_vel_->publish(cmd);

        // -----------------------------------------------------------
        // Publish best trajectory for visualization
        // -----------------------------------------------------------
        auto traj = simulate_trajectory(best_v, best_yr);

        auto traj_msg = geometry_msgs::msg::PoseArray();
        traj_msg.header.stamp = this->now();
        traj_msg.header.frame_id = "map";
        traj_msg.poses.resize(traj.size());

        for (size_t i = 0; i < traj.size(); i++) {
            auto& p = traj_msg.poses[i];
            p.position.x = traj[i].x;
            p.position.y = traj[i].y;
            p.position.z = 0.0;
            p.orientation.z = std::sin(traj[i].yaw / 2.0f);
            p.orientation.w = std::cos(traj[i].yaw / 2.0f);
        }

        pub_traj_->publish(traj_msg);
    }

    // -----------------------------------------------------------
    // Members
    // -----------------------------------------------------------
    std::mutex mtx_;

    DWAConfig cfg_;
    float goal_tolerance_ = 0.5f;

    // State: [x, y, yaw, v, omega]
    float state_[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float goal_[2]  = {0.0f, 0.0f};
    std::vector<float> obstacles_;  // flat array [x0, y0, x1, y1, ...]

    bool odom_received_ = false;
    bool goal_received_ = false;

    // CUDA device memory
    float* d_ob_         = nullptr;
    float* d_costs_      = nullptr;
    float* d_ctrl_v_     = nullptr;
    float* d_ctrl_yr_    = nullptr;
    float* d_traj_end_x_ = nullptr;
    float* d_traj_end_y_ = nullptr;
    int*   d_min_idx_    = nullptr;

    int max_nv_;
    int max_nyr_;
    int max_samples_;
    int max_obstacles_;

    // ROS2
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          sub_odom_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr   sub_goal_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr     sub_obstacles_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr            pub_cmd_vel_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr        pub_traj_;
    rclcpp::TimerBase::SharedPtr                                       timer_;
};

// ===================================================================
// main
// ===================================================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DWANode>());
    rclcpp::shutdown();
    return 0;
}
