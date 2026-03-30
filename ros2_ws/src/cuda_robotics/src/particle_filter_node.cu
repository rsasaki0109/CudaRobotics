/*************************************************************************
 * particle_filter_node.cu
 * ROS2 node wrapping the CUDA-parallelized Particle Filter
 *
 * Subscribes:
 *   /odom       (nav_msgs/Odometry)       - robot odometry
 *   /landmarks  (geometry_msgs/PoseArray)  - observed landmark positions
 *
 * Publishes:
 *   /pf/pose      (geometry_msgs/PoseStamped)           - estimated pose
 *   /pf/particles (geometry_msgs/PoseArray)             - particle cloud
 *
 * Timer: 10 Hz
 ************************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include <cmath>
#include <vector>
#include <random>
#include <mutex>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define PI 3.141592653f
#define NP 1000
#define NTH (NP / 2)
#define MAX_LANDMARKS 64
#define THREADS 256
#define BLOCKS ((NP + THREADS - 1) / THREADS)

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
// Observation struct (distance + landmark position)
// ---------------------------------------------------------------------------
struct Observation {
    float d;
    float lx;
    float ly;
};

// ===================================================================
// CUDA Kernels (inline from CudaRobotics particle_filter.cu)
// ===================================================================

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void predict_and_weight_kernel(
    float* px,
    float* pw,
    const float u_v,
    const float u_omega,
    const float rsim_0,
    const float rsim_1,
    const Observation* obs,
    const int n_obs,
    const float Q,
    curandState* rng_states,
    const float dt,
    const int np)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    curandState local_rng = rng_states[ip];

    float ud_v     = u_v     + curand_normal(&local_rng) * rsim_0;
    float ud_omega = u_omega + curand_normal(&local_rng) * rsim_1;

    float x   = px[0 * np + ip];
    float y   = px[1 * np + ip];
    float yaw = px[2 * np + ip];
    float v   = px[3 * np + ip];

    x   += dt * cosf(yaw) * ud_v;
    y   += dt * sinf(yaw) * ud_v;
    yaw += dt * ud_omega;
    v    = ud_v;

    px[0 * np + ip] = x;
    px[1 * np + ip] = y;
    px[2 * np + ip] = yaw;
    px[3 * np + ip] = v;

    float w = pw[ip];
    float inv_coeff = 1.0f / sqrtf(2.0f * PI * Q);

    for (int i = 0; i < n_obs; i++) {
        float dx = x - obs[i].lx;
        float dy = y - obs[i].ly;
        float prez = sqrtf(dx * dx + dy * dy);
        float dz = prez - obs[i].d;
        w *= inv_coeff * expf(-dz * dz / (2.0f * Q));
    }

    pw[ip] = w;
    rng_states[ip] = local_rng;
}

__global__ void normalize_weights_kernel(float* pw, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = 0.0f;
    for (int i = tid; i < np; i += blockDim.x) {
        val += pw[i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float total = sdata[0];
    if (total < 1e-30f) total = 1e-30f;

    for (int i = tid; i < np; i += blockDim.x) {
        pw[i] /= total;
    }
}

__global__ void weighted_mean_kernel(const float* px, const float* pw,
                                     float* x_est, int np) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i];
        s0 += px[0 * np + i] * w;
        s1 += px[1 * np + i] * w;
        s2 += px[2 * np + i] * w;
        s3 += px[3 * np + i] * w;
    }
    sdata[tid * 4 + 0] = s0;
    sdata[tid * 4 + 1] = s1;
    sdata[tid * 4 + 2] = s2;
    sdata[tid * 4 + 3] = s3;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid * 4 + 0] += sdata[(tid + s) * 4 + 0];
            sdata[tid * 4 + 1] += sdata[(tid + s) * 4 + 1];
            sdata[tid * 4 + 2] += sdata[(tid + s) * 4 + 2];
            sdata[tid * 4 + 3] += sdata[(tid + s) * 4 + 3];
        }
        __syncthreads();
    }

    if (tid == 0) {
        x_est[0] = sdata[0];
        x_est[1] = sdata[1];
        x_est[2] = sdata[2];
        x_est[3] = sdata[3];
    }
}

__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0];
    for (int i = 1; i < np; i++) {
        wcum[i] = wcum[i - 1] + pw[i];
    }
}

__global__ void resample_kernel(const float* px_in, float* px_out,
                                const float* wcum, float base_step,
                                float rand_offset, int np) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= np) return;

    float target = base_step * ip + rand_offset;

    int lo = 0, hi = np - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (wcum[mid] < target) lo = mid + 1;
        else hi = mid;
    }

    px_out[0 * np + ip] = px_in[0 * np + lo];
    px_out[1 * np + ip] = px_in[1 * np + lo];
    px_out[2 * np + ip] = px_in[2 * np + lo];
    px_out[3 * np + ip] = px_in[3 * np + lo];
}

// ===================================================================
// ROS2 Node
// ===================================================================

class ParticleFilterNode : public rclcpp::Node {
public:
    ParticleFilterNode() : Node("particle_filter_node") {
        // -----------------------------------------------------------
        // Parameters
        // -----------------------------------------------------------
        this->declare_parameter("num_particles", NP);
        this->declare_parameter("dt", 0.1);
        this->declare_parameter("measurement_noise_q", 0.01);
        this->declare_parameter("motion_noise_v", 1.0);
        this->declare_parameter("motion_noise_yaw", 0.5236);  // 30 deg
        this->declare_parameter("max_range", 20.0);

        np_          = this->get_parameter("num_particles").as_int();
        dt_          = static_cast<float>(this->get_parameter("dt").as_double());
        Q_           = static_cast<float>(this->get_parameter("measurement_noise_q").as_double());
        rsim_v_      = static_cast<float>(this->get_parameter("motion_noise_v").as_double());
        rsim_yaw_    = static_cast<float>(this->get_parameter("motion_noise_yaw").as_double());
        max_range_   = static_cast<float>(this->get_parameter("max_range").as_double());

        // -----------------------------------------------------------
        // CUDA allocation
        // -----------------------------------------------------------
        int blocks = (np_ + THREADS - 1) / THREADS;

        CUDA_CHECK(cudaMalloc(&d_px_,     4 * np_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_px_tmp_, 4 * np_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_px_, 0,   4 * np_ * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_pw_, np_ * sizeof(float)));
        std::vector<float> pw_init(np_, 1.0f / np_);
        CUDA_CHECK(cudaMemcpy(d_pw_, pw_init.data(), np_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_obs_, MAX_LANDMARKS * sizeof(Observation)));

        CUDA_CHECK(cudaMalloc(&d_rng_, np_ * sizeof(curandState)));
        init_curand_kernel<<<blocks, THREADS>>>(d_rng_, 42ULL, np_);

        CUDA_CHECK(cudaMalloc(&d_xEst_, 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wcum_, np_ * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());

        h_px_.resize(4 * np_);
        h_pw_.resize(np_);

        // -----------------------------------------------------------
        // ROS2 subscribers
        // -----------------------------------------------------------
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&ParticleFilterNode::odom_callback, this, std::placeholders::_1));

        sub_landmarks_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/landmarks", 10,
            std::bind(&ParticleFilterNode::landmarks_callback, this, std::placeholders::_1));

        // -----------------------------------------------------------
        // ROS2 publishers
        // -----------------------------------------------------------
        pub_pose_      = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/pose", 10);
        pub_particles_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/particles", 10);

        // -----------------------------------------------------------
        // Timer at 10 Hz
        // -----------------------------------------------------------
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&ParticleFilterNode::timer_callback, this));

        RCLCPP_INFO(this->get_logger(),
                    "ParticleFilterNode started: %d particles, dt=%.2f, Q=%.4f",
                    np_, dt_, Q_);
    }

    ~ParticleFilterNode() override {
        cudaFree(d_px_);
        cudaFree(d_px_tmp_);
        cudaFree(d_pw_);
        cudaFree(d_obs_);
        cudaFree(d_rng_);
        cudaFree(d_xEst_);
        cudaFree(d_wcum_);
    }

private:
    // -----------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        odom_v_     = static_cast<float>(msg->twist.twist.linear.x);
        odom_omega_ = static_cast<float>(msg->twist.twist.angular.z);
        odom_received_ = true;
    }

    void landmarks_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        landmarks_.clear();
        // Each pose in the array represents a landmark with an observed
        // distance stored in position.z (convention), and landmark
        // coordinates in position.x, position.y.
        for (const auto& pose : msg->poses) {
            Observation obs;
            obs.d  = static_cast<float>(pose.position.z);   // observed distance
            obs.lx = static_cast<float>(pose.position.x);   // landmark x
            obs.ly = static_cast<float>(pose.position.y);   // landmark y
            landmarks_.push_back(obs);
        }
    }

    // -----------------------------------------------------------
    // Main filter step (10 Hz timer)
    // -----------------------------------------------------------
    void timer_callback() {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!odom_received_) return;

        int blocks = (np_ + THREADS - 1) / THREADS;

        // Upload observations
        int n_obs = static_cast<int>(landmarks_.size());
        if (n_obs > MAX_LANDMARKS) n_obs = MAX_LANDMARKS;
        if (n_obs > 0) {
            CUDA_CHECK(cudaMemcpy(d_obs_, landmarks_.data(),
                                  n_obs * sizeof(Observation),
                                  cudaMemcpyHostToDevice));
        }

        // Predict + weight update
        predict_and_weight_kernel<<<blocks, THREADS>>>(
            d_px_, d_pw_,
            odom_v_, odom_omega_,
            rsim_v_, rsim_yaw_,
            d_obs_, n_obs, Q_,
            d_rng_, dt_, np_);

        // Normalize
        normalize_weights_kernel<<<1, THREADS, THREADS * sizeof(float)>>>(
            d_pw_, np_);

        // Weighted mean estimate
        weighted_mean_kernel<<<1, THREADS, THREADS * 4 * sizeof(float)>>>(
            d_px_, d_pw_, d_xEst_, np_);

        float h_xEst[4];
        CUDA_CHECK(cudaMemcpy(h_xEst, d_xEst_, 4 * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Read back particles and weights for resampling check + visualization
        CUDA_CHECK(cudaMemcpy(h_px_.data(), d_px_, 4 * np_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pw_.data(), d_pw_, np_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Resampling
        float neff_denom = 0.0f;
        for (int i = 0; i < np_; i++) neff_denom += h_pw_[i] * h_pw_[i];
        float neff = 1.0f / neff_denom;

        if (neff < NTH) {
            cumsum_kernel<<<1, 1>>>(d_pw_, d_wcum_, np_);

            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            float rand_offset = uni(gen_) / np_;
            float base_step   = 1.0f / np_;

            resample_kernel<<<blocks, THREADS>>>(
                d_px_, d_px_tmp_, d_wcum_, base_step, rand_offset, np_);

            CUDA_CHECK(cudaMemcpy(d_px_, d_px_tmp_, 4 * np_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice));

            std::vector<float> pw_uniform(np_, 1.0f / np_);
            CUDA_CHECK(cudaMemcpy(d_pw_, pw_uniform.data(),
                                  np_ * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Re-read particles for visualization after resample
            CUDA_CHECK(cudaMemcpy(h_px_.data(), d_px_, 4 * np_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // -----------------------------------------------------------
        // Publish estimated pose
        // -----------------------------------------------------------
        auto pose_msg = geometry_msgs::msg::PoseStamped();
        pose_msg.header.stamp = this->now();
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = h_xEst[0];
        pose_msg.pose.position.y = h_xEst[1];
        pose_msg.pose.position.z = 0.0;

        float yaw = h_xEst[2];
        pose_msg.pose.orientation.z = std::sin(yaw / 2.0f);
        pose_msg.pose.orientation.w = std::cos(yaw / 2.0f);

        pub_pose_->publish(pose_msg);

        // -----------------------------------------------------------
        // Publish particle cloud
        // -----------------------------------------------------------
        auto particles_msg = geometry_msgs::msg::PoseArray();
        particles_msg.header.stamp = this->now();
        particles_msg.header.frame_id = "map";
        particles_msg.poses.resize(np_);

        for (int i = 0; i < np_; i++) {
            auto& p = particles_msg.poses[i];
            p.position.x = h_px_[0 * np_ + i];
            p.position.y = h_px_[1 * np_ + i];
            p.position.z = 0.0;
            float pyaw = h_px_[2 * np_ + i];
            p.orientation.z = std::sin(pyaw / 2.0f);
            p.orientation.w = std::cos(pyaw / 2.0f);
        }

        pub_particles_->publish(particles_msg);
    }

    // -----------------------------------------------------------
    // Members
    // -----------------------------------------------------------
    std::mutex mtx_;

    // Parameters
    int   np_;
    float dt_;
    float Q_;
    float rsim_v_;
    float rsim_yaw_;
    float max_range_;

    // Odometry state
    float odom_v_     = 0.0f;
    float odom_omega_ = 0.0f;
    bool  odom_received_ = false;

    // Landmark observations
    std::vector<Observation> landmarks_;

    // CUDA device memory
    float*       d_px_      = nullptr;
    float*       d_px_tmp_  = nullptr;
    float*       d_pw_      = nullptr;
    Observation* d_obs_     = nullptr;
    curandState* d_rng_     = nullptr;
    float*       d_xEst_    = nullptr;
    float*       d_wcum_    = nullptr;

    // Host buffers
    std::vector<float> h_px_;
    std::vector<float> h_pw_;
    std::mt19937 gen_{std::random_device{}()};

    // ROS2
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          sub_odom_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr     sub_landmarks_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr      pub_pose_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr        pub_particles_;
    rclcpp::TimerBase::SharedPtr                                       timer_;
};

// ===================================================================
// main
// ===================================================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ParticleFilterNode>());
    rclcpp::shutdown();
    return 0;
}
