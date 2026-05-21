/*************************************************************************
 * esdf_node.cu
 * ROS2 node wrapping the CUDA Jump Flooding Algorithm (JFA) ESDF
 *
 * Subscribes:
 *   /map  (nav_msgs/OccupancyGrid)  - input occupancy grid
 *
 * Publishes:
 *   /esdf (nav_msgs/OccupancyGrid)  - distance to nearest occupied cell,
 *                                     encoded as uint8 (0..100 := 0..max_dist
 *                                     meters; -1 in occupied cells)
 *
 * On every /map message, rebuilds the ESDF on the GPU and publishes the
 * result. Grid is reallocated on resolution / size changes.
 *
 * Background: comparison_esdf.cu shows ~53,000x per-cell speedup over
 * brute-force CPU. This node exposes the same kernel as a ROS2 service.
 ************************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <chrono>
#include <cfloat>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
} } while (0)

// -------------------------------------------------------------------------
// JFA kernels (mirror src/comparison_esdf.cu, parameterised on dynamic grid)
// -------------------------------------------------------------------------
__global__ void jfa_init_kernel(const unsigned char* __restrict__ occ,
                                int* __restrict__ seed, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    seed[idx] = occ[idx] ? idx : -1;
}

__global__ void jfa_step_kernel(const int* __restrict__ seed_in,
                                int* __restrict__ seed_out,
                                int W, int H, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    int best = seed_in[idx];
    float best_d2 = FLT_MAX;
    if (best >= 0) {
        int bx = best % W, by = best / W;
        int ex = x - bx, ey = y - by;
        best_d2 = static_cast<float>(ex * ex + ey * ey);
    }
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx * k;
            int ny = y + dy * k;
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            int s = seed_in[ny * W + nx];
            if (s < 0) continue;
            int sx = s % W, sy = s / W;
            int ex = x - sx, ey = y - sy;
            float d2 = static_cast<float>(ex * ex + ey * ey);
            if (d2 < best_d2) { best = s; best_d2 = d2; }
        }
    }
    seed_out[idx] = best;
}

__global__ void jfa_to_int8_kernel(const int* __restrict__ seed,
                                   const unsigned char* __restrict__ occ,
                                   signed char* __restrict__ out,
                                   int W, int H, float res, float max_dist) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    if (occ[idx]) { out[idx] = -1; return; }  // mark occupied
    int s = seed[idx];
    float d;
    if (s < 0) {
        d = max_dist;
    } else {
        int sx = s % W, sy = s / W;
        int dx = x - sx, dy = y - sy;
        d = sqrtf(static_cast<float>(dx * dx + dy * dy)) * res;
    }
    float scaled = d / max_dist * 100.0f;
    if (scaled > 100.0f) scaled = 100.0f;
    out[idx] = static_cast<signed char>(scaled);
}

// -------------------------------------------------------------------------
// ROS2 Node
// -------------------------------------------------------------------------
class EsdfNode : public rclcpp::Node {
public:
    EsdfNode() : Node("esdf_node") {
        this->declare_parameter("max_distance", 10.0);
        this->declare_parameter("occupancy_threshold", 50);
        max_dist_ = static_cast<float>(this->get_parameter("max_distance").as_double());
        occ_threshold_ = static_cast<int>(this->get_parameter("occupancy_threshold").as_int());

        sub_map_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(),
            std::bind(&EsdfNode::map_callback, this, std::placeholders::_1));

        pub_esdf_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/esdf", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());

        RCLCPP_INFO(this->get_logger(),
                    "esdf_node started: max_distance=%.2f m, occupancy_threshold=%d",
                    max_dist_, occ_threshold_);
    }

    ~EsdfNode() override {
        free_buffers();
    }

private:
    void free_buffers() {
        if (d_occ_)    { cudaFree(d_occ_);    d_occ_    = nullptr; }
        if (d_seed_a_) { cudaFree(d_seed_a_); d_seed_a_ = nullptr; }
        if (d_seed_b_) { cudaFree(d_seed_b_); d_seed_b_ = nullptr; }
        if (d_out_)    { cudaFree(d_out_);    d_out_    = nullptr; }
    }

    void ensure_buffers(int W, int H) {
        if (W == alloc_W_ && H == alloc_H_) return;
        free_buffers();
        size_t cells = static_cast<size_t>(W) * H;
        CUDA_CHECK(cudaMalloc(&d_occ_,    cells * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_seed_a_, cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seed_b_, cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_,    cells * sizeof(signed char)));
        alloc_W_ = W;
        alloc_H_ = H;
        RCLCPP_INFO(this->get_logger(), "Allocated buffers for %dx%d grid",
                    W, H);
    }

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        int W = static_cast<int>(msg->info.width);
        int H = static_cast<int>(msg->info.height);
        float res = msg->info.resolution;
        if (W <= 0 || H <= 0) {
            RCLCPP_WARN(this->get_logger(), "Empty map received");
            return;
        }
        ensure_buffers(W, H);

        std::vector<unsigned char> occ(static_cast<size_t>(W) * H);
        const auto& data = msg->data;
        for (size_t i = 0; i < occ.size(); i++) {
            int8_t v = (i < data.size()) ? data[i] : 0;
            occ[i] = (v >= occ_threshold_) ? 1u : 0u;
        }
        CUDA_CHECK(cudaMemcpy(d_occ_, occ.data(), occ.size(),
                              cudaMemcpyHostToDevice));

        dim3 blk(16, 16);
        dim3 grd((W + 15) / 16, (H + 15) / 16);

        auto t0 = std::chrono::high_resolution_clock::now();
        jfa_init_kernel<<<grd, blk>>>(d_occ_, d_seed_a_, W, H);
        int* in_ptr = d_seed_a_;
        int* out_ptr = d_seed_b_;
        int kmax = std::max(W, H) / 2;
        for (int k = kmax; k >= 1; k /= 2) {
            jfa_step_kernel<<<grd, blk>>>(in_ptr, out_ptr, W, H, k);
            std::swap(in_ptr, out_ptr);
        }
        jfa_to_int8_kernel<<<grd, blk>>>(in_ptr, d_occ_, d_out_, W, H, res, max_dist_);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        nav_msgs::msg::OccupancyGrid out;
        out.header = msg->header;
        out.info   = msg->info;
        out.data.resize(static_cast<size_t>(W) * H);
        CUDA_CHECK(cudaMemcpy(out.data.data(), d_out_, out.data.size(),
                              cudaMemcpyDeviceToHost));
        pub_esdf_->publish(out);
        RCLCPP_INFO(this->get_logger(),
                    "Published ESDF: %dx%d cells, JFA %.3f ms (max_dist=%.1f m)",
                    W, H, ms, max_dist_);
    }

    // -----------------------------------------------------------
    // Members
    // -----------------------------------------------------------
    float max_dist_       = 10.0f;
    int   occ_threshold_  = 50;

    unsigned char* d_occ_    = nullptr;
    int*           d_seed_a_ = nullptr;
    int*           d_seed_b_ = nullptr;
    signed char*   d_out_    = nullptr;
    int alloc_W_ = 0;
    int alloc_H_ = 0;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr     pub_esdf_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EsdfNode>());
    rclcpp::shutdown();
    return 0;
}
