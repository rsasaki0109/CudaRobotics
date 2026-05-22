/*************************************************************************
 * voxel_node.cu
 * ROS2 node wrapping the CUDA 3D voxel-map log-odds raycast
 * (mirrors src/comparison_voxel_map.cu kernels)
 *
 * Subscribes:
 *   /points  (sensor_msgs/PointCloud2)  - LiDAR hits in sensor frame.
 *                                         Expects float32 x,y,z at the
 *                                         standard offsets 0/4/8.
 *   /odom    (nav_msgs/Odometry)        - sensor pose (used as origin
 *                                         and yaw for the rays)
 *
 * Publishes:
 *   /voxel_map (nav_msgs/OccupancyGrid) - top-down max-over-Z projection
 *                                         of the log-odds voxel grid,
 *                                         encoded 0..100 from sigmoid(L)
 *
 * Each /points message launches the same atomicAdd log-odds kernel used
 * in comparison_voxel_map.cu (one thread per ray) and then a small
 * reduce-Z kernel projects the grid to 2D for visualization.
 ************************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
} } while (0)

// -------------------------------------------------------------------------
// Voxel grid parameters (compile-time defaults; overridable via parameters)
// -------------------------------------------------------------------------
__constant__ float c_L_OCC  =  0.85f;
__constant__ float c_L_FREE = -0.40f;
__constant__ float c_L_MIN  = -4.0f;
__constant__ float c_L_MAX  =  4.0f;

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------
__device__ static void atomic_add_clamped(float* addr, float delta) {
    float old = atomicAdd(addr, delta);
    float updated = old + delta;
    if (updated > c_L_MAX) atomicAdd(addr, c_L_MAX - updated);
    else if (updated < c_L_MIN) atomicAdd(addr, c_L_MIN - updated);
}

__global__ void clear_grid_kernel(float* grid, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grid[i] = 0.0f;
}

__global__ void raycast_pointcloud_kernel(
    const float* __restrict__ pts_world,  // (n_pts, 3) world frame
    int n_pts,
    float ox, float oy, float oz,          // sensor origin (world frame)
    float* __restrict__ grid,
    int NX, int NY, int NZ,
    float rx, float ry, float rz,
    float max_range)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_pts) return;
    float hx = pts_world[rid * 3 + 0];
    float hy = pts_world[rid * 3 + 1];
    float hz = pts_world[rid * 3 + 2];
    float dx = hx - ox, dy = hy - oy, dz = hz - oz;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist < 1e-3f) return;
    bool hit = dist < max_range - 1e-3f;
    if (!hit) {
        // truncate at max_range, treat as miss
        float s = max_range / dist;
        hx = ox + dx * s; hy = oy + dy * s; hz = oz + dz * s;
        dist = max_range;
        dx = hx - ox; dy = hy - oy; dz = hz - oz;
    }
    float t_end = dist;
    float ux = dx / dist, uy = dy / dist, uz = dz / dist;

    float fx = ox / rx, fy = oy / ry, fz = oz / rz;
    int gx = static_cast<int>(floorf(fx));
    int gy = static_cast<int>(floorf(fy));
    int gz = static_cast<int>(floorf(fz));
    int step_x = (ux > 0.0f) ? 1 : -1;
    int step_y = (uy > 0.0f) ? 1 : -1;
    int step_z = (uz > 0.0f) ? 1 : -1;
    float inv_dx = (fabsf(ux) > 1e-7f) ? 1.0f / fabsf(ux) : 1e30f;
    float inv_dy = (fabsf(uy) > 1e-7f) ? 1.0f / fabsf(uy) : 1e30f;
    float inv_dz = (fabsf(uz) > 1e-7f) ? 1.0f / fabsf(uz) : 1e30f;
    float t_max_x = (ux > 0.0f) ? (gx + 1 - fx) * rx * inv_dx : (fx - gx) * rx * inv_dx;
    float t_max_y = (uy > 0.0f) ? (gy + 1 - fy) * ry * inv_dy : (fy - gy) * ry * inv_dy;
    float t_max_z = (uz > 0.0f) ? (gz + 1 - fz) * rz * inv_dz : (fz - gz) * rz * inv_dz;
    float dt_x = rx * inv_dx, dt_y = ry * inv_dy, dt_z = rz * inv_dz;

    int max_iter = NX + NY + NZ + 8;
    for (int it = 0; it < max_iter; it++) {
        if (gx >= 0 && gx < NX && gy >= 0 && gy < NY && gz >= 0 && gz < NZ) {
            int idx = (gz * NY + gy) * NX + gx;
            atomic_add_clamped(&grid[idx], c_L_FREE);
        }
        float t_next = fminf(fminf(t_max_x, t_max_y), t_max_z);
        if (t_next >= t_end) break;
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) { gx += step_x; t_max_x += dt_x; }
        else if (t_max_y <= t_max_z) { gy += step_y; t_max_y += dt_y; }
        else { gz += step_z; t_max_z += dt_z; }
        if (gx < 0 || gx >= NX || gy < 0 || gy >= NY || gz < 0 || gz >= NZ) break;
    }
    if (hit) {
        int hgx = static_cast<int>(floorf(hx / rx));
        int hgy = static_cast<int>(floorf(hy / ry));
        int hgz = static_cast<int>(floorf(hz / rz));
        if (hgx >= 0 && hgx < NX && hgy >= 0 && hgy < NY && hgz >= 0 && hgz < NZ) {
            int idx = (hgz * NY + hgy) * NX + hgx;
            atomic_add_clamped(&grid[idx], c_L_OCC - c_L_FREE);
        }
    }
}

__global__ void project_max_z_kernel(const float* __restrict__ grid,
                                     signed char* __restrict__ out_2d,
                                     int NX, int NY, int NZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    float best = -1.0e9f;
    for (int z = 0; z < NZ; z++) {
        float v = grid[(z * NY + y) * NX + x];
        if (v > best) best = v;
    }
    float p = 1.0f / (1.0f + expf(-best));
    int v = static_cast<int>(p * 100.0f);
    if (v < 0) v = 0; if (v > 100) v = 100;
    out_2d[y * NX + x] = static_cast<signed char>(v);
}

// -------------------------------------------------------------------------
// ROS2 Node
// -------------------------------------------------------------------------
class VoxelNode : public rclcpp::Node {
public:
    VoxelNode() : Node("voxel_node") {
        this->declare_parameter("nx", 128);
        this->declare_parameter("ny", 128);
        this->declare_parameter("nz", 16);
        this->declare_parameter("world_x", 20.0);
        this->declare_parameter("world_y", 20.0);
        this->declare_parameter("world_z", 5.0);
        this->declare_parameter("max_range", 20.0);
        this->declare_parameter("origin_x", -10.0);
        this->declare_parameter("origin_y", -10.0);
        this->declare_parameter("origin_z", 0.0);

        NX_ = static_cast<int>(this->get_parameter("nx").as_int());
        NY_ = static_cast<int>(this->get_parameter("ny").as_int());
        NZ_ = static_cast<int>(this->get_parameter("nz").as_int());
        float wx = static_cast<float>(this->get_parameter("world_x").as_double());
        float wy = static_cast<float>(this->get_parameter("world_y").as_double());
        float wz = static_cast<float>(this->get_parameter("world_z").as_double());
        rx_ = wx / NX_;
        ry_ = wy / NY_;
        rz_ = wz / NZ_;
        max_range_ = static_cast<float>(this->get_parameter("max_range").as_double());
        ox_ = static_cast<float>(this->get_parameter("origin_x").as_double());
        oy_ = static_cast<float>(this->get_parameter("origin_y").as_double());
        oz_ = static_cast<float>(this->get_parameter("origin_z").as_double());

        size_t cells = static_cast<size_t>(NX_) * NY_ * NZ_;
        CUDA_CHECK(cudaMalloc(&d_grid_, cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_2d_, static_cast<size_t>(NX_) * NY_ * sizeof(signed char)));
        max_points_ = 200000;
        CUDA_CHECK(cudaMalloc(&d_pts_, max_points_ * 3 * sizeof(float)));

        int threads = 256;
        int blocks = (cells + threads - 1) / threads;
        clear_grid_kernel<<<blocks, threads>>>(d_grid_, cells);
        CUDA_CHECK(cudaDeviceSynchronize());

        sub_points_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points", 10,
            std::bind(&VoxelNode::points_callback, this, std::placeholders::_1));
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&VoxelNode::odom_callback, this, std::placeholders::_1));
        pub_map_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/voxel_map", 10);

        RCLCPP_INFO(this->get_logger(),
                    "voxel_node started: %dx%dx%d (res %.2f x %.2f x %.2f m), "
                    "world origin (%.2f, %.2f, %.2f), max_range=%.1f m",
                    NX_, NY_, NZ_, rx_, ry_, rz_, ox_, oy_, oz_, max_range_);
    }

    ~VoxelNode() override {
        if (d_grid_)   cudaFree(d_grid_);
        if (d_out_2d_) cudaFree(d_out_2d_);
        if (d_pts_)    cudaFree(d_pts_);
    }

private:
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        sx_ = static_cast<float>(msg->pose.pose.position.x);
        sy_ = static_cast<float>(msg->pose.pose.position.y);
        sz_ = static_cast<float>(msg->pose.pose.position.z);
        odom_received_ = true;
    }

    void points_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "Waiting for /odom before processing /points");
            return;
        }
        // Extract x,y,z (assume float32 at offsets 0,4,8 — standard layout)
        size_t row_step = msg->point_step;
        size_t n_total = (msg->width * msg->height);
        if (n_total == 0) return;
        size_t n = std::min(n_total, max_points_);
        std::vector<float> pts(n * 3);
        const uint8_t* data = msg->data.data();
        for (size_t i = 0; i < n; i++) {
            float x, y, z;
            std::memcpy(&x, data + i * row_step + 0, sizeof(float));
            std::memcpy(&y, data + i * row_step + 4, sizeof(float));
            std::memcpy(&z, data + i * row_step + 8, sizeof(float));
            // points are sensor-frame; transform to world by adding sensor origin
            pts[i * 3 + 0] = sx_ + x;
            pts[i * 3 + 1] = sy_ + y;
            pts[i * 3 + 2] = sz_ + z;
        }
        CUDA_CHECK(cudaMemcpy(d_pts_, pts.data(), pts.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        auto t0 = std::chrono::high_resolution_clock::now();
        // Sensor origin in voxel-grid frame (shift by -origin)
        float sx_g = sx_ - ox_;
        float sy_g = sy_ - oy_;
        float sz_g = sz_ - oz_;
        // Transform points to grid frame
        std::vector<float> pts_grid(n * 3);
        for (size_t i = 0; i < n; i++) {
            pts_grid[i * 3 + 0] = pts[i * 3 + 0] - ox_;
            pts_grid[i * 3 + 1] = pts[i * 3 + 1] - oy_;
            pts_grid[i * 3 + 2] = pts[i * 3 + 2] - oz_;
        }
        CUDA_CHECK(cudaMemcpy(d_pts_, pts_grid.data(), pts_grid.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (static_cast<int>(n) + threads - 1) / threads;
        raycast_pointcloud_kernel<<<blocks, threads>>>(
            d_pts_, static_cast<int>(n),
            sx_g, sy_g, sz_g,
            d_grid_, NX_, NY_, NZ_,
            rx_, ry_, rz_, max_range_);

        dim3 blk2d(16, 16);
        dim3 grd2d((NX_ + 15) / 16, (NY_ + 15) / 16);
        project_max_z_kernel<<<grd2d, blk2d>>>(d_grid_, d_out_2d_, NX_, NY_, NZ_);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        nav_msgs::msg::OccupancyGrid out;
        out.header = msg->header;
        out.info.resolution = rx_;
        out.info.width = NX_;
        out.info.height = NY_;
        out.info.origin.position.x = ox_;
        out.info.origin.position.y = oy_;
        out.info.origin.position.z = oz_;
        out.data.resize(static_cast<size_t>(NX_) * NY_);
        CUDA_CHECK(cudaMemcpy(out.data.data(), d_out_2d_,
                              out.data.size(), cudaMemcpyDeviceToHost));
        pub_map_->publish(out);
        RCLCPP_INFO(this->get_logger(),
                    "Published voxel_map: %dx%d (z-max proj of %dx%dx%d), %zu rays, %.3f ms",
                    NX_, NY_, NX_, NY_, NZ_, n, ms);
    }

    // -----------------------------------------------------------
    std::mutex mtx_;
    int NX_ = 128, NY_ = 128, NZ_ = 16;
    float rx_ = 0.156f, ry_ = 0.156f, rz_ = 0.3125f;
    float ox_ = -10.0f, oy_ = -10.0f, oz_ = 0.0f;
    float max_range_ = 20.0f;
    float sx_ = 0.0f, sy_ = 0.0f, sz_ = 0.0f;
    bool odom_received_ = false;

    float* d_grid_   = nullptr;
    signed char* d_out_2d_ = nullptr;
    float* d_pts_    = nullptr;
    size_t max_points_ = 0;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       sub_odom_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr     pub_map_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VoxelNode>());
    rclcpp::shutdown();
    return 0;
}
