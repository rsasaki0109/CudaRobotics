#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

namespace cudabot {

struct PointXYZ { float x, y, z; };
struct PointNormal { float x, y, z, nx, ny, nz; };

class CudaPointCloud {
public:
    CudaPointCloud() : d_x_(nullptr), d_y_(nullptr), d_z_(nullptr), n_(0), capacity_(0) {}

    ~CudaPointCloud() {
        free();
    }

    CudaPointCloud(const CudaPointCloud& other) : d_x_(nullptr), d_y_(nullptr), d_z_(nullptr), n_(0), capacity_(0) {
        if (other.n_ > 0) {
            reserve(other.n_);
            n_ = other.n_;
            CUDA_CHECK(cudaMemcpy(d_x_, other.d_x_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_y_, other.d_y_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_z_, other.d_z_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }

    CudaPointCloud& operator=(const CudaPointCloud& other) {
        if (this != &other) {
            if (other.n_ > 0) {
                reserve(other.n_);
                n_ = other.n_;
                CUDA_CHECK(cudaMemcpy(d_x_, other.d_x_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_y_, other.d_y_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_z_, other.d_z_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
            } else {
                n_ = 0;
            }
        }
        return *this;
    }

    void reserve(int cap) {
        if (cap <= capacity_) return;
        float *new_x, *new_y, *new_z;
        CUDA_CHECK(cudaMalloc(&new_x, cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_y, cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&new_z, cap * sizeof(float)));
        if (n_ > 0 && d_x_) {
            CUDA_CHECK(cudaMemcpy(new_x, d_x_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(new_y, d_y_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(new_z, d_z_, n_ * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        free();
        d_x_ = new_x; d_y_ = new_y; d_z_ = new_z;
        capacity_ = cap;
    }

    void upload(const std::vector<PointXYZ>& points) {
        int n = (int)points.size();
        reserve(n);
        n_ = n;
        std::vector<float> hx(n), hy(n), hz(n);
        for (int i = 0; i < n; i++) {
            hx[i] = points[i].x;
            hy[i] = points[i].y;
            hz[i] = points[i].z;
        }
        CUDA_CHECK(cudaMemcpy(d_x_, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z_, hz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void upload(const float* hx, const float* hy, const float* hz, int n) {
        reserve(n);
        n_ = n;
        CUDA_CHECK(cudaMemcpy(d_x_, hx, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_, hy, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z_, hz, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void download(std::vector<PointXYZ>& points) const {
        points.resize(n_);
        std::vector<float> hx(n_), hy(n_), hz(n_);
        CUDA_CHECK(cudaMemcpy(hx.data(), d_x_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), d_y_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hz.data(), d_z_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n_; i++) {
            points[i].x = hx[i]; points[i].y = hy[i]; points[i].z = hz[i];
        }
    }

    void setSize(int n) { n_ = n; }
    int size() const { return n_; }
    int capacity() const { return capacity_; }
    float* d_x() { return d_x_; }
    float* d_y() { return d_y_; }
    float* d_z() { return d_z_; }
    const float* d_x() const { return d_x_; }
    const float* d_y() const { return d_y_; }
    const float* d_z() const { return d_z_; }

private:
    void free() {
        if (d_x_) { cudaFree(d_x_); d_x_ = nullptr; }
        if (d_y_) { cudaFree(d_y_); d_y_ = nullptr; }
        if (d_z_) { cudaFree(d_z_); d_z_ = nullptr; }
        capacity_ = 0;
    }

    float *d_x_, *d_y_, *d_z_;
    int n_, capacity_;
};

// Voxel Grid Filter
CudaPointCloud voxel_grid_filter(const CudaPointCloud& input, float leaf_size);

// Statistical Outlier Removal
CudaPointCloud statistical_outlier_removal(const CudaPointCloud& input, int k = 20, float std_mul = 1.0f);

// Normal Estimation
void estimate_normals(const CudaPointCloud& input, float* d_nx, float* d_ny, float* d_nz, int k = 20);

// GICP
void gicp_align(const CudaPointCloud& source, const CudaPointCloud& target,
                float* R, float* t, int max_iter = 50, float tolerance = 1e-4f);

// RANSAC plane detection
void ransac_plane(const CudaPointCloud& input, float* plane_coeffs,
                  float distance_threshold = 0.01f, int max_iterations = 1000);

} // namespace cudabot
