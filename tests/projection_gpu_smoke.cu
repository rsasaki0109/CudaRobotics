#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "cudarobotics/lie_group_math.cuh"
#include "cudarobotics/projection.hpp"

namespace {

__global__ void projection_kernel(float* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float x[6] = {0.1f, -0.2f, 0.4f, 0.03f, -0.02f, 0.1f};
    const float point[3] = {0.5f, -0.3f, 4.0f};
    const float K[9] = {400.0f, 0.0f, 200.0f,
                        0.0f, 420.0f, 100.0f,
                        0.0f, 0.0f, 1.0f};
    float T[16];
    float camera_point[3];
    float measurement[2];
    float residual[2];
    float pose_jacobian[12];
    float point_jacobian[6];
    cudarobotics::lie::se3_exp(x, T);
    cudarobotics::projection::transform_inverse(T, false, point, camera_point);
    cudarobotics::projection::reproject(camera_point, K, measurement);
    cudarobotics::projection::reprojection_error(
        T, point, measurement, K, residual, pose_jacobian, point_jacobian);
    output[0] = residual[0];
    output[1] = residual[1];
    output[2] = camera_point[0];
    output[3] = camera_point[1];
    output[4] = camera_point[2];
    for (int i = 0; i < 12; ++i) output[5 + i] = pose_jacobian[i];
}

}  // namespace

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || device_count == 0) {
        std::printf("CUDA device unavailable; skipping projection GPU smoke.\n");
        return 77;
    }
    CUDA_CHECK(status);
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), 17 * sizeof(float)));
    projection_kernel<<<1, 1>>>(device_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float device_values[17];
    CUDA_CHECK(cudaMemcpy(device_values, device_output, sizeof(device_values), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_output));

    const float x[6] = {0.1f, -0.2f, 0.4f, 0.03f, -0.02f, 0.1f};
    const float point[3] = {0.5f, -0.3f, 4.0f};
    const float K[9] = {400.0f, 0.0f, 200.0f,
                        0.0f, 420.0f, 100.0f,
                        0.0f, 0.0f, 1.0f};
    float T[16];
    float camera_point[3];
    float measurement[2];
    float residual[2];
    float pose_jacobian[12];
    float point_jacobian[6];
    cudarobotics::lie::se3_exp(x, T);
    cudarobotics::projection::transform_inverse(T, false, point, camera_point);
    cudarobotics::projection::reproject(camera_point, K, measurement);
    cudarobotics::projection::reprojection_error(
        T, point, measurement, K, residual, pose_jacobian, point_jacobian);
    float max_error = 0.0f;
    const float expected[5] = {residual[0], residual[1], camera_point[0], camera_point[1], camera_point[2]};
    for (int i = 0; i < 5; ++i) max_error = fmaxf(max_error, fabsf(expected[i] - device_values[i]));
    for (int i = 0; i < 12; ++i) max_error = fmaxf(max_error, fabsf(pose_jacobian[i] - device_values[5 + i]));
    std::printf("CUDA projection smoke max error: %.6g\n", max_error);
    if (max_error > 2.0e-5f) return 1;
    std::printf("CUDA projection smoke: PASS\n");
    return 0;
}
