#include <cmath>
#include <cstdio>

#include "cuda_check.cuh"
#include "cudarobotics/imu_graph.hpp"
#include "cudarobotics/imu_preintegration.hpp"

namespace {

using cudarobotics::imu::ImuBias;
using cudarobotics::imu::ImuFactorLinearization;
using cudarobotics::imu::ImuFactor15Linearization;
using cudarobotics::imu::ImuPreintegrator;
using cudarobotics::imu::NavState;

__global__ void build_and_linearize_kernel(
    ImuPreintegrator* pre,
    const NavState* state_i,
    const NavState* state_j,
    const ImuBias* query_bias,
    ImuBias linearization_bias,
    ImuFactorLinearization* output,
    ImuFactor15Linearization* output15) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const float acc0[3] = {0.5f, -0.2f, 9.6f};
    const float gyro0[3] = {0.03f, -0.04f, 0.02f};
    const float acc1[3] = {0.6f, -0.1f, 9.7f};
    const float gyro1[3] = {0.04f, -0.02f, 0.03f};
    pre->reset(9.81f, linearization_bias);
    pre->update(acc0, gyro0, 0.01f);
    pre->update(acc1, gyro1, 0.015f);
    cudarobotics::imu::linearize_imu_factor(
        *pre, *state_i, *state_j, *query_bias, output);
    cudarobotics::imu::linearize_imu_factor_15(
        *pre, *state_i, *state_j, *query_bias, output15);
}

void make_fixture(ImuPreintegrator* pre,
                  NavState* state_i,
                  NavState* state_j,
                  ImuBias* linearization_bias,
                  ImuBias* query_bias) {
    for (int i = 0; i < 3; ++i) {
        linearization_bias->accel[i] = 0.02f;
        linearization_bias->gyro[i] = -0.01f;
    }
    *query_bias = *linearization_bias;
    query_bias->accel[0] += 0.003f;
    query_bias->gyro[1] -= 0.002f;

    pre->reset(9.81f, *linearization_bias);
    const float acc0[3] = {0.5f, -0.2f, 9.6f};
    const float gyro0[3] = {0.03f, -0.04f, 0.02f};
    const float acc1[3] = {0.6f, -0.1f, 9.7f};
    const float gyro1[3] = {0.04f, -0.02f, 0.03f};
    pre->update(acc0, gyro0, 0.01f);
    pre->update(acc1, gyro1, 0.015f);

    cudarobotics::imu::identity_state(state_i);
    const float state_i_rot[3] = {0.08f, -0.03f, 0.04f};
    cudarobotics::lie::so3_exp(state_i_rot, state_i->R);
    state_i->p[0] = 0.3f;
    state_i->p[1] = -0.2f;
    state_i->p[2] = 0.5f;
    state_i->v[0] = 0.4f;
    state_i->v[1] = 0.1f;
    state_i->v[2] = -0.2f;

    pre->predict(*state_i, *query_bias, state_j);
    state_j->p[0] += 0.002f;
    state_j->v[1] -= 0.001f;
}

float max_abs(const float* a, const float* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; ++i) {
        result = fmaxf(result, fabsf(a[i] - b[i]));
    }
    return result;
}

float max3(float a, float b, float c) {
    return fmaxf(a, fmaxf(b, c));
}

}  // namespace

int main() {
    int device_count = 0;
    cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice || device_count == 0) {
        std::printf("CUDA device unavailable; skipping IMU GPU smoke.\n");
        return 77;
    }
    CUDA_CHECK(device_status);

    ImuPreintegrator host_pre;
    NavState state_i;
    NavState state_j;
    ImuBias linearization_bias;
    ImuBias query_bias;
    make_fixture(&host_pre, &state_i, &state_j,
                 &linearization_bias, &query_bias);

    ImuFactorLinearization host_output;
    cudarobotics::imu::linearize_imu_factor(
        host_pre, state_i, state_j, query_bias, &host_output);

    ImuPreintegrator* device_pre = nullptr;
    NavState* device_state_i = nullptr;
    NavState* device_state_j = nullptr;
    ImuBias* device_query_bias = nullptr;
    ImuFactorLinearization* device_output = nullptr;
    ImuFactor15Linearization* device_output15 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_pre),
                          sizeof(ImuPreintegrator)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_state_i),
                          sizeof(NavState)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_state_j),
                          sizeof(NavState)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_query_bias),
                          sizeof(ImuBias)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output),
                          sizeof(ImuFactorLinearization)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output15),
                          sizeof(ImuFactor15Linearization)));
    CUDA_CHECK(cudaMemcpy(device_state_i, &state_i, sizeof(NavState),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_state_j, &state_j, sizeof(NavState),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_query_bias, &query_bias, sizeof(ImuBias),
                          cudaMemcpyHostToDevice));

    build_and_linearize_kernel<<<1, 1>>>(
        device_pre, device_state_i, device_state_j, device_query_bias,
        linearization_bias, device_output, device_output15);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    ImuPreintegrator device_pre_host;
    ImuFactorLinearization device_output_host;
    ImuFactor15Linearization device_output15_host;
    CUDA_CHECK(cudaMemcpy(&device_pre_host, device_pre,
                          sizeof(ImuPreintegrator), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&device_output_host, device_output,
                          sizeof(ImuFactorLinearization),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&device_output15_host, device_output15,
                          sizeof(ImuFactor15Linearization),
                          cudaMemcpyDeviceToHost));

    const float delta_error = max3(
        max_abs(host_pre.delta.R, device_pre_host.delta.R, 9),
        max_abs(host_pre.delta.p, device_pre_host.delta.p, 3),
        max_abs(host_pre.delta.v, device_pre_host.delta.v, 3));
    const float residual_error =
        max_abs(host_output.residual, device_output_host.residual, 9);
    const float state_i_error =
        max_abs(host_output.J_state_i, device_output_host.J_state_i, 81);
    const float state_j_error =
        max_abs(host_output.J_state_j, device_output_host.J_state_j, 81);
    const float bias_error =
        max_abs(host_output.J_bias_i, device_output_host.J_bias_i, 54);
    ImuFactor15Linearization host_output15;
    cudarobotics::imu::linearize_imu_factor_15(
        host_pre, state_i, state_j, query_bias, &host_output15);
    const float block_residual_error =
        max_abs(host_output15.residual, device_output15_host.residual, 9);
    const float block_from_error =
        max_abs(host_output15.J_from, device_output15_host.J_from, 135);
    const float block_to_error =
        max_abs(host_output15.J_to, device_output15_host.J_to, 135);

    CUDA_CHECK(cudaFree(device_pre));
    CUDA_CHECK(cudaFree(device_state_i));
    CUDA_CHECK(cudaFree(device_state_j));
    CUDA_CHECK(cudaFree(device_query_bias));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_output15));

    std::printf("IMU GPU smoke errors: delta=%.6g residual=%.6g state_i=%.6g "
                "state_j=%.6g bias=%.6g block_residual=%.6g "
                "block_from=%.6g block_to=%.6g\n",
                delta_error, residual_error, state_i_error,
                state_j_error, bias_error, block_residual_error,
                block_from_error, block_to_error);
    const float tolerance = 2.0e-5f;
    if (delta_error > tolerance || residual_error > tolerance ||
        state_i_error > tolerance || state_j_error > tolerance ||
        bias_error > tolerance || block_residual_error > tolerance ||
        block_from_error > tolerance || block_to_error > tolerance) {
        std::fprintf(stderr, "host/device IMU linearization mismatch\n");
        return 1;
    }
    std::printf("CUDA IMU preintegration/factor smoke: PASS\n");
    return 0;
}
