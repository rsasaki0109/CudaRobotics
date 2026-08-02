#include <cmath>
#include <cstdio>

#include "cudarobotics/imu_preintegration.hpp"

namespace {

using cudarobotics::imu::ImuBias;
using cudarobotics::imu::ImuFactorLinearization;
using cudarobotics::imu::ImuPreintegrator;
using cudarobotics::imu::NavDelta;
using cudarobotics::imu::NavState;

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) {
        std::printf("  PASS: %s\n", name);
    } else {
        std::printf("  FAIL: %s\n", name);
        ++failures;
    }
}

float max_abs(const float* a, const float* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; ++i) result = fmaxf(result, fabsf(a[i] - b[i]));
    return result;
}

void make_bias(float accel, float gyro, ImuBias* bias) {
    for (int i = 0; i < 3; ++i) {
        bias->accel[i] = accel;
        bias->gyro[i] = gyro;
    }
}

void perturb_state(const NavState& state, const float* dx, NavState* result) {
    NavDelta delta;
    cudarobotics::imu::tangent_to_delta(dx, &delta);
    cudarobotics::imu::nav_state_retract(state, delta, result);
}

void residual_only(const ImuPreintegrator& pre,
                   const NavState& state_i,
                   const NavState& state_j,
                   const ImuBias& bias,
                   NavDelta* residual) {
    NavState predicted;
    pre.predict(state_i, bias, &predicted);
    cudarobotics::imu::nav_state_local(state_j, predicted, residual);
}

void numerical_jacobian_state_i(const ImuPreintegrator& pre,
                                const NavState& state_i,
                                const NavState& state_j,
                                const ImuBias& bias,
                                float* J) {
    const float eps = 1.0e-2f;
    NavDelta base;
    residual_only(pre, state_i, state_j, bias, &base);
    for (int c = 0; c < 9; ++c) {
        float dx[9] = {0.0f};
        dx[c] = eps;
        NavState perturbed;
        perturb_state(state_i, dx, &perturbed);
        NavDelta next;
        residual_only(pre, perturbed, state_j, bias, &next);
        NavDelta local;
        cudarobotics::imu::delta_local(base, next, &local);
        float y[9];
        cudarobotics::imu::delta_to_tangent(local, y);
        for (int r = 0; r < 9; ++r) J[9 * r + c] = y[r] / eps;
    }
}

void numerical_jacobian_state_j(const ImuPreintegrator& pre,
                                const NavState& state_i,
                                const NavState& state_j,
                                const ImuBias& bias,
                                float* J) {
    const float eps = 1.0e-2f;
    NavDelta base;
    residual_only(pre, state_i, state_j, bias, &base);
    for (int c = 0; c < 9; ++c) {
        float dx[9] = {0.0f};
        dx[c] = eps;
        NavState perturbed;
        perturb_state(state_j, dx, &perturbed);
        NavDelta next;
        residual_only(pre, state_i, perturbed, bias, &next);
        NavDelta local;
        cudarobotics::imu::delta_local(base, next, &local);
        float y[9];
        cudarobotics::imu::delta_to_tangent(local, y);
        for (int r = 0; r < 9; ++r) J[9 * r + c] = y[r] / eps;
    }
}

void numerical_jacobian_bias(const ImuPreintegrator& pre,
                             const NavState& state_i,
                             const NavState& state_j,
                             const ImuBias& bias,
                             float* J) {
    const float eps = 1.0e-2f;
    NavDelta base;
    residual_only(pre, state_i, state_j, bias, &base);
    for (int c = 0; c < 6; ++c) {
        ImuBias perturbed = bias;
        if (c < 3) perturbed.accel[c] += eps;
        else perturbed.gyro[c - 3] += eps;
        NavDelta next;
        residual_only(pre, state_i, state_j, perturbed, &next);
        NavDelta local;
        cudarobotics::imu::delta_local(base, next, &local);
        float y[9];
        cudarobotics::imu::delta_to_tangent(local, y);
        for (int r = 0; r < 9; ++r) J[6 * r + c] = y[r] / eps;
    }
}

void test_measurement_update() {
    std::printf("[test_measurement_update]\n");
    ImuBias bias;
    make_bias(0.0f, 0.0f, &bias);
    ImuPreintegrator pre;
    pre.reset(9.81f, bias);
    const float acc[3] = {1.0f, 0.0f, 0.0f};
    const float gyro[3] = {0.0f, 0.0f, 0.0f};
    pre.update(acc, gyro, 0.1f);
    check(fabsf(pre.delta.p[0] - 0.005f) < 1.0e-6f,
          "constant acceleration position");
    check(fabsf(pre.delta.v[0] - 0.1f) < 1.0e-6f,
          "constant acceleration velocity");
    check(fabsf(pre.delta.p[1]) < 1.0e-6f && fabsf(pre.delta.p[2]) < 1.0e-6f,
          "constant acceleration lateral position");
    check(fabsf(pre.total_dt - 0.1f) < 1.0e-6f,
          "preintegration time accumulation");
}

void test_rotation_update() {
    std::printf("[test_rotation_update]\n");
    ImuBias bias;
    make_bias(0.0f, 0.0f, &bias);
    ImuPreintegrator pre;
    pre.reset(9.81f, bias);
    const float acc[3] = {0.0f, 0.0f, 0.0f};
    const float gyro[3] = {0.0f, 0.0f, 1.0f};
    pre.update(acc, gyro, 0.2f);
    float expected[9];
    const float angle[3] = {0.0f, 0.0f, 0.2f};
    cudarobotics::lie::so3_exp(angle, expected);
    check(max_abs(pre.delta.R, expected, 9) < 2.0e-6f,
          "constant angular velocity rotation");
}

void test_factor_jacobians() {
    std::printf("[test_factor_jacobians]\n");
    ImuBias linearization_bias;
    make_bias(0.02f, -0.01f, &linearization_bias);
    ImuPreintegrator pre;
    pre.reset(9.81f, linearization_bias);
    const float acc0[3] = {0.5f, -0.2f, 9.6f};
    const float gyro0[3] = {0.03f, -0.04f, 0.02f};
    const float acc1[3] = {0.6f, -0.1f, 9.7f};
    const float gyro1[3] = {0.04f, -0.02f, 0.03f};
    pre.update(acc0, gyro0, 0.01f);
    pre.update(acc1, gyro1, 0.015f);

    NavState state_i;
    cudarobotics::imu::identity_state(&state_i);
    const float state_i_rot[3] = {0.08f, -0.03f, 0.04f};
    cudarobotics::lie::so3_exp(state_i_rot, state_i.R);
    state_i.p[0] = 0.3f; state_i.p[1] = -0.2f; state_i.p[2] = 0.5f;
    state_i.v[0] = 0.4f; state_i.v[1] = 0.1f; state_i.v[2] = -0.2f;

    NavState state_j;
    ImuBias query_bias = linearization_bias;
    query_bias.accel[0] += 0.003f;
    query_bias.gyro[1] -= 0.002f;
    pre.predict(state_i, query_bias, &state_j);
    state_j.p[0] += 0.002f;
    state_j.v[1] -= 0.001f;

    ImuFactorLinearization analytic;
    cudarobotics::imu::linearize_imu_factor(
        pre, state_i, state_j, query_bias, &analytic);
    float numeric_i[81];
    float numeric_j[81];
    float numeric_b[54];
    numerical_jacobian_state_i(pre, state_i, state_j, query_bias, numeric_i);
    numerical_jacobian_state_j(pre, state_i, state_j, query_bias, numeric_j);
    numerical_jacobian_bias(pre, state_i, state_j, query_bias, numeric_b);
    const float error_i = max_abs(analytic.J_state_i, numeric_i, 81);
    const float error_j = max_abs(analytic.J_state_j, numeric_j, 81);
    const float error_b = max_abs(analytic.J_bias_i, numeric_b, 54);
    check(error_i < 3.0e-3f,
          "IMU factor state-i Jacobian");
    check(error_j < 3.0e-3f,
          "IMU factor state-j Jacobian");
    check(error_b < 3.0e-3f,
          "IMU factor bias Jacobian");
}

}  // namespace

int main() {
    std::printf("=== test_imu_preintegration ===\n");
    test_measurement_update();
    test_rotation_update();
    test_factor_jacobians();
    std::printf("===============================\n");
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
