// Kinematics helpers ported from MathematicalRobotics' mathR/kinematics.
// All transforms are row-major and map a source-frame vector into a
// destination frame.  The routines are fixed-size so they are callable from
// host C++ and CUDA kernels.

#pragma once

#include "cudarobotics/lie_group_math.cuh"

#if defined(__CUDACC__)
#define CUDAROBOTICS_KIN_HD __host__ __device__
#else
#define CUDAROBOTICS_KIN_HD
#endif

namespace cudarobotics {
namespace kinematics {

CUDAROBOTICS_KIN_HD static inline void transform_velocity_3d(
    const float* T_ba, const float* velocity_a, float* velocity_b) {
    float R[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) R[3 * row + col] = T_ba[4 * row + col];
    }
    const float t[3] = {T_ba[3], T_ba[7], T_ba[11]};
    float rotated_v[3];
    float rotated_omega[3];
    float cross_matrix[9];
    float cross_term[3];
    lie::mat3_vec(R, velocity_a, rotated_v);
    lie::mat3_vec(R, velocity_a + 3, rotated_omega);
    lie::skew(t, cross_matrix);
    lie::mat3_vec(cross_matrix, rotated_omega, cross_term);
    for (int i = 0; i < 3; ++i) {
        velocity_b[i] = rotated_v[i] + cross_term[i];
        velocity_b[3 + i] = rotated_omega[i];
    }
}

CUDAROBOTICS_KIN_HD static inline void transform_velocity_2d(
    const float* T_ba, const float* velocity_a, float* velocity_b) {
    const float c = T_ba[0];
    const float minus_s = T_ba[1];
    const float s = T_ba[3];
    const float tx = T_ba[2];
    const float ty = T_ba[5];
    const float vx = velocity_a[0];
    const float vy = velocity_a[1];
    const float omega = velocity_a[2];
    velocity_b[0] = c * vx + minus_s * vy + ty * omega;
    velocity_b[1] = s * vx + c * vy - tx * omega;
    velocity_b[2] = omega;
}

CUDAROBOTICS_KIN_HD static inline void transform_velocity_split(
    const float* T_ba, const float* velocity_a, const float* omega_a,
    float* velocity_b) {
    float R[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) R[3 * row + col] = T_ba[4 * row + col];
    }
    const float t[3] = {T_ba[3], T_ba[7], T_ba[11]};
    float rv[3];
    float rw[3];
    float skew_t[9];
    float cross[3];
    lie::mat3_vec(R, velocity_a, rv);
    lie::mat3_vec(R, omega_a, rw);
    lie::skew(t, skew_t);
    lie::mat3_vec(skew_t, rw, cross);
    for (int i = 0; i < 3; ++i) {
        velocity_b[i] = rv[i] + cross[i];
        velocity_b[3 + i] = rw[i];
    }
}

struct ImuInput {
    float acceleration[3] = {0.0f, 0.0f, 0.0f};
    float angular_velocity[3] = {0.0f, 0.0f, 0.0f};
};

CUDAROBOTICS_KIN_HD static inline void transform_imu(
    const float* T_ba, const ImuInput& imu_a, const float* angular_acceleration,
    ImuInput* imu_b) {
    float R[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) R[3 * row + col] = T_ba[4 * row + col];
    }
    const float t[3] = {T_ba[3], T_ba[7], T_ba[11]};
    float omega_b[3];
    float alpha_b[3];
    float R_omega[3];
    float skew_omega[9];
    float skew_t[9];
    float centripetal[3];
    float angular_term[3];
    float rotated_accel[3];
    lie::mat3_vec(R, imu_a.angular_velocity, omega_b);
    lie::mat3_vec(R, angular_acceleration, alpha_b);
    lie::mat3_vec(R, imu_a.acceleration, rotated_accel);
    lie::skew(omega_b, skew_omega);
    lie::mat3_vec(skew_omega, t, R_omega);
    lie::mat3_vec(skew_omega, R_omega, centripetal);
    lie::skew(t, skew_t);
    lie::mat3_vec(skew_t, alpha_b, angular_term);
    for (int i = 0; i < 3; ++i) {
        imu_b->angular_velocity[i] = omega_b[i];
        imu_b->acceleration[i] = rotated_accel[i] - centripetal[i] + angular_term[i];
    }
}

struct ImuKinematicState {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float rotation[9] = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f};
    float velocity[3] = {0.0f, 0.0f, 0.0f};
    float angular_velocity[3] = {0.0f, 0.0f, 0.0f};
};

CUDAROBOTICS_KIN_HD static inline void imu_input_kinematic_model(
    const ImuInput& input,
    const ImuKinematicState& state,
    float dt,
    float* delta) {
    if (delta == nullptr) return;
    for (int i = 0; i < 3; ++i) {
        delta[i] = state.velocity[i] * dt;
        delta[6 + i] = state.rotation[3 * i + 0] * input.acceleration[0] * dt +
                       state.rotation[3 * i + 1] * input.acceleration[1] * dt +
                       state.rotation[3 * i + 2] * input.acceleration[2] * dt;
        delta[9 + i] = input.angular_velocity[i] - state.angular_velocity[i];
    }
    for (int i = 0; i < 3; ++i) delta[3 + i] = input.angular_velocity[i] * dt;
}

CUDAROBOTICS_KIN_HD static inline void imu_state_from_vector(
    const float* vector, ImuKinematicState* state) {
    if (vector == nullptr || state == nullptr) return;
    for (int i = 0; i < 3; ++i) {
        state->position[i] = vector[i];
        state->velocity[i] = vector[6 + i];
        state->angular_velocity[i] = vector[9 + i];
    }
    lie::so3_exp(vector + 3, state->rotation);
}

CUDAROBOTICS_KIN_HD static inline void imu_state_pose_matrix(
    const ImuKinematicState& state, float* T) {
    if (T == nullptr) return;
    lie::mat4_identity(T);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) T[4 * row + col] = state.rotation[3 * row + col];
        T[4 * row + 3] = state.position[row];
    }
}

CUDAROBOTICS_KIN_HD static inline void imu_state_retract(
    const ImuKinematicState& state, const float* delta,
    ImuKinematicState* result) {
    for (int i = 0; i < 3; ++i) {
        result->position[i] = state.position[i] + delta[i];
        result->velocity[i] = state.velocity[i] + delta[6 + i];
        result->angular_velocity[i] = state.angular_velocity[i] + delta[9 + i];
    }
    float dR[9];
    lie::so3_exp(delta + 3, dR);
    lie::mat3_mul(state.rotation, dR, result->rotation);
}

}  // namespace kinematics
}  // namespace cudarobotics

#undef CUDAROBOTICS_KIN_HD
