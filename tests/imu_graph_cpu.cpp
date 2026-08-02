#include <cmath>
#include <cstdio>

#include "cudarobotics/imu_graph.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) std::printf("  PASS: %s\n", name);
    else {
        std::printf("  FAIL: %s\n", name);
        ++failures;
    }
}

void make_identity(cudarobotics::imu::NavState* state) {
    cudarobotics::imu::identity_state(state);
}

float max_abs(const float* values, int count) {
    float result = 0.0f;
    for (int i = 0; i < count; ++i) result = fmaxf(result, fabsf(values[i]));
    return result;
}

void test_15dof_graph() {
    std::printf("[test_imu_factor_graph15]\n");
    cudarobotics::imu::ImuPreintegrator preintegrator;
    const float gravity[3] = {0.0f, 0.0f, 0.0f};
    const float identity[9] = {1.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f,
                               0.0f, 0.0f, 1.0f};
    const float lever[3] = {0.0f, 0.0f, 0.0f};
    cudarobotics::imu::ImuBias zero_bias;
    cudarobotics::imu::identity_bias(&zero_bias);
    preintegrator.reset_with_calibration(gravity, zero_bias, identity, lever);
    const float acceleration[3] = {0.0f, 0.0f, 0.0f};
    const float gyro[3] = {0.0f, 0.0f, 0.0f};
    preintegrator.update(acceleration, gyro, 0.1f);

    cudarobotics::imu::NavState truth_state;
    make_identity(&truth_state);
    cudarobotics::imu::NavStateBias truth;
    truth.state = truth_state;
    truth.bias = zero_bias;
    cudarobotics::imu::NavStateBias initial = truth;
    const float perturbation[9] = {0.12f, -0.08f, 0.05f,
                                   0.25f, -0.10f, 0.12f,
                                   -0.18f, 0.06f, 0.10f};
    cudarobotics::imu::NavDelta perturbation_delta;
    cudarobotics::imu::tangent_to_delta(perturbation, &perturbation_delta);
    cudarobotics::imu::nav_state_retract(truth_state, perturbation_delta,
                                         &initial.state);

    cudarobotics::imu::ImuFactorGraph15 graph;
    graph.add_vertex(truth, true);
    graph.add_vertex(initial, false);
    cudarobotics::imu::ImuGraphEdge edge;
    edge.from = 0;
    edge.to = 1;
    edge.preintegrator = preintegrator;
    for (int i = 0; i < 9; ++i) edge.information[9 * i + i] = 100.0f;
    graph.add_edge(edge);
    const float initial_score = graph.score();
    cudarobotics::imu::ImuGraphOptions options;
    options.max_iterations = 12;
    options.min_iterations = 2;
    options.damping = 1.0e-5f;
    options.max_step = 0.5f;
    const cudarobotics::imu::ImuGraphSummary summary = graph.solve(options);
    check(summary.finite && summary.iterations > 0, "15-DoF graph returns finite summary");
    check(graph.score() < initial_score * 1.0e-4f,
          "IMU factor graph reduces NavState residual");
    float recovered[9];
    cudarobotics::imu::NavDelta error;
    cudarobotics::imu::nav_state_local(truth.state, graph.vertices()[1].state, &error);
    cudarobotics::imu::delta_to_tangent(error, recovered);
    float max_error = 0.0f;
    for (float value : recovered) max_error = fmaxf(max_error, fabsf(value));
    check(max_error < 2.0e-3f, "15-DoF graph recovers NavState orientation/position/velocity");
}

void test_auxiliary_factors() {
    std::printf("[test_imu_auxiliary_factors]\n");
    cudarobotics::imu::ImuBias bias{{0.1f, -0.2f, 0.3f}, {0.4f, -0.5f, 0.6f}};
    cudarobotics::imu::ImuBias target{{0.0f, -0.1f, 0.2f}, {0.3f, -0.4f, 0.5f}};
    cudarobotics::imu::BiasFactorLinearization bias_prior;
    cudarobotics::imu::linearize_bias_prior(bias, target, &bias_prior);
    check(std::fabs(bias_prior.residual[0] - 0.1f) < 1.0e-6f &&
              std::fabs(bias_prior.residual[5] - 0.1f) < 1.0e-6f &&
              bias_prior.J_from[0] == 1.0f && bias_prior.J_to[0] == 0.0f,
          "bias prior factor matches residual/Jacobian contract");
    cudarobotics::imu::BiasFactorLinearization bias_change;
    cudarobotics::imu::linearize_bias_change(bias, target, &bias_change);
    check(bias_change.J_from[0] == 1.0f && bias_change.J_to[0] == -1.0f,
          "bias change factor has opposite endpoint Jacobians");

    cudarobotics::imu::NavState state_i;
    cudarobotics::imu::NavState state_j;
    make_identity(&state_i);
    make_identity(&state_j);
    state_i.p[0] = 1.0f;
    state_i.v[0] = 2.0f;
    state_j.p[0] = 3.0f;
    state_j.v[0] = 2.0f;
    cudarobotics::imu::NavStateFactorLinearization nav_prior;
    cudarobotics::imu::linearize_nav_state_prior(state_i, state_i, &nav_prior);
    check(max_abs(nav_prior.residual, 9) < 1.0e-6f,
          "navigation-state prior is zero at its measurement");
    cudarobotics::imu::PositionVelocityFactorLinearization posvel;
    check(cudarobotics::imu::linearize_position_velocity(state_i, state_j, 1.0f, &posvel) &&
              max_abs(posvel.residual, 3) < 1.0e-6f,
          "position-velocity factor matches constant-velocity motion");
    cudarobotics::imu::NavDelta measurement;
    cudarobotics::imu::nav_state_local(state_i, state_j, &measurement);
    cudarobotics::imu::NavTransitionFactorLinearization transition;
    cudarobotics::imu::linearize_nav_transition(state_i, state_j, measurement, &transition);
    check(max_abs(transition.residual, 9) < 1.0e-6f,
          "navigation transition factor is zero at its measurement");
    float xytheta[3];
    cudarobotics::imu::nav_state_to_2d(state_j, xytheta);
    check(std::fabs(xytheta[0] - 3.0f) < 1.0e-6f &&
              std::fabs(xytheta[1]) < 1.0e-6f && std::fabs(xytheta[2]) < 1.0e-6f,
          "navigation state projects to planar pose");
}

}  // namespace

int main() {
    std::printf("=== test_imu_graph ===\n");
    test_15dof_graph();
    test_auxiliary_factors();
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
