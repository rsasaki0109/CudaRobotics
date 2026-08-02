#include <array>
#include <cmath>
#include <cstdio>

#include "cudarobotics/graph_optimization.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) {
        std::printf("  PASS: %s\n", name);
    } else {
        std::printf("  FAIL: %s\n", name);
        ++failures;
    }
}

std::array<float, 16> exp_pose(const float* x) {
    std::array<float, 16> T{};
    cudarobotics::lie::se3_exp(x, T.data());
    return T;
}

std::array<float, 16> relative(const std::array<float, 16>& a,
                               const std::array<float, 16>& b) {
    std::array<float, 16> inverse{};
    std::array<float, 16> result{};
    cudarobotics::graph::rigid_inverse(a.data(), inverse.data());
    cudarobotics::graph::mat4_multiply(inverse.data(), b.data(), result.data());
    return result;
}

void test_pose_graph() {
    std::printf("[test_pose_graph6]\n");
    const float x1[6] = {1.0f, 0.1f, -0.2f, 0.03f, -0.04f, 0.20f};
    const float x2[6] = {2.0f, 0.4f, -0.1f, 0.05f, -0.02f, 0.38f};
    const float d1[6] = {0.15f, -0.08f, 0.04f, 0.02f, -0.01f, 0.03f};
    const float d2[6] = {-0.12f, 0.06f, -0.03f, -0.01f, 0.02f, -0.025f};
    const float zero[6] = {0, 0, 0, 0, 0, 0};
    const std::array<float, 16> identity = exp_pose(zero);
    const std::array<float, 16> truth1 = exp_pose(x1);
    const std::array<float, 16> truth2 = exp_pose(x2);
    const std::array<float, 16> delta1 = exp_pose(d1);
    const std::array<float, 16> delta2 = exp_pose(d2);
    std::array<float, 16> initial1{};
    std::array<float, 16> initial2{};
    cudarobotics::graph::mat4_multiply(truth1.data(), delta1.data(), initial1.data());
    cudarobotics::graph::mat4_multiply(truth2.data(), delta2.data(), initial2.data());

    cudarobotics::graph::PoseGraph6 graph;
    graph.add_vertex(identity, true, 0);
    graph.add_vertex(initial1, false, 1);
    graph.add_vertex(initial2, false, 2);
    cudarobotics::graph::PoseEdge e01;
    e01.from = 0;
    e01.to = 1;
    e01.measurement = relative(identity, truth1);
    e01.information.fill(0.0f);
    for (int i = 0; i < 6; ++i) e01.information[6 * i + i] = 1.0f;
    cudarobotics::graph::PoseEdge e12 = e01;
    e12.from = 1;
    e12.to = 2;
    e12.measurement = relative(truth1, truth2);
    graph.add_edge(e01);
    graph.add_edge(e12);

    const float initial_score = graph.score();
    cudarobotics::graph::SolverOptions options;
    options.max_iterations = 15;
    options.min_iterations = 2;
    options.min_score_change = 1.0e-8f;
    options.damping = 1.0e-5f;
    options.max_step = 0.5f;
    const cudarobotics::graph::SolveSummary summary = graph.solve(options);
    check(summary.finite && summary.iterations > 0, "graph solve returns finite summary");
    check(graph.score() < initial_score * 1.0e-4f,
          "SE(3) graph score decreases to numerical zero");

    float error1[6];
    float error2[6];
    cudarobotics::lie::se3_log(
        relative(truth1, graph.vertices()[1].T).data(), error1);
    cudarobotics::lie::se3_log(
        relative(truth2, graph.vertices()[2].T).data(), error2);
    float max_error = 0.0f;
    for (int i = 0; i < 6; ++i) {
        max_error = fmaxf(max_error, fabsf(error1[i]));
        max_error = fmaxf(max_error, fabsf(error2[i]));
    }
    check(max_error < 2.0e-3f, "graph poses recover the reference trajectory");
}

void test_pose_graph_2d() {
    std::printf("[test_pose_graph2]\n");
    const float x1[3] = {1.0f, -0.2f, 0.3f};
    const float x2[3] = {2.0f, 0.4f, 0.7f};
    const float d1[3] = {0.15f, -0.08f, 0.04f};
    const float d2[3] = {-0.12f, 0.06f, -0.025f};
    const float zero[3] = {0, 0, 0};
    std::array<float, 9> identity{};
    std::array<float, 9> truth1{};
    std::array<float, 9> truth2{};
    std::array<float, 9> delta1{};
    std::array<float, 9> delta2{};
    std::array<float, 9> initial1{};
    std::array<float, 9> initial2{};
    cudarobotics::lie::se2_exp(zero, identity.data());
    cudarobotics::lie::se2_exp(x1, truth1.data());
    cudarobotics::lie::se2_exp(x2, truth2.data());
    cudarobotics::lie::se2_exp(d1, delta1.data());
    cudarobotics::lie::se2_exp(d2, delta2.data());
    cudarobotics::graph::mat3_matrix_multiply(truth1.data(), delta1.data(), initial1.data());
    cudarobotics::graph::mat3_matrix_multiply(truth2.data(), delta2.data(), initial2.data());

    cudarobotics::graph::PoseGraph2 graph;
    graph.add_vertex(identity, true, 0);
    graph.add_vertex(initial1, false, 1);
    graph.add_vertex(initial2, false, 2);
    cudarobotics::graph::PoseEdge2 e01;
    e01.from = 0;
    e01.to = 1;
    e01.measurement = identity;
    cudarobotics::graph::rigid_inverse_2d(identity.data(), e01.measurement.data());
    cudarobotics::graph::mat3_matrix_multiply(
        e01.measurement.data(), truth1.data(), e01.measurement.data());
    e01.information.fill(0.0f);
    for (int i = 0; i < 3; ++i) e01.information[3 * i + i] = 1.0f;
    cudarobotics::graph::PoseEdge2 e12 = e01;
    e12.from = 1;
    e12.to = 2;
    float truth1_inverse[9];
    cudarobotics::graph::rigid_inverse_2d(truth1.data(), truth1_inverse);
    cudarobotics::graph::mat3_matrix_multiply(
        truth1_inverse, truth2.data(), e12.measurement.data());
    graph.add_edge(e01);
    graph.add_edge(e12);

    const float initial_score = graph.score();
    cudarobotics::graph::SolverOptions options;
    options.max_iterations = 15;
    options.min_iterations = 2;
    options.min_score_change = 1.0e-8f;
    options.damping = 1.0e-5f;
    options.max_step = 0.5f;
    const cudarobotics::graph::SolveSummary summary = graph.solve(options);
    check(summary.finite && summary.iterations > 0, "SE(2) graph solve returns finite summary");
    check(graph.score() < initial_score * 1.0e-4f,
          "SE(2) graph score decreases to numerical zero");
    float error1[3];
    float error2[3];
    float truth1_inverse_again[9];
    float truth2_inverse[9];
    std::array<float, 9> relative1{};
    std::array<float, 9> relative2{};
    cudarobotics::graph::rigid_inverse_2d(truth1.data(), truth1_inverse_again);
    cudarobotics::graph::rigid_inverse_2d(truth2.data(), truth2_inverse);
    cudarobotics::graph::mat3_matrix_multiply(
        truth1_inverse_again, graph.vertices()[1].T.data(), relative1.data());
    cudarobotics::graph::mat3_matrix_multiply(
        truth2_inverse, graph.vertices()[2].T.data(), relative2.data());
    cudarobotics::lie::se2_log(relative1.data(), error1);
    cudarobotics::lie::se2_log(relative2.data(), error2);
    float max_error = 0.0f;
    for (int i = 0; i < 3; ++i) {
        max_error = fmaxf(max_error, fabsf(error1[i]));
        max_error = fmaxf(max_error, fabsf(error2[i]));
    }
    check(max_error < 2.0e-3f, "SE(2) graph poses recover the reference trajectory");
}

}  // namespace

int main() {
    std::printf("=== test_graph_optimization ===\n");
    test_pose_graph();
    test_pose_graph_2d();
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
