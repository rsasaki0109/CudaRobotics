#include <cmath>
#include <cstdio>
#include <sstream>

#include "cudarobotics/g2o_io.hpp"
#include "cudarobotics/graph_optimization.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) std::printf("  PASS: %s\n", name);
    else {
        std::printf("  FAIL: %s\n", name);
        ++failures;
    }
}

void test_g2o_replay() {
    std::printf("[test_g2o_graph_replay]\n");
    std::istringstream input(
        "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
        "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1\n"
        "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 "
        "100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 100 0 0 100 0 100\n");
    cudarobotics::g2o::PoseQuatGraph parsed;
    check(cudarobotics::g2o::parse_pose_quat(input, &parsed), "parse g2o replay fixture");
    cudarobotics::g2o::Se3Graph se3;
    cudarobotics::g2o::pose_quat_to_se3(parsed, &se3);
    check(se3.vertices.size() == 2 && se3.edges.size() == 1,
          "g2o replay fixture has expected graph size");

    std::array<float, 16> initial = se3.vertices[1].T;
    initial[3] += 0.2f;
    cudarobotics::graph::PoseGraph6 graph;
    graph.add_vertex(se3.vertices[0].T, true, se3.vertices[0].id);
    graph.add_vertex(initial, false, se3.vertices[1].id);
    cudarobotics::graph::PoseEdge edge;
    edge.from = 0;
    edge.to = 1;
    edge.measurement = se3.edges[0].measurement;
    edge.information = se3.edges[0].information;
    graph.add_edge(edge);
    const float before = graph.score();
    cudarobotics::graph::SolverOptions options;
    options.max_iterations = 8;
    options.min_iterations = 2;
    options.damping = 1.0e-5f;
    graph.solve(options);
    check(graph.score() < before * 1.0e-4f, "g2o replay converges in native graph solver");
    check(fabsf(graph.vertices()[1].T[3] - 1.0f) < 2.0e-3f,
          "g2o replay preserves metric translation");
}

void test_g2o_se2_replay() {
    std::printf("[test_g2o_se2_graph_replay]\n");
    std::istringstream input(
        "VERTEX_SE2 0 0 0 0\n"
        "VERTEX_SE2 1 1 0 0\n"
        "EDGE_SE2 0 1 1 0 0 100 0 0 100 0 100\n");
    cudarobotics::g2o::Se2Graph parsed;
    check(cudarobotics::g2o::parse_se2(input, &parsed), "parse SE2 replay fixture");
    cudarobotics::graph::PoseGraph2 graph;
    std::array<float, 9> initial = parsed.vertices[1].T;
    initial[2] += 0.2f;
    graph.add_vertex(parsed.vertices[0].T, true, parsed.vertices[0].id);
    graph.add_vertex(initial, false, parsed.vertices[1].id);
    cudarobotics::graph::PoseEdge2 edge;
    edge.from = 0;
    edge.to = 1;
    edge.measurement = parsed.edges[0].measurement;
    edge.information = parsed.edges[0].information;
    graph.add_edge(edge);
    const float before = graph.score();
    cudarobotics::graph::SolverOptions options;
    options.max_iterations = 8;
    options.min_iterations = 2;
    options.damping = 1.0e-5f;
    graph.solve(options);
    check(graph.score() < before * 1.0e-4f, "SE2 g2o replay converges in native graph solver");
    check(fabsf(graph.vertices()[1].T[2] - 1.0f) < 2.0e-3f,
          "SE2 g2o replay preserves metric translation");
}

}  // namespace

int main() {
    std::printf("=== test_graph_g2o ===\n");
    test_g2o_replay();
    test_g2o_se2_replay();
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
