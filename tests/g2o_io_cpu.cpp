#include <cmath>
#include <cstdio>
#include <sstream>

#include "cudarobotics/g2o_io.hpp"

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

float max_abs_rotation_and_homogeneous(const float* T) {
    float result = 0.0f;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k) value += T[4 * k + row] * T[4 * k + col];
            if (row == col) value -= 1.0f;
            result = fmaxf(result, fabsf(value));
        }
    }
    result = fmaxf(result, fabsf(T[12]));
    result = fmaxf(result, fabsf(T[13]));
    result = fmaxf(result, fabsf(T[14]));
    result = fmaxf(result, fabsf(T[15] - 1.0f));
    return result;
}

void test_se2() {
    std::printf("[test_se2_g2o]\n");
    std::istringstream input(
        "# a tiny SE2 graph\n"
        "VERTEX_SE2 4 1.0 -2.0 0.3\n"
        "VERTEX_SE2 9 1.4 -1.5 0.5\n"
        "EDGE_SE2 4 9 0.4 0.5 0.2 1 2 3 4 5 6\n");
    cudarobotics::g2o::Se2Graph graph;
    check(cudarobotics::g2o::parse_se2(input, &graph), "parse SE2 stream");
    check(graph.vertices.size() == 2 && graph.edges.size() == 1,
          "SE2 vertex/edge counts");
    check(graph.vertices[0].id == 4 && graph.edges[0].from == 4 &&
              graph.edges[0].to == 9,
          "SE2 ids are preserved");
    check(fabsf(graph.edges[0].information[0] - 1.0f) < 1.0e-6f &&
              fabsf(graph.edges[0].information[1] - 2.0f) < 1.0e-6f &&
              fabsf(graph.edges[0].information[3] - 2.0f) < 1.0e-6f &&
              fabsf(graph.edges[0].information[8] - 6.0f) < 1.0e-6f,
          "SE2 upper information matrix is mirrored");
}

void test_se3() {
    std::printf("[test_se3_g2o]\n");
    std::istringstream input(
        "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
        "VERTEX_SE3:QUAT 1 1 2 3 0 0 0.707106781 0.707106781\n"
        "EDGE_SE3:QUAT 0 1 1 2 3 0 0 0.707106781 0.707106781 "
        "1 0 0 0 0 0 2 0 0 0 0 3 0 0 0 4 0 0 5 0 6\n");
    cudarobotics::g2o::PoseQuatGraph pose_graph;
    check(cudarobotics::g2o::parse_pose_quat(input, &pose_graph),
          "parse SE3 quaternion stream");
    cudarobotics::g2o::Se3Graph graph;
    check(cudarobotics::g2o::pose_quat_to_se3(pose_graph, &graph),
          "convert SE3 quaternion graph");
    check(graph.vertices.size() == 2 && graph.edges.size() == 1,
          "SE3 vertex/edge counts");
    const float homogeneous_error =
        max_abs_rotation_and_homogeneous(graph.vertices[1].T.data());
    check(homogeneous_error < 3.0e-5f,
          "SE3 quaternion vertex is homogeneous");
    check(fabsf(graph.vertices[1].T[3] - 1.0f) < 1.0e-6f &&
              fabsf(graph.vertices[1].T[7] - 2.0f) < 1.0e-6f &&
              fabsf(graph.vertices[1].T[11] - 3.0f) < 1.0e-6f,
          "SE3 translation is preserved");
    check(fabsf(graph.edges[0].information[0] - 1.0f) < 1.0e-6f &&
              fabsf(graph.edges[0].information[7] - 2.0f) < 1.0e-6f &&
              fabsf(graph.edges[0].information[35] - 6.0f) < 1.0e-6f,
          "SE3 information matrix is mirrored");
}

}  // namespace

int main() {
    std::printf("=== test_g2o_io ===\n");
    test_se2();
    test_se3();
    if (failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", failures);
    return 1;
}
