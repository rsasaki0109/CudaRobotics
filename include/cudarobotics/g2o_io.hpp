// g2o_io.hpp
//
// Native C++/CUDA-friendly readers for the small g2o interchange subset used
// by MathematicalRobotics.  The parser is intentionally dependency-free and
// keeps the measurements in the same row-major SE(2)/SE(3) representation as
// lie_group_math.cuh.
//
// The upper-triangular information-matrix convention follows the original
// mathR/utilities/g2o_io.py implementation.  That file originated from the
// python-graphslam project (Jeff Irion and contributors).

#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cudarobotics/lie_group_math.cuh"

namespace cudarobotics {
namespace g2o {

struct Se2Vertex {
    int id = -1;
    std::array<float, 9> T{};
};

struct Se2Edge {
    int from = -1;
    int to = -1;
    std::array<float, 9> measurement{};
    std::array<float, 9> information{};
};

struct Se3Vertex {
    int id = -1;
    std::array<float, 16> T{};
};

struct Se3Edge {
    int from = -1;
    int to = -1;
    std::array<float, 16> measurement{};
    std::array<float, 36> information{};
};

struct PoseQuatVertex {
    int id = -1;
    // g2o order: tx, ty, tz, qx, qy, qz, qw.
    std::array<float, 7> pose{};
};

struct PoseQuatEdge {
    int from = -1;
    int to = -1;
    // g2o order: tx, ty, tz, qx, qy, qz, qw.
    std::array<float, 7> measurement{};
    std::array<float, 36> information{};
};

struct Se2Graph {
    std::vector<Se2Vertex> vertices;
    std::vector<Se2Edge> edges;
};

struct Se3Graph {
    std::vector<Se3Vertex> vertices;
    std::vector<Se3Edge> edges;
};

struct PoseQuatGraph {
    std::vector<PoseQuatVertex> vertices;
    std::vector<PoseQuatEdge> edges;
};

inline void upper_triangle_to_full(const std::vector<float>& upper,
                                   int dimension,
                                   float* full) {
    for (int i = 0; i < dimension * dimension; ++i) full[i] = 0.0f;
    int index = 0;
    for (int row = 0; row < dimension; ++row) {
        for (int col = row; col < dimension; ++col) {
            const float value = index < static_cast<int>(upper.size())
                                    ? upper[index]
                                    : 0.0f;
            full[row * dimension + col] = value;
            full[col * dimension + row] = value;
            ++index;
        }
    }
}

inline bool parse_se2(std::istream& input, Se2Graph* graph) {
    if (graph == nullptr) return false;
    graph->vertices.clear();
    graph->edges.clear();
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        std::string tag;
        stream >> tag;
        if (tag == "VERTEX_SE2") {
            Se2Vertex vertex;
            float x, y, theta;
            if (!(stream >> vertex.id >> x >> y >> theta)) return false;
            const float state[3] = {x, y, theta};
            lie::se2_exp(state, vertex.T.data());
            graph->vertices.push_back(vertex);
        } else if (tag == "EDGE_SE2") {
            Se2Edge edge;
            float x, y, theta;
            if (!(stream >> edge.from >> edge.to >> x >> y >> theta)) return false;
            const float state[3] = {x, y, theta};
            lie::se2_exp(state, edge.measurement.data());
            std::vector<float> upper;
            float value;
            while (stream >> value) upper.push_back(value);
            if (upper.size() != 6) return false;
            upper_triangle_to_full(upper, 3, edge.information.data());
            graph->edges.push_back(edge);
        }
    }
    return !graph->vertices.empty() || !graph->edges.empty();
}

inline bool parse_pose_quat(std::istream& input, PoseQuatGraph* graph) {
    if (graph == nullptr) return false;
    graph->vertices.clear();
    graph->edges.clear();
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        std::string tag;
        stream >> tag;
        if (tag == "VERTEX_SE3:QUAT") {
            PoseQuatVertex vertex;
            if (!(stream >> vertex.id)) return false;
            for (float& value : vertex.pose) {
                if (!(stream >> value)) return false;
            }
            graph->vertices.push_back(vertex);
        } else if (tag == "EDGE_SE3:QUAT") {
            PoseQuatEdge edge;
            if (!(stream >> edge.from >> edge.to)) return false;
            for (float& value : edge.measurement) {
                if (!(stream >> value)) return false;
            }
            std::vector<float> upper;
            float value;
            while (stream >> value) upper.push_back(value);
            if (upper.size() != 21) return false;
            upper_triangle_to_full(upper, 6, edge.information.data());
            graph->edges.push_back(edge);
        }
    }
    return !graph->vertices.empty() || !graph->edges.empty();
}

inline bool pose_quat_to_se3(const PoseQuatGraph& source, Se3Graph* graph) {
    if (graph == nullptr) return false;
    graph->vertices.clear();
    graph->edges.clear();
    graph->vertices.reserve(source.vertices.size());
    graph->edges.reserve(source.edges.size());
    for (const PoseQuatVertex& source_vertex : source.vertices) {
        Se3Vertex vertex;
        vertex.id = source_vertex.id;
        float R[9];
        lie::quaternion_to_mat3(source_vertex.pose.data() + 3,
                                R);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                vertex.T[4 * row + col] = R[3 * row + col];
            }
        }
        vertex.T[3] = source_vertex.pose[0];
        vertex.T[7] = source_vertex.pose[1];
        vertex.T[11] = source_vertex.pose[2];
        vertex.T[12] = 0.0f;
        vertex.T[13] = 0.0f;
        vertex.T[14] = 0.0f;
        vertex.T[15] = 1.0f;
        graph->vertices.push_back(vertex);
    }
    for (const PoseQuatEdge& source_edge : source.edges) {
        Se3Edge edge;
        edge.from = source_edge.from;
        edge.to = source_edge.to;
        float R[9];
        lie::quaternion_to_mat3(source_edge.measurement.data() + 3,
                                R);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                edge.measurement[4 * row + col] = R[3 * row + col];
            }
        }
        edge.measurement[3] = source_edge.measurement[0];
        edge.measurement[7] = source_edge.measurement[1];
        edge.measurement[11] = source_edge.measurement[2];
        edge.measurement[12] = 0.0f;
        edge.measurement[13] = 0.0f;
        edge.measurement[14] = 0.0f;
        edge.measurement[15] = 1.0f;
        edge.information = source_edge.information;
        graph->edges.push_back(edge);
    }
    return true;
}

inline bool load_se2(const std::string& filename, Se2Graph* graph) {
    std::ifstream input(filename.c_str());
    return input.good() && parse_se2(input, graph);
}

inline bool load_pose_quat(const std::string& filename,
                           PoseQuatGraph* graph) {
    std::ifstream input(filename.c_str());
    return input.good() && parse_pose_quat(input, graph);
}

inline bool load_se3(const std::string& filename, Se3Graph* graph) {
    PoseQuatGraph pose_quat;
    if (!load_pose_quat(filename, &pose_quat)) return false;
    return pose_quat_to_se3(pose_quat, graph);
}

}  // namespace g2o
}  // namespace cudarobotics
