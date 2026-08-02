// graph_optimization.hpp
//
// A small native SE(3) graph optimizer corresponding to the reusable part of
// mathR/graph_optimization/graph_solver.py.  It deliberately separates graph
// storage from visualization and Python/SciPy sparse dependencies.  Residuals
// use the same right-retracted pose update as MathematicalRobotics' camera and
// pose vertices; central differences make the implementation useful for
// custom factors while the normal-equation assembly remains fixed-size per
// edge.  The existing CUDA pose-graph executables use the same Lie primitives
// and PCG strategy for large GPU graphs.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "cudarobotics/lie_group_math.cuh"
#include "cudarobotics/robust_loss.cuh"

namespace cudarobotics {
namespace graph {

enum class RobustLossKind { kL2, kHuber, kPseudoHuber, kCauchy };

struct PoseVertex {
    int id = -1;
    std::array<float, 16> T{};
    bool constant = false;
};

struct PoseEdge {
    int from = -1;
    int to = -1;
    std::array<float, 16> measurement{};
    std::array<float, 36> information{};
    RobustLossKind loss = RobustLossKind::kL2;
    float loss_delta = 1.0f;
};

struct SolverOptions {
    int max_iterations = 20;
    int min_iterations = 3;
    float min_score_change = 1.0e-5f;
    float damping = 1.0e-6f;
    float finite_difference_step = 1.0e-5f;
    float max_step = 0.0f;
};

struct SolveSummary {
    int iterations = 0;
    float initial_score = 0.0f;
    float final_score = 0.0f;
    bool finite = true;
};

inline void mat4_multiply(const float* A, const float* B, float* C) {
    float result[16];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) value += A[4 * row + k] * B[4 * k + col];
            result[4 * row + col] = value;
        }
    }
    for (int i = 0; i < 16; ++i) C[i] = result[i];
}

inline void rigid_inverse(const float* T, float* inverse) {
    lie::mat4_identity(inverse);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) inverse[4 * row + col] = T[4 * col + row];
    }
    const float t[3] = {T[3], T[7], T[11]};
    float R[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) R[3 * row + col] = T[4 * row + col];
    }
    float minus_rt[3];
    lie::mat3_transpose_vec(R, t, minus_rt);
    inverse[3] = -minus_rt[0];
    inverse[7] = -minus_rt[1];
    inverse[11] = -minus_rt[2];
}

inline void pose_error(const PoseEdge& edge,
                       const std::array<float, 16>& pose_from,
                       const std::array<float, 16>& pose_to,
                       float* residual) {
    float from_inverse[16];
    float relative[16];
    float measurement_inverse[16];
    float error[16];
    rigid_inverse(pose_from.data(), from_inverse);
    mat4_multiply(from_inverse, pose_to.data(), relative);
    rigid_inverse(edge.measurement.data(), measurement_inverse);
    mat4_multiply(measurement_inverse, relative, error);
    lie::se3_log(error, residual);
}

inline robust::LossEvaluation evaluate_loss(RobustLossKind kind,
                                            float squared_error,
                                            float delta) {
    switch (kind) {
        case RobustLossKind::kHuber:
            return robust::huber(squared_error, delta);
        case RobustLossKind::kPseudoHuber:
            return robust::pseudo_huber(squared_error, delta);
        case RobustLossKind::kCauchy:
            return robust::cauchy(squared_error, delta);
        case RobustLossKind::kL2:
        default:
            return robust::l2(squared_error);
    }
}

inline bool solve_dense_system(std::vector<float> A,
                               std::vector<float> b,
                               std::vector<float>* x) {
    const int n = static_cast<int>(b.size());
    if (static_cast<int>(A.size()) != n * n || x == nullptr) return false;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        float pivot_abs = fabsf(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            const float candidate = fabsf(A[row * n + col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (!(pivot_abs > 1.0e-12f) || !std::isfinite(pivot_abs)) return false;
        if (pivot != col) {
            for (int j = col; j < n; ++j) std::swap(A[col * n + j], A[pivot * n + j]);
            std::swap(b[col], b[pivot]);
        }
        for (int row = col + 1; row < n; ++row) {
            const float factor = A[row * n + col] / A[col * n + col];
            if (factor == 0.0f) continue;
            A[row * n + col] = 0.0f;
            for (int j = col + 1; j < n; ++j) A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }
    x->assign(n, 0.0f);
    for (int row = n - 1; row >= 0; --row) {
        float value = b[row];
        for (int col = row + 1; col < n; ++col) value -= A[row * n + col] * (*x)[col];
        (*x)[row] = value / A[row * n + row];
    }
    return true;
}

class PoseGraph6 {
public:
    int add_vertex(const std::array<float, 16>& T,
                   bool constant = false,
                   int id = -1) {
        PoseVertex vertex;
        vertex.id = id >= 0 ? id : static_cast<int>(vertices_.size());
        vertex.T = T;
        vertex.constant = constant;
        vertices_.push_back(vertex);
        return static_cast<int>(vertices_.size()) - 1;
    }

    int add_vertex(const float* T, bool constant = false, int id = -1) {
        std::array<float, 16> value{};
        for (int i = 0; i < 16; ++i) value[i] = T[i];
        return add_vertex(value, constant, id);
    }

    void add_edge(const PoseEdge& edge) { edges_.push_back(edge); }

    const std::vector<PoseVertex>& vertices() const { return vertices_; }
    std::vector<PoseVertex>& vertices() { return vertices_; }
    const std::vector<PoseEdge>& edges() const { return edges_; }
    std::vector<PoseEdge>& edges() { return edges_; }

    void residual(const PoseEdge& edge, float* output) const {
        pose_error(edge, vertices_[edge.from].T, vertices_[edge.to].T, output);
    }

    float score() const {
        float total = 0.0f;
        for (const PoseEdge& edge : edges_) {
            float r[6];
            residual(edge, r);
            float weighted[6] = {};
            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 6; ++col) weighted[row] += edge.information[6 * row + col] * r[col];
            }
            float e2 = 0.0f;
            for (int i = 0; i < 6; ++i) e2 += r[i] * weighted[i];
            total += evaluate_loss(edge.loss, e2, edge.loss_delta).rho0;
        }
        return total;
    }

    bool solve_once(const SolverOptions& options, std::vector<float>* dx, float* score_out) const {
        if (dx == nullptr) return false;
        std::vector<int> offsets(vertices_.size(), -1);
        int parameter_count = 0;
        for (size_t i = 0; i < vertices_.size(); ++i) {
            if (!vertices_[i].constant) {
                offsets[i] = parameter_count;
                parameter_count += 6;
            }
        }
        if (parameter_count == 0) return false;
        std::vector<float> H(parameter_count * parameter_count, 0.0f);
        std::vector<float> g(parameter_count, 0.0f);
        float score = 0.0f;
        for (const PoseEdge& edge : edges_) {
            float r[6];
            residual(edge, r);
            float Ji[36] = {};
            float Jj[36] = {};
            finite_difference(edge, edge.from, options.finite_difference_step, Ji);
            finite_difference(edge, edge.to, options.finite_difference_step, Jj);
            float weighted_r[6] = {};
            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 6; ++col) weighted_r[row] += edge.information[6 * row + col] * r[col];
            }
            float e2 = 0.0f;
            for (int i = 0; i < 6; ++i) e2 += r[i] * weighted_r[i];
            const robust::LossEvaluation loss = evaluate_loss(edge.loss, e2, edge.loss_delta);
            score += loss.rho0;
            const float weight = fmaxf(loss.rho1, 0.0f);
            accumulate_block(edge.from, edge.from, offsets, Ji, Ji, edge.information, weight, &H);
            accumulate_block(edge.from, edge.to, offsets, Ji, Jj, edge.information, weight, &H);
            accumulate_block(edge.to, edge.from, offsets, Jj, Ji, edge.information, weight, &H);
            accumulate_block(edge.to, edge.to, offsets, Jj, Jj, edge.information, weight, &H);
            accumulate_gradient(edge.from, offsets, Ji, weighted_r, weight, &g);
            accumulate_gradient(edge.to, offsets, Jj, weighted_r, weight, &g);
        }
        for (int i = 0; i < parameter_count; ++i) H[i * parameter_count + i] += options.damping;
        std::vector<float> rhs(parameter_count);
        for (int i = 0; i < parameter_count; ++i) rhs[i] = -g[i];
        if (!solve_dense_system(H, rhs, dx)) return false;
        if (score_out != nullptr) *score_out = score;
        return true;
    }

    SolveSummary solve(const SolverOptions& options = SolverOptions()) {
        SolveSummary summary;
        summary.initial_score = score();
        float previous = summary.initial_score;
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            std::vector<float> dx;
            float linearized_score = 0.0f;
            if (!solve_once(options, &dx, &linearized_score)) {
                summary.finite = false;
                break;
            }
            if (options.max_step > 0.0f) {
                float max_abs_step = 0.0f;
                for (float value : dx) max_abs_step = fmaxf(max_abs_step, fabsf(value));
                if (max_abs_step > options.max_step) {
                    const float scale = options.max_step / max_abs_step;
                    for (float& value : dx) value *= scale;
                }
            }
            apply(dx);
            const float current = score();
            summary.iterations = iteration + 1;
            if (iteration == 0) summary.initial_score = linearized_score;
            summary.final_score = current;
            if (iteration + 1 >= options.min_iterations &&
                fabsf(previous - current) < options.min_score_change) break;
            previous = current;
        }
        if (summary.iterations == 0) summary.final_score = score();
        return summary;
    }

    void apply(const std::vector<float>& dx) {
        int offset = 0;
        for (PoseVertex& vertex : vertices_) {
            if (vertex.constant) continue;
            float step[6];
            for (int i = 0; i < 6; ++i) step[i] = dx[offset + i];
            float increment[16];
            float updated[16];
            lie::se3_exp(step, increment);
            mat4_multiply(vertex.T.data(), increment, updated);
            for (int i = 0; i < 16; ++i) vertex.T[i] = updated[i];
            offset += 6;
        }
    }

private:
    void finite_difference(const PoseEdge& edge,
                           int vertex_index,
                           float step_size,
                           float* J) const {
        const std::array<float, 16>& original = vertices_[vertex_index].T;
        for (int column = 0; column < 6; ++column) {
            float plus_step[6] = {};
            float minus_step[6] = {};
            plus_step[column] = step_size;
            minus_step[column] = -step_size;
            float plus_increment[16];
            float minus_increment[16];
            std::array<float, 16> plus_pose = original;
            std::array<float, 16> minus_pose = original;
            lie::se3_exp(plus_step, plus_increment);
            lie::se3_exp(minus_step, minus_increment);
            mat4_multiply(original.data(), plus_increment, plus_pose.data());
            mat4_multiply(original.data(), minus_increment, minus_pose.data());
            const std::array<float, 16>* from_plus = &vertices_[edge.from].T;
            const std::array<float, 16>* to_plus = &vertices_[edge.to].T;
            const std::array<float, 16>* from_minus = &vertices_[edge.from].T;
            const std::array<float, 16>* to_minus = &vertices_[edge.to].T;
            if (vertex_index == edge.from) {
                from_plus = &plus_pose;
                from_minus = &minus_pose;
            } else if (vertex_index == edge.to) {
                to_plus = &plus_pose;
                to_minus = &minus_pose;
            } else {
                for (int row = 0; row < 6; ++row) J[row * 6 + column] = 0.0f;
                continue;
            }
            float r_plus[6];
            float r_minus[6];
            pose_error(edge, *from_plus, *to_plus, r_plus);
            pose_error(edge, *from_minus, *to_minus, r_minus);
            for (int row = 0; row < 6; ++row) J[row * 6 + column] =
                (r_plus[row] - r_minus[row]) / (2.0f * step_size);
        }
    }

    static void accumulate_block(int row_vertex,
                                 int col_vertex,
                                 const std::vector<int>& offsets,
                                 const float* row_jacobian,
                                 const float* col_jacobian,
                                 const std::array<float, 36>& information,
                                 float weight,
                                 std::vector<float>* H) {
        if (offsets[row_vertex] < 0 || offsets[col_vertex] < 0) return;
        const int parameter_count = static_cast<int>(std::sqrt(static_cast<float>(H->size())));
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 6; ++col) {
                float value = 0.0f;
                for (int a = 0; a < 6; ++a) {
                    for (int b = 0; b < 6; ++b) {
                        value += row_jacobian[a * 6 + row] * information[a * 6 + b] *
                                 col_jacobian[b * 6 + col];
                    }
                }
                (*H)[(offsets[row_vertex] + row) * parameter_count +
                     offsets[col_vertex] + col] += weight * value;
            }
        }
    }

    static void accumulate_gradient(int vertex,
                                    const std::vector<int>& offsets,
                                    const float* J,
                                    const float* weighted_r,
                                    float weight,
                                    std::vector<float>* g) {
        if (offsets[vertex] < 0) return;
        for (int col = 0; col < 6; ++col) {
            float value = 0.0f;
            for (int row = 0; row < 6; ++row) value += J[row * 6 + col] * weighted_r[row];
            (*g)[offsets[vertex] + col] += weight * value;
        }
    }

    std::vector<PoseVertex> vertices_;
    std::vector<PoseEdge> edges_;
};

// The upstream graph demos also provide a reusable SE(2) specialization.  It
// follows the same right-retracted update and dense reference solve as the
// SE(3) graph above; large CUDA workloads should use the existing GPU graph
// executables instead.
struct PoseVertex2 {
    int id = -1;
    std::array<float, 9> T{};
    bool constant = false;
};

struct PoseEdge2 {
    int from = -1;
    int to = -1;
    std::array<float, 9> measurement{};
    std::array<float, 9> information{};
    RobustLossKind loss = RobustLossKind::kL2;
    float loss_delta = 1.0f;
};

inline void mat3_matrix_multiply(const float* A, const float* B, float* C) {
    float result[9];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            float value = 0.0f;
            for (int k = 0; k < 3; ++k) value += A[3 * row + k] * B[3 * k + col];
            result[3 * row + col] = value;
        }
    }
    for (int i = 0; i < 9; ++i) C[i] = result[i];
}

inline void rigid_inverse_2d(const float* T, float* inverse) {
    inverse[0] = T[0];
    inverse[1] = T[3];
    inverse[2] = -(T[0] * T[2] + T[3] * T[5]);
    inverse[3] = T[1];
    inverse[4] = T[4];
    inverse[5] = -(T[1] * T[2] + T[4] * T[5]);
    inverse[6] = 0.0f;
    inverse[7] = 0.0f;
    inverse[8] = 1.0f;
}

inline void pose_error_2d(const PoseEdge2& edge,
                          const std::array<float, 9>& pose_from,
                          const std::array<float, 9>& pose_to,
                          float* residual) {
    float from_inverse[9];
    float relative[9];
    float measurement_inverse[9];
    float error[9];
    rigid_inverse_2d(pose_from.data(), from_inverse);
    mat3_matrix_multiply(from_inverse, pose_to.data(), relative);
    rigid_inverse_2d(edge.measurement.data(), measurement_inverse);
    mat3_matrix_multiply(measurement_inverse, relative, error);
    lie::se2_log(error, residual);
}

class PoseGraph2 {
public:
    int add_vertex(const std::array<float, 9>& T,
                   bool constant = false,
                   int id = -1) {
        PoseVertex2 vertex;
        vertex.id = id >= 0 ? id : static_cast<int>(vertices_.size());
        vertex.T = T;
        vertex.constant = constant;
        vertices_.push_back(vertex);
        return static_cast<int>(vertices_.size()) - 1;
    }

    int add_vertex(const float* T, bool constant = false, int id = -1) {
        std::array<float, 9> value{};
        for (int i = 0; i < 9; ++i) value[i] = T[i];
        return add_vertex(value, constant, id);
    }

    void add_edge(const PoseEdge2& edge) { edges_.push_back(edge); }

    const std::vector<PoseVertex2>& vertices() const { return vertices_; }
    std::vector<PoseVertex2>& vertices() { return vertices_; }
    const std::vector<PoseEdge2>& edges() const { return edges_; }
    std::vector<PoseEdge2>& edges() { return edges_; }

    void residual(const PoseEdge2& edge, float* output) const {
        pose_error_2d(edge, vertices_[edge.from].T, vertices_[edge.to].T, output);
    }

    float score() const {
        float total = 0.0f;
        for (const PoseEdge2& edge : edges_) {
            float r[3];
            residual(edge, r);
            float weighted[3] = {};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    weighted[row] += edge.information[3 * row + col] * r[col];
                }
            }
            float e2 = 0.0f;
            for (int i = 0; i < 3; ++i) e2 += r[i] * weighted[i];
            total += evaluate_loss(edge.loss, e2, edge.loss_delta).rho0;
        }
        return total;
    }

    bool solve_once(const SolverOptions& options,
                    std::vector<float>* dx,
                    float* score_out) const {
        if (dx == nullptr) return false;
        std::vector<int> offsets(vertices_.size(), -1);
        int parameter_count = 0;
        for (size_t i = 0; i < vertices_.size(); ++i) {
            if (!vertices_[i].constant) {
                offsets[i] = parameter_count;
                parameter_count += 3;
            }
        }
        if (parameter_count == 0) return false;
        std::vector<float> H(parameter_count * parameter_count, 0.0f);
        std::vector<float> g(parameter_count, 0.0f);
        float score = 0.0f;
        for (const PoseEdge2& edge : edges_) {
            float r[3];
            residual(edge, r);
            float Ji[9] = {};
            float Jj[9] = {};
            finite_difference(edge, edge.from, options.finite_difference_step, Ji);
            finite_difference(edge, edge.to, options.finite_difference_step, Jj);
            float weighted_r[3] = {};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    weighted_r[row] += edge.information[3 * row + col] * r[col];
                }
            }
            float e2 = 0.0f;
            for (int i = 0; i < 3; ++i) e2 += r[i] * weighted_r[i];
            const robust::LossEvaluation loss = evaluate_loss(edge.loss, e2, edge.loss_delta);
            score += loss.rho0;
            const float weight = fmaxf(loss.rho1, 0.0f);
            accumulate_block(edge.from, edge.from, offsets, Ji, Ji,
                             edge.information, weight, &H);
            accumulate_block(edge.from, edge.to, offsets, Ji, Jj,
                             edge.information, weight, &H);
            accumulate_block(edge.to, edge.from, offsets, Jj, Ji,
                             edge.information, weight, &H);
            accumulate_block(edge.to, edge.to, offsets, Jj, Jj,
                             edge.information, weight, &H);
            accumulate_gradient(edge.from, offsets, Ji, weighted_r, weight, &g);
            accumulate_gradient(edge.to, offsets, Jj, weighted_r, weight, &g);
        }
        for (int i = 0; i < parameter_count; ++i) H[i * parameter_count + i] += options.damping;
        std::vector<float> rhs(parameter_count);
        for (int i = 0; i < parameter_count; ++i) rhs[i] = -g[i];
        if (!solve_dense_system(H, rhs, dx)) return false;
        if (score_out != nullptr) *score_out = score;
        return true;
    }

    SolveSummary solve(const SolverOptions& options = SolverOptions()) {
        SolveSummary summary;
        summary.initial_score = score();
        float previous = summary.initial_score;
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            std::vector<float> dx;
            float linearized_score = 0.0f;
            if (!solve_once(options, &dx, &linearized_score)) {
                summary.finite = false;
                break;
            }
            if (options.max_step > 0.0f) {
                float max_abs_step = 0.0f;
                for (float value : dx) max_abs_step = fmaxf(max_abs_step, fabsf(value));
                if (max_abs_step > options.max_step) {
                    const float scale = options.max_step / max_abs_step;
                    for (float& value : dx) value *= scale;
                }
            }
            apply(dx);
            const float current = score();
            summary.iterations = iteration + 1;
            if (iteration == 0) summary.initial_score = linearized_score;
            summary.final_score = current;
            if (iteration + 1 >= options.min_iterations &&
                fabsf(previous - current) < options.min_score_change) break;
            previous = current;
        }
        if (summary.iterations == 0) summary.final_score = score();
        return summary;
    }

    void apply(const std::vector<float>& dx) {
        int offset = 0;
        for (PoseVertex2& vertex : vertices_) {
            if (vertex.constant) continue;
            float step[3];
            for (int i = 0; i < 3; ++i) step[i] = dx[offset + i];
            float increment[9];
            float updated[9];
            lie::se2_exp(step, increment);
            mat3_matrix_multiply(vertex.T.data(), increment, updated);
            for (int i = 0; i < 9; ++i) vertex.T[i] = updated[i];
            offset += 3;
        }
    }

private:
    void finite_difference(const PoseEdge2& edge,
                           int vertex_index,
                           float step_size,
                           float* J) const {
        const std::array<float, 9>& original = vertices_[vertex_index].T;
        for (int column = 0; column < 3; ++column) {
            float plus_step[3] = {};
            float minus_step[3] = {};
            plus_step[column] = step_size;
            minus_step[column] = -step_size;
            float plus_increment[9];
            float minus_increment[9];
            std::array<float, 9> plus_pose = original;
            std::array<float, 9> minus_pose = original;
            lie::se2_exp(plus_step, plus_increment);
            lie::se2_exp(minus_step, minus_increment);
            mat3_matrix_multiply(original.data(), plus_increment, plus_pose.data());
            mat3_matrix_multiply(original.data(), minus_increment, minus_pose.data());
            const std::array<float, 9>* from_plus = &vertices_[edge.from].T;
            const std::array<float, 9>* to_plus = &vertices_[edge.to].T;
            const std::array<float, 9>* from_minus = &vertices_[edge.from].T;
            const std::array<float, 9>* to_minus = &vertices_[edge.to].T;
            if (vertex_index == edge.from) {
                from_plus = &plus_pose;
                from_minus = &minus_pose;
            } else if (vertex_index == edge.to) {
                to_plus = &plus_pose;
                to_minus = &minus_pose;
            } else {
                for (int row = 0; row < 3; ++row) J[row * 3 + column] = 0.0f;
                continue;
            }
            float r_plus[3];
            float r_minus[3];
            pose_error_2d(edge, *from_plus, *to_plus, r_plus);
            pose_error_2d(edge, *from_minus, *to_minus, r_minus);
            for (int row = 0; row < 3; ++row) {
                J[row * 3 + column] = (r_plus[row] - r_minus[row]) / (2.0f * step_size);
            }
        }
    }

    static void accumulate_block(int row_vertex,
                                 int col_vertex,
                                 const std::vector<int>& offsets,
                                 const float* row_jacobian,
                                 const float* col_jacobian,
                                 const std::array<float, 9>& information,
                                 float weight,
                                 std::vector<float>* H) {
        if (offsets[row_vertex] < 0 || offsets[col_vertex] < 0) return;
        const int parameter_count = static_cast<int>(std::sqrt(static_cast<float>(H->size())));
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                float value = 0.0f;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        value += row_jacobian[a * 3 + row] * information[a * 3 + b] *
                                 col_jacobian[b * 3 + col];
                    }
                }
                (*H)[(offsets[row_vertex] + row) * parameter_count +
                     offsets[col_vertex] + col] += weight * value;
            }
        }
    }

    static void accumulate_gradient(int vertex,
                                    const std::vector<int>& offsets,
                                    const float* J,
                                    const float* weighted_r,
                                    float weight,
                                    std::vector<float>* g) {
        if (offsets[vertex] < 0) return;
        for (int col = 0; col < 3; ++col) {
            float value = 0.0f;
            for (int row = 0; row < 3; ++row) value += J[row * 3 + col] * weighted_r[row];
            (*g)[offsets[vertex] + col] += weight * value;
        }
    }

    std::vector<PoseVertex2> vertices_;
    std::vector<PoseEdge2> edges_;
};

}  // namespace graph
}  // namespace cudarobotics
