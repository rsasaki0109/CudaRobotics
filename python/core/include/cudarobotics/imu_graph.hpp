// 15-DoF IMU factor graph backend.
//
// MathematicalRobotics exposes an IMU factor over NavState=(R,p,v) and a
// six-dimensional bias.  This adapter connects that factor API to a small
// dense Gauss-Newton graph backend: each vertex owns 9 navigation tangent
// variables plus 6 bias variables, while an edge uses the analytic
// preintegrator Jacobians already shared by CPU and CUDA code.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "cudarobotics/imu_preintegration.hpp"

namespace cudarobotics {
namespace imu {

struct NavStateBias {
    NavState state;
    ImuBias bias;
};

struct ImuGraphEdge {
    int from = -1;
    int to = -1;
    ImuPreintegrator preintegrator;
    float information[81] = {};
};

struct ImuFactor15Linearization {
    float residual[9];
    // [state(9), bias(6)] for the source vertex and state-only for target.
    float J_from[135];
    float J_to[135];
};

struct BiasFactorLinearization {
    float residual[6];
    float J_from[36];
    float J_to[36];
};

struct NavStateFactorLinearization {
    float residual[9];
    float J_from[81];
    float J_to[81];
};

struct PositionVelocityFactorLinearization {
    float residual[3];
    float J_from[27];
    float J_to[27];
};

struct NavTransitionFactorLinearization {
    float residual[9];
    float J_from[81];
    float J_to[81];
};

#if defined(__CUDACC__)
#define CUDAROBOTICS_IMU_GRAPH_HD __host__ __device__
#else
#define CUDAROBOTICS_IMU_GRAPH_HD
#endif

CUDAROBOTICS_IMU_GRAPH_HD static inline void linearize_imu_factor_15(
    const ImuPreintegrator& preintegrator,
    const NavState& state_i,
    const NavState& state_j,
    const ImuBias& bias,
    ImuFactor15Linearization* output) {
    if (output == nullptr) return;
    ImuFactorLinearization base;
    linearize_imu_factor(preintegrator, state_i, state_j, bias, &base);
    for (int i = 0; i < 9; ++i) output->residual[i] = base.residual[i];
    for (int i = 0; i < 135; ++i) {
        output->J_from[i] = 0.0f;
        output->J_to[i] = 0.0f;
    }
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            output->J_from[15 * row + col] = base.J_state_i[9 * row + col];
            output->J_to[15 * row + col] = base.J_state_j[9 * row + col];
        }
        for (int col = 0; col < 6; ++col)
            output->J_from[15 * row + 9 + col] = base.J_bias_i[6 * row + col];
    }
}

CUDAROBOTICS_IMU_GRAPH_HD static inline void linearize_bias_prior(
    const ImuBias& bias,
    const ImuBias& measurement,
    BiasFactorLinearization* output) {
    if (output == nullptr) return;
    for (int i = 0; i < 6; ++i) {
        const float current = i < 3 ? bias.accel[i] : bias.gyro[i - 3];
        const float target = i < 3 ? measurement.accel[i] : measurement.gyro[i - 3];
        output->residual[i] = current - target;
    }
    for (int i = 0; i < 36; ++i) {
        output->J_from[i] = 0.0f;
        output->J_to[i] = 0.0f;
    }
    for (int i = 0; i < 6; ++i) output->J_from[6 * i + i] = 1.0f;
}

CUDAROBOTICS_IMU_GRAPH_HD static inline void linearize_bias_change(
    const ImuBias& bias_from,
    const ImuBias& bias_to,
    BiasFactorLinearization* output) {
    if (output == nullptr) return;
    for (int i = 0; i < 6; ++i) {
        const float from = i < 3 ? bias_from.accel[i] : bias_from.gyro[i - 3];
        const float to = i < 3 ? bias_to.accel[i] : bias_to.gyro[i - 3];
        output->residual[i] = from - to;
    }
    for (int i = 0; i < 36; ++i) {
        output->J_from[i] = 0.0f;
        output->J_to[i] = 0.0f;
    }
    for (int i = 0; i < 6; ++i) {
        output->J_from[6 * i + i] = 1.0f;
        output->J_to[6 * i + i] = -1.0f;
    }
}

CUDAROBOTICS_IMU_GRAPH_HD static inline void linearize_nav_state_prior(
    const NavState& state,
    const NavState& measurement,
    NavStateFactorLinearization* output) {
    if (output == nullptr) return;
    NavDelta delta;
    nav_state_local(state, measurement, &delta, output->J_from, output->J_to);
    delta_to_tangent(delta, output->residual);
}

CUDAROBOTICS_IMU_GRAPH_HD static inline bool linearize_position_velocity(
    const NavState& state_from,
    const NavState& state_to,
    float dt,
    PositionVelocityFactorLinearization* output) {
    if (output == nullptr || fabsf(dt) < 1.0e-12f) return false;
    const float inv_dt = 1.0f / dt;
    for (int row = 0; row < 3; ++row) {
        output->residual[row] = state_from.v[row] -
            (state_to.p[row] - state_from.p[row]) * inv_dt;
    }
    for (int i = 0; i < 27; ++i) {
        output->J_from[i] = 0.0f;
        output->J_to[i] = 0.0f;
    }
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            output->J_from[9 * row + 3 + col] = state_from.R[3 * row + col] * inv_dt;
            output->J_from[9 * row + 6 + col] = state_from.R[3 * row + col];
            output->J_to[9 * row + 3 + col] = -state_to.R[3 * row + col] * inv_dt;
        }
    }
    return true;
}

CUDAROBOTICS_IMU_GRAPH_HD static inline void linearize_nav_transition(
    const NavState& state_from,
    const NavState& state_to,
    const NavDelta& measurement,
    NavTransitionFactorLinearization* output) {
    if (output == nullptr) return;
    NavDelta actual;
    float J_actual_from[81];
    float J_actual_to[81];
    nav_state_local(state_from, state_to, &actual,
                    J_actual_from, J_actual_to);
    NavDelta error;
    float J_measurement[81];
    float J_error_actual[81];
    delta_local(measurement, actual, &error,
                J_measurement, J_error_actual);
    delta_to_tangent(error, output->residual);
    mat9_mul(J_error_actual, J_actual_from, output->J_from);
    mat9_mul(J_error_actual, J_actual_to, output->J_to);
}

CUDAROBOTICS_IMU_GRAPH_HD static inline void nav_state_to_2d(
    const NavState& state, float* output) {
    if (output == nullptr) return;
    output[0] = state.p[0];
    output[1] = state.p[1];
    output[2] = atan2f(state.R[3], state.R[0]);
}

struct ImuGraphOptions {
    int max_iterations = 15;
    int min_iterations = 2;
    float min_score_change = 1.0e-6f;
    float damping = 1.0e-6f;
    float max_step = 0.0f;
};

struct ImuGraphSummary {
    int iterations = 0;
    float initial_score = 0.0f;
    float final_score = 0.0f;
    bool finite = true;
};

inline bool solve_imu_dense(std::vector<float> A,
                            std::vector<float> b,
                            std::vector<float>* x) {
    const int n = static_cast<int>(b.size());
    if (x == nullptr || static_cast<int>(A.size()) != n * n) return false;
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row)
            if (fabsf(A[row * n + col]) > fabsf(A[pivot * n + col])) pivot = row;
        if (fabsf(A[pivot * n + col]) < 1.0e-12f) return false;
        if (pivot != col) {
            for (int j = col; j < n; ++j) std::swap(A[col * n + j], A[pivot * n + j]);
            std::swap(b[col], b[pivot]);
        }
        for (int row = col + 1; row < n; ++row) {
            const float factor = A[row * n + col] / A[col * n + col];
            for (int j = col; j < n; ++j) A[row * n + j] -= factor * A[col * n + j];
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

class ImuFactorGraph15 {
public:
    int add_vertex(const NavStateBias& vertex, bool constant = false) {
        vertices_.push_back(vertex);
        constants_.push_back(constant);
        return static_cast<int>(vertices_.size()) - 1;
    }

    void add_edge(const ImuGraphEdge& edge) {
        ImuGraphEdge copy = edge;
        bool has_information = false;
        for (float value : copy.information) if (value != 0.0f) has_information = true;
        if (!has_information) for (int i = 0; i < 9; ++i) copy.information[9 * i + i] = 1.0f;
        edges_.push_back(copy);
    }

    const std::vector<NavStateBias>& vertices() const { return vertices_; }
    std::vector<NavStateBias>& vertices() { return vertices_; }
    const std::vector<ImuGraphEdge>& edges() const { return edges_; }

    float score() const {
        float total = 0.0f;
        for (const ImuGraphEdge& edge : edges_) {
            ImuFactorLinearization linearization;
            linearize_imu_factor(edge.preintegrator,
                                 vertices_[edge.from].state,
                                 vertices_[edge.to].state,
                                 vertices_[edge.from].bias,
                                 &linearization);
            float weighted[9] = {};
            for (int row = 0; row < 9; ++row)
                for (int col = 0; col < 9; ++col) weighted[row] += edge.information[9 * row + col] * linearization.residual[col];
            for (int i = 0; i < 9; ++i) total += linearization.residual[i] * weighted[i];
        }
        return total;
    }

    ImuGraphSummary solve(const ImuGraphOptions& options = ImuGraphOptions()) {
        ImuGraphSummary summary;
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
                float max_abs = 0.0f;
                for (float value : dx) max_abs = fmaxf(max_abs, fabsf(value));
                if (max_abs > options.max_step)
                    for (float& value : dx) value *= options.max_step / max_abs;
            }
            apply(dx);
            summary.iterations = iteration + 1;
            summary.final_score = score();
            if (summary.iterations >= options.min_iterations &&
                fabsf(previous - summary.final_score) < options.min_score_change) break;
            previous = summary.final_score;
        }
        if (summary.iterations == 0) summary.final_score = score();
        return summary;
    }

private:
    bool solve_once(const ImuGraphOptions& options,
                    std::vector<float>* dx,
                    float* score_out) const {
        std::vector<int> offsets(vertices_.size(), -1);
        int parameter_count = 0;
        for (size_t i = 0; i < vertices_.size(); ++i) {
            if (!constants_[i]) {
                offsets[i] = parameter_count;
                parameter_count += 15;
            }
        }
        if (parameter_count == 0) return false;
        std::vector<float> H(parameter_count * parameter_count, 0.0f);
        std::vector<float> g(parameter_count, 0.0f);
        float score = 0.0f;
        for (const ImuGraphEdge& edge : edges_) {
            ImuFactorLinearization linearization;
            linearize_imu_factor(edge.preintegrator,
                                 vertices_[edge.from].state,
                                 vertices_[edge.to].state,
                                 vertices_[edge.from].bias,
                                 &linearization);
            float J_from[135] = {};
            float J_to[135] = {};
            for (int row = 0; row < 9; ++row) {
                for (int col = 0; col < 9; ++col) {
                    J_from[15 * row + col] = linearization.J_state_i[9 * row + col];
                    J_to[15 * row + col] = linearization.J_state_j[9 * row + col];
                }
                for (int col = 0; col < 6; ++col) J_from[15 * row + 9 + col] = linearization.J_bias_i[6 * row + col];
            }
            float weighted_residual[9] = {};
            for (int row = 0; row < 9; ++row)
                for (int col = 0; col < 9; ++col) weighted_residual[row] += edge.information[9 * row + col] * linearization.residual[col];
            float e2 = 0.0f;
            for (int i = 0; i < 9; ++i) e2 += linearization.residual[i] * weighted_residual[i];
            score += e2;
            accumulate_block(edge.from, edge.from, offsets, J_from, J_from, edge.information, &H);
            accumulate_block(edge.from, edge.to, offsets, J_from, J_to, edge.information, &H);
            accumulate_block(edge.to, edge.from, offsets, J_to, J_from, edge.information, &H);
            accumulate_block(edge.to, edge.to, offsets, J_to, J_to, edge.information, &H);
            accumulate_gradient(edge.from, offsets, J_from, weighted_residual, &g);
            accumulate_gradient(edge.to, offsets, J_to, weighted_residual, &g);
        }
        for (int i = 0; i < parameter_count; ++i) H[i * parameter_count + i] += options.damping;
        std::vector<float> rhs(parameter_count);
        for (int i = 0; i < parameter_count; ++i) rhs[i] = -g[i];
        if (!solve_imu_dense(H, rhs, dx)) return false;
        if (score_out != nullptr) *score_out = score;
        return true;
    }

    static void accumulate_block(int row_vertex,
                                 int col_vertex,
                                 const std::vector<int>& offsets,
                                 const float* row_jacobian,
                                 const float* col_jacobian,
                                 const float* information,
                                 std::vector<float>* H) {
        if (offsets[row_vertex] < 0 || offsets[col_vertex] < 0) return;
        const int n = static_cast<int>(std::sqrt(static_cast<float>(H->size())));
        for (int row = 0; row < 15; ++row) {
            for (int col = 0; col < 15; ++col) {
                float value = 0.0f;
                for (int a = 0; a < 9; ++a)
                    for (int b = 0; b < 9; ++b) value += row_jacobian[15 * a + row] * information[9 * a + b] * col_jacobian[15 * b + col];
                (*H)[(offsets[row_vertex] + row) * n + offsets[col_vertex] + col] += value;
            }
        }
    }

    static void accumulate_gradient(int vertex,
                                    const std::vector<int>& offsets,
                                    const float* J,
                                    const float* weighted_residual,
                                    std::vector<float>* g) {
        if (offsets[vertex] < 0) return;
        for (int col = 0; col < 15; ++col) {
            for (int row = 0; row < 9; ++row) (*g)[offsets[vertex] + col] += J[15 * row + col] * weighted_residual[row];
        }
    }

    void apply(const std::vector<float>& dx) {
        int offset = 0;
        for (size_t i = 0; i < vertices_.size(); ++i) {
            if (constants_[i]) continue;
            NavDelta delta;
            tangent_to_delta(&dx[offset], &delta);
            NavState updated;
            nav_state_retract(vertices_[i].state, delta, &updated);
            vertices_[i].state = updated;
            for (int j = 0; j < 3; ++j) {
                vertices_[i].bias.accel[j] += dx[offset + 9 + j];
                vertices_[i].bias.gyro[j] += dx[offset + 12 + j];
            }
            offset += 15;
        }
    }

    std::vector<NavStateBias> vertices_;
    std::vector<bool> constants_;
    std::vector<ImuGraphEdge> edges_;
};

}  // namespace imu
}  // namespace cudarobotics

#undef CUDAROBOTICS_IMU_GRAPH_HD
