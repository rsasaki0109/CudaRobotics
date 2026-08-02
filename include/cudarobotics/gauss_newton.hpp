// Dense Gauss-Newton assembly corresponding to mathR/optimization/gauss_newton.py.
// Callers provide residual blocks and Jacobians, so the same implementation is
// usable for ordinary vectors and Lie-group states (via a custom plus callback).

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "cudarobotics/robust_loss.cuh"

namespace cudarobotics {
namespace optimization {

struct ResidualBlock {
    std::vector<float> residual;
    std::vector<float> jacobian;  // row-major: residual.size() x parameter_count
    robust::LossEvaluation loss = {0.0f, 1.0f, 0.0f};
    bool use_loss = false;
};

struct GaussNewtonOptions {
    int max_iterations = 30;
    int min_iterations = 2;
    float min_score_change = 1.0e-5f;
    float damping = 1.0e-8f;
    float max_step = 0.0f;
};

inline bool solve_linear_system(std::vector<float> A,
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

inline bool gauss_newton_step(int parameter_count,
                              const std::vector<ResidualBlock>& blocks,
                              float damping,
                              std::vector<float>* dx,
                              float* score = nullptr) {
    if (parameter_count <= 0 || dx == nullptr) return false;
    std::vector<float> H(parameter_count * parameter_count, 0.0f);
    std::vector<float> g(parameter_count, 0.0f);
    float total_score = 0.0f;
    for (const ResidualBlock& block : blocks) {
        const int m = static_cast<int>(block.residual.size());
        if (static_cast<int>(block.jacobian.size()) != m * parameter_count) return false;
        float squared_error = 0.0f;
        for (float value : block.residual) squared_error += value * value;
        const robust::LossEvaluation loss = block.use_loss
            ? block.loss : robust::l2(squared_error);
        total_score += block.use_loss ? loss.rho0 : squared_error;
        const float weight = fmaxf(block.use_loss ? loss.rho1 : 1.0f, 0.0f);
        for (int col = 0; col < parameter_count; ++col) {
            for (int row = 0; row < m; ++row) g[col] += weight * block.jacobian[row * parameter_count + col] * block.residual[row];
            for (int other = 0; other < parameter_count; ++other) {
                for (int row = 0; row < m; ++row)
                    H[col * parameter_count + other] += weight * block.jacobian[row * parameter_count + col] * block.jacobian[row * parameter_count + other];
            }
        }
    }
    for (int i = 0; i < parameter_count; ++i) H[i * parameter_count + i] += damping;
    std::vector<float> rhs(parameter_count);
    for (int i = 0; i < parameter_count; ++i) rhs[i] = -g[i];
    if (!solve_linear_system(H, rhs, dx)) return false;
    if (score != nullptr) *score = total_score;
    return true;
}

class GaussNewtonSolver {
public:
    using LinearizeFunction = std::function<void(const std::vector<float>&,
                                                  std::vector<ResidualBlock>*)>;
    using PlusFunction = std::function<void(const std::vector<float>&,
                                            const std::vector<float>&,
                                            std::vector<float>*)>;

    explicit GaussNewtonSolver(int parameter_count,
                               LinearizeFunction linearize,
                               PlusFunction plus = PlusFunction())
        : parameter_count_(parameter_count), linearize_(linearize), plus_(plus) {}

    std::vector<float> solve(const std::vector<float>& initial,
                             const GaussNewtonOptions& options = GaussNewtonOptions(),
                             int* iterations_out = nullptr,
                             float* final_score_out = nullptr) const {
        std::vector<float> x = initial;
        float previous = 1.0e30f;
        int iterations = 0;
        float final_score = previous;
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            std::vector<ResidualBlock> blocks;
            linearize_(x, &blocks);
            std::vector<float> dx;
            float score = 0.0f;
            if (!gauss_newton_step(parameter_count_, blocks, options.damping, &dx, &score)) break;
            if (options.max_step > 0.0f) {
                float max_abs = 0.0f;
                for (float value : dx) max_abs = fmaxf(max_abs, fabsf(value));
                if (max_abs > options.max_step)
                    for (float& value : dx) value *= options.max_step / max_abs;
            }
            std::vector<float> updated;
            if (plus_) plus_(x, dx, &updated);
            else {
                updated = x;
                for (int i = 0; i < parameter_count_; ++i) updated[i] += dx[i];
            }
            x.swap(updated);
            final_score = score;
            iterations = iteration + 1;
            if (iterations >= options.min_iterations && fabsf(previous - score) < options.min_score_change) break;
            previous = score;
        }
        if (iterations_out != nullptr) *iterations_out = iterations;
        if (final_score_out != nullptr) *final_score_out = final_score;
        return x;
    }

private:
    int parameter_count_;
    LinearizeFunction linearize_;
    PlusFunction plus_;
};

}  // namespace optimization
}  // namespace cudarobotics
