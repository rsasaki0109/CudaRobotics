// Deterministic central finite differences for vector-valued CPU functions.

#pragma once

#include <functional>
#include <vector>

namespace cudarobotics {
namespace math {

using VectorFunction = std::function<std::vector<float>(const std::vector<float>&)>;

inline bool numerical_derivative(const VectorFunction& function,
                                 const std::vector<float>& input,
                                 std::vector<float>* jacobian,
                                 // Float arithmetic needs a slightly larger
                                 // default than the double-precision Python
                                 // reference to avoid cancellation.
                                 float step = 1.0e-4f) {
    if (!function || jacobian == nullptr || input.empty() || step <= 0.0f) return false;
    const std::vector<float> reference = function(input);
    const int rows = static_cast<int>(reference.size());
    const int cols = static_cast<int>(input.size());
    jacobian->assign(rows * cols, 0.0f);
    for (int col = 0; col < cols; ++col) {
        std::vector<float> plus = input;
        std::vector<float> minus = input;
        plus[col] += step;
        minus[col] -= step;
        const std::vector<float> plus_value = function(plus);
        const std::vector<float> minus_value = function(minus);
        if (static_cast<int>(plus_value.size()) != rows || static_cast<int>(minus_value.size()) != rows) return false;
        for (int row = 0; row < rows; ++row) (*jacobian)[row * cols + col] =
            (plus_value[row] - minus_value[row]) / (2.0f * step);
    }
    return true;
}

}  // namespace math
}  // namespace cudarobotics
