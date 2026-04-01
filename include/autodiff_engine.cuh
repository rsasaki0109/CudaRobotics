#pragma once
#include <cmath>
#include <cuda_runtime.h>

namespace cudabot {

// Forward-mode autodiff via dual numbers
// DualNumber.val = f(x), DualNumber.deriv = f'(x)
template <typename T = float>
struct DualNumber {
    T val;
    T deriv;

    __host__ __device__ DualNumber() : val(0), deriv(0) {}
    __host__ __device__ DualNumber(T v, T d = 0) : val(v), deriv(d) {}

    // 変数を作る: x = DualNumber(value, 1.0) で ∂/∂x を追跡
    __host__ __device__ static DualNumber variable(T v) { return DualNumber(v, T(1)); }
    __host__ __device__ static DualNumber constant(T v) { return DualNumber(v, T(0)); }
};

// 算術演算子
template <typename T>
__host__ __device__ DualNumber<T> operator+(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val + b.val, a.deriv + b.deriv};
}
template <typename T>
__host__ __device__ DualNumber<T> operator-(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val - b.val, a.deriv - b.deriv};
}
template <typename T>
__host__ __device__ DualNumber<T> operator*(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val * b.val, a.val * b.deriv + a.deriv * b.val};
}
template <typename T>
__host__ __device__ DualNumber<T> operator/(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val / b.val, (a.deriv * b.val - a.val * b.deriv) / (b.val * b.val)};
}

// 比較演算子（val のみで比較）
template <typename T>
__host__ __device__ bool operator<(const DualNumber<T>& a, const DualNumber<T>& b) { return a.val < b.val; }
template <typename T>
__host__ __device__ bool operator>(const DualNumber<T>& a, const DualNumber<T>& b) { return a.val > b.val; }
template <typename T>
__host__ __device__ bool operator<=(const DualNumber<T>& a, const DualNumber<T>& b) { return a.val <= b.val; }

// スカラーとの演算
template <typename T>
__host__ __device__ DualNumber<T> operator*(T s, const DualNumber<T>& a) { return {s * a.val, s * a.deriv}; }
template <typename T>
__host__ __device__ DualNumber<T> operator*(const DualNumber<T>& a, T s) { return {a.val * s, a.deriv * s}; }
template <typename T>
__host__ __device__ DualNumber<T> operator+(const DualNumber<T>& a, T s) { return {a.val + s, a.deriv}; }
template <typename T>
__host__ __device__ DualNumber<T> operator-(const DualNumber<T>& a, T s) { return {a.val - s, a.deriv}; }

// 数学関数
template <typename T>
__host__ __device__ DualNumber<T> sin(const DualNumber<T>& a) {
    return {std::sin(a.val), a.deriv * std::cos(a.val)};
}
template <typename T>
__host__ __device__ DualNumber<T> cos(const DualNumber<T>& a) {
    return {std::cos(a.val), -a.deriv * std::sin(a.val)};
}
template <typename T>
__host__ __device__ DualNumber<T> tan(const DualNumber<T>& a) {
    T c = std::cos(a.val);
    return {std::tan(a.val), a.deriv / (c * c)};
}
template <typename T>
__host__ __device__ DualNumber<T> sqrt(const DualNumber<T>& a) {
    T s = std::sqrt(a.val);
    return {s, a.deriv / (T(2) * s + T(1e-10))};
}
template <typename T>
__host__ __device__ DualNumber<T> exp(const DualNumber<T>& a) {
    T e = std::exp(a.val);
    return {e, a.deriv * e};
}
template <typename T>
__host__ __device__ DualNumber<T> log(const DualNumber<T>& a) {
    return {std::log(a.val), a.deriv / a.val};
}
template <typename T>
__host__ __device__ DualNumber<T> atan2(const DualNumber<T>& y, const DualNumber<T>& x) {
    T denom = x.val * x.val + y.val * y.val + T(1e-10);
    return {std::atan2(y.val, x.val), (x.val * y.deriv - y.val * x.deriv) / denom};
}
template <typename T>
__host__ __device__ DualNumber<T> abs(const DualNumber<T>& a) {
    return {std::abs(a.val), a.val >= 0 ? a.deriv : -a.deriv};
}
// clamp (val ベースで clamp、微分は範囲内のみ伝播)
template <typename T>
__host__ __device__ DualNumber<T> clamp(const DualNumber<T>& a, T lo, T hi) {
    if (a.val < lo) return {lo, T(0)};
    if (a.val > hi) return {hi, T(0)};
    return a;
}

using Dualf = DualNumber<float>;
using Duald = DualNumber<double>;

// ヤコビアン計算ヘルパー
// f: R^n -> R^m の関数に対して、1変数ずつ DualNumber で微分を取る
// 使い方: 各入力変数を順番に variable() にして forward pass を実行

} // namespace cudabot
