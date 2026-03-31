#pragma once
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Benchmark optimization functions (all __device__, arbitrary dimension D)
// ============================================================================

// Rastrigin function
// Optimal: f(0, ..., 0) = 0
// Search range: [-5.12, 5.12]^D
__device__ inline float rastrigin(const float* x, int D) {
    float sum = 10.0f * D;
    for (int i = 0; i < D; i++)
        sum += x[i] * x[i] - 10.0f * cosf(2.0f * M_PI * x[i]);
    return sum;
}

// Rosenbrock function
// Optimal: f(1, ..., 1) = 0
// Search range: [-5, 10]^D
__device__ inline float rosenbrock(const float* x, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D - 1; i++) {
        float xi = x[i];
        float xi1 = x[i + 1];
        sum += 100.0f * (xi1 - xi * xi) * (xi1 - xi * xi) + (1.0f - xi) * (1.0f - xi);
    }
    return sum;
}

// Ackley function
// Optimal: f(0, ..., 0) = 0
// Search range: [-5, 5]^D
__device__ inline float ackley(const float* x, int D) {
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < D; i++) {
        sum1 += x[i] * x[i];
        sum2 += cosf(2.0f * M_PI * x[i]);
    }
    return -20.0f * expf(-0.2f * sqrtf(sum1 / D)) - expf(sum2 / D) + 20.0f + M_E;
}

// Schwefel function
// Optimal: f(420.9687, ..., 420.9687) ~ 0
// Search range: [-500, 500]^D
__device__ inline float schwefel(const float* x, int D) {
    float sum = 418.9829f * D;
    for (int i = 0; i < D; i++)
        sum -= x[i] * sinf(sqrtf(fabsf(x[i])));
    return sum;
}

// Evaluate benchmark function by ID
// 0 = Rastrigin, 1 = Rosenbrock, 2 = Ackley, 3 = Schwefel
__device__ inline float evaluate_benchmark(const float* x, int D, int func_id) {
    switch (func_id) {
        case 0: return rastrigin(x, D);
        case 1: return rosenbrock(x, D);
        case 2: return ackley(x, D);
        case 3: return schwefel(x, D);
        default: return rastrigin(x, D);
    }
}
