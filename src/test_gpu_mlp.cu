#include "gpu_mlp.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

using namespace cudabot;

// ===== Test 1: XOR problem =====
// 2->4->1 MLP, 1000 steps, loss < 0.01
bool test_xor() {
    printf("[Test 1] XOR learning (2->4->1 MLP)\n");

    // XOR data: {(0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0}
    // Map to [-1, 1]: {(-1,-1)->-1, (-1,1)->1, (1,-1)->1, (1,1)->-1}
    float h_input[] = {-1, -1,  -1, 1,  1, -1,  1, 1};
    float h_target[] = {-1,  1,  1, -1};
    int batch_size = 4;

    float* d_input;
    float* d_target;
    cudaMalloc(&d_input, 8 * sizeof(float));
    cudaMalloc(&d_target, 4 * sizeof(float));
    cudaMemcpy(d_input, h_input, 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 2 input, 4 hidden, 1 hidden layer, 1 output
    GpuMLP mlp(2, 4, 1, 1);
    mlp.init_random(42);

    float loss = 0;
    for (int step = 0; step < 1000; step++) {
        loss = mlp.train_step(d_input, d_target, batch_size, 0.01f);
        if (step % 200 == 0) {
            printf("  Step %d: loss = %.6f\n", step, loss);
        }
    }
    printf("  Final loss: %.6f\n", loss);

    // Verify predictions
    float* d_output;
    cudaMalloc(&d_output, 4 * sizeof(float));
    mlp.forward_batch(d_input, d_output, batch_size, 1);  // tanh activation
    float h_output[4];
    cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("  Predictions: ");
    for (int i = 0; i < 4; i++) printf("%.3f ", h_output[i]);
    printf("\n  Targets:     ");
    for (int i = 0; i < 4; i++) printf("%.3f ", h_target[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_output);

    bool pass = loss < 0.01f;
    printf("  %s (loss=%.6f %s 0.01)\n\n", pass ? "PASS" : "FAIL", loss, pass ? "<" : ">=");
    return pass;
}

// ===== Test 2: SDF learning (circle SDF approximation) =====
bool test_sdf() {
    printf("[Test 2] SDF learning (circle SDF)\n");

    // Circle centered at (0,0) with radius 1.0
    // SDF(x,y) = sqrt(x^2 + y^2) - 1.0
    // Generate training data on a grid
    int grid_n = 8;
    int n_samples = grid_n * grid_n;
    std::vector<float> h_input(n_samples * 2);
    std::vector<float> h_target(n_samples);

    int idx = 0;
    for (int i = 0; i < grid_n; i++) {
        for (int j = 0; j < grid_n; j++) {
            float x = -2.0f + 4.0f * i / (grid_n - 1);
            float y = -2.0f + 4.0f * j / (grid_n - 1);
            h_input[idx * 2] = x;
            h_input[idx * 2 + 1] = y;
            h_target[idx] = std::sqrt(x * x + y * y) - 1.0f;
            idx++;
        }
    }

    float* d_input;
    float* d_target;
    cudaMalloc(&d_input, n_samples * 2 * sizeof(float));
    cudaMalloc(&d_target, n_samples * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), n_samples * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);

    // 2 input, 16 hidden, 2 hidden layers, 1 output
    GpuMLP mlp(2, 16, 2, 1);
    mlp.init_random(123);

    float loss = 0;
    for (int step = 0; step < 8000; step++) {
        loss = mlp.train_step(d_input, d_target, n_samples, 0.002f);
        if (step % 1600 == 0) {
            printf("  Step %d: loss = %.6f\n", step, loss);
        }
    }
    printf("  Final loss: %.6f\n", loss);

    cudaFree(d_input);
    cudaFree(d_target);

    bool pass = loss < 0.05f;
    printf("  %s (loss=%.6f %s 0.05)\n\n", pass ? "PASS" : "FAIL", loss, pass ? "<" : ">=");
    return pass;
}

// ===== Test 3: Batch inference speed test =====
bool test_batch_speed() {
    printf("[Test 3] Batch inference speed (1M points)\n");

    int N = 1000000;
    int input_dim = 4;
    int output_dim = 2;

    // Create random input on device
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * input_dim * sizeof(float));
    cudaMalloc(&d_output, N * output_dim * sizeof(float));

    // Initialize input to zeros (just for speed test)
    cudaMemset(d_input, 0, N * input_dim * sizeof(float));

    // 4 input, 32 hidden, 2 layers, 2 output
    GpuMLP mlp(input_dim, 32, 2, output_dim);
    mlp.init_random(42);

    // Warmup
    mlp.forward_batch(d_input, d_output, N);

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mlp.forward_batch(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("  1M forward passes: %.2f ms\n", ms);
    printf("  Throughput: %.2f M samples/sec\n", (float)N / (ms * 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    bool pass = true;  // Speed test always passes
    printf("  PASS\n\n");
    return pass;
}

int main() {
    printf("========================================\n");
    printf("  GPU MLP Tests\n");
    printf("========================================\n\n");

    bool all_pass = true;
    all_pass &= test_xor();
    all_pass &= test_sdf();
    all_pass &= test_batch_speed();

    if (all_pass) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
