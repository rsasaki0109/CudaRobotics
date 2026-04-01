#pragma once
#include <cuda_runtime.h>

#define NN_HIDDEN1 32
#define NN_HIDDEN2 16
#define MAX_WEIGHTS 1024

// Fixed topology NN: input -> 32 -> 16 -> output
// Activation: tanh (hidden layers), linear (output layer)
// Weight layout: [W1(in*32), b1(32), W2(32*16), b2(16), W3(16*out), b3(out)]

__device__ inline void nn_forward(
    const float* weights,
    const float* input,
    float* output,
    int input_dim, int output_dim)
{
    float h1[NN_HIDDEN1];
    float h2[NN_HIDDEN2];

    // Layer 1: input -> hidden1 (tanh)
    int offset = 0;
    for (int j = 0; j < NN_HIDDEN1; j++) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            sum += weights[offset + j * input_dim + i] * input[i];
        }
        offset += NN_HIDDEN1 * input_dim;  // end of W1 (only after last j)
        // bias
        sum += weights[input_dim * NN_HIDDEN1 + j];
        h1[j] = tanhf(sum);
    }
    // Fix: offset properly
    offset = input_dim * NN_HIDDEN1 + NN_HIDDEN1;

    // Layer 2: hidden1 -> hidden2 (tanh)
    for (int j = 0; j < NN_HIDDEN2; j++) {
        float sum = 0.0f;
        for (int i = 0; i < NN_HIDDEN1; i++) {
            sum += weights[offset + j * NN_HIDDEN1 + i] * h1[i];
        }
        sum += weights[offset + NN_HIDDEN1 * NN_HIDDEN2 + j];
        h2[j] = tanhf(sum);
    }
    offset += NN_HIDDEN1 * NN_HIDDEN2 + NN_HIDDEN2;

    // Layer 3: hidden2 -> output (linear)
    for (int j = 0; j < output_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < NN_HIDDEN2; i++) {
            sum += weights[offset + j * NN_HIDDEN2 + i] * h2[i];
        }
        sum += weights[offset + NN_HIDDEN2 * output_dim + j];
        output[j] = sum;  // linear output
    }
}

// Compute total number of weights for given input/output dims
__host__ __device__ inline int nn_total_weights(int input_dim, int output_dim) {
    return input_dim * NN_HIDDEN1 + NN_HIDDEN1
         + NN_HIDDEN1 * NN_HIDDEN2 + NN_HIDDEN2
         + NN_HIDDEN2 * output_dim + output_dim;
}
