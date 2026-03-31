#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

namespace cudabot {

// 固定アーキテクチャ MLP
// Layer sizes は template parameter ではなく、コンストラクタで指定
// 重みは1次元 float 配列に flatten して GPU メモリに格納
//
// メモリレイアウト:
//   weights[]: [W0 (in*h0), b0 (h0), W1 (h0*h1), b1 (h1), ..., Wn (h_{n-1}*out), bn (out)]
//
// Forward pass: 1スレッドが1入力ベクトルに対して全レイヤーを計算
// 中間活性化は thread-local array (スタック) に格納

struct MLPConfig {
    int n_layers;          // 隠れ層の数
    int input_dim;
    int output_dim;
    int hidden_dim;        // 全隠れ層同一サイズ（簡略化）
    int total_weights;     // 全重み+バイアスの総数

    // 計算: input_dim*hidden_dim + hidden_dim  (第1層)
    //      + (n_layers-1) * (hidden_dim*hidden_dim + hidden_dim)  (中間層)
    //      + hidden_dim*output_dim + output_dim  (出力層)
};

static constexpr int MLP_MAX_HIDDEN = 128;
static constexpr int MLP_MAX_LAYERS = 8;
static constexpr int MLP_MAX_OUTPUT = 32;

__device__ inline float mlp_activate(float x, int activation) {
    if (activation == 0) return x > 0.0f ? x : 0.0f;
    if (activation == 1) return tanhf(x);
    if (activation == 2) return 1.0f / (1.0f + expf(-x));
    return x;
}

__device__ inline float mlp_activation_grad(float pre, float act, int activation) {
    if (activation == 0) return pre > 0.0f ? 1.0f : 0.0f;
    if (activation == 1) return 1.0f - act * act;
    if (activation == 2) return act * (1.0f - act);
    return 1.0f;
}

// デバイス側 forward pass
// weights: GPU メモリ上の重み配列
// input: 入力ベクトル（thread-local）
// output: 出力ベクトル（thread-local）
// scratch: 中間バッファ（thread-local、サイズ = hidden_dim * 2）
// activation: 0=ReLU, 1=tanh, 2=sigmoid
__device__ inline void mlp_forward(
    const float* weights,
    const float* input, int input_dim,
    float* output, int output_dim,
    int hidden_dim, int n_layers,
    float* scratch,  // サイズ hidden_dim * 2
    int activation = 0  // 0=ReLU
) {
    float* buf_a = scratch;
    float* buf_b = scratch + hidden_dim;

    // Layer 0: input -> hidden
    int offset = 0;
    const float* W = weights + offset;
    offset += input_dim * hidden_dim;
    const float* b = weights + offset;
    offset += hidden_dim;

    for (int j = 0; j < hidden_dim; j++) {
        float sum = b[j];
        for (int i = 0; i < input_dim; i++) {
            sum += W[i * hidden_dim + j] * input[i];
        }
        // activation
        if (activation == 0) sum = sum > 0.0f ? sum : 0.0f;           // ReLU
        else if (activation == 1) sum = tanhf(sum);                    // tanh
        else if (activation == 2) sum = 1.0f / (1.0f + expf(-sum));   // sigmoid
        buf_a[j] = sum;
    }

    // Hidden layers
    float* src = buf_a;
    float* dst = buf_b;
    for (int l = 1; l < n_layers; l++) {
        W = weights + offset;
        offset += hidden_dim * hidden_dim;
        b = weights + offset;
        offset += hidden_dim;

        for (int j = 0; j < hidden_dim; j++) {
            float sum = b[j];
            for (int i = 0; i < hidden_dim; i++) {
                sum += W[i * hidden_dim + j] * src[i];
            }
            if (activation == 0) sum = sum > 0.0f ? sum : 0.0f;
            else if (activation == 1) sum = tanhf(sum);
            else if (activation == 2) sum = 1.0f / (1.0f + expf(-sum));
            dst[j] = sum;
        }
        // swap
        float* tmp = src; src = dst; dst = tmp;
    }

    // Output layer (no activation)
    W = weights + offset;
    offset += hidden_dim * output_dim;
    b = weights + offset;
    // offset += output_dim;

    for (int j = 0; j < output_dim; j++) {
        float sum = b[j];
        for (int i = 0; i < hidden_dim; i++) {
            sum += W[i * output_dim + j] * src[i];
        }
        output[j] = sum;
    }
}

// Forward pass kernel for batch inference
__global__ inline void mlp_forward_batch_kernel(
    const float* weights,
    const float* d_input, float* d_output,
    int input_dim, int output_dim,
    int hidden_dim, int n_layers,
    int N, int activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* my_input = d_input + idx * input_dim;
    float* my_output = d_output + idx * output_dim;

    // Allocate scratch on stack (max hidden_dim = 256)
    float scratch[512];  // hidden_dim * 2, max 256
    mlp_forward(weights, my_input, input_dim, my_output, output_dim,
                hidden_dim, n_layers, scratch, activation);
}

__device__ inline void mlp_forward_perturbed(
    const float* weights,
    const float* input, int input_dim,
    float* output, int output_dim,
    int hidden_dim, int n_layers,
    float* scratch,
    int perturbed_idx, float delta,
    int activation = 0
) {
    float* buf_a = scratch;
    float* buf_b = scratch + hidden_dim;
    int offset = 0;

    for (int j = 0; j < hidden_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            int idx = offset + i * hidden_dim + j;
            float w = weights[idx] + (idx == perturbed_idx ? delta : 0.0f);
            sum += w * input[i];
        }
        int bias_idx = offset + input_dim * hidden_dim + j;
        float bias = weights[bias_idx] + (bias_idx == perturbed_idx ? delta : 0.0f);
        sum += bias;
        if (activation == 0) sum = sum > 0.0f ? sum : 0.0f;
        else if (activation == 1) sum = tanhf(sum);
        else if (activation == 2) sum = 1.0f / (1.0f + expf(-sum));
        buf_a[j] = sum;
    }
    offset += input_dim * hidden_dim + hidden_dim;

    float* src = buf_a;
    float* dst = buf_b;
    for (int l = 1; l < n_layers; l++) {
        for (int j = 0; j < hidden_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                int idx = offset + i * hidden_dim + j;
                float w = weights[idx] + (idx == perturbed_idx ? delta : 0.0f);
                sum += w * src[i];
            }
            int bias_idx = offset + hidden_dim * hidden_dim + j;
            float bias = weights[bias_idx] + (bias_idx == perturbed_idx ? delta : 0.0f);
            sum += bias;
            if (activation == 0) sum = sum > 0.0f ? sum : 0.0f;
            else if (activation == 1) sum = tanhf(sum);
            else if (activation == 2) sum = 1.0f / (1.0f + expf(-sum));
            dst[j] = sum;
        }
        offset += hidden_dim * hidden_dim + hidden_dim;
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    for (int j = 0; j < output_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            int idx = offset + i * output_dim + j;
            float w = weights[idx] + (idx == perturbed_idx ? delta : 0.0f);
            sum += w * src[i];
        }
        int bias_idx = offset + hidden_dim * output_dim + j;
        float bias = weights[bias_idx] + (bias_idx == perturbed_idx ? delta : 0.0f);
        output[j] = sum + bias;
    }
}

// Training kernel: finite-difference gradient per weight against a frozen snapshot.
__global__ inline void mlp_train_gradient_kernel(
    const float* weights,
    float* d_grads,
    const float* d_input, const float* d_target,
    int input_dim, int output_dim,
    int hidden_dim, int n_layers,
    int batch_size, int total_weights
) {
    int wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid >= total_weights) return;

    float eps = 1e-3f;
    float grad = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        const float* inp = d_input + b * input_dim;
        const float* tgt = d_target + b * output_dim;

        float out_plus[16];
        float out_minus[16];
        float scratch[512];

        mlp_forward_perturbed(weights, inp, input_dim, out_plus, output_dim,
                              hidden_dim, n_layers, scratch, wid, eps, 1);
        mlp_forward_perturbed(weights, inp, input_dim, out_minus, output_dim,
                              hidden_dim, n_layers, scratch, wid, -eps, 1);

        float loss_plus = 0.0f, loss_minus = 0.0f;
        for (int j = 0; j < output_dim; j++) {
            float dp = out_plus[j] - tgt[j];
            float dm = out_minus[j] - tgt[j];
            loss_plus += dp * dp;
            loss_minus += dm * dm;
        }
        grad += (loss_plus - loss_minus) / (2.0f * eps);
    }
    d_grads[wid] = grad / batch_size;
}

__global__ inline void mlp_apply_gradients_kernel(
    float* weights, const float* d_grads, int total_weights, float lr
) {
    int wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid >= total_weights) return;
    weights[wid] -= lr * d_grads[wid];
}

__global__ inline void mlp_loss_kernel(
    const float* weights,
    const float* d_input, const float* d_target,
    int input_dim, int output_dim,
    int hidden_dim, int n_layers,
    int batch_size, float* d_loss
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float total_loss = 0.0f;
    float scratch[512];
    float out[16];
    for (int b = 0; b < batch_size; b++) {
        const float* inp = d_input + b * input_dim;
        const float* tgt = d_target + b * output_dim;
        mlp_forward(weights, inp, input_dim, out, output_dim,
                    hidden_dim, n_layers, scratch, 1);
        for (int j = 0; j < output_dim; j++) {
            float d = out[j] - tgt[j];
            total_loss += d * d;
        }
    }
    *d_loss = total_loss / batch_size;
}

__global__ inline void mlp_backprop_batch_kernel(
    const float* weights,
    float* d_grads,
    const float* d_input,
    const float* d_target,
    int input_dim,
    int output_dim,
    int hidden_dim,
    int n_layers,
    int batch_size,
    int activation,
    float* d_loss
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= batch_size) return;

    if (hidden_dim > MLP_MAX_HIDDEN || n_layers > MLP_MAX_LAYERS || output_dim > MLP_MAX_OUTPUT) {
        return;
    }

    const float* input = d_input + sample_idx * input_dim;
    const float* target = d_target + sample_idx * output_dim;
    float inv_batch = 1.0f / batch_size;

    float pre[MLP_MAX_LAYERS][MLP_MAX_HIDDEN];
    float actv[MLP_MAX_LAYERS][MLP_MAX_HIDDEN];
    float out[MLP_MAX_OUTPUT];
    float delta_out[MLP_MAX_OUTPUT];
    float delta_cur[MLP_MAX_HIDDEN];
    float delta_prev[MLP_MAX_HIDDEN];

    int offset = 0;

    // input -> first hidden
    for (int j = 0; j < hidden_dim; j++) {
        float sum = weights[offset + input_dim * hidden_dim + j];
        for (int i = 0; i < input_dim; i++) {
            sum += weights[offset + i * hidden_dim + j] * input[i];
        }
        pre[0][j] = sum;
        actv[0][j] = mlp_activate(sum, activation);
    }
    offset += input_dim * hidden_dim + hidden_dim;

    // hidden -> hidden
    for (int l = 1; l < n_layers; l++) {
        int layer_offset = offset;
        for (int j = 0; j < hidden_dim; j++) {
            float sum = weights[layer_offset + hidden_dim * hidden_dim + j];
            for (int i = 0; i < hidden_dim; i++) {
                sum += weights[layer_offset + i * hidden_dim + j] * actv[l - 1][i];
            }
            pre[l][j] = sum;
            actv[l][j] = mlp_activate(sum, activation);
        }
        offset += hidden_dim * hidden_dim + hidden_dim;
    }

    int output_w_offset = offset;
    int output_b_offset = output_w_offset + hidden_dim * output_dim;

    float sample_loss = 0.0f;
    for (int j = 0; j < output_dim; j++) {
        float sum = weights[output_b_offset + j];
        for (int i = 0; i < hidden_dim; i++) {
            sum += weights[output_w_offset + i * output_dim + j] * actv[n_layers - 1][i];
        }
        out[j] = sum;
        float err = out[j] - target[j];
        delta_out[j] = err;
        sample_loss += err * err;
    }
    atomicAdd(d_loss, sample_loss * inv_batch);

    // output gradients
    for (int j = 0; j < output_dim; j++) {
        atomicAdd(&d_grads[output_b_offset + j], delta_out[j] * inv_batch);
        for (int i = 0; i < hidden_dim; i++) {
            atomicAdd(&d_grads[output_w_offset + i * output_dim + j],
                      delta_out[j] * actv[n_layers - 1][i] * inv_batch);
        }
    }

    // last hidden delta
    for (int i = 0; i < hidden_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < output_dim; j++) {
            sum += weights[output_w_offset + i * output_dim + j] * delta_out[j];
        }
        delta_cur[i] = sum * mlp_activation_grad(pre[n_layers - 1][i], actv[n_layers - 1][i], activation);
    }

    // hidden -> hidden gradients, back to first hidden
    int layer_offset = output_w_offset;
    for (int l = n_layers - 1; l >= 1; l--) {
        layer_offset -= hidden_dim * hidden_dim + hidden_dim;
        int bias_offset = layer_offset + hidden_dim * hidden_dim;
        for (int j = 0; j < hidden_dim; j++) {
            atomicAdd(&d_grads[bias_offset + j], delta_cur[j] * inv_batch);
            for (int i = 0; i < hidden_dim; i++) {
                atomicAdd(&d_grads[layer_offset + i * hidden_dim + j],
                          delta_cur[j] * actv[l - 1][i] * inv_batch);
            }
        }

        for (int i = 0; i < hidden_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden_dim; j++) {
                sum += weights[layer_offset + i * hidden_dim + j] * delta_cur[j];
            }
            delta_prev[i] = sum * mlp_activation_grad(pre[l - 1][i], actv[l - 1][i], activation);
        }
        for (int i = 0; i < hidden_dim; i++) delta_cur[i] = delta_prev[i];
    }

    // input -> first hidden gradients
    int first_bias_offset = input_dim * hidden_dim;
    for (int j = 0; j < hidden_dim; j++) {
        atomicAdd(&d_grads[first_bias_offset + j], delta_cur[j] * inv_batch);
        for (int i = 0; i < input_dim; i++) {
            atomicAdd(&d_grads[i * hidden_dim + j], delta_cur[j] * input[i] * inv_batch);
        }
    }
}

__global__ inline void mlp_backprop_output_grad_kernel(
    const float* weights,
    float* d_grads,
    const float* d_input,
    const float* d_output_grad,
    int input_dim,
    int output_dim,
    int hidden_dim,
    int n_layers,
    int batch_size,
    int activation
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= batch_size) return;

    if (hidden_dim > MLP_MAX_HIDDEN || n_layers > MLP_MAX_LAYERS || output_dim > MLP_MAX_OUTPUT) {
        return;
    }

    const float* input = d_input + sample_idx * input_dim;
    const float* output_grad = d_output_grad + sample_idx * output_dim;
    float inv_batch = 1.0f / batch_size;

    float pre[MLP_MAX_LAYERS][MLP_MAX_HIDDEN];
    float actv[MLP_MAX_LAYERS][MLP_MAX_HIDDEN];
    float delta_out[MLP_MAX_OUTPUT];
    float delta_cur[MLP_MAX_HIDDEN];
    float delta_prev[MLP_MAX_HIDDEN];

    int offset = 0;

    for (int j = 0; j < hidden_dim; j++) {
        float sum = weights[offset + input_dim * hidden_dim + j];
        for (int i = 0; i < input_dim; i++) {
            sum += weights[offset + i * hidden_dim + j] * input[i];
        }
        pre[0][j] = sum;
        actv[0][j] = mlp_activate(sum, activation);
    }
    offset += input_dim * hidden_dim + hidden_dim;

    for (int l = 1; l < n_layers; l++) {
        int layer_offset = offset;
        for (int j = 0; j < hidden_dim; j++) {
            float sum = weights[layer_offset + hidden_dim * hidden_dim + j];
            for (int i = 0; i < hidden_dim; i++) {
                sum += weights[layer_offset + i * hidden_dim + j] * actv[l - 1][i];
            }
            pre[l][j] = sum;
            actv[l][j] = mlp_activate(sum, activation);
        }
        offset += hidden_dim * hidden_dim + hidden_dim;
    }

    int output_w_offset = offset;
    int output_b_offset = output_w_offset + hidden_dim * output_dim;

    for (int j = 0; j < output_dim; j++) {
        delta_out[j] = output_grad[j];
        atomicAdd(&d_grads[output_b_offset + j], delta_out[j] * inv_batch);
        for (int i = 0; i < hidden_dim; i++) {
            atomicAdd(&d_grads[output_w_offset + i * output_dim + j],
                      delta_out[j] * actv[n_layers - 1][i] * inv_batch);
        }
    }

    for (int i = 0; i < hidden_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < output_dim; j++) {
            sum += weights[output_w_offset + i * output_dim + j] * delta_out[j];
        }
        delta_cur[i] = sum * mlp_activation_grad(pre[n_layers - 1][i], actv[n_layers - 1][i], activation);
    }

    int layer_offset = output_w_offset;
    for (int l = n_layers - 1; l >= 1; l--) {
        layer_offset -= hidden_dim * hidden_dim + hidden_dim;
        int bias_offset = layer_offset + hidden_dim * hidden_dim;
        for (int j = 0; j < hidden_dim; j++) {
            atomicAdd(&d_grads[bias_offset + j], delta_cur[j] * inv_batch);
            for (int i = 0; i < hidden_dim; i++) {
                atomicAdd(&d_grads[layer_offset + i * hidden_dim + j],
                          delta_cur[j] * actv[l - 1][i] * inv_batch);
            }
        }

        for (int i = 0; i < hidden_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden_dim; j++) {
                sum += weights[layer_offset + i * hidden_dim + j] * delta_cur[j];
            }
            delta_prev[i] = sum * mlp_activation_grad(pre[l - 1][i], actv[l - 1][i], activation);
        }
        for (int i = 0; i < hidden_dim; i++) delta_cur[i] = delta_prev[i];
    }

    int first_bias_offset = input_dim * hidden_dim;
    for (int j = 0; j < hidden_dim; j++) {
        atomicAdd(&d_grads[first_bias_offset + j], delta_cur[j] * inv_batch);
        for (int i = 0; i < input_dim; i++) {
            atomicAdd(&d_grads[i * hidden_dim + j], delta_cur[j] * input[i] * inv_batch);
        }
    }
}

// Xavier initialization kernel
__global__ inline void mlp_init_weights_kernel(
    float* weights, int total_weights,
    int input_dim, int hidden_dim, int output_dim, int n_layers,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_weights) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Determine which layer this weight belongs to and scale accordingly
    float scale = sqrtf(2.0f / (float)hidden_dim);

    // First layer
    int first_layer_w = input_dim * hidden_dim;
    if (idx < first_layer_w) {
        scale = sqrtf(2.0f / (float)input_dim);
    }

    weights[idx] = curand_normal(&state) * scale;

    // Zero out biases
    int offset = 0;
    // Layer 0
    offset += input_dim * hidden_dim;
    if (idx >= offset && idx < offset + hidden_dim) { weights[idx] = 0.0f; return; }
    offset += hidden_dim;
    // Hidden layers
    for (int l = 1; l < n_layers; l++) {
        offset += hidden_dim * hidden_dim;
        if (idx >= offset && idx < offset + hidden_dim) { weights[idx] = 0.0f; return; }
        offset += hidden_dim;
    }
    // Output layer
    offset += hidden_dim * output_dim;
    if (idx >= offset && idx < offset + output_dim) { weights[idx] = 0.0f; return; }
}

// ホスト側 MLP 管理クラス
class GpuMLP {
public:
    GpuMLP(int input_dim, int hidden_dim, int n_layers, int output_dim)
        : d_weights_(nullptr), d_rng_(nullptr)
    {
        config_.input_dim = input_dim;
        config_.hidden_dim = hidden_dim;
        config_.n_layers = n_layers;
        config_.output_dim = output_dim;

        // Calculate total weights
        config_.total_weights = input_dim * hidden_dim + hidden_dim;  // first layer
        for (int i = 1; i < n_layers; i++) {
            config_.total_weights += hidden_dim * hidden_dim + hidden_dim;  // hidden layers
        }
        config_.total_weights += hidden_dim * output_dim + output_dim;  // output layer

        cudaMalloc(&d_weights_, config_.total_weights * sizeof(float));
    }

    ~GpuMLP() {
        if (d_weights_) cudaFree(d_weights_);
        if (d_rng_) cudaFree(d_rng_);
    }

    void init_random(unsigned long long seed = 42) {
        int threads = 256;
        int blocks = (config_.total_weights + threads - 1) / threads;
        mlp_init_weights_kernel<<<blocks, threads>>>(
            d_weights_, config_.total_weights,
            config_.input_dim, config_.hidden_dim, config_.output_dim,
            config_.n_layers, seed);
        cudaDeviceSynchronize();
    }

    void load_weights(const std::vector<float>& weights) {
        cudaMemcpy(d_weights_, weights.data(), weights.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    std::vector<float> get_weights() const {
        std::vector<float> w(config_.total_weights);
        cudaMemcpy(w.data(), d_weights_, config_.total_weights * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return w;
    }

    float* device_weights() const { return d_weights_; }
    MLPConfig config() const { return config_; }

    // バッチ推論（N 入力を同時に推論）
    void forward_batch(const float* d_input, float* d_output, int N, int activation = 0) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        mlp_forward_batch_kernel<<<blocks, threads>>>(
            d_weights_, d_input, d_output,
            config_.input_dim, config_.output_dim,
            config_.hidden_dim, config_.n_layers,
            N, activation);
        cudaDeviceSynchronize();
    }

    // SGD 学習ステップ
    float train_step(const float* d_input, const float* d_target,
                     int batch_size, float lr = 0.001f) {
        float* d_grads;
        float* d_loss;
        cudaMalloc(&d_grads, config_.total_weights * sizeof(float));
        cudaMalloc(&d_loss, sizeof(float));

        int threads = 256;
        int blocks = (config_.total_weights + threads - 1) / threads;
        mlp_train_gradient_kernel<<<blocks, threads>>>(
            d_weights_, d_grads, d_input, d_target,
            config_.input_dim, config_.output_dim,
            config_.hidden_dim, config_.n_layers,
            batch_size, config_.total_weights);
        mlp_apply_gradients_kernel<<<blocks, threads>>>(
            d_weights_, d_grads, config_.total_weights, lr);
        mlp_loss_kernel<<<1, 1>>>(
            d_weights_, d_input, d_target,
            config_.input_dim, config_.output_dim,
            config_.hidden_dim, config_.n_layers,
            batch_size, d_loss);
        cudaDeviceSynchronize();

        float loss;
        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_grads);
        cudaFree(d_loss);
        return loss;
    }

    float train_step_backprop(const float* d_input, const float* d_target,
                              int batch_size, float lr = 0.001f, int activation = 0) {
        float* d_grads;
        float* d_loss;
        cudaMalloc(&d_grads, config_.total_weights * sizeof(float));
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemset(d_grads, 0, config_.total_weights * sizeof(float));
        cudaMemset(d_loss, 0, sizeof(float));

        int threads = 128;
        int blocks = (batch_size + threads - 1) / threads;
        mlp_backprop_batch_kernel<<<blocks, threads>>>(
            d_weights_, d_grads, d_input, d_target,
            config_.input_dim, config_.output_dim,
            config_.hidden_dim, config_.n_layers,
            batch_size, activation, d_loss);

        int grad_threads = 256;
        int grad_blocks = (config_.total_weights + grad_threads - 1) / grad_threads;
        mlp_apply_gradients_kernel<<<grad_blocks, grad_threads>>>(
            d_weights_, d_grads, config_.total_weights, lr);
        cudaDeviceSynchronize();

        float loss = 0.0f;
        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_grads);
        cudaFree(d_loss);
        return loss;
    }

    void apply_output_grad(const float* d_input, const float* d_output_grad,
                           int batch_size, float lr = 0.001f, int activation = 0) {
        float* d_grads;
        cudaMalloc(&d_grads, config_.total_weights * sizeof(float));
        cudaMemset(d_grads, 0, config_.total_weights * sizeof(float));

        int threads = 128;
        int blocks = (batch_size + threads - 1) / threads;
        mlp_backprop_output_grad_kernel<<<blocks, threads>>>(
            d_weights_, d_grads, d_input, d_output_grad,
            config_.input_dim, config_.output_dim,
            config_.hidden_dim, config_.n_layers,
            batch_size, activation);

        int grad_threads = 256;
        int grad_blocks = (config_.total_weights + grad_threads - 1) / grad_threads;
        mlp_apply_gradients_kernel<<<grad_blocks, grad_threads>>>(
            d_weights_, d_grads, config_.total_weights, lr);
        cudaDeviceSynchronize();
        cudaFree(d_grads);
    }

private:
    MLPConfig config_;
    float* d_weights_;
    curandState* d_rng_;
};

} // namespace cudabot
