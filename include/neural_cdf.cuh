#pragma once
// ============================================================================
// Neural Configuration-Space Distance Field (CDF) for the 7-DOF arm.
//
// Literature-faithful component for the One-Step CDF-MPPI baseline
// (arXiv:2509.00836). The CDF is f_c(q) = min joint-space distance from a
// configuration q to the contact set {q' : robot(q') in collision}. We encode
// it with the repo's flat-weight GpuMLP (gpu_mlp.cuh), mirroring the neural-SDF
// trainer (neural_sdf_nav.cuh) but lifted to 7-D config space with
// workspace-collision-derived ground truth.
//
// Design notes (see paper/diff_mppi_baseline_literature_2026-06.md):
//  - Ground truth f_c(q) ~= max(margin_ws(q), 0) / ||grad_q margin_ws(q)||, a
//    first-order (Newton-step) estimate of the joint-space distance to contact
//    from the workspace signed margin. This is DENSE and correct near contact,
//    unlike nearest-neighbour over a sampled contact set, which is hopeless in
//    7-D (curse of dimensionality: the sampled contact manifold is too sparse,
//    inflating distances so the activation gate never fires). The MLP smooths
//    and accelerates this field. Targets are generated host-side in the .cu.
//  - Activation = tanh (gradient is load-bearing; ReLU finite-difference is
//    noisy).
//  - eikonal ||grad f_c|| = 1 is NOT enforced in training; the gradient is
//    normalized at runtime, so the angle-based cost only ever sees a direction.
//  - Runtime gradient via 8-row batched finite difference (q and q+eps e_j),
//    one forward_batch call per control step.
//
// Joint metric: plain Euclidean in raw radians (no angle wrap); the controller
// clamps q to joint limits (real Franka has hard limits), keeping queries in
// the trained domain.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_mlp.cuh"

namespace cudabot {

static constexpr int NCDF_INPUT_DIM = 7;
static constexpr int NCDF_HIDDEN_DIM = 64;
static constexpr int NCDF_HIDDEN_LAYERS = 3;
static constexpr int NCDF_OUTPUT_DIM = 1;
static constexpr int NCDF_ACTIVATION = 1;  // 1 = tanh (smooth gradients)

// Franka Emika Panda joint limits (rad). Used for sampling + input encoding.
static constexpr float NCDF_Q_LO[NCDF_INPUT_DIM] = {
    -2.8973f, -1.7628f, -2.8973f, -3.0718f, -2.8973f, -0.0175f, -2.8973f};
static constexpr float NCDF_Q_HI[NCDF_INPUT_DIM] = {
     2.8973f,  1.7628f,  2.8973f, -0.0698f,  2.8973f,  3.7525f,  2.8973f};

// Encode a raw config to [-1, 1] per joint (mirror encode_sdf_input).
// Uses a function-local copy of the limits so runtime indexing is device-safe.
__host__ __device__ inline void encode_cdf_input(
    const float q[NCDF_INPUT_DIM], float out[NCDF_INPUT_DIM])
{
    const float lo[NCDF_INPUT_DIM] = {-2.8973f,-1.7628f,-2.8973f,-3.0718f,-2.8973f,-0.0175f,-2.8973f};
    const float hi[NCDF_INPUT_DIM] = { 2.8973f, 1.7628f, 2.8973f,-0.0698f, 2.8973f, 3.7525f, 2.8973f};
    for (int j = 0; j < NCDF_INPUT_DIM; j++) {
        out[j] = 2.0f * (q[j] - lo[j]) / (hi[j] - lo[j]) - 1.0f;
    }
}

// ============================================================================
// NeuralCdf: trains MLP(q) -> normalized f_c and serves value+gradient.
// ============================================================================
class NeuralCdf {
public:
    NeuralCdf()
        : mlp_(NCDF_INPUT_DIM, NCDF_HIDDEN_DIM, NCDF_HIDDEN_LAYERS, NCDF_OUTPUT_DIM),
          scale_(1.0f)
    {
        cudaMalloc(&d_in8_, 8 * NCDF_INPUT_DIM * sizeof(float));
        cudaMalloc(&d_out8_, 8 * NCDF_OUTPUT_DIM * sizeof(float));
    }
    ~NeuralCdf() {
        if (d_in8_) cudaFree(d_in8_);
        if (d_out8_) cudaFree(d_out8_);
    }

    GpuMLP& mlp() { return mlp_; }
    float scale() const { return scale_; }

    // Train on raw queries (Nq*7) with raw target distances (Nq). Holds out a
    // fraction for RMSE reporting. Returns held-out RMSE (raw distance units).
    float train(const std::vector<float>& query_raw,
                const std::vector<float>& target_raw,
                int epochs = 1200, int batch_size = 256, float lr = 1.0e-3f,
                float holdout_frac = 0.15f, unsigned int seed = 17,
                bool verbose = true)
    {
        int N = static_cast<int>(target_raw.size());
        int Nh = static_cast<int>(N * (1.0f - holdout_frac));
        // Target normalization scale = mean absolute target (>= 0.25). Targets
        // are signed margins, so use mean-abs (not signed mean).
        double mean = 0.0;
        for (int i = 0; i < Nh; i++) mean += std::fabs(target_raw[i]);
        mean /= std::max(1, Nh);
        scale_ = std::max(0.25f, static_cast<float>(mean));

        // Encode inputs + normalize targets (training split only).
        std::vector<float> enc(Nh * NCDF_INPUT_DIM);
        std::vector<float> tgt(Nh);
        for (int i = 0; i < Nh; i++) {
            encode_cdf_input(&query_raw[i * NCDF_INPUT_DIM], &enc[i * NCDF_INPUT_DIM]);
            tgt[i] = target_raw[i] / scale_;
        }

        mlp_.init_random(seed);

        float* d_in = nullptr;
        float* d_tg = nullptr;
        cudaMalloc(&d_in, batch_size * NCDF_INPUT_DIM * sizeof(float));
        cudaMalloc(&d_tg, batch_size * sizeof(float));

        std::vector<int> idx(Nh);
        for (int i = 0; i < Nh; i++) idx[i] = i;
        unsigned int rng = seed;
        auto next_rand = [&rng]() { rng = rng * 1664525u + 1013904223u; return rng; };

        std::vector<float> bin(batch_size * NCDF_INPUT_DIM);
        std::vector<float> btg(batch_size);
        float last_loss = 0.0f;
        for (int ep = 0; ep < epochs; ep++) {
            // Fisher-Yates shuffle (Math.random-free, deterministic).
            for (int i = Nh - 1; i > 0; i--) {
                int j = next_rand() % (i + 1);
                std::swap(idx[i], idx[j]);
            }
            for (int i = 0; i < batch_size; i++) {
                int s = idx[i % Nh];
                for (int k = 0; k < NCDF_INPUT_DIM; k++)
                    bin[i * NCDF_INPUT_DIM + k] = enc[s * NCDF_INPUT_DIM + k];
                btg[i] = tgt[s];
            }
            cudaMemcpy(d_in, bin.data(), bin.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tg, btg.data(), btg.size() * sizeof(float), cudaMemcpyHostToDevice);
            float lr_ep = lr * (1.0f - 0.5f * ep / std::max(1, epochs));  // linear decay to 0.5x
            last_loss = mlp_.train_step_backprop(d_in, d_tg, batch_size, lr_ep, NCDF_ACTIVATION);
            if (verbose && (ep % 200 == 0 || ep == epochs - 1))
                printf("  [cdf train] epoch %4d  loss=%.5f  lr=%.2e\n", ep, last_loss, lr_ep);
        }
        cudaFree(d_in);
        cudaFree(d_tg);

        // Held-out RMSE in raw distance units.
        float rmse = holdout_rmse(query_raw, target_raw, Nh, N);
        if (verbose) printf("  [cdf train] held-out RMSE = %.4f rad (scale=%.3f)\n", rmse, scale_);
        return rmse;
    }

    // Single-config value (raw distance).
    float value(const float q[NCDF_INPUT_DIM]) {
        float fc; float grad[NCDF_INPUT_DIM];
        value_and_grad(q, fc, grad);
        return fc;
    }

    // Value + raw-space gradient via 8-row batched forward FD. One forward_batch.
    // eps is a smoothing secant width (rad): too small drowns the MLP gradient
    // in training noise; ~0.05 rad matches the field's near-contact ramp scale.
    void value_and_grad(const float q[NCDF_INPUT_DIM], float& fc, float grad[NCDF_INPUT_DIM]) {
        const float eps = 5.0e-2f;
        float rows[8 * NCDF_INPUT_DIM];
        float enc[NCDF_INPUT_DIM];
        // row 0 = q
        encode_cdf_input(q, enc);
        for (int k = 0; k < NCDF_INPUT_DIM; k++) rows[k] = enc[k];
        // rows 1..7 = q + eps e_j
        for (int j = 0; j < NCDF_INPUT_DIM; j++) {
            float qp[NCDF_INPUT_DIM];
            for (int k = 0; k < NCDF_INPUT_DIM; k++) qp[k] = q[k];
            qp[j] += eps;
            encode_cdf_input(qp, enc);
            for (int k = 0; k < NCDF_INPUT_DIM; k++) rows[(j + 1) * NCDF_INPUT_DIM + k] = enc[k];
        }
        cudaMemcpy(d_in8_, rows, 8 * NCDF_INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
        mlp_.forward_batch(d_in8_, d_out8_, 8, NCDF_ACTIVATION);
        float out[8];
        cudaMemcpy(out, d_out8_, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fc = out[0] * scale_;
        for (int j = 0; j < NCDF_INPUT_DIM; j++)
            grad[j] = (out[j + 1] - out[0]) * scale_ / eps;
    }

private:
    float holdout_rmse(const std::vector<float>& query_raw,
                       const std::vector<float>& target_raw, int from, int to) {
        double sse = 0.0; int n = 0;
        for (int i = from; i < to; i++) {
            float pred = value(&query_raw[i * NCDF_INPUT_DIM]);
            float e = pred - target_raw[i];
            sse += e * e; n++;
        }
        return n > 0 ? static_cast<float>(std::sqrt(sse / n)) : 0.0f;
    }

    GpuMLP mlp_;
    float scale_;
    float* d_in8_ = nullptr;
    float* d_out8_ = nullptr;
};

}  // namespace cudabot
