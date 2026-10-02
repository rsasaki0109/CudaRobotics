// Checks include/mppi_reduction.cuh against the legacy serial MPPI kernels and
// prints per-call timings. Exits non-zero on a numerical mismatch.
#include "mppi_reduction.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

// Legacy kernels as they appeared in mppi.cu / diff_mppi.cu before the switch.
__global__ void legacy_softmin(const float* costs, float* weights, int K, float lambda)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float min_cost = FLT_MAX;
    for (int k = 0; k < K; k++) min_cost = fminf(min_cost, costs[k]);
    float sum_w = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = expf(-(costs[k] - min_cost) / lambda);
        weights[k] = w;
        sum_w += w;
    }
    if (sum_w > 0.0f) for (int k = 0; k < K; k++) weights[k] /= sum_w;
}

__global__ void legacy_update(float* nominal, const float* perturbed, const float* weights, int K, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float a = 0.0f, b = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = weights[k];
        a += w * perturbed[k * T * 2 + t * 2 + 0];
        b += w * perturbed[k * T * 2 + t * 2 + 1];
    }
    nominal[t * 2 + 0] = a;
    nominal[t * 2 + 1] = b;
}

template <class Launch>
float time_ms(Launch launch, int iterations = 200)
{
    for (int i = 0; i < 10; ++i) launch();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / iterations;
}

int main()
{
    constexpr int T = 30;
    constexpr int controls = T * 2;
    constexpr float lambda = 10.0f;
    bool ok = true;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> cost_dist(50.0f, 400.0f);
    std::normal_distribution<float> ctrl_dist(0.0f, 1.0f);

    std::printf("K,legacy_softmin_ms,parallel_softmin_ms,softmin_speedup,"
                "legacy_update_ms,parallel_update_ms,update_speedup,max_weight_rel_err,max_nominal_abs_err\n");
    for (int K : {1024, 4096, 16384, 65536}) {
        std::vector<float> costs(K), perturbed(static_cast<size_t>(K) * controls);
        for (float& c : costs) c = cost_dist(rng);
        for (float& u : perturbed) u = ctrl_dist(rng);

        float *d_costs, *d_perturbed, *d_w_ref, *d_w_new, *d_nom_ref, *d_nom_new, *d_min;
        CUDA_CHECK(cudaMalloc(&d_costs, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perturbed, perturbed.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w_ref, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w_new, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nom_ref, controls * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nom_new, controls * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_costs, costs.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_perturbed, perturbed.data(), perturbed.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        float legacy_softmin_ms = time_ms([&] { legacy_softmin<<<1, 1>>>(d_costs, d_w_ref, K, lambda); });
        float parallel_softmin_ms = time_ms([&] {
            cudabot::launch_softmin_weights(d_costs, d_w_new, K, lambda, d_min);
        });
        float legacy_update_ms = time_ms([&] {
            legacy_update<<<(T + 255) / 256, 256>>>(d_nom_ref, d_perturbed, d_w_ref, K, T);
        });
        float parallel_update_ms = time_ms([&] {
            cudabot::launch_weighted_control_update(d_perturbed, d_w_ref, d_nom_new, K, controls);
        });

        std::vector<float> w_ref(K), w_new(K), nom_ref(controls), nom_new(controls);
        float min_gpu = 0.0f;
        CUDA_CHECK(cudaMemcpy(w_ref.data(), d_w_ref, K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(w_new.data(), d_w_new, K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nom_ref.data(), d_nom_ref, controls * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(nom_new.data(), d_nom_new, controls * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&min_gpu, d_min, sizeof(float), cudaMemcpyDeviceToHost));

        const float w_max = *std::max_element(w_ref.begin(), w_ref.end());
        float w_err = 0.0f, nom_err = 0.0f;
        for (int k = 0; k < K; ++k) w_err = std::max(w_err, std::fabs(w_ref[k] - w_new[k]) / w_max);
        for (int j = 0; j < controls; ++j) nom_err = std::max(nom_err, std::fabs(nom_ref[j] - nom_new[j]));
        const float min_ref = *std::min_element(costs.begin(), costs.end());

        std::printf("%d,%.4f,%.4f,%.1f,%.4f,%.4f,%.1f,%.3g,%.3g\n", K,
                    legacy_softmin_ms, parallel_softmin_ms, legacy_softmin_ms / parallel_softmin_ms,
                    legacy_update_ms, parallel_update_ms, legacy_update_ms / parallel_update_ms,
                    w_err, nom_err);
        if (w_err > 1e-4f || nom_err > 1e-4f || min_gpu != min_ref) {
            std::fprintf(stderr, "mismatch at K=%d (min %g vs %g)\n", K, min_gpu, min_ref);
            ok = false;
        }

        cudaFree(d_costs); cudaFree(d_perturbed); cudaFree(d_w_ref); cudaFree(d_w_new);
        cudaFree(d_nom_ref); cudaFree(d_nom_new); cudaFree(d_min);
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
