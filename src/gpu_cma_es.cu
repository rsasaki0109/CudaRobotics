// gpu_cma_es.cu
//
// GPU-accelerated CMA-ES style black-box optimisation.
//
// The covariance matrix adaptation stays on the host because the matrix is
// tiny (10x10).  The expensive part for robotics-style black-box search is
// evaluating many candidate controllers / calibration vectors / planner
// parameters, so each CUDA thread evaluates one candidate.  This demo runs
// three independent benchmark objectives in parallel: Rosenbrock, Rastrigin,
// and Ackley.
//
// Output: gif/gpu_cma_es.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

constexpr int DIM = 10;
constexpr int N_TASKS = 3;
constexpr int POP = 32768;
constexpr int MU = 256;
constexpr int N_GEN = 72;
constexpr int N_BENCH = 25;
constexpr int CPU_BENCH_REPEAT = 3;
constexpr float X_MIN = -5.5f;
constexpr float X_MAX = 5.5f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 12;

enum TaskId {
    TASK_ROSENBROCK = 0,
    TASK_RASTRIGIN = 1,
    TASK_ACKLEY = 2
};

static const char* task_name(int task) {
    switch (task) {
        case TASK_ROSENBROCK: return "Rosenbrock";
        case TASK_RASTRIGIN: return "Rastrigin";
        case TASK_ACKLEY: return "Ackley";
        default: return "unknown";
    }
}

static inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

__host__ __device__ static inline float objective_value(int task, const float* x) {
    if (task == TASK_ROSENBROCK) {
        float sum = 0.0f;
        for (int i = 0; i < DIM - 1; i++) {
            float a = x[i];
            float b = x[i + 1];
            float t1 = b - a * a;
            float t2 = 1.0f - a;
            sum += 100.0f * t1 * t1 + t2 * t2;
        }
        return sum;
    }
    if (task == TASK_RASTRIGIN) {
        float sum = 10.0f * DIM;
        for (int i = 0; i < DIM; i++) {
            float xi = x[i];
            sum += xi * xi - 10.0f * cosf(2.0f * PI_F * xi);
        }
        return sum;
    }

    float sum_sq = 0.0f;
    float sum_cos = 0.0f;
    for (int i = 0; i < DIM; i++) {
        float xi = x[i];
        sum_sq += xi * xi;
        sum_cos += cosf(2.0f * PI_F * xi);
    }
    float a = -20.0f * expf(-0.2f * sqrtf(sum_sq / DIM));
    float b = -expf(sum_cos / DIM);
    return a + b + 20.0f + 2.71828182845904523536f;
}

__global__ void evaluate_kernel(const float* __restrict__ candidates,
                                float* __restrict__ fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = N_TASKS * POP;
    if (idx >= n) return;
    int task = idx / POP;
    const float* x = candidates + idx * DIM;
    fitness[idx] = objective_value(task, x);
}

struct CmaState {
    std::array<float, DIM> mean;
    std::array<float, DIM> best_x;
    std::array<float, DIM * DIM> cov;
    std::array<float, DIM * DIM> chol;
    std::vector<float> history;
    float sigma = 1.0f;
    float best = 1.0e30f;
};

struct BenchResult {
    float gpu_ms = 0.0f;
    double cpu_ms = 0.0;
    double speedup = 0.0;
    double max_rel_error = 0.0;
};

static void init_state(CmaState& st, int task) {
    st.cov.fill(0.0f);
    st.chol.fill(0.0f);
    for (int i = 0; i < DIM; i++) {
        st.cov[i * DIM + i] = 1.0f;
        st.chol[i * DIM + i] = 1.0f;
    }

    for (int i = 0; i < DIM; i++) {
        if (task == TASK_ROSENBROCK) {
            st.mean[i] = (i % 2 == 0) ? -2.4f : 2.2f;
            st.sigma = 1.35f;
        } else if (task == TASK_RASTRIGIN) {
            st.mean[i] = (i % 2 == 0) ? 3.9f : -3.6f;
            st.sigma = 2.15f;
        } else {
            st.mean[i] = (i % 2 == 0) ? 4.2f : -4.1f;
            st.sigma = 2.25f;
        }
        st.best_x[i] = st.mean[i];
    }
    st.history.clear();
}

static bool cholesky_lower(const std::array<float, DIM * DIM>& a_in,
                           std::array<float, DIM * DIM>& l_out) {
    l_out.fill(0.0f);
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = a_in[i * DIM + j];
            for (int k = 0; k < j; k++) {
                sum -= (double)l_out[i * DIM + k] * l_out[j * DIM + k];
            }
            if (i == j) {
                if (sum <= 1.0e-9) return false;
                l_out[i * DIM + j] = (float)std::sqrt(sum);
            } else {
                float denom = l_out[j * DIM + j];
                if (std::fabs(denom) < 1.0e-12f) return false;
                l_out[i * DIM + j] = (float)(sum / denom);
            }
        }
    }
    return true;
}

static void refresh_cholesky(CmaState& st) {
    for (int attempt = 0; attempt < 5; attempt++) {
        if (cholesky_lower(st.cov, st.chol)) return;
        float jitter = 1.0e-4f * std::pow(10.0f, (float)attempt);
        for (int i = 0; i < DIM; i++) st.cov[i * DIM + i] += jitter;
    }
    st.cov.fill(0.0f);
    st.chol.fill(0.0f);
    for (int i = 0; i < DIM; i++) {
        st.cov[i * DIM + i] = 1.0f;
        st.chol[i * DIM + i] = 1.0f;
    }
}

static std::vector<float> make_weights() {
    std::vector<float> w(MU);
    float sum = 0.0f;
    for (int i = 0; i < MU; i++) {
        w[i] = std::log((float)MU + 0.5f) - std::log((float)(i + 1));
        sum += w[i];
    }
    for (int i = 0; i < MU; i++) w[i] /= sum;
    return w;
}

static void sample_population(const std::vector<CmaState>& states,
                              std::mt19937& rng,
                              std::vector<float>& candidates) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    candidates.resize(N_TASKS * POP * DIM);
    float z[DIM];
    float y[DIM];

    for (int task = 0; task < N_TASKS; task++) {
        const CmaState& st = states[task];
        for (int p = 0; p < POP; p++) {
            for (int d = 0; d < DIM; d++) z[d] = normal(rng);
            for (int i = 0; i < DIM; i++) {
                float v = 0.0f;
                for (int j = 0; j <= i; j++) v += st.chol[i * DIM + j] * z[j];
                y[i] = v;
            }
            float* dst = candidates.data() + (task * POP + p) * DIM;
            for (int d = 0; d < DIM; d++) {
                dst[d] = clampf(st.mean[d] + st.sigma * y[d], X_MIN, X_MAX);
            }
        }
    }
}

static float evaluate_gpu(const std::vector<float>& candidates,
                          std::vector<float>& fitness,
                          float* d_candidates,
                          float* d_fitness) {
    const size_t cand_bytes = candidates.size() * sizeof(float);
    const size_t fit_bytes = N_TASKS * POP * sizeof(float);
    fitness.resize(N_TASKS * POP);

    CUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(), cand_bytes,
                          cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    evaluate_kernel<<<(N_TASKS * POP + 255) / 256, 256>>>(d_candidates, d_fitness);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(fitness.data(), d_fitness, fit_bytes, cudaMemcpyDeviceToHost));
    return ms;
}

static BenchResult benchmark_eval(const std::vector<float>& candidates,
                                  float* d_candidates,
                                  float* d_fitness) {
    BenchResult br;
    std::vector<float> gpu_fit(N_TASKS * POP);
    std::vector<float> cpu_fit(N_TASKS * POP);
    const size_t cand_bytes = candidates.size() * sizeof(float);
    const size_t fit_bytes = gpu_fit.size() * sizeof(float);

    CUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(), cand_bytes,
                          cudaMemcpyHostToDevice));
    evaluate_kernel<<<(N_TASKS * POP + 255) / 256, 256>>>(d_candidates, d_fitness);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < N_BENCH; i++) {
        evaluate_kernel<<<(N_TASKS * POP + 255) / 256, 256>>>(d_candidates, d_fitness);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float total_gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaGetLastError());
    br.gpu_ms = total_gpu_ms / N_BENCH;
    CUDA_CHECK(cudaMemcpy(gpu_fit.data(), d_fitness, fit_bytes, cudaMemcpyDeviceToHost));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < CPU_BENCH_REPEAT; r++) {
        for (int idx = 0; idx < N_TASKS * POP; idx++) {
            int task = idx / POP;
            cpu_fit[idx] = objective_value(task, candidates.data() + idx * DIM);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    br.cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
              / CPU_BENCH_REPEAT;
    br.speedup = br.cpu_ms / std::max(1.0e-9, (double)br.gpu_ms);

    double max_rel = 0.0;
    for (int idx = 0; idx < N_TASKS * POP; idx++) {
        double denom = std::max(1.0, std::fabs((double)cpu_fit[idx]));
        double rel = std::fabs((double)cpu_fit[idx] - (double)gpu_fit[idx]) / denom;
        max_rel = std::max(max_rel, rel);
    }
    br.max_rel_error = max_rel;
    return br;
}

static void update_state(CmaState& st,
                         int task,
                         const std::vector<float>& candidates,
                         const std::vector<float>& fitness,
                         const std::vector<float>& weights,
                         std::vector<int>& order) {
    order.resize(POP);
    std::iota(order.begin(), order.end(), 0);
    const int base = task * POP;
    const float* fit = fitness.data() + base;
    std::nth_element(order.begin(), order.begin() + MU, order.end(),
                     [fit](int a, int b) { return fit[a] < fit[b]; });
    std::sort(order.begin(), order.begin() + MU,
              [fit](int a, int b) { return fit[a] < fit[b]; });

    std::array<float, DIM> old_mean = st.mean;
    std::array<float, DIM> new_mean;
    new_mean.fill(0.0f);
    for (int k = 0; k < MU; k++) {
        const float* x = candidates.data() + (base + order[k]) * DIM;
        for (int d = 0; d < DIM; d++) new_mean[d] += weights[k] * x[d];
    }

    std::array<float, DIM * DIM> elite_cov;
    elite_cov.fill(0.0f);
    float inv_sigma = 1.0f / std::max(1.0e-4f, st.sigma);
    for (int k = 0; k < MU; k++) {
        const float* x = candidates.data() + (base + order[k]) * DIM;
        float y[DIM];
        for (int d = 0; d < DIM; d++) y[d] = (x[d] - old_mean[d]) * inv_sigma;
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j <= i; j++) {
                elite_cov[i * DIM + j] += weights[k] * y[i] * y[j];
            }
        }
    }

    constexpr float COV_LR = 0.32f;
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j <= i; j++) {
            float v = (1.0f - COV_LR) * st.cov[i * DIM + j]
                    + COV_LR * elite_cov[i * DIM + j];
            if (i == j) v = std::max(v, 1.0e-4f);
            st.cov[i * DIM + j] = v;
            st.cov[j * DIM + i] = v;
        }
    }
    st.mean = new_mean;

    float best_now = fit[order[0]];
    if (best_now < st.best) {
        st.best = best_now;
        const float* x = candidates.data() + (base + order[0]) * DIM;
        for (int d = 0; d < DIM; d++) st.best_x[d] = x[d];
        st.sigma *= 0.955f;
    } else {
        st.sigma *= 1.035f;
    }
    st.sigma = clampf(st.sigma, 0.018f, 3.0f);
    st.history.push_back(st.best);
    refresh_cholesky(st);
}

static cv::Point project_xy(float x, float y, const cv::Rect& r) {
    float u = (x - X_MIN) / (X_MAX - X_MIN);
    float v = (y - X_MIN) / (X_MAX - X_MIN);
    u = clampf(u, 0.0f, 1.0f);
    v = clampf(v, 0.0f, 1.0f);
    int px = r.x + (int)(u * r.width);
    int py = r.y + r.height - (int)(v * r.height);
    return cv::Point(px, py);
}

static cv::Scalar task_color(int task) {
    if (task == TASK_ROSENBROCK) return cv::Scalar(90, 220, 255);
    if (task == TASK_RASTRIGIN) return cv::Scalar(255, 165, 80);
    return cv::Scalar(120, 230, 125);
}

static void draw_cov_ellipse(cv::Mat& img, const CmaState& st,
                             const cv::Rect& r, const cv::Scalar& color) {
    float a = st.sigma * st.sigma * st.cov[0 * DIM + 0];
    float b = st.sigma * st.sigma * st.cov[0 * DIM + 1];
    float c = st.sigma * st.sigma * st.cov[1 * DIM + 1];
    float tr = a + c;
    float det = a * c - b * b;
    float disc = std::sqrt(std::max(0.0f, tr * tr * 0.25f - det));
    float l1 = std::max(1.0e-8f, tr * 0.5f + disc);
    float l2 = std::max(1.0e-8f, tr * 0.5f - disc);
    float angle = 0.5f * std::atan2(2.0f * b, a - c);
    float sx = (float)r.width / (X_MAX - X_MIN);
    float sy = (float)r.height / (X_MAX - X_MIN);
    cv::Point center = project_xy(st.mean[0], st.mean[1], r);
    cv::Size axes((int)(2.0f * std::sqrt(l1) * sx),
                  (int)(2.0f * std::sqrt(l2) * sy));
    axes.width = std::max(3, std::min(axes.width, r.width));
    axes.height = std::max(3, std::min(axes.height, r.height));
    cv::ellipse(img, center, axes, angle * 180.0 / PI_F, 0.0, 360.0,
                color, 1, cv::LINE_AA);
}

static float log_metric(float x) {
    return std::log10(std::max(1.0e-8f, x));
}

static void draw_history(cv::Mat& img,
                         const std::vector<CmaState>& states,
                         const cv::Rect& r) {
    cv::rectangle(img, r, cv::Scalar(30, 30, 34), -1);
    cv::rectangle(img, r, cv::Scalar(70, 70, 78), 1);
    cv::putText(img, "best objective, log10 scale",
                cv::Point(r.x + 10, r.y + 22), cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(225, 225, 225), 1, cv::LINE_AA);

    const float y_min = -8.0f;
    const float y_max = 6.0f;
    for (int grid = -6; grid <= 4; grid += 2) {
        float t = ((float)grid - y_min) / (y_max - y_min);
        int y = r.y + r.height - 20 - (int)(t * (r.height - 42));
        cv::line(img, cv::Point(r.x + 44, y), cv::Point(r.x + r.width - 12, y),
                 cv::Scalar(46, 46, 52), 1);
        cv::putText(img, cv::format("%d", grid), cv::Point(r.x + 12, y + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
    }

    for (int task = 0; task < N_TASKS; task++) {
        const auto& h = states[task].history;
        if (h.size() < 2) continue;
        std::vector<cv::Point> pts;
        for (size_t i = 0; i < h.size(); i++) {
            float x01 = (float)i / std::max<size_t>(1, h.size() - 1);
            float y01 = (log_metric(h[i]) - y_min) / (y_max - y_min);
            y01 = clampf(y01, 0.0f, 1.0f);
            int x = r.x + 44 + (int)(x01 * (r.width - 60));
            int y = r.y + r.height - 20 - (int)(y01 * (r.height - 42));
            pts.emplace_back(x, y);
        }
        cv::polylines(img, pts, false, task_color(task), 2, cv::LINE_AA);
        cv::putText(img, cv::format("%s %.3g", task_name(task), states[task].best),
                    cv::Point(r.x + 210 + 210 * task, r.y + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, task_color(task), 1, cv::LINE_AA);
    }
}

static cv::Mat draw_frame(const std::vector<CmaState>& states,
                          const std::vector<float>& candidates,
                          int gen,
                          float eval_ms,
                          const BenchResult& bench) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 18, 22));
    cv::putText(img, cv::format("GPU CMA-ES  generation %02d / %d", gen, N_GEN),
                cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    cv::putText(img,
                cv::format("%d objectives x %d candidates x %dD   GPU eval %.3f ms/gen   CPU eval %.3f ms   %.1fx",
                           N_TASKS, POP, DIM, eval_ms, bench.cpu_ms, bench.speedup),
                cv::Point(18, 54), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(210, 210, 215), 1, cv::LINE_AA);
    cv::putText(img, cv::format("max CPU/GPU relative fitness error %.2e",
                                bench.max_rel_error),
                cv::Point(680, 28), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(175, 205, 255), 1, cv::LINE_AA);

    const int panel_y = 78;
    const int panel_h = 290;
    const int panel_gap = 14;
    const int panel_w = (PANEL_W - 2 * 18 - 2 * panel_gap) / 3;
    for (int task = 0; task < N_TASKS; task++) {
        cv::Rect r(18 + task * (panel_w + panel_gap), panel_y, panel_w, panel_h);
        cv::rectangle(img, r, cv::Scalar(28, 28, 32), -1);
        cv::rectangle(img, r, cv::Scalar(70, 70, 78), 1);
        cv::line(img, project_xy(0.0f, X_MIN, r), project_xy(0.0f, X_MAX, r),
                 cv::Scalar(45, 45, 50), 1);
        cv::line(img, project_xy(X_MIN, 0.0f, r), project_xy(X_MAX, 0.0f, r),
                 cv::Scalar(45, 45, 50), 1);

        int stride = std::max(1, POP / 900);
        cv::Scalar base = task_color(task);
        for (int p = 0; p < POP; p += stride) {
            const float* x = candidates.data() + (task * POP + p) * DIM;
            cv::Scalar c(base[0] * 0.45, base[1] * 0.45, base[2] * 0.45);
            cv::circle(img, project_xy(x[0], x[1], r), 1, c, -1, cv::LINE_AA);
        }
        draw_cov_ellipse(img, states[task], r, base);
        cv::circle(img, project_xy(states[task].mean[0], states[task].mean[1], r),
                   5, cv::Scalar(245, 245, 245), -1, cv::LINE_AA);
        cv::drawMarker(img, project_xy(states[task].best_x[0], states[task].best_x[1], r),
                       base, cv::MARKER_CROSS, 15, 2, cv::LINE_AA);
        float opt = (task == TASK_ROSENBROCK) ? 1.0f : 0.0f;
        cv::drawMarker(img, project_xy(opt, opt, r), cv::Scalar(255, 255, 255),
                       cv::MARKER_TILTED_CROSS, 12, 1, cv::LINE_AA);

        cv::putText(img, task_name(task), cv::Point(r.x + 10, r.y + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, base, 1, cv::LINE_AA);
        cv::putText(img, cv::format("best %.4g  sigma %.3f",
                                    states[task].best, states[task].sigma),
                    cv::Point(r.x + 10, r.y + panel_h - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(220, 220, 225),
                    1, cv::LINE_AA);
    }

    draw_history(img, states, cv::Rect(18, 390, PANEL_W - 36, 206));
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<CmaState> states(N_TASKS);
    for (int task = 0; task < N_TASKS; task++) init_state(states[task], task);
    const std::vector<float> weights = make_weights();

    std::vector<float> candidates;
    std::vector<float> fitness;
    float* d_candidates = nullptr;
    float* d_fitness = nullptr;
    CUDA_CHECK(cudaMalloc(&d_candidates, N_TASKS * POP * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fitness, N_TASKS * POP * sizeof(float)));

    std::mt19937 rng(24052026);
    sample_population(states, rng, candidates);
    BenchResult bench = benchmark_eval(candidates, d_candidates, d_fitness);
    std::printf("GPU CMA-ES population eval: %d objectives x %d candidates x %dD\n",
                N_TASKS, POP, DIM);
    std::printf("GPU objective eval %.3f ms, CPU %.3f ms, speedup %.1fx, max rel error %.3e\n",
                bench.gpu_ms, bench.cpu_ms, bench.speedup, bench.max_rel_error);

    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_cma_es.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));

    double total_eval_ms = 0.0;
    std::vector<int> order;
    for (int gen = 0; gen < N_GEN; gen++) {
        if (gen > 0) sample_population(states, rng, candidates);
        float eval_ms = evaluate_gpu(candidates, fitness, d_candidates, d_fitness);
        total_eval_ms += eval_ms;

        for (int task = 0; task < N_TASKS; task++) {
            update_state(states[task], task, candidates, fitness, weights, order);
        }

        if (gen % 6 == 0 || gen == N_GEN - 1) {
            std::printf("gen %02d  eval %.3f ms  Rosen %.4g  Rastr %.4g  Ackley %.4g\n",
                        gen, eval_ms,
                        states[TASK_ROSENBROCK].best,
                        states[TASK_RASTRIGIN].best,
                        states[TASK_ACKLEY].best);
        }

        cv::Mat frame = draw_frame(states, candidates, gen + 1, eval_ms, bench);
        video.write(frame);
    }

    for (int hold = 0; hold < 18; hold++) {
        cv::Mat frame = draw_frame(states, candidates, N_GEN, (float)(total_eval_ms / N_GEN), bench);
        video.write(frame);
    }
    video.release();

    std::printf("Average GPU objective eval %.3f ms/generation\n",
                total_eval_ms / N_GEN);
    for (int task = 0; task < N_TASKS; task++) {
        std::printf("%s final best %.6g  first dims [%.3f %.3f %.3f]\n",
                    task_name(task), states[task].best,
                    states[task].best_x[0], states[task].best_x[1],
                    states[task].best_x[2]);
    }

    cudabot::avi_to_gif("gif/gpu_cma_es.avi", "gif/gpu_cma_es.gif", 8, 640);
    std::printf("GIF saved to gif/gpu_cma_es.gif\n");

    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_fitness));
    return 0;
}
