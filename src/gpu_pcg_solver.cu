// gpu_pcg_solver.cu
//
// Generic CUDA Jacobi-PCG solver demo for sparse SPD systems.
//
// The matrix is a CSR 2D Poisson-like grid operator with a positive
// spatially varying diagonal term. A known smooth field x_true is used to
// build b = A x_true, then CPU and GPU PCG solve A x = b from x=0.
//
// This is intentionally a reusable solver-shaped demo: CSR SpMV, Jacobi
// preconditioning, BLAS-like vector updates, residual tracking, and a visual
// convergence plot. The same primitives can back future BA / pose-graph
// sparse normal-equation solvers.
//
// Output: gif/gpu_pcg_solver.gif

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_blas.cuh"
#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

using blas::axpy_kernel;
using blas::copy_kernel;
using blas::dot_kernel;
using blas::xpay_kernel;

constexpr int GRID = 512;
constexpr int N = GRID * GRID;
constexpr int MAX_ITERS = 120;
constexpr int SNAP_STRIDE = 4;
constexpr float REL_TOL = 1.0e-7f;
constexpr int THREADS = 256;
constexpr int DOT_BLOCKS = 256;
constexpr int PANEL_W = 960;
constexpr int PANEL_H = 620;
constexpr int VIDEO_FPS = 10;

struct CsrMatrix {
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<float> values;
    std::vector<float> diag_inv;
};

struct SolveStats {
    int iterations = 0;
    float rel_residual = 1.0f;
    float rmse = 0.0f;
    double ms = 0.0;
};

__global__ void spmv_csr_kernel(int n,
                                const int* __restrict__ row_ptr,
                                const int* __restrict__ col_ind,
                                const float* __restrict__ values,
                                const float* __restrict__ x,
                                float* __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float sum = 0.0f;
    int begin = row_ptr[row];
    int end = row_ptr[row + 1];
    for (int k = begin; k < end; k++) sum += values[k] * x[col_ind[k]];
    y[row] = sum;
}

__global__ void init_pcg_kernel(int n,
                                const float* __restrict__ b,
                                const float* __restrict__ diag_inv,
                                float* __restrict__ x,
                                float* __restrict__ r,
                                float* __restrict__ z,
                                float* __restrict__ p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 0.0f;
    r[i] = b[i];
    z[i] = diag_inv[i] * r[i];
    p[i] = z[i];
}

__global__ void apply_jacobi_kernel(int n,
                                    const float* __restrict__ diag_inv,
                                    const float* __restrict__ r,
                                    float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = diag_inv[i] * r[i];
}

static inline float clampf(float x, float lo, float hi) {
    return std::min(hi, std::max(lo, x));
}

static CsrMatrix build_grid_matrix() {
    CsrMatrix A;
    A.row_ptr.reserve(N + 1);
    A.col_ind.reserve(N * 5);
    A.values.reserve(N * 5);
    A.diag_inv.resize(N);
    A.row_ptr.push_back(0);

    for (int y = 0; y < GRID; y++) {
        for (int x = 0; x < GRID; x++) {
            int row = y * GRID + x;
            float fx = static_cast<float>(x) / (GRID - 1);
            float fy = static_cast<float>(y) / (GRID - 1);
            float diagonal = 4.0f
                           + 0.75f
                           + 0.20f * std::sin(6.0f * fx)
                           + 0.15f * std::cos(5.0f * fy);
            if (x > 0) {
                A.col_ind.push_back(row - 1);
                A.values.push_back(-1.0f);
            }
            if (y > 0) {
                A.col_ind.push_back(row - GRID);
                A.values.push_back(-1.0f);
            }
            A.col_ind.push_back(row);
            A.values.push_back(diagonal);
            if (x + 1 < GRID) {
                A.col_ind.push_back(row + 1);
                A.values.push_back(-1.0f);
            }
            if (y + 1 < GRID) {
                A.col_ind.push_back(row + GRID);
                A.values.push_back(-1.0f);
            }
            A.diag_inv[row] = 1.0f / diagonal;
            A.row_ptr.push_back(static_cast<int>(A.col_ind.size()));
        }
    }
    return A;
}

static std::vector<float> make_true_field() {
    std::vector<float> x(N);
    for (int y = 0; y < GRID; y++) {
        for (int xi = 0; xi < GRID; xi++) {
            float fx = static_cast<float>(xi) / (GRID - 1);
            float fy = static_cast<float>(y) / (GRID - 1);
            float sx = std::sin(2.0f * static_cast<float>(M_PI) * fx);
            float sy = std::sin(3.0f * static_cast<float>(M_PI) * fy);
            float dx = fx - 0.62f;
            float dy = fy - 0.38f;
            float blob = std::exp(-(dx * dx + dy * dy) / 0.014f);
            float ridge = std::exp(-std::pow(fx + 0.45f * fy - 0.72f, 2.0f) / 0.018f);
            x[y * GRID + xi] = 0.75f * sx * sy + 0.85f * blob - 0.38f * ridge;
        }
    }
    return x;
}

static void csr_matvec_cpu(const CsrMatrix& A,
                           const std::vector<float>& x,
                           std::vector<float>& y) {
    y.assign(N, 0.0f);
    for (int row = 0; row < N; row++) {
        float sum = 0.0f;
        for (int k = A.row_ptr[row]; k < A.row_ptr[row + 1]; k++) {
            sum += A.values[k] * x[A.col_ind[k]];
        }
        y[row] = sum;
    }
}

static double dot_cpu(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (int i = 0; i < N; i++) s += static_cast<double>(a[i]) * b[i];
    return s;
}

static float rmse_cpu(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double d = static_cast<double>(a[i]) - b[i];
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s / N));
}

static SolveStats solve_cpu(const CsrMatrix& A,
                            const std::vector<float>& b,
                            const std::vector<float>& truth,
                            std::vector<float>& x,
                            std::vector<float>& history) {
    std::vector<float> r(N), z(N), p(N), Ap(N);
    x.assign(N, 0.0f);
    r = b;
    for (int i = 0; i < N; i++) {
        z[i] = A.diag_inv[i] * r[i];
        p[i] = z[i];
    }
    double bnorm = std::sqrt(std::max(1.0e-30, dot_cpu(b, b)));
    double rz = dot_cpu(r, z);
    history.clear();
    history.push_back(1.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    int iter = 0;
    float rel = 1.0f;
    for (; iter < MAX_ITERS; iter++) {
        csr_matvec_cpu(A, p, Ap);
        double pAp = dot_cpu(p, Ap);
        if (pAp <= 1.0e-30) break;
        double alpha = rz / pAp;
        for (int i = 0; i < N; i++) {
            x[i] += static_cast<float>(alpha * p[i]);
            r[i] -= static_cast<float>(alpha * Ap[i]);
        }
        double rr = dot_cpu(r, r);
        rel = static_cast<float>(std::sqrt(rr) / bnorm);
        history.push_back(rel);
        if (rel < REL_TOL) {
            iter++;
            break;
        }
        for (int i = 0; i < N; i++) z[i] = A.diag_inv[i] * r[i];
        double rz_new = dot_cpu(r, z);
        double beta = rz_new / rz;
        for (int i = 0; i < N; i++) p[i] = z[i] + static_cast<float>(beta * p[i]);
        rz = rz_new;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    SolveStats s;
    s.iterations = iter;
    s.rel_residual = rel;
    s.rmse = rmse_cpu(x, truth);
    s.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return s;
}

static float dot_gpu(int n, const float* a, const float* b, float* scratch) {
    CUDA_CHECK(cudaMemset(scratch, 0, sizeof(float)));
    dot_kernel<<<DOT_BLOCKS, THREADS>>>(n, a, b, scratch);
    CUDA_CHECK(cudaGetLastError());
    float h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h, scratch, sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static SolveStats solve_gpu(int n,
                            const int* d_row_ptr,
                            const int* d_col_ind,
                            const float* d_values,
                            const float* d_diag_inv,
                            const float* d_b,
                            float* d_x,
                            float* d_r,
                            float* d_z,
                            float* d_p,
                            float* d_Ap,
                            float* d_scratch,
                            float bnorm,
                            const std::vector<float>& truth,
                            std::vector<float>& x_host,
                            std::vector<float>* history,
                            std::vector<std::vector<float>>* snapshots) {
    int blocks = (n + THREADS - 1) / THREADS;
    init_pcg_kernel<<<blocks, THREADS>>>(n, d_b, d_diag_inv, d_x, d_r, d_z, d_p);
    CUDA_CHECK(cudaGetLastError());

    float rz = dot_gpu(n, d_r, d_z, d_scratch);
    float rel = 1.0f;
    if (history) {
        history->clear();
        history->push_back(1.0f);
    }
    if (snapshots) {
        snapshots->clear();
        x_host.assign(n, 0.0f);
        snapshots->push_back(x_host);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int iter = 0;
    int last_snapshot_iter = 0;
    for (; iter < MAX_ITERS; iter++) {
        spmv_csr_kernel<<<blocks, THREADS>>>(n, d_row_ptr, d_col_ind, d_values, d_p, d_Ap);
        CUDA_CHECK(cudaGetLastError());
        float pAp = dot_gpu(n, d_p, d_Ap, d_scratch);
        if (pAp <= 1.0e-30f) break;
        float alpha = rz / pAp;
        axpy_kernel<<<blocks, THREADS>>>(n, alpha, d_p, d_x);
        axpy_kernel<<<blocks, THREADS>>>(n, -alpha, d_Ap, d_r);
        CUDA_CHECK(cudaGetLastError());

        float rr = dot_gpu(n, d_r, d_r, d_scratch);
        rel = std::sqrt(std::max(0.0f, rr)) / bnorm;
        if (history) history->push_back(rel);
        if (snapshots && (((iter + 1) % SNAP_STRIDE == 0) || iter + 1 == MAX_ITERS || rel < REL_TOL)) {
            CUDA_CHECK(cudaMemcpy(x_host.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
            snapshots->push_back(x_host);
            last_snapshot_iter = iter + 1;
        }
        if (rel < REL_TOL) {
            iter++;
            break;
        }

        apply_jacobi_kernel<<<blocks, THREADS>>>(n, d_diag_inv, d_r, d_z);
        CUDA_CHECK(cudaGetLastError());
        float rz_new = dot_gpu(n, d_r, d_z, d_scratch);
        float beta = rz_new / rz;
        xpay_kernel<<<blocks, THREADS>>>(n, beta, d_z, d_p);
        CUDA_CHECK(cudaGetLastError());
        rz = rz_new;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (snapshots && last_snapshot_iter != iter) {
        snapshots->push_back(x_host);
    }

    SolveStats s;
    s.iterations = iter;
    s.rel_residual = rel;
    s.rmse = rmse_cpu(x_host, truth);
    s.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return s;
}

static cv::Mat field_image(const std::vector<float>& field,
                           float lo,
                           float hi,
                           const cv::Size& size) {
    cv::Mat raw(GRID, GRID, CV_32F);
    for (int y = 0; y < GRID; y++) {
        float* row = raw.ptr<float>(y);
        for (int x = 0; x < GRID; x++) row[x] = field[y * GRID + x];
    }
    cv::Mat norm8;
    float scale = 255.0f / std::max(1.0e-6f, hi - lo);
    raw.convertTo(norm8, CV_8U, scale, -lo * scale);
    cv::Mat color;
    cv::applyColorMap(norm8, color, cv::COLORMAP_TURBO);
    cv::Mat resized;
    cv::resize(color, resized, size, 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

static std::vector<float> abs_error_field(const std::vector<float>& a,
                                          const std::vector<float>& b) {
    std::vector<float> e(N);
    for (int i = 0; i < N; i++) e[i] = std::fabs(a[i] - b[i]);
    return e;
}

static void draw_curve(cv::Mat& img,
                       const std::vector<float>& gpu_hist,
                       const std::vector<float>& cpu_hist,
                       const cv::Rect& r,
                       int upto_iter) {
    cv::rectangle(img, r, cv::Scalar(25, 27, 31), -1);
    cv::rectangle(img, r, cv::Scalar(80, 84, 92), 1);
    cv::putText(img, "relative residual", cv::Point(r.x + 14, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);

    for (int g = 0; g <= 4; g++) {
        int y = r.y + r.height - 24 - g * (r.height - 58) / 4;
        cv::line(img, cv::Point(r.x + 46, y), cv::Point(r.x + r.width - 14, y),
                 cv::Scalar(45, 48, 55), 1);
    }

    auto draw_one = [&](const std::vector<float>& h, cv::Scalar color, int limit) {
        if (h.size() < 2) return;
        int max_i = std::min<int>(limit, static_cast<int>(h.size()) - 1);
        if (max_i < 1) return;
        std::vector<cv::Point> pts;
        for (int i = 0; i <= max_i; i++) {
            float x01 = static_cast<float>(i) / std::max(1, MAX_ITERS);
            float logv = std::log10(std::max(1.0e-8f, h[i]));
            float y01 = clampf((logv - (-8.0f)) / 8.0f, 0.0f, 1.0f);
            int px = r.x + 46 + static_cast<int>(x01 * (r.width - 64));
            int py = r.y + r.height - 24 - static_cast<int>((1.0f - y01) * (r.height - 58));
            pts.emplace_back(px, py);
        }
        cv::polylines(img, pts, false, color, 2, cv::LINE_AA);
    };
    draw_one(cpu_hist, cv::Scalar(160, 170, 185), static_cast<int>(cpu_hist.size()) - 1);
    draw_one(gpu_hist, cv::Scalar(80, 210, 130), upto_iter);
    cv::putText(img, "GPU", cv::Point(r.x + 166, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 210, 130), 1);
    cv::putText(img, "CPU", cv::Point(r.x + 218, r.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(170, 180, 195), 1);
}

static cv::Mat draw_frame(const std::vector<float>& truth,
                          const std::vector<float>& solution,
                          const std::vector<float>& gpu_hist,
                          const std::vector<float>& cpu_hist,
                          const SolveStats& gpu,
                          const SolveStats& cpu,
                          int iter_hint,
                          int nnz,
                          float field_lo,
                          float field_hi) {
    cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 19, 23));
    cv::putText(img, "GPU PCG sparse SPD solver", cv::Point(22, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.72, cv::Scalar(245, 245, 245), 1,
                cv::LINE_AA);
    cv::putText(img,
                cv::format("%d unknowns, %d nnz   GPU %.2f ms   CPU %.2f ms   %.1fx",
                           N, nnz, gpu.ms, cpu.ms, cpu.ms / std::max(1.0e-9, gpu.ms)),
                cv::Point(22, 56), cv::FONT_HERSHEY_SIMPLEX, 0.46,
                cv::Scalar(210, 214, 220), 1, cv::LINE_AA);

    cv::Rect r_true(24, 84, 278, 278);
    cv::Rect r_sol(342, 84, 278, 278);
    cv::Rect r_err(660, 84, 278, 278);
    field_image(truth, field_lo, field_hi, r_true.size()).copyTo(img(r_true));
    field_image(solution, field_lo, field_hi, r_sol.size()).copyTo(img(r_sol));
    std::vector<float> err = abs_error_field(solution, truth);
    float err_hi = std::max(0.02f, *std::max_element(err.begin(), err.end()));
    field_image(err, 0.0f, err_hi, r_err.size()).copyTo(img(r_err));
    cv::rectangle(img, r_true, cv::Scalar(80, 84, 92), 1);
    cv::rectangle(img, r_sol, cv::Scalar(80, 84, 92), 1);
    cv::rectangle(img, r_err, cv::Scalar(80, 84, 92), 1);
    cv::putText(img, "ground truth", cv::Point(r_true.x + 10, r_true.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(245, 245, 245), 1);
    cv::putText(img, cv::format("GPU PCG iter %03d", iter_hint),
                cv::Point(r_sol.x + 10, r_sol.y + 24), cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(245, 245, 245), 1);
    cv::putText(img, "absolute error", cv::Point(r_err.x + 10, r_err.y + 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(245, 245, 245), 1);

    cv::Rect curve(24, 398, 596, 186);
    draw_curve(img, gpu_hist, cpu_hist, curve, iter_hint);

    cv::Rect stats(652, 408, 286, 162);
    cv::rectangle(img, stats, cv::Scalar(29, 31, 36), -1);
    cv::rectangle(img, stats, cv::Scalar(76, 80, 88), 1);
    cv::putText(img, cv::format("GPU iters %d", gpu.iterations),
                cv::Point(stats.x + 14, stats.y + 34), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
    cv::putText(img, cv::format("rel residual %.2e", gpu.rel_residual),
                cv::Point(stats.x + 14, stats.y + 66), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(80, 210, 130), 1, cv::LINE_AA);
    cv::putText(img, cv::format("RMSE %.5f", gpu.rmse),
                cv::Point(stats.x + 14, stats.y + 98), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(90, 170, 255), 1, cv::LINE_AA);
    cv::putText(img, cv::format("CPU/GPU %.1fx", cpu.ms / std::max(1.0e-9, gpu.ms)),
                cv::Point(stats.x + 14, stats.y + 130), cv::FONT_HERSHEY_SIMPLEX,
                0.50, cv::Scalar(220, 224, 230), 1, cv::LINE_AA);
    return img;
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    CsrMatrix A = build_grid_matrix();
    std::vector<float> truth = make_true_field();
    std::vector<float> b;
    csr_matvec_cpu(A, truth, b);
    double bnorm_cpu = std::sqrt(std::max(1.0e-30, dot_cpu(b, b)));

    std::vector<float> cpu_x;
    std::vector<float> cpu_hist;
    SolveStats cpu = solve_cpu(A, b, truth, cpu_x, cpu_hist);

    int* d_row_ptr = nullptr;
    int* d_col_ind = nullptr;
    float* d_values = nullptr;
    float* d_diag_inv = nullptr;
    float* d_b = nullptr;
    float* d_x = nullptr;
    float* d_r = nullptr;
    float* d_z = nullptr;
    float* d_p = nullptr;
    float* d_Ap = nullptr;
    float* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, A.row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, A.col_ind.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, A.values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diag_inv, A.diag_inv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ap, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), A.row_ptr.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, A.col_ind.data(), A.col_ind.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, A.values.data(), A.values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diag_inv, A.diag_inv.data(), A.diag_inv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> gpu_x(N);
    std::vector<float> gpu_hist;
    std::vector<std::vector<float>> snapshots;
    SolveStats gpu = solve_gpu(N, d_row_ptr, d_col_ind, d_values, d_diag_inv, d_b,
                               d_x, d_r, d_z, d_p, d_Ap, d_scratch,
                               static_cast<float>(bnorm_cpu), truth, gpu_x,
                               &gpu_hist, &snapshots);

    int nnz = static_cast<int>(A.values.size());
    std::printf("GPU PCG sparse SPD: %d unknowns, %d nnz\n", N, nnz);
    std::printf("GPU %.3f ms, CPU %.3f ms, speedup %.1fx\n",
                gpu.ms, cpu.ms, cpu.ms / std::max(1.0e-9, gpu.ms));
    std::printf("GPU iters %d, rel residual %.3e, RMSE %.6f\n",
                gpu.iterations, gpu.rel_residual, gpu.rmse);
    std::printf("CPU iters %d, rel residual %.3e, RMSE %.6f\n",
                cpu.iterations, cpu.rel_residual, cpu.rmse);

    float field_lo = *std::min_element(truth.begin(), truth.end());
    float field_hi = *std::max_element(truth.begin(), truth.end());

    int mkdir_ret = std::system("mkdir -p gif");
    (void)mkdir_ret;
    cv::VideoWriter video("gif/gpu_pcg_solver.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS, cv::Size(PANEL_W, PANEL_H));
    if (!video.isOpened()) {
        std::fprintf(stderr, "failed to open gif/gpu_pcg_solver.avi\n");
        return 1;
    }

    for (int hold = 0; hold < 8; hold++) {
        video.write(draw_frame(truth, snapshots.front(), gpu_hist, cpu_hist,
                               gpu, cpu, 0, nnz, field_lo, field_hi));
    }
    for (size_t si = 0; si < snapshots.size(); si++) {
        int iter_hint = std::min(MAX_ITERS, static_cast<int>(si) * SNAP_STRIDE);
        cv::Mat frame = draw_frame(truth, snapshots[si], gpu_hist, cpu_hist,
                                   gpu, cpu, iter_hint, nnz, field_lo, field_hi);
        video.write(frame);
        if (si + 1 == snapshots.size()) {
            for (int hold = 0; hold < 12; hold++) video.write(frame);
        }
    }
    video.release();
    cudabot::avi_to_gif("gif/gpu_pcg_solver.avi", "gif/gpu_pcg_solver.gif", 10, 720);
    std::printf("GIF saved to gif/gpu_pcg_solver.gif\n");

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_diag_inv));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_scratch));
    return 0;
}
