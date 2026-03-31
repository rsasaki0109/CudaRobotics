/*************************************************************************
    emcl2 Comparison: Standard MCL (fails) vs emcl2 (recovers) after kidnap
    Left: standard MCL, no expansion reset - loses track after kidnap
    Right: emcl2 with expansion resetting - recovers
    Same map, same kidnap event at t=10s
 ************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#define PI 3.141592653f
#define DT 0.1f
#define SIM_TIME 30.0f

#define GRID_W 200
#define GRID_H 200
#define GRID_RES 0.1f

#define N_PARTICLES 500
#define NUM_BEAMS 36
#define BEAM_ANGLE_STEP (2.0f * PI / NUM_BEAMS)
#define MAX_RANGE 10.0f

#define SIGMA_HIT 0.2f
#define Z_HIT 0.9f
#define Z_RAND 0.1f

#define ALPHA1 0.1f
#define ALPHA2 0.1f
#define ALPHA3 0.1f
#define ALPHA4 0.1f

#define EXPANSION_RESET_THRESHOLD 0.001f
#define EXPANSION_NOISE_XY 2.0f
#define EXPANSION_NOISE_TH 1.0f

#define VIS_SCALE 2
#define PANEL_W (GRID_W * VIS_SCALE)
#define PANEL_H (GRID_H * VIS_SCALE)

#define KIDNAP_TIME 10.0f

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void predict_kernel(float* px, float* py, float* ptheta,
    float v, float omega, float dt, curandState* rng_states, int np) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    curandState local_rng = rng_states[idx];
    float v_hat = v + curand_normal(&local_rng) * (ALPHA1 * fabsf(v) + ALPHA2 * fabsf(omega));
    float omega_hat = omega + curand_normal(&local_rng) * (ALPHA3 * fabsf(v) + ALPHA4 * fabsf(omega));
    float theta = ptheta[idx];
    if (fabsf(omega_hat) < 1e-6f) {
        px[idx] += v_hat * cosf(theta) * dt; py[idx] += v_hat * sinf(theta) * dt;
    } else {
        float r = v_hat / omega_hat;
        px[idx] += r * (sinf(theta + omega_hat * dt) - sinf(theta));
        py[idx] += r * (cosf(theta) - cosf(theta + omega_hat * dt));
    }
    ptheta[idx] += omega_hat * dt;
    float th = ptheta[idx];
    while (th > PI) th -= 2.0f * PI; while (th < -PI) th += 2.0f * PI;
    ptheta[idx] = th;
    rng_states[idx] = local_rng;
}

__global__ void compute_likelihood_kernel(float* px, float* py, float* ptheta, float* pw,
    const float* lf, const float* beams, int w, int h, float res, float ox, float oy, int np) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    float x = px[idx], y = py[idx], theta = ptheta[idx];
    float log_w = 0.0f;
    for (int b = 0; b < NUM_BEAMS; b++) {
        float range = beams[b]; if (range >= MAX_RANGE) continue;
        float ba = theta + (float)b * BEAM_ANGLE_STEP - PI;
        int gx = (int)((x + range * cosf(ba) - ox) / res);
        int gy = (int)((y + range * sinf(ba) - oy) / res);
        if (gx >= 0 && gx < w && gy >= 0 && gy < h) log_w += logf(fmaxf(lf[gy * w + gx], 1e-10f));
        else log_w += logf(Z_RAND / MAX_RANGE);
    }
    pw[idx] = expf(log_w);
}

__global__ void check_reset_kernel(const float* pw, int np, float* out_max, float* out_sum) {
    extern __shared__ float sd[];
    float* sm = sd; float* ss = sd + blockDim.x;
    int tid = threadIdx.x;
    float lm = 0, ls = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i]; if (w > lm) lm = w; ls += w;
    }
    sm[tid] = lm; ss[tid] = ls; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { if (sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s]; ss[tid] += ss[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { *out_max = sm[0]; *out_sum = ss[0]; }
}

__global__ void expansion_reset_kernel(float* px, float* py, float* ptheta, float* pw,
    float nxy, float nth, const int* occ, int gw, int gh, float ox, float oy, float res,
    curandState* rng, int np) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    curandState lr = rng[idx];
    px[idx] += curand_normal(&lr) * nxy; py[idx] += curand_normal(&lr) * nxy;
    ptheta[idx] += curand_normal(&lr) * nth;
    int gx = (int)((px[idx] - ox) / res); int gy = (int)((py[idx] - oy) / res);
    if (gx < 2) gx = 2; if (gx >= gw - 2) gx = gw - 3;
    if (gy < 2) gy = 2; if (gy >= gh - 2) gy = gh - 3;
    if (occ[gy * gw + gx] == 1) {
        for (int a = 0; a < 100; a++) {
            int rx = (int)(curand_uniform(&lr) * (gw - 4)) + 2;
            int ry = (int)(curand_uniform(&lr) * (gh - 4)) + 2;
            if (occ[ry * gw + rx] == 0) {
                px[idx] = ox + (rx + 0.5f) * res; py[idx] = oy + (ry + 0.5f) * res; break;
            }
        }
    }
    float th = ptheta[idx];
    while (th > PI) th -= 2 * PI; while (th < -PI) th += 2 * PI;
    ptheta[idx] = th; pw[idx] = 1.0f / np; rng[idx] = lr;
}

__global__ void sensor_reset_kernel(float* px, float* py, float* ptheta, float* pw,
    const int* occ, int gw, int gh, float ox, float oy, float res,
    curandState* rng, int np, int n_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_reset) return;
    curandState lr = rng[idx];
    for (int a = 0; a < 200; a++) {
        int rx = (int)(curand_uniform(&lr) * (gw - 4)) + 2;
        int ry = (int)(curand_uniform(&lr) * (gh - 4)) + 2;
        if (occ[ry * gw + rx] == 0) {
            px[idx] = ox + (rx + 0.5f) * res; py[idx] = oy + (ry + 0.5f) * res;
            ptheta[idx] = curand_uniform(&lr) * 2.0f * PI - PI; pw[idx] = 1.0f / np; break;
        }
    }
    rng[idx] = lr;
}

__global__ void normalize_kernel(float* pw, int np, float total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np) return;
    if (total > 1e-30f) pw[idx] /= total; else pw[idx] = 1.0f / np;
}

__global__ void cumsum_kernel(const float* pw, float* wcum, int np) {
    wcum[0] = pw[0]; for (int i = 1; i < np; i++) wcum[i] = wcum[i - 1] + pw[i];
}

__global__ void resample_kernel(const float* px_in, const float* py_in, const float* pt_in,
    float* px_out, float* py_out, float* pt_out, const float* wcum, float bs, float ro, int np) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x; if (ip >= np) return;
    float t = bs * ip + ro; int lo = 0, hi = np - 1;
    while (lo < hi) { int m = (lo + hi) / 2; if (wcum[m] < t) lo = m + 1; else hi = m; }
    px_out[ip] = px_in[lo]; py_out[ip] = py_in[lo]; pt_out[ip] = pt_in[lo];
}

__global__ void weighted_mean_kernel(const float* px, const float* py, const float* pt,
    const float* pw, float* ox, float* oy, float* ot, int np) {
    extern __shared__ float sd[];
    float* sx = sd; float* sy = sd + blockDim.x; float* sc = sd + 2 * blockDim.x; float* ss = sd + 3 * blockDim.x;
    int tid = threadIdx.x;
    float vx = 0, vy = 0, vc = 0, vs = 0;
    for (int i = tid; i < np; i += blockDim.x) {
        float w = pw[i]; vx += px[i] * w; vy += py[i] * w; vc += cosf(pt[i]) * w; vs += sinf(pt[i]) * w;
    }
    sx[tid] = vx; sy[tid] = vy; sc[tid] = vc; ss[tid] = vs; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sx[tid] += sx[tid + s]; sy[tid] += sy[tid + s]; sc[tid] += sc[tid + s]; ss[tid] += ss[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { *ox = sx[0]; *oy = sy[0]; *ot = atan2f(ss[0], sc[0]); }
}

__global__ void build_lf_kernel(const int* occ, float* lf, int w, int h, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;
    int cy = idx / w, cx = idx % w;
    float md = 1e6f; int sr = (int)(3.0f * sigma / GRID_RES) + 1; if (sr > 50) sr = 50;
    for (int dy = -sr; dy <= sr; dy++) for (int dx = -sr; dx <= sr; dx++) {
        int nx = cx + dx, ny = cy + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h && occ[ny * w + nx] == 1) {
            float d = sqrtf((float)(dx * dx + dy * dy)) * GRID_RES; if (d < md) md = d;
        }
    }
    lf[idx] = Z_HIT * expf(-0.5f * md * md / (sigma * sigma)) + Z_RAND / MAX_RANGE;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
void build_map(std::vector<int>& occ, int w, int h) {
    occ.assign(w * h, 0);
    for (int x = 0; x < w; x++) { occ[x] = 1; occ[w + x] = 1; occ[(h - 1) * w + x] = 1; occ[(h - 2) * w + x] = 1; }
    for (int y = 0; y < h; y++) { occ[y * w] = 1; occ[y * w + 1] = 1; occ[y * w + w - 1] = 1; occ[y * w + w - 2] = 1; }
    for (int x = 40; x < 80; x++) for (int t = 0; t < 3; t++) occ[(60 + t) * w + x] = 1;
    for (int y = 80; y < 140; y++) for (int t = 0; t < 3; t++) occ[y * w + 130 + t] = 1;
    for (int x = 30; x < 55; x++) for (int t = 0; t < 3; t++) occ[(130 + t) * w + x] = 1;
    for (int y = 130; y < 165; y++) for (int t = 0; t < 3; t++) occ[y * w + 30 + t] = 1;
}

void simulate_lidar(const std::vector<int>& occ, float rx, float ry, float rt,
    float ox, float oy, float res, int gw, int gh, float* beams) {
    float step = res * 0.5f;
    for (int b = 0; b < NUM_BEAMS; b++) {
        float a = rt + (float)b * BEAM_ANGLE_STEP - PI;
        float ca = cosf(a), sa = sinf(a), r = 0; bool hit = false;
        while (r < MAX_RANGE) { r += step; int gx = (int)((rx + r * ca - ox) / res); int gy = (int)((ry + r * sa - oy) / res);
            if (gx < 0 || gx >= gw || gy < 0 || gy >= gh) { hit = true; break; }
            if (occ[gy * gw + gx]) { hit = true; break; }
        }
        beams[b] = hit ? r : MAX_RANGE;
    }
}

cv::Point2i w2p(float wx, float wy, int offx = 0) {
    int gx = (int)(wx / GRID_RES), gy = (int)(wy / GRID_RES);
    return cv::Point2i(gx * VIS_SCALE + offx, (GRID_H - 1 - gy) * VIS_SCALE);
}

// ---------------------------------------------------------------------------
// PF state on GPU
// ---------------------------------------------------------------------------
struct GPUParticles {
    float *d_px, *d_py, *d_pt, *d_pw;
    float *d_px2, *d_py2, *d_pt2;
    float *d_wcum;
    curandState *d_rng;
    float *d_ex, *d_ey, *d_et;
    float *d_max_w, *d_sum_w;

    void alloc(int np, unsigned long long seed) {
        int blk = (np + 255) / 256;
        CUDA_CHECK(cudaMalloc(&d_px, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_py, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pt, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pw, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_px2, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_py2, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pt2, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wcum, np * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng, np * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_ex, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ey, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_et, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_max_w, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sum_w, sizeof(float)));
        init_curand_kernel<<<blk, 256>>>(d_rng, seed, np);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void free_all() {
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_pt); cudaFree(d_pw);
        cudaFree(d_px2); cudaFree(d_py2); cudaFree(d_pt2);
        cudaFree(d_wcum); cudaFree(d_rng);
        cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_et);
        cudaFree(d_max_w); cudaFree(d_sum_w);
    }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Comparison: Standard MCL vs emcl2" << std::endl;

    float origin_x = 0, origin_y = 0;
    std::vector<int> h_occ;
    build_map(h_occ, GRID_W, GRID_H);

    int* d_occ; float* d_lf;
    CUDA_CHECK(cudaMalloc(&d_occ, GRID_W * GRID_H * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_occ, h_occ.data(), GRID_W * GRID_H * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_lf, GRID_W * GRID_H * sizeof(float)));
    build_lf_kernel<<<(GRID_W * GRID_H + 255) / 256, 256>>>(d_occ, d_lf, GRID_W, GRID_H, SIGMA_HIT);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_beams;
    CUDA_CHECK(cudaMalloc(&d_beams, NUM_BEAMS * sizeof(float)));

    const int np = N_PARTICLES;
    const int thr = 256;
    const int blk = (np + thr - 1) / thr;

    // Two particle filters
    GPUParticles std_pf, emcl_pf;
    std_pf.alloc(np, 42ULL);
    emcl_pf.alloc(np, 84ULL);

    std::mt19937 gen(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    float gt_x = 5.0f, gt_y = 5.0f, gt_theta = 0.0f;
    float robot_v = 1.0f, robot_omega = 0.0f;

    // Initialize both PFs near ground truth
    std::vector<float> h_px(np), h_py(np), h_pt(np), h_pw(np, 1.0f / np);
    for (int i = 0; i < np; i++) {
        h_px[i] = gt_x + gauss(gen) * 1.0f; h_py[i] = gt_y + gauss(gen) * 1.0f;
        h_pt[i] = gt_theta + gauss(gen) * 0.5f;
    }
    auto upload = [&](GPUParticles& pf) {
        CUDA_CHECK(cudaMemcpy(pf.d_px, h_px.data(), np * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pf.d_py, h_py.data(), np * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pf.d_pt, h_pt.data(), np * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pf.d_pw, h_pw.data(), np * sizeof(float), cudaMemcpyHostToDevice));
    };
    upload(std_pf); upload(emcl_pf);

    // Pre-render map
    cv::Mat map_img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int gy = 0; gy < GRID_H; gy++)
        for (int gx = 0; gx < GRID_W; gx++)
            if (h_occ[gy * GRID_W + gx]) {
                int px = gx * VIS_SCALE, py = (GRID_H - 1 - gy) * VIS_SCALE;
                cv::rectangle(map_img, cv::Point(px, py), cv::Point(px + VIS_SCALE - 1, py + VIS_SCALE - 1), cv::Scalar(0, 0, 0), -1);
            }

    cv::VideoWriter video("gif/comparison_emcl2.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(PANEL_W * 2, PANEL_H));

    float h_beams[NUM_BEAMS];
    bool kidnapped = false;
    int step = 0;

    // Lambda to run one PF step
    auto run_pf = [&](GPUParticles& pf, bool do_expansion) {
        predict_kernel<<<blk, thr>>>(pf.d_px, pf.d_py, pf.d_pt, robot_v, robot_omega, DT, pf.d_rng, np);
        compute_likelihood_kernel<<<blk, thr>>>(pf.d_px, pf.d_py, pf.d_pt, pf.d_pw,
            d_lf, d_beams, GRID_W, GRID_H, GRID_RES, origin_x, origin_y, np);
        check_reset_kernel<<<1, thr, 2 * thr * sizeof(float)>>>(pf.d_pw, np, pf.d_max_w, pf.d_sum_w);

        float h_max, h_sum;
        CUDA_CHECK(cudaMemcpy(&h_max, pf.d_max_w, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sum, pf.d_sum_w, sizeof(float), cudaMemcpyDeviceToHost));

        if (do_expansion && h_max < EXPANSION_RESET_THRESHOLD) {
            expansion_reset_kernel<<<blk, thr>>>(pf.d_px, pf.d_py, pf.d_pt, pf.d_pw,
                EXPANSION_NOISE_XY, EXPANSION_NOISE_TH, d_occ, GRID_W, GRID_H,
                origin_x, origin_y, GRID_RES, pf.d_rng, np);
            int nr = np / 4;
            sensor_reset_kernel<<<(nr + 255) / 256, 256>>>(pf.d_px, pf.d_py, pf.d_pt, pf.d_pw,
                d_occ, GRID_W, GRID_H, origin_x, origin_y, GRID_RES, pf.d_rng, np, nr);
            compute_likelihood_kernel<<<blk, thr>>>(pf.d_px, pf.d_py, pf.d_pt, pf.d_pw,
                d_lf, d_beams, GRID_W, GRID_H, GRID_RES, origin_x, origin_y, np);
            check_reset_kernel<<<1, thr, 2 * thr * sizeof(float)>>>(pf.d_pw, np, pf.d_max_w, pf.d_sum_w);
            CUDA_CHECK(cudaMemcpy(&h_sum, pf.d_sum_w, sizeof(float), cudaMemcpyDeviceToHost));
        }

        normalize_kernel<<<blk, thr>>>(pf.d_pw, np, h_sum);
        weighted_mean_kernel<<<1, thr, 4 * thr * sizeof(float)>>>(
            pf.d_px, pf.d_py, pf.d_pt, pf.d_pw, pf.d_ex, pf.d_ey, pf.d_et, np);

        // Resample
        std::vector<float> tmp_pw(np);
        CUDA_CHECK(cudaMemcpy(tmp_pw.data(), pf.d_pw, np * sizeof(float), cudaMemcpyDeviceToHost));
        float nd = 0; for (int i = 0; i < np; i++) nd += tmp_pw[i] * tmp_pw[i];
        if (1.0f / nd < np / 2) {
            cumsum_kernel<<<1, 1>>>(pf.d_pw, pf.d_wcum, np);
            float ro = uni(gen) / np;
            resample_kernel<<<blk, thr>>>(pf.d_px, pf.d_py, pf.d_pt,
                pf.d_px2, pf.d_py2, pf.d_pt2, pf.d_wcum, 1.0f / np, ro, np);
            CUDA_CHECK(cudaMemcpy(pf.d_px, pf.d_px2, np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(pf.d_py, pf.d_py2, np * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(pf.d_pt, pf.d_pt2, np * sizeof(float), cudaMemcpyDeviceToDevice));
            std::vector<float> uw(np, 1.0f / np);
            CUDA_CHECK(cudaMemcpy(pf.d_pw, uw.data(), np * sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    auto draw_panel = [&](cv::Mat& panel, GPUParticles& pf, const char* label, float gt_x, float gt_y, float gt_t) {
        panel = map_img.clone();
        std::vector<float> lx(np), ly(np);
        CUDA_CHECK(cudaMemcpy(lx.data(), pf.d_px, np * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ly.data(), pf.d_py, np * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < np; i++) {
            cv::Point2i p = w2p(lx[i], ly[i]);
            if (p.x >= 0 && p.x < PANEL_W && p.y >= 0 && p.y < PANEL_H)
                cv::circle(panel, p, 2, cv::Scalar(0, 0, 255), -1);
        }
        float ex, ey, et;
        CUDA_CHECK(cudaMemcpy(&ex, pf.d_ex, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&ey, pf.d_ey, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&et, pf.d_et, sizeof(float), cudaMemcpyDeviceToHost));
        cv::circle(panel, w2p(ex, ey), 6, cv::Scalar(255, 0, 0), -1);
        cv::circle(panel, w2p(gt_x, gt_y), 6, cv::Scalar(0, 200, 0), -1);
        cv::putText(panel, label, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        float err = sqrtf((ex - gt_x) * (ex - gt_x) + (ey - gt_y) * (ey - gt_y));
        char buf[64]; snprintf(buf, sizeof(buf), "err=%.1fm", err);
        cv::putText(panel, buf, cv::Point(5, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 200), 1);
    };

    float time_val = 0;
    float map_w = GRID_W * GRID_RES, map_h = GRID_H * GRID_RES;

    while (time_val <= SIM_TIME) {
        time_val += DT; step++;

        if (gt_x > map_w - 3 || gt_x < 3) robot_omega = 0.3f;
        if (gt_y > map_h - 3 || gt_y < 3) robot_omega = 0.3f;
        if (step % 50 == 0) robot_omega = (uni(gen) - 0.5f) * 0.6f;

        if (time_val >= KIDNAP_TIME && !kidnapped) {
            kidnapped = true;
            gt_x = 15.0f; gt_y = 15.0f; gt_theta = PI / 2.0f;
            robot_omega = 0.1f;
            printf("[%.1fs] KIDNAPPED!\n", time_val);
        }

        gt_theta += robot_omega * DT;
        gt_x += robot_v * cosf(gt_theta) * DT;
        gt_y += robot_v * sinf(gt_theta) * DT;
        if (gt_x < 1) { gt_x = 1; robot_omega = 0.5f; }
        if (gt_x > map_w - 1) { gt_x = map_w - 1; robot_omega = 0.5f; }
        if (gt_y < 1) { gt_y = 1; robot_omega = 0.5f; }
        if (gt_y > map_h - 1) { gt_y = map_h - 1; robot_omega = 0.5f; }

        simulate_lidar(h_occ, gt_x, gt_y, gt_theta, origin_x, origin_y, GRID_RES, GRID_W, GRID_H, h_beams);
        CUDA_CHECK(cudaMemcpy(d_beams, h_beams, NUM_BEAMS * sizeof(float), cudaMemcpyHostToDevice));

        run_pf(std_pf, false);  // Standard MCL: NO expansion reset
        run_pf(emcl_pf, true);  // emcl2: WITH expansion reset

        // Visualization
        cv::Mat left, right;
        draw_panel(left, std_pf, "Standard MCL (no reset)", gt_x, gt_y, gt_theta);
        draw_panel(right, emcl_pf, "emcl2 (expansion reset)", gt_x, gt_y, gt_theta);

        if (kidnapped) {
            cv::putText(left, "KIDNAPPED", cv::Point(PANEL_W / 2 - 60, PANEL_H - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
            cv::putText(right, "KIDNAPPED", cv::Point(PANEL_W / 2 - 60, PANEL_H - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        char tbuf[32]; snprintf(tbuf, sizeof(tbuf), "t=%.1fs", time_val);
        cv::putText(left, tbuf, cv::Point(PANEL_W - 80, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(right, tbuf, cv::Point(PANEL_W - 80, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
    }

    video.release();
    system("ffmpeg -y -i gif/comparison_emcl2.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_emcl2.gif 2>/dev/null");
    printf("GIF saved to gif/comparison_emcl2.gif\n");

    std_pf.free_all(); emcl_pf.free_all();
    cudaFree(d_occ); cudaFree(d_lf); cudaFree(d_beams);
    return 0;
}
