/*************************************************************************
    > File Name: frenet_optimal_trajectory.cu
    > CUDA-parallelized Frenet Optimal Trajectory Planning
    > Based on original C++ implementation by TAI Lei
    > Each candidate path (di, Ti, tv) = 1 GPU thread
 ************************************************************************/

#include <iostream>
#include <limits>
#include <vector>
#include <array>
#include <cmath>
#include <cfloat>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Eigen>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Constants (same as original)
// ---------------------------------------------------------------------------
#define SIM_LOOP 500
#define MAX_SPEED  (50.0f / 3.6f)
#define MAX_ACCEL  2.0f
#define MAX_CURVATURE  1.0f
#define MAX_ROAD_WIDTH  7.0f
#define D_ROAD_W  1.0f
#define DT  0.2f
#define MAXT  5.0f
#define MINT  4.0f
#define TARGET_SPEED  (30.0f / 3.6f)
#define D_T_S  (5.0f / 3.6f)
#define N_S_SAMPLE  1
#define ROBOT_RADIUS  1.5f

#define KJ  0.1f
#define KT  0.1f
#define KD  1.0f
#define KLAT  1.0f
#define KLON  1.0f

#define MAX_TRAJ_POINTS 30  // max points per trajectory
#define MAX_PATHS 512       // max candidate paths
#define MAX_SPLINE_SEGS 256 // max spline segments
#define MAX_OBS 32          // max obstacles

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Host: Spline (same as original, for host-side computation)
// ---------------------------------------------------------------------------
using Vec_f = std::vector<float>;
using Poi_f = std::array<float, 2>;

Vec_f vec_diff(Vec_f input) {
    Vec_f output;
    for (unsigned int i = 1; i < input.size(); i++)
        output.push_back(input[i] - input[i - 1]);
    return output;
}

Vec_f cum_sum(Vec_f input) {
    Vec_f output;
    float temp = 0;
    for (unsigned int i = 0; i < input.size(); i++) {
        temp += input[i];
        output.push_back(temp);
    }
    return output;
}

// Minimal spline class (host only) — replicates include/cubic_spline.h logic
class Spline {
public:
    Vec_f x, y, h, a, b, c, d;
    int nx;

    Spline() : nx(0) {}

    Spline(Vec_f x_, Vec_f y_) : x(x_), y(y_), nx(x_.size()), h(vec_diff(x_)), a(y_) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(nx, nx);
        A(0, 0) = 1;
        for (int i = 0; i < nx - 1; i++) {
            if (i != nx - 2) A(i + 1, i + 1) = 2 * (h[i] + h[i + 1]);
            A(i + 1, i) = h[i];
            A(i, i + 1) = h[i];
        }
        A(0, 1) = 0.0;
        A(nx - 1, nx - 2) = 0.0;
        A(nx - 1, nx - 1) = 1.0;

        Eigen::VectorXf B = Eigen::VectorXf::Zero(nx);
        for (int i = 0; i < nx - 2; i++)
            B(i + 1) = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i];

        Eigen::VectorXf c_eigen = A.colPivHouseholderQr().solve(B);
        float* cp = c_eigen.data();
        c.assign(cp, cp + c_eigen.rows());

        for (int i = 0; i < nx - 1; i++) {
            d.push_back((c[i + 1] - c[i]) / (3.0f * h[i]));
            b.push_back((a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3.0f);
        }
    }

    int bisect(float t, int start, int end) {
        int mid = (start + end) / 2;
        if (t == x[mid] || end - start <= 1) return mid;
        else if (t > x[mid]) return bisect(t, mid, end);
        else return bisect(t, start, mid);
    }

    float calc(float t) {
        int seg = bisect(t, 0, nx);
        float dx = t - x[seg];
        return a[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx * dx * dx;
    }

    float calc_d(float t) {
        int seg = bisect(t, 0, nx - 1);
        float dx = t - x[seg];
        return b[seg] + 2 * c[seg] * dx + 3 * d[seg] * dx * dx;
    }

    float calc_dd(float t) {
        int seg = bisect(t, 0, nx);
        float dx = t - x[seg];
        return 2 * c[seg] + 6 * d[seg] * dx;
    }
};

class Spline2D {
public:
    Spline sx, sy;
    Vec_f s;

    Spline2D() {}

    Spline2D(Vec_f x, Vec_f y) {
        Vec_f dx = vec_diff(x);
        Vec_f dy = vec_diff(y);
        Vec_f ds;
        for (unsigned int i = 0; i < dx.size(); i++)
            ds.push_back(sqrtf(dx[i] * dx[i] + dy[i] * dy[i]));
        s.push_back(0);
        Vec_f cum_ds = cum_sum(ds);
        s.insert(s.end(), cum_ds.begin(), cum_ds.end());
        sx = Spline(s, x);
        sy = Spline(s, y);
    }

    Poi_f calc_position(float s_t) {
        return {{sx.calc(s_t), sy.calc(s_t)}};
    }
    float calc_yaw(float s_t) {
        return atan2f(sy.calc_d(s_t), sx.calc_d(s_t));
    }
    float calc_curvature(float s_t) {
        float dx = sx.calc_d(s_t), ddx = sx.calc_dd(s_t);
        float dy = sy.calc_d(s_t), ddy = sy.calc_dd(s_t);
        return (ddy * dx - ddx * dy) / (dx * dx + dy * dy);
    }
};

// ---------------------------------------------------------------------------
// Device: spline coefficients (flat arrays transferred from host)
// ---------------------------------------------------------------------------
struct DeviceSpline {
    float* knots;  // x values (breakpoints)
    float* a;
    float* b;
    float* c;
    float* d;
    int n_segs;    // number of segments (nx - 1)
    int nx;
    float s_max;   // max s value
};

// Device: bisect search in spline knots
__device__ int d_bisect(const float* knots, float t, int nx) {
    int lo = 0, hi = nx - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (t > knots[mid]) lo = mid;
        else hi = mid;
    }
    return lo;
}

__device__ float d_spline_calc(const DeviceSpline& sp, float t) {
    int seg = d_bisect(sp.knots, t, sp.nx);
    float dx = t - sp.knots[seg];
    return sp.a[seg] + sp.b[seg] * dx + sp.c[seg] * dx * dx + sp.d[seg] * dx * dx * dx;
}

__device__ float d_spline_calc_d(const DeviceSpline& sp, float t) {
    int seg = d_bisect(sp.knots, t, sp.nx);
    float dx = t - sp.knots[seg];
    return sp.b[seg] + 2.0f * sp.c[seg] * dx + 3.0f * sp.d[seg] * dx * dx;
}

// ---------------------------------------------------------------------------
// Device: solve quintic polynomial coefficients (a3, a4, a5) via Cramer's rule
//   System: A * [a3, a4, a5]^T = B
//   where A and B come from boundary conditions
// ---------------------------------------------------------------------------
__device__ void d_solve_quintic(float T, float a0, float a1, float a2,
                                float xe, float vxe, float axe,
                                float& a3, float& a4, float& a5) {
    float T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;

    // A = [[T3, T4, T5], [3T2, 4T3, 5T4], [6T, 12T2, 20T3]]
    float b0 = xe - a0 - a1 * T - a2 * T2;
    float b1 = vxe - a1 - 2.0f * a2 * T;
    float b2 = axe - 2.0f * a2;

    // Cramer's rule for 3x3
    float detA = T3 * (4.0f * T3 * 20.0f * T3 - 5.0f * T4 * 12.0f * T2)
               - T4 * (3.0f * T2 * 20.0f * T3 - 5.0f * T4 * 6.0f * T)
               + T5 * (3.0f * T2 * 12.0f * T2 - 4.0f * T3 * 6.0f * T);

    if (fabsf(detA) < 1e-10f) { a3 = a4 = a5 = 0; return; }

    float detA3 = b0 * (4.0f * T3 * 20.0f * T3 - 5.0f * T4 * 12.0f * T2)
                - T4 * (b1 * 20.0f * T3 - 5.0f * T4 * b2)
                + T5 * (b1 * 12.0f * T2 - 4.0f * T3 * b2);

    float detA4 = T3 * (b1 * 20.0f * T3 - 5.0f * T4 * b2)
                - b0 * (3.0f * T2 * 20.0f * T3 - 5.0f * T4 * 6.0f * T)
                + T5 * (3.0f * T2 * b2 - b1 * 6.0f * T);

    float detA5 = T3 * (4.0f * T3 * b2 - b1 * 12.0f * T2)
                - T4 * (3.0f * T2 * b2 - b1 * 6.0f * T)
                + b0 * (3.0f * T2 * 12.0f * T2 - 4.0f * T3 * 6.0f * T);

    a3 = detA3 / detA;
    a4 = detA4 / detA;
    a5 = detA5 / detA;
}

// ---------------------------------------------------------------------------
// Device: solve quartic polynomial coefficients (a3, a4) via 2x2 system
//   [3T2  4T3] [a3]   [vxe - a1 - 2*a2*T]
//   [6T  12T2] [a4] = [axe - 2*a2        ]
// ---------------------------------------------------------------------------
__device__ void d_solve_quartic(float T, float a1, float a2,
                                float vxe, float axe,
                                float& a3, float& a4) {
    float T2 = T * T, T3 = T2 * T;
    float det = 3.0f * T2 * 12.0f * T2 - 4.0f * T3 * 6.0f * T;
    if (fabsf(det) < 1e-10f) { a3 = a4 = 0; return; }

    float b0 = vxe - a1 - 2.0f * a2 * T;
    float b1 = axe - 2.0f * a2;

    a3 = (12.0f * T2 * b0 - 4.0f * T3 * b1) / det;
    a4 = (3.0f * T2 * b1 - 6.0f * T * b0) / det;
}

// ---------------------------------------------------------------------------
// Device: polynomial evaluation helpers
// ---------------------------------------------------------------------------
__device__ float d_quintic_eval(float a0, float a1, float a2,
                                float a3, float a4, float a5, float t) {
    return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t;
}
__device__ float d_quintic_d1(float a1, float a2, float a3, float a4, float a5, float t) {
    return a1 + 2*a2*t + 3*a3*t*t + 4*a4*t*t*t + 5*a5*t*t*t*t;
}
__device__ float d_quintic_d2(float a2, float a3, float a4, float a5, float t) {
    return 2*a2 + 6*a3*t + 12*a4*t*t + 20*a5*t*t*t;
}
__device__ float d_quintic_d3(float a3, float a4, float a5, float t) {
    return 6*a3 + 24*a4*t + 60*a5*t*t;
}

__device__ float d_quartic_eval(float a0, float a1, float a2,
                                float a3, float a4, float t) {
    return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t;
}
__device__ float d_quartic_d1(float a1, float a2, float a3, float a4, float t) {
    return a1 + 2*a2*t + 3*a3*t*t + 4*a4*t*t*t;
}
__device__ float d_quartic_d2(float a2, float a3, float a4, float t) {
    return 2*a2 + 6*a3*t + 12*a4*t*t;
}
__device__ float d_quartic_d3(float a3, float a4, float t) {
    return 6*a3 + 24*a4*t;
}

// ---------------------------------------------------------------------------
// Kernel: frenet path generation + global conversion + collision + cost
//   Grid: (di_idx, Ti_idx, tv_idx) → 1D thread index
// ---------------------------------------------------------------------------
struct PathResult {
    float cost;
    int n_points;
    float x[MAX_TRAJ_POINTS];
    float y[MAX_TRAJ_POINTS];
    // frenet state at t=DT (index 1) for state update
    float s1, d1, d_d1, d_dd1, s_d1;
};

__global__ void frenet_paths_kernel(
    // current frenet state
    float c_speed, float c_d, float c_d_d, float c_d_dd, float s0,
    // sampling grid info
    int n_di, int n_Ti, int n_tv,
    float di_start, float di_step,
    float Ti_start, float Ti_step,
    float tv_start, float tv_step,
    // spline data (sx and sy)
    DeviceSpline dsx, DeviceSpline dsy,
    float s_max,
    // obstacles [n_ob x 2]
    const float* obs, int n_ob,
    // output
    PathResult* results,
    int max_results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_di * n_Ti * n_tv;
    if (idx >= total || idx >= max_results) return;

    int i_di = idx / (n_Ti * n_tv);
    int rem  = idx % (n_Ti * n_tv);
    int i_Ti = rem / n_tv;
    int i_tv = rem % n_tv;

    float di = di_start + i_di * di_step;
    float Ti = Ti_start + i_Ti * Ti_step;
    float tv = tv_start + i_tv * tv_step;

    PathResult& res = results[idx];
    res.cost = FLT_MAX;
    res.n_points = 0;

    // --- Lateral: quintic polynomial ---
    float lat_a0 = c_d;
    float lat_a1 = c_d_d;
    float lat_a2 = c_d_dd / 2.0f;
    float lat_a3, lat_a4, lat_a5;
    d_solve_quintic(Ti, lat_a0, lat_a1, lat_a2, di, 0.0f, 0.0f,
                    lat_a3, lat_a4, lat_a5);

    // --- Longitudinal: quartic polynomial ---
    float lon_a0 = s0;
    float lon_a1 = c_speed;
    float lon_a2 = 0.0f;  // axs / 2.0
    float lon_a3, lon_a4;
    d_solve_quartic(Ti, lon_a1, lon_a2, tv, 0.0f, lon_a3, lon_a4);

    // --- Generate trajectory points ---
    int np = 0;
    float Jp = 0, Js = 0;
    float max_speed_val = -FLT_MAX;
    float max_accel_val = -FLT_MAX;
    float max_curv_val  = -FLT_MAX;

    float prev_x = 0, prev_y = 0;
    bool first = true;
    bool valid = true;

    float saved_s1 = 0, saved_d1 = 0, saved_d_d1 = 0, saved_d_dd1 = 0, saved_s_d1 = 0;

    for (float t = 0; t < Ti && np < MAX_TRAJ_POINTS; t += DT) {
        float d_val  = d_quintic_eval(lat_a0, lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, t);
        float d_d3   = d_quintic_d3(lat_a3, lat_a4, lat_a5, t);

        float s_val  = d_quartic_eval(lon_a0, lon_a1, lon_a2, lon_a3, lon_a4, t);
        float s_d    = d_quartic_d1(lon_a1, lon_a2, lon_a3, lon_a4, t);
        float s_dd   = d_quartic_d2(lon_a2, lon_a3, lon_a4, t);
        float s_d3   = d_quartic_d3(lon_a3, lon_a4, t);

        Jp += d_d3 * d_d3;
        Js += s_d3 * s_d3;
        if (s_d > max_speed_val) max_speed_val = s_d;
        if (s_dd > max_accel_val) max_accel_val = s_dd;

        // save state at second point (index 1)
        if (np == 1) {
            saved_s1 = s_val;
            saved_d1 = d_val;
            saved_d_d1 = d_quintic_d1(lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, t);
            saved_d_dd1 = d_quintic_d2(lat_a2, lat_a3, lat_a4, lat_a5, t);
            saved_s_d1 = s_d;
        }

        // convert to global via spline
        if (s_val >= s_max) { valid = false; break; }

        float gx_s = d_spline_calc(dsx, s_val);
        float gy_s = d_spline_calc(dsy, s_val);
        float dx_s = d_spline_calc_d(dsx, s_val);
        float dy_s = d_spline_calc_d(dsy, s_val);
        float iyaw = atan2f(dy_s, dx_s);

        float gx = gx_s + d_val * cosf(iyaw + M_PIf / 2.0f);
        float gy = gy_s + d_val * sinf(iyaw + M_PIf / 2.0f);

        res.x[np] = gx;
        res.y[np] = gy;

        // curvature from consecutive points
        if (!first) {
            float ddx = gx - prev_x;
            float ddy = gy - prev_y;
            float ds = sqrtf(ddx * ddx + ddy * ddy);
            if (np >= 2 && ds > 1e-6f) {
                // approximate curvature from yaw change
                float yaw_now  = atan2f(gy - prev_y, gx - prev_x);
                // We'll compute curvature below after all points
            }
        }
        prev_x = gx;
        prev_y = gy;
        first = false;
        np++;
    }

    if (!valid || np < 3) { return; }

    // --- compute curvature from generated global path ---
    for (int i = 0; i < np - 1; i++) {
        float ddx = res.x[i + 1] - res.x[i];
        float ddy = res.y[i + 1] - res.y[i];
        float ds = sqrtf(ddx * ddx + ddy * ddy);
        if (i > 0 && ds > 1e-6f) {
            float yaw_prev = atan2f(res.y[i] - res.y[i - 1], res.x[i] - res.x[i - 1]);
            float yaw_curr = atan2f(ddy, ddx);
            float curv = fabsf(yaw_curr - yaw_prev) / ds;
            if (curv > max_curv_val) max_curv_val = curv;
        }
    }

    // --- constraint checks ---
    if (max_speed_val >= MAX_SPEED) return;
    if (max_accel_val >= MAX_ACCEL) return;
    if (max_curv_val >= MAX_CURVATURE) return;

    // --- collision check ---
    for (int io = 0; io < n_ob; io++) {
        float ox = obs[io * 2 + 0];
        float oy = obs[io * 2 + 1];
        for (int ip = 0; ip < np; ip++) {
            float ddx = res.x[ip] - ox;
            float ddy = res.y[ip] - oy;
            if (ddx * ddx + ddy * ddy <= ROBOT_RADIUS * ROBOT_RADIUS) {
                return;  // collision
            }
        }
    }

    // --- cost ---
    float d_last = d_quintic_eval(lat_a0, lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, Ti - DT);
    float s_d_last = d_quartic_d1(lon_a1, lon_a2, lon_a3, lon_a4, Ti - DT);
    float ds_cost = TARGET_SPEED - s_d_last;

    float cd = KJ * Jp + KT * Ti + KD * d_last * d_last;
    float cv = KJ * Js + KT * Ti + KD * ds_cost * ds_cost;
    float cf = KLAT * cd + KLON * cv;

    res.cost = cf;
    res.n_points = np;
    res.s1 = saved_s1;
    res.d1 = saved_d1;
    res.d_d1 = saved_d_d1;
    res.d_dd1 = saved_d_dd1;
    res.s_d1 = saved_s_d1;
}

// ---------------------------------------------------------------------------
// Kernel: find minimum cost path (reduction)
// ---------------------------------------------------------------------------
__global__ void find_best_path_kernel(const PathResult* results, int n,
                                      int* best_idx) {
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int*   sidx = (int*)(smem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    float best_val = FLT_MAX;
    int   best_i = -1;

    for (int i = tid; i < n; i += blockDim.x) {
        if (results[i].cost < best_val) {
            best_val = results[i].cost;
            best_i = i;
        }
    }
    sval[tid] = best_val;
    sidx[tid] = best_i;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] < sval[tid]) {
            sval[tid] = sval[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) *best_idx = sidx[0];
}

// ---------------------------------------------------------------------------
// Host: upload spline coefficients to device
// ---------------------------------------------------------------------------
struct DeviceSplineBuffers {
    float *knots, *a, *b, *c, *d;
    int nx, n_segs;
    float s_max;

    void upload(const Spline& sp) {
        nx = sp.nx;
        n_segs = sp.nx - 1;
        CUDA_CHECK(cudaMalloc(&knots, nx * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a, nx * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b, n_segs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c, nx * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d, n_segs * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(knots, sp.x.data(), nx * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(a, sp.a.data(), nx * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b, sp.b.data(), n_segs * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c, sp.c.data(), nx * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d, sp.d.data(), n_segs * sizeof(float), cudaMemcpyHostToDevice));
    }

    DeviceSpline to_device_spline() {
        DeviceSpline ds;
        ds.knots = knots;
        ds.a = a; ds.b = b; ds.c = c; ds.d = d;
        ds.n_segs = n_segs;
        ds.nx = nx;
        return ds;
    }

    void free() {
        cudaFree(knots); cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(d);
    }
};

// ---------------------------------------------------------------------------
// Host: visualization
// ---------------------------------------------------------------------------
cv::Point2i cv_offset(float x, float y,
                      int image_width = 2000, int image_height = 2000) {
    cv::Point2i output;
    output.x = int(x * 100) + 300;
    output.y = image_height - int(y * 100) - image_height / 3;
    return output;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // waypoints
    Vec_f wx = {0.0f, 10.0f, 20.5f, 35.0f, 70.5f};
    Vec_f wy = {0.0f, -6.0f, 5.0f, 6.5f, 0.0f};

    // obstacles
    std::vector<Poi_f> obstacles = {
        {{20.0f, 10.0f}}, {{30.0f, 6.0f}}, {{30.0f, 8.0f}},
        {{35.0f, 8.0f}},  {{50.0f, 3.0f}}
    };
    int n_ob = (int)obstacles.size();

    // build reference spline on host
    Spline2D csp_obj(wx, wy);
    Vec_f r_x, r_y;
    for (float i = 0; i < csp_obj.s.back(); i += 0.1f) {
        Poi_f p = csp_obj.calc_position(i);
        r_x.push_back(p[0]);
        r_y.push_back(p[1]);
    }

    // upload spline coefficients to device
    DeviceSplineBuffers sx_buf, sy_buf;
    sx_buf.upload(csp_obj.sx);
    sy_buf.upload(csp_obj.sy);
    DeviceSpline dsx = sx_buf.to_device_spline();
    DeviceSpline dsy = sy_buf.to_device_spline();
    float s_max = csp_obj.s.back();

    // upload obstacles
    float* d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, n_ob * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs, obstacles.data(), n_ob * 2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // compute sampling grid dimensions
    int n_di = (int)(2.0f * MAX_ROAD_WIDTH / D_ROAD_W) + 1;
    int n_Ti = (int)((MAXT - MINT) / DT) + 1;
    int n_tv = 2 * N_S_SAMPLE + 1;
    int total_paths = n_di * n_Ti * n_tv;

    std::cout << "Frenet Optimal Trajectory with CUDA ("
              << total_paths << " candidate paths)" << std::endl;

    // allocate device results
    PathResult* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, total_paths * sizeof(PathResult)));

    int* d_best_idx;
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));

    // host result buffer
    PathResult h_best;

    // frenet state
    float c_speed = 10.0f / 3.6f;
    float c_d = 2.0f, c_d_d = 0.0f, c_d_dd = 0.0f;
    float s0 = 0.0f;

    cv::namedWindow("frenet", cv::WINDOW_NORMAL);
    int threads = 256;

    for (int iter = 0; iter < SIM_LOOP; iter++) {
        float di_start = -MAX_ROAD_WIDTH;
        float Ti_start = MINT;
        float tv_start = TARGET_SPEED - D_T_S * N_S_SAMPLE;

        int blocks = (total_paths + threads - 1) / threads;

        frenet_paths_kernel<<<blocks, threads>>>(
            c_speed, c_d, c_d_d, c_d_dd, s0,
            n_di, n_Ti, n_tv,
            di_start, D_ROAD_W,
            Ti_start, DT,
            tv_start, D_T_S,
            dsx, dsy, s_max,
            d_obs, n_ob,
            d_results, total_paths);

        int red_threads = 256;
        size_t smem_sz = red_threads * (sizeof(float) + sizeof(int));
        find_best_path_kernel<<<1, red_threads, smem_sz>>>(
            d_results, total_paths, d_best_idx);

        int h_best_idx;
        CUDA_CHECK(cudaMemcpy(&h_best_idx, d_best_idx, sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (h_best_idx < 0) {
            std::cout << "No valid path found at iter " << iter << std::endl;
            break;
        }

        CUDA_CHECK(cudaMemcpy(&h_best, d_results + h_best_idx,
                              sizeof(PathResult), cudaMemcpyDeviceToHost));

        if (h_best.n_points < 2) {
            std::cout << "Best path too short at iter " << iter << std::endl;
            break;
        }

        // update frenet state
        s0     = h_best.s1;
        c_d    = h_best.d1;
        c_d_d  = h_best.d_d1;
        c_d_dd = h_best.d_dd1;
        c_speed = h_best.s_d1;

        // check goal
        float dx_goal = h_best.x[1] - r_x.back();
        float dy_goal = h_best.y[1] - r_y.back();
        if (dx_goal * dx_goal + dy_goal * dy_goal <= 1.0f) break;

        // --- visualization ---
        cv::Mat bg(2000, 8000, CV_8UC3, cv::Scalar(255, 255, 255));

        // reference path
        for (unsigned int j = 1; j < r_x.size(); j++) {
            cv::line(bg,
                     cv_offset(r_x[j - 1], r_y[j - 1], bg.cols, bg.rows),
                     cv_offset(r_x[j], r_y[j], bg.cols, bg.rows),
                     cv::Scalar(0, 0, 0), 10);
        }

        // best trajectory
        for (int j = 0; j < h_best.n_points; j++) {
            cv::circle(bg,
                       cv_offset(h_best.x[j], h_best.y[j], bg.cols, bg.rows),
                       40, cv::Scalar(255, 0, 0), -1);
        }

        // current position
        cv::circle(bg,
                   cv_offset(h_best.x[0], h_best.y[0], bg.cols, bg.rows),
                   50, cv::Scalar(0, 255, 0), -1);

        // obstacles
        for (int j = 0; j < n_ob; j++) {
            cv::circle(bg,
                       cv_offset(obstacles[j][0], obstacles[j][1], bg.cols, bg.rows),
                       40, cv::Scalar(0, 0, 255), 5);
        }

        cv::putText(bg,
                    "Speed: " + std::to_string(c_speed * 3.6f).substr(0, 4) + "km/h",
                    cv::Point2i((int)bg.cols / 2, (int)(bg.rows * 0.1)),
                    cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 0), 10);

        cv::imshow("frenet", bg);
        cv::waitKey(5);
    }

    // cleanup
    sx_buf.free();
    sy_buf.free();
    cudaFree(d_obs);
    cudaFree(d_results);
    cudaFree(d_best_idx);

    return 0;
}
