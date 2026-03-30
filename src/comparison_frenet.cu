/*************************************************************************
    Frenet Optimal Trajectory: CPU vs CUDA side-by-side comparison
    Left panel:  CPU sequential nested loops over (di, Ti, tv)
    Right panel: CUDA parallel (1 thread per candidate path)
    Output: gif/comparison_frenet.gif
 ************************************************************************/

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------------
// Constants
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

#define MAX_TRAJ_POINTS 30
#define MAX_PATHS 512
#define MAX_OBS 32

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

using Vec_f = std::vector<float>;
using Poi_f = std::array<float, 2>;

// ---------------------------------------------------------------------------
// Host: Cubic Spline (no Eigen -- Thomas algorithm for tridiagonal solve)
// ---------------------------------------------------------------------------
static Vec_f vec_diff(const Vec_f& input) {
    Vec_f output;
    for (unsigned int i = 1; i < input.size(); i++)
        output.push_back(input[i] - input[i - 1]);
    return output;
}

static Vec_f cum_sum(const Vec_f& input) {
    Vec_f output;
    float temp = 0;
    for (unsigned int i = 0; i < input.size(); i++) {
        temp += input[i];
        output.push_back(temp);
    }
    return output;
}

class Spline {
public:
    Vec_f x, y, h, a, b, c, d;
    int nx;

    Spline() : nx(0) {}

    Spline(Vec_f x_, Vec_f y_) : x(x_), y(y_), nx(x_.size()), h(vec_diff(x_)), a(y_) {
        // Build tridiagonal system for natural cubic spline
        // A * c = B  where A is tridiagonal
        int n = nx;
        std::vector<float> diag(n, 0.0f), upper(n, 0.0f), lower(n, 0.0f), rhs(n, 0.0f);

        diag[0] = 1.0f;
        diag[n - 1] = 1.0f;
        for (int i = 1; i < n - 1; i++) {
            lower[i] = h[i - 1];
            diag[i] = 2.0f * (h[i - 1] + h[i]);
            upper[i] = h[i];
            rhs[i] = 3.0f * (a[i + 1] - a[i]) / h[i] - 3.0f * (a[i] - a[i - 1]) / h[i - 1];
        }

        // Thomas algorithm (forward sweep)
        std::vector<float> cp(n, 0.0f), dp(n, 0.0f);
        cp[0] = upper[0] / diag[0];
        dp[0] = rhs[0] / diag[0];
        for (int i = 1; i < n; i++) {
            float m = diag[i] - lower[i] * cp[i - 1];
            cp[i] = upper[i] / m;
            dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m;
        }

        // Back substitution
        c.resize(n, 0.0f);
        c[n - 1] = dp[n - 1];
        for (int i = n - 2; i >= 0; i--)
            c[i] = dp[i] - cp[i] * c[i + 1];

        // Compute b and d
        for (int i = 0; i < n - 1; i++) {
            d.push_back((c[i + 1] - c[i]) / (3.0f * h[i]));
            b.push_back((a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0f * c[i]) / 3.0f);
        }
    }

    int bisect(float t, int start, int end) const {
        int mid = (start + end) / 2;
        if (t == x[mid] || end - start <= 1) return mid;
        else if (t > x[mid]) return bisect(t, mid, end);
        else return bisect(t, start, mid);
    }

    float calc(float t) const {
        int seg = bisect(t, 0, nx);
        float dx = t - x[seg];
        return a[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx * dx * dx;
    }
    float calc_d(float t) const {
        int seg = bisect(t, 0, nx - 1);
        float dx = t - x[seg];
        return b[seg] + 2 * c[seg] * dx + 3 * d[seg] * dx * dx;
    }
    float calc_dd(float t) const {
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

    Poi_f calc_position(float s_t) { return {{sx.calc(s_t), sy.calc(s_t)}}; }
    float calc_yaw(float s_t) { return atan2f(sy.calc_d(s_t), sx.calc_d(s_t)); }
    float calc_curvature(float s_t) {
        float ddx = sx.calc_d(s_t), dddx = sx.calc_dd(s_t);
        float ddy = sy.calc_d(s_t), dddy = sy.calc_dd(s_t);
        return (dddy * ddx - dddx * ddy) / (ddx * ddx + ddy * ddy);
    }
};

// ---------------------------------------------------------------------------
// Host: quintic/quartic polynomial solvers (CPU side)
// ---------------------------------------------------------------------------
static void cpu_solve_quintic(float T, float a0, float a1, float a2,
                              float xe, float vxe, float axe,
                              float& a3, float& a4, float& a5) {
    float T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
    float detA = T3 * (4.0f * T3 * 20.0f * T3 - 5.0f * T4 * 12.0f * T2)
               - T4 * (3.0f * T2 * 20.0f * T3 - 5.0f * T4 * 6.0f * T)
               + T5 * (3.0f * T2 * 12.0f * T2 - 4.0f * T3 * 6.0f * T);
    if (fabsf(detA) < 1e-10f) { a3 = a4 = a5 = 0; return; }
    float b0 = xe - a0 - a1 * T - a2 * T2;
    float b1 = vxe - a1 - 2.0f * a2 * T;
    float b2 = axe - 2.0f * a2;
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

static void cpu_solve_quartic(float T, float a1, float a2,
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

// Host: polynomial evaluation
static float cpu_quintic_eval(float a0, float a1, float a2, float a3, float a4, float a5, float t) {
    return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t;
}
static float cpu_quintic_d1(float a1, float a2, float a3, float a4, float a5, float t) {
    return a1 + 2*a2*t + 3*a3*t*t + 4*a4*t*t*t + 5*a5*t*t*t*t;
}
static float cpu_quintic_d2(float a2, float a3, float a4, float a5, float t) {
    return 2*a2 + 6*a3*t + 12*a4*t*t + 20*a5*t*t*t;
}
static float cpu_quintic_d3(float a3, float a4, float a5, float t) {
    return 6*a3 + 24*a4*t + 60*a5*t*t;
}
static float cpu_quartic_eval(float a0, float a1, float a2, float a3, float a4, float t) {
    return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t;
}
static float cpu_quartic_d1(float a1, float a2, float a3, float a4, float t) {
    return a1 + 2*a2*t + 3*a3*t*t + 4*a4*t*t*t;
}
static float cpu_quartic_d2(float a2, float a3, float a4, float t) {
    return 2*a2 + 6*a3*t + 12*a4*t*t;
}
static float cpu_quartic_d3(float a3, float a4, float t) {
    return 6*a3 + 24*a4*t;
}

// ---------------------------------------------------------------------------
// PathResult struct shared by CPU and GPU
// ---------------------------------------------------------------------------
struct PathResult {
    float cost;
    int n_points;
    float x[MAX_TRAJ_POINTS];
    float y[MAX_TRAJ_POINTS];
    float s1, d1, d_d1, d_dd1, s_d1;
};

// ---------------------------------------------------------------------------
// CPU: frenet path planning (sequential nested loops)
// ---------------------------------------------------------------------------
void cpu_calc_frenet_paths(
    float c_speed, float c_d, float c_d_d, float c_d_dd, float s0,
    const Spline2D& csp, float s_max,
    const std::vector<Poi_f>& obstacles,
    PathResult& best_result)
{
    best_result.cost = FLT_MAX;
    best_result.n_points = 0;
    int n_ob = (int)obstacles.size();

    for (float di = -MAX_ROAD_WIDTH; di <= MAX_ROAD_WIDTH; di += D_ROAD_W) {
        for (float Ti = MINT; Ti <= MAXT; Ti += DT) {
            for (int i_tv = -N_S_SAMPLE; i_tv <= N_S_SAMPLE; i_tv++) {
                float tv = TARGET_SPEED + i_tv * D_T_S;

                // Lateral: quintic
                float lat_a0 = c_d, lat_a1 = c_d_d, lat_a2 = c_d_dd / 2.0f;
                float lat_a3, lat_a4, lat_a5;
                cpu_solve_quintic(Ti, lat_a0, lat_a1, lat_a2, di, 0.0f, 0.0f,
                                  lat_a3, lat_a4, lat_a5);

                // Longitudinal: quartic
                float lon_a0 = s0, lon_a1 = c_speed, lon_a2 = 0.0f;
                float lon_a3, lon_a4;
                cpu_solve_quartic(Ti, lon_a1, lon_a2, tv, 0.0f, lon_a3, lon_a4);

                // Generate trajectory
                PathResult res;
                res.cost = FLT_MAX;
                int np = 0;
                float Jp = 0, Js = 0;
                float max_speed_val = -FLT_MAX;
                float max_accel_val = -FLT_MAX;
                bool valid = true;

                float saved_s1 = 0, saved_d1 = 0, saved_d_d1 = 0, saved_d_dd1 = 0, saved_s_d1 = 0;

                for (float t = 0; t < Ti && np < MAX_TRAJ_POINTS; t += DT) {
                    float d_val = cpu_quintic_eval(lat_a0, lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, t);
                    float d_d3  = cpu_quintic_d3(lat_a3, lat_a4, lat_a5, t);
                    float s_val = cpu_quartic_eval(lon_a0, lon_a1, lon_a2, lon_a3, lon_a4, t);
                    float s_d   = cpu_quartic_d1(lon_a1, lon_a2, lon_a3, lon_a4, t);
                    float s_dd  = cpu_quartic_d2(lon_a2, lon_a3, lon_a4, t);
                    float s_d3  = cpu_quartic_d3(lon_a3, lon_a4, t);

                    Jp += d_d3 * d_d3;
                    Js += s_d3 * s_d3;
                    if (s_d > max_speed_val) max_speed_val = s_d;
                    if (s_dd > max_accel_val) max_accel_val = s_dd;

                    if (np == 1) {
                        saved_s1 = s_val;
                        saved_d1 = d_val;
                        saved_d_d1 = cpu_quintic_d1(lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, t);
                        saved_d_dd1 = cpu_quintic_d2(lat_a2, lat_a3, lat_a4, lat_a5, t);
                        saved_s_d1 = s_d;
                    }

                    if (s_val >= s_max || s_val < 0) { valid = false; break; }

                    float gx_s = csp.sx.calc(s_val);
                    float gy_s = csp.sy.calc(s_val);
                    float dx_s = csp.sx.calc_d(s_val);
                    float dy_s = csp.sy.calc_d(s_val);
                    float iyaw = atan2f(dy_s, dx_s);

                    res.x[np] = gx_s + d_val * cosf(iyaw + M_PIf / 2.0f);
                    res.y[np] = gy_s + d_val * sinf(iyaw + M_PIf / 2.0f);
                    np++;
                }

                if (!valid || np < 3) continue;

                // Curvature check
                float max_curv_val = -FLT_MAX;
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

                if (max_speed_val >= MAX_SPEED) continue;
                if (max_accel_val >= MAX_ACCEL) continue;
                if (max_curv_val >= MAX_CURVATURE) continue;

                // Collision check
                bool collision = false;
                for (int io = 0; io < n_ob && !collision; io++) {
                    for (int ip = 0; ip < np; ip++) {
                        float ddx = res.x[ip] - obstacles[io][0];
                        float ddy = res.y[ip] - obstacles[io][1];
                        if (ddx * ddx + ddy * ddy <= ROBOT_RADIUS * ROBOT_RADIUS) {
                            collision = true; break;
                        }
                    }
                }
                if (collision) continue;

                // Cost
                float d_last = cpu_quintic_eval(lat_a0, lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, Ti - DT);
                float s_d_last = cpu_quartic_d1(lon_a1, lon_a2, lon_a3, lon_a4, Ti - DT);
                float ds_cost = TARGET_SPEED - s_d_last;
                float cd = KJ * Jp + KT * Ti + KD * d_last * d_last;
                float cv = KJ * Js + KT * Ti + KD * ds_cost * ds_cost;
                float cf = KLAT * cd + KLON * cv;

                if (cf < best_result.cost) {
                    best_result.cost = cf;
                    best_result.n_points = np;
                    for (int k = 0; k < np; k++) {
                        best_result.x[k] = res.x[k];
                        best_result.y[k] = res.y[k];
                    }
                    best_result.s1 = saved_s1;
                    best_result.d1 = saved_d1;
                    best_result.d_d1 = saved_d_d1;
                    best_result.d_dd1 = saved_d_dd1;
                    best_result.s_d1 = saved_s_d1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Device: spline structures and evaluation (same as frenet_optimal_trajectory.cu)
// ---------------------------------------------------------------------------
struct DeviceSpline {
    float* knots;
    float* a;
    float* b;
    float* c;
    float* d;
    int n_segs;
    int nx;
    float s_max;
};

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

// Device: polynomial solvers
__device__ void d_solve_quintic(float T, float a0, float a1, float a2,
                                float xe, float vxe, float axe,
                                float& a3, float& a4, float& a5) {
    float T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
    float detA = T3 * (4.0f * T3 * 20.0f * T3 - 5.0f * T4 * 12.0f * T2)
               - T4 * (3.0f * T2 * 20.0f * T3 - 5.0f * T4 * 6.0f * T)
               + T5 * (3.0f * T2 * 12.0f * T2 - 4.0f * T3 * 6.0f * T);
    if (fabsf(detA) < 1e-10f) { a3 = a4 = a5 = 0; return; }
    float b0 = xe - a0 - a1 * T - a2 * T2;
    float b1 = vxe - a1 - 2.0f * a2 * T;
    float b2 = axe - 2.0f * a2;
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

// Device: polynomial evaluation
__device__ float d_quintic_eval(float a0, float a1, float a2, float a3, float a4, float a5, float t) {
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
__device__ float d_quartic_eval(float a0, float a1, float a2, float a3, float a4, float t) {
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
// CUDA Kernel: frenet path generation + check + cost (1 thread per path)
// ---------------------------------------------------------------------------
__global__ void frenet_paths_kernel(
    float c_speed, float c_d, float c_d_d, float c_d_dd, float s0,
    int n_di, int n_Ti, int n_tv,
    float di_start, float di_step,
    float Ti_start, float Ti_step,
    float tv_start, float tv_step,
    DeviceSpline dsx, DeviceSpline dsy,
    float s_max,
    const float* obs, int n_ob,
    PathResult* results, int max_results)
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

    // Lateral: quintic
    float lat_a0 = c_d, lat_a1 = c_d_d, lat_a2 = c_d_dd / 2.0f;
    float lat_a3, lat_a4, lat_a5;
    d_solve_quintic(Ti, lat_a0, lat_a1, lat_a2, di, 0.0f, 0.0f, lat_a3, lat_a4, lat_a5);

    // Longitudinal: quartic
    float lon_a0 = s0, lon_a1 = c_speed, lon_a2 = 0.0f;
    float lon_a3, lon_a4;
    d_solve_quartic(Ti, lon_a1, lon_a2, tv, 0.0f, lon_a3, lon_a4);

    int np = 0;
    float Jp = 0, Js = 0;
    float max_speed_val = -FLT_MAX, max_accel_val = -FLT_MAX, max_curv_val = -FLT_MAX;
    float prev_x = 0, prev_y = 0;
    bool first = true, valid = true;
    float saved_s1 = 0, saved_d1 = 0, saved_d_d1 = 0, saved_d_dd1 = 0, saved_s_d1 = 0;

    for (float t = 0; t < Ti && np < MAX_TRAJ_POINTS; t += DT) {
        float d_val = d_quintic_eval(lat_a0, lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, t);
        float d_d3  = d_quintic_d3(lat_a3, lat_a4, lat_a5, t);
        float s_val = d_quartic_eval(lon_a0, lon_a1, lon_a2, lon_a3, lon_a4, t);
        float s_d   = d_quartic_d1(lon_a1, lon_a2, lon_a3, lon_a4, t);
        float s_dd  = d_quartic_d2(lon_a2, lon_a3, lon_a4, t);
        float s_d3  = d_quartic_d3(lon_a3, lon_a4, t);

        Jp += d_d3 * d_d3;
        Js += s_d3 * s_d3;
        if (s_d > max_speed_val) max_speed_val = s_d;
        if (s_dd > max_accel_val) max_accel_val = s_dd;

        if (np == 1) {
            saved_s1 = s_val; saved_d1 = d_val;
            saved_d_d1 = d_quintic_d1(lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, t);
            saved_d_dd1 = d_quintic_d2(lat_a2, lat_a3, lat_a4, lat_a5, t);
            saved_s_d1 = s_d;
        }

        if (s_val >= s_max) { valid = false; break; }

        float gx_s = d_spline_calc(dsx, s_val);
        float gy_s = d_spline_calc(dsy, s_val);
        float dx_s = d_spline_calc_d(dsx, s_val);
        float dy_s = d_spline_calc_d(dsy, s_val);
        float iyaw = atan2f(dy_s, dx_s);

        res.x[np] = gx_s + d_val * cosf(iyaw + M_PIf / 2.0f);
        res.y[np] = gy_s + d_val * sinf(iyaw + M_PIf / 2.0f);
        prev_x = res.x[np]; prev_y = res.y[np];
        first = false;
        np++;
    }

    if (!valid || np < 3) return;

    // Curvature
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

    if (max_speed_val >= MAX_SPEED) return;
    if (max_accel_val >= MAX_ACCEL) return;
    if (max_curv_val >= MAX_CURVATURE) return;

    // Collision
    for (int io = 0; io < n_ob; io++) {
        float ox = obs[io * 2 + 0], oy = obs[io * 2 + 1];
        for (int ip = 0; ip < np; ip++) {
            float ddx = res.x[ip] - ox, ddy = res.y[ip] - oy;
            if (ddx * ddx + ddy * ddy <= ROBOT_RADIUS * ROBOT_RADIUS) return;
        }
    }

    // Cost
    float d_last = d_quintic_eval(lat_a0, lat_a1, lat_a2, lat_a3, lat_a4, lat_a5, Ti - DT);
    float s_d_last = d_quartic_d1(lon_a1, lon_a2, lon_a3, lon_a4, Ti - DT);
    float ds_cost = TARGET_SPEED - s_d_last;
    float cd = KJ * Jp + KT * Ti + KD * d_last * d_last;
    float cv = KJ * Js + KT * Ti + KD * ds_cost * ds_cost;
    res.cost = KLAT * cd + KLON * cv;
    res.n_points = np;
    res.s1 = saved_s1; res.d1 = saved_d1;
    res.d_d1 = saved_d_d1; res.d_dd1 = saved_d_dd1; res.s_d1 = saved_s_d1;
}

// ---------------------------------------------------------------------------
// CUDA Kernel: find best path (reduction)
// ---------------------------------------------------------------------------
__global__ void find_best_path_kernel(const PathResult* results, int n, int* best_idx) {
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int*   sidx = (int*)(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    float best_val = FLT_MAX; int best_i = -1;
    for (int i = tid; i < n; i += blockDim.x) {
        if (results[i].cost < best_val) { best_val = results[i].cost; best_i = i; }
    }
    sval[tid] = best_val; sidx[tid] = best_i; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] < sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *best_idx = sidx[0];
}

// ---------------------------------------------------------------------------
// Host: upload spline to device
// ---------------------------------------------------------------------------
struct DeviceSplineBuffers {
    float *knots, *a, *b, *c, *d;
    int nx, n_segs;

    void upload(const Spline& sp) {
        nx = sp.nx; n_segs = sp.nx - 1;
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
        ds.knots = knots; ds.a = a; ds.b = b; ds.c = c; ds.d = d;
        ds.n_segs = n_segs; ds.nx = nx;
        return ds;
    }
    void free_mem() {
        cudaFree(knots); cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(d);
    }
};

// ---------------------------------------------------------------------------
// Visualization helper
// ---------------------------------------------------------------------------
static cv::Point2i cv_offset(float x, float y, int w, int h) {
    return cv::Point2i(int(x * 100) + 300, h - int(y * 100) - h / 3);
}

static void draw_frenet_scene(cv::Mat& img,
    const Vec_f& r_x, const Vec_f& r_y,
    const PathResult& best,
    const std::vector<Poi_f>& obstacles,
    const char* label, double ms)
{
    // Reference path (black)
    for (unsigned int j = 1; j < r_x.size(); j++) {
        cv::line(img,
                 cv_offset(r_x[j - 1], r_y[j - 1], img.cols, img.rows),
                 cv_offset(r_x[j], r_y[j], img.cols, img.rows),
                 cv::Scalar(0, 0, 0), 5);
    }

    // Best trajectory (blue)
    for (int j = 0; j < best.n_points; j++) {
        cv::circle(img,
                   cv_offset(best.x[j], best.y[j], img.cols, img.rows),
                   20, cv::Scalar(255, 0, 0), -1);
    }

    // Current position (green)
    if (best.n_points > 0) {
        cv::circle(img,
                   cv_offset(best.x[0], best.y[0], img.cols, img.rows),
                   30, cv::Scalar(0, 255, 0), -1);
    }

    // Obstacles (red)
    for (auto& o : obstacles) {
        cv::circle(img,
                   cv_offset(o[0], o[1], img.cols, img.rows),
                   25, cv::Scalar(0, 0, 255), 5);
    }

    // Label and timing
    cv::putText(img, label, cv::Point(20, 60),
                cv::FONT_HERSHEY_SIMPLEX, 1.8, cv::Scalar(0, 0, 0), 3);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f ms/step", ms);
    cv::putText(img, buf, cv::Point(20, 130),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 200), 2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "Frenet Optimal Trajectory: CPU vs CUDA Comparison" << std::endl;

    // Waypoints
    Vec_f wx = {0.0f, 10.0f, 20.5f, 35.0f, 70.5f};
    Vec_f wy = {0.0f, -6.0f, 5.0f, 6.5f, 0.0f};

    // Obstacles
    std::vector<Poi_f> obstacles = {
        {{20.0f, 10.0f}}, {{30.0f, 6.0f}}, {{30.0f, 8.0f}},
        {{35.0f, 8.0f}},  {{50.0f, 3.0f}}
    };
    int n_ob = (int)obstacles.size();

    // Build reference spline
    Spline2D csp(wx, wy);
    Vec_f r_x, r_y;
    for (float i = 0; i < csp.s.back(); i += 0.1f) {
        Poi_f p = csp.calc_position(i);
        r_x.push_back(p[0]);
        r_y.push_back(p[1]);
    }
    float s_max = csp.s.back();

    // Upload spline to device
    DeviceSplineBuffers sx_buf, sy_buf;
    sx_buf.upload(csp.sx);
    sy_buf.upload(csp.sy);
    DeviceSpline dsx = sx_buf.to_device_spline();
    DeviceSpline dsy = sy_buf.to_device_spline();

    // Upload obstacles to device
    float* d_obs;
    CUDA_CHECK(cudaMalloc(&d_obs, n_ob * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_obs, obstacles.data(), n_ob * 2 * sizeof(float), cudaMemcpyHostToDevice));

    // Sampling grid
    int n_di = (int)(2.0f * MAX_ROAD_WIDTH / D_ROAD_W) + 1;
    int n_Ti = (int)((MAXT - MINT) / DT) + 1;
    int n_tv = 2 * N_S_SAMPLE + 1;
    int total_paths = n_di * n_Ti * n_tv;

    std::cout << "Candidate paths per step: " << total_paths << std::endl;

    // Allocate device results
    PathResult* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, total_paths * sizeof(PathResult)));
    int* d_best_idx;
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));

    // Frenet state (independent for CPU and CUDA)
    float cpu_c_speed = 10.0f / 3.6f, cpu_c_d = 2.0f, cpu_c_d_d = 0.0f, cpu_c_d_dd = 0.0f, cpu_s0 = 0.0f;
    float gpu_c_speed = 10.0f / 3.6f, gpu_c_d = 2.0f, gpu_c_d_d = 0.0f, gpu_c_d_dd = 0.0f, gpu_s0 = 0.0f;

    int W = 4000, H = 1000;
    cv::VideoWriter video(
        "gif/comparison_frenet.avi",
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(W * 2, H));

    if (!video.isOpened()) {
        std::cerr << "Failed to open video writer" << std::endl;
        return 1;
    }

    int threads = 256;

    for (int iter = 0; iter < SIM_LOOP; iter++) {
        // ===================== CPU =====================
        PathResult cpu_best;
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_calc_frenet_paths(cpu_c_speed, cpu_c_d, cpu_c_d_d, cpu_c_d_dd, cpu_s0,
                              csp, s_max, obstacles, cpu_best);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (cpu_best.n_points < 2) {
            std::cout << "CPU: no valid path at iter " << iter << std::endl;
            break;
        }

        // Update CPU state
        cpu_s0     = cpu_best.s1;
        cpu_c_d    = cpu_best.d1;
        cpu_c_d_d  = cpu_best.d_d1;
        cpu_c_d_dd = cpu_best.d_dd1;
        cpu_c_speed = cpu_best.s_d1;

        // ===================== CUDA =====================
        float di_start = -MAX_ROAD_WIDTH;
        float Ti_start = MINT;
        float tv_start = TARGET_SPEED - D_T_S * N_S_SAMPLE;
        int blocks = (total_paths + threads - 1) / threads;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        frenet_paths_kernel<<<blocks, threads>>>(
            gpu_c_speed, gpu_c_d, gpu_c_d_d, gpu_c_d_dd, gpu_s0,
            n_di, n_Ti, n_tv,
            di_start, D_ROAD_W, Ti_start, DT, tv_start, D_T_S,
            dsx, dsy, s_max,
            d_obs, n_ob,
            d_results, total_paths);

        int red_threads = 256;
        size_t smem_sz = red_threads * (sizeof(float) + sizeof(int));
        find_best_path_kernel<<<1, red_threads, smem_sz>>>(d_results, total_paths, d_best_idx);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        int h_best_idx;
        CUDA_CHECK(cudaMemcpy(&h_best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_best_idx < 0) {
            std::cout << "CUDA: no valid path at iter " << iter << std::endl;
            break;
        }

        PathResult gpu_best;
        CUDA_CHECK(cudaMemcpy(&gpu_best, d_results + h_best_idx, sizeof(PathResult), cudaMemcpyDeviceToHost));

        if (gpu_best.n_points < 2) {
            std::cout << "CUDA: best path too short at iter " << iter << std::endl;
            break;
        }

        // Update CUDA state
        gpu_s0     = gpu_best.s1;
        gpu_c_d    = gpu_best.d1;
        gpu_c_d_d  = gpu_best.d_d1;
        gpu_c_d_dd = gpu_best.d_dd1;
        gpu_c_speed = gpu_best.s_d1;

        // Check goal
        float dx_cpu = cpu_best.x[1] - r_x.back();
        float dy_cpu = cpu_best.y[1] - r_y.back();
        float dx_gpu = gpu_best.x[1] - r_x.back();
        float dy_gpu = gpu_best.y[1] - r_y.back();
        bool cpu_done = (dx_cpu * dx_cpu + dy_cpu * dy_cpu <= 1.0f);
        bool gpu_done = (dx_gpu * dx_gpu + dy_gpu * dy_gpu <= 1.0f);

        // ===================== Visualization =====================
        cv::Mat left(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat right(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

        draw_frenet_scene(left, r_x, r_y, cpu_best, obstacles, "CPU (C++)", cpu_ms);
        draw_frenet_scene(right, r_x, r_y, gpu_best, obstacles, "CUDA (GPU)", (double)cuda_ms);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);

        if (iter % 10 == 0)
            printf("Iter %3d  CPU: %.2f ms  CUDA: %.2f ms\n", iter, cpu_ms, (double)cuda_ms);

        if (cpu_done && gpu_done) break;
    }

    video.release();
    std::cout << "Video saved to gif/comparison_frenet.avi" << std::endl;

    // Convert to gif
    system("ffmpeg -y -i gif/comparison_frenet.avi "
           "-vf 'fps=15,scale=800:-1:flags=lanczos' -loop 0 "
           "gif/comparison_frenet.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_frenet.gif" << std::endl;

    // Cleanup
    sx_buf.free_mem();
    sy_buf.free_mem();
    cudaFree(d_obs);
    cudaFree(d_results);
    cudaFree(d_best_idx);

    return 0;
}
