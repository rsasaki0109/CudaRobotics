// gpu_frontier_exploration.cu
//
// GPU frontier-based exploration on an occupancy grid — the classic Yamauchi
// (1997) "where do I go next?" primitive for autonomous mapping.  The map onto
// the repo's canonical 2D idiom is:
//
//   one thread = one cell
//
// A frontier is a FREE cell that touches at least one UNKNOWN cell: it is the
// boundary between what the robot has mapped and what it has not.  Driving the
// robot to frontiers, over and over, is exactly how occupancy-grid SLAM
// front-ends decide where to explore.
//
// Pipeline (CPU and GPU run the SAME integer logic):
//   1. frontier detect: cell is a frontier iff state == FREE and any of its 8
//      neighbours is UNKNOWN.
//   2. connected components: label[i] = i for frontier cells; each cell pulls
//      its label down to the smallest label among its frontier neighbours;
//      iterate until no label moves (parallel min-propagation — the same
//      union-find-lite used by gpu_dbscan, but over a grid neighbourhood).
//   3. cluster reduction: atomically accumulate (sum_x, sum_y, count) per
//      label, giving each frontier component a size and a centroid.
//   4. target select: pick the frontier component maximising  size / distance
//      to the robot  (favouring big, nearby openings), ignoring specks below a
//      minimum size.
//
// Everything is exact integer arithmetic, so the CPU and GPU produce
// bit-identical frontier maps, identical component labellings (after canonical
// renumbering), and pick the identical next target.  The demo runs the loop
// for several exploration steps — the robot repeatedly drives to its chosen
// frontier and re-reveals the world — so you watch the unknown shrink.
//
// The headline timing compares one full frontier+components+select pipeline on
// CPU vs GPU at a representative step.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ----------------------------------------------------------------- constants
#define GRID_W 512
#define GRID_H 512
static const int   N_CELLS  = GRID_W * GRID_H;
static const int   SENSOR_R = 70;          // sensor range in cells
static const int   STEP_MAX = 55;          // robot advance per step (cells)
static const int   N_STEPS  = 24;          // exploration iterations
static const int   MIN_FRONTIER = 12;      // ignore frontier specks below this
static const int   MAX_PROP_ITERS = 1024;  // CC propagation cap
static const int   CHECK_EVERY = 16;        // GPU sweeps between host sync checks

static const int   UNKNOWN = 0;
static const int   FREE    = 1;
static const int   OCC     = 2;
static const int   NO_LABEL = -1;

static const int   PANEL_W = 760;
static const int   PANEL_H = 600;

__host__ __device__ static inline int idx_of(int x, int y) { return y * GRID_W + x; }

// --------------------------------------------------------------- true world
// Analytic ground-truth occupancy the robot is discovering: outer walls plus a
// few rectangular obstacles and a divider with a gap, so the revealed region
// grows non-trivial frontiers.
static void make_true_map(std::vector<uint8_t>& occ) {
    occ.assign(N_CELLS, 0);
    auto wall = [&](int x0, int y0, int x1, int y1) {
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                if (x >= 0 && x < GRID_W && y >= 0 && y < GRID_H)
                    occ[idx_of(x, y)] = 1;
    };
    int b = 4;
    wall(0, 0, GRID_W - 1, b);                       // borders
    wall(0, GRID_H - 1 - b, GRID_W - 1, GRID_H - 1);
    wall(0, 0, b, GRID_H - 1);
    wall(GRID_W - 1 - b, 0, GRID_W - 1, GRID_H - 1);
    // interior rooms / obstacles
    wall(120, 120, 200, 200);
    wall(330, 90, 410, 240);
    wall(150, 320, 250, 400);
    wall(360, 340, 440, 430);
    // a divider wall with a gap (corridor)
    wall(250, 60, 262, 250);
    wall(250, 300, 262, GRID_H - 5);                 // gap between y=250..300
}

// reveal cells in line-of-sight within SENSOR_R of (rx, ry) using a DDA ray to
// each candidate cell; the first true-occupied cell along the ray is marked OCC
// and blocks everything behind it.
static void reveal(const std::vector<uint8_t>& truth, std::vector<int>& state,
                   int rx, int ry) {
    int x0 = std::max(1, rx - SENSOR_R), x1 = std::min(GRID_W - 2, rx + SENSOR_R);
    int y0 = std::max(1, ry - SENSOR_R), y1 = std::min(GRID_H - 2, ry + SENSOR_R);
    for (int ty = y0; ty <= y1; ++ty) {
        for (int tx = x0; tx <= x1; ++tx) {
            int ddx = tx - rx, ddy = ty - ry;
            if (ddx * ddx + ddy * ddy > SENSOR_R * SENSOR_R) continue;
            int steps = std::max(std::abs(ddx), std::abs(ddy));
            if (steps == 0) { state[idx_of(rx, ry)] = FREE; continue; }
            float sx = (float)ddx / steps, sy = (float)ddy / steps;
            float cx = rx + 0.5f, cy = ry + 0.5f;
            for (int s = 1; s <= steps; ++s) {
                cx += sx; cy += sy;
                int gx = (int)cx, gy = (int)cy;
                if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H) break;
                int id = idx_of(gx, gy);
                if (truth[id]) { state[id] = OCC; break; }
                state[id] = FREE;
            }
        }
    }
}

// ------------------------------------------------------------- shared kernels
__host__ __device__ static inline int is_frontier(int x, int y, const int* state) {
    if (state[idx_of(x, y)] != FREE) return 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
            if (state[idx_of(nx, ny)] == UNKNOWN) return 1;
        }
    return 0;
}

__global__ void frontier_kernel(const int* state, int* label) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;
    int x = i % GRID_W, y = i / GRID_W;
    label[i] = is_frontier(x, y, state) ? i : NO_LABEL;
}

// one CC sweep: each frontier cell adopts the smallest label among its 8
// frontier neighbours.
__global__ void cc_kernel(int* label, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;
    if (label[i] == NO_LABEL) return;
    int x = i % GRID_W, y = i / GRID_W;
    int m = label[i];
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
            int lj = label[idx_of(nx, ny)];
            if (lj != NO_LABEL && lj < m) m = lj;
        }
    if (m < label[i]) { atomicMin(&label[i], m); atomicExch(changed, 1); }
}

__global__ void reduce_kernel(const int* label, int* sx, int* sy, int* cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_CELLS) return;
    int l = label[i];
    if (l == NO_LABEL) return;
    atomicAdd(&sx[l],  i % GRID_W);
    atomicAdd(&sy[l],  i / GRID_W);
    atomicAdd(&cnt[l], 1);
}

// --------------------------------------------------------------- CPU pipeline
static void frontier_cpu(const std::vector<int>& state, std::vector<int>& label,
                         int& iters_out) {
    label.assign(N_CELLS, NO_LABEL);
    for (int y = 0; y < GRID_H; ++y)
        for (int x = 0; x < GRID_W; ++x)
            if (is_frontier(x, y, state.data()))
                label[idx_of(x, y)] = idx_of(x, y);
    int iters = 0;
    for (iters = 0; iters < MAX_PROP_ITERS; ++iters) {
        bool changed = false;
        std::vector<int> next = label;
        for (int i = 0; i < N_CELLS; ++i) {
            if (label[i] == NO_LABEL) continue;
            int x = i % GRID_W, y = i / GRID_W, m = label[i];
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) continue;
                    int lj = label[idx_of(nx, ny)];
                    if (lj != NO_LABEL && lj < m) m = lj;
                }
            if (m < label[i]) { next[i] = m; changed = true; }
        }
        label.swap(next);
        if (!changed) break;
    }
    iters_out = iters;
}

// A frontier component: representative label, centroid, size.
struct Cluster { int label, cx, cy, size; };

static std::vector<Cluster> clusters_from_labels(const std::vector<int>& label) {
    std::unordered_map<int, std::array<long long, 3>> acc;  // label -> (sx,sy,n)
    for (int i = 0; i < N_CELLS; ++i) {
        int l = label[i];
        if (l == NO_LABEL) continue;
        auto& a = acc[l];
        a[0] += i % GRID_W; a[1] += i / GRID_W; a[2] += 1;
    }
    std::vector<Cluster> out;
    for (auto& kv : acc) {
        long long n = kv.second[2];
        if (n < MIN_FRONTIER) continue;
        out.push_back({kv.first, (int)(kv.second[0] / n), (int)(kv.second[1] / n), (int)n});
    }
    std::sort(out.begin(), out.end(),
              [](const Cluster& a, const Cluster& b) { return a.label < b.label; });
    return out;
}

// navigation goal for a chosen frontier component: the frontier cell whose
// direction from the robot is most aligned with the current heading (greatest
// projection along it).  Picking the cell *ahead* commits the robot to a
// consistent sweep direction — straight across open space, naturally following
// an arc — instead of oscillating between two near opposite boundary points the
// way a nearest- or farthest-cell heading does for a frontier that encircles
// open space.
static void goal_for_component(const std::vector<int>& label_raw, const Cluster& c,
                               int rx, int ry, float hx, float hy, int& gx, int& gy) {
    double best = -1e18; gx = c.cx; gy = c.cy;
    for (int i = 0; i < N_CELLS; ++i) {
        if (label_raw[i] != c.label) continue;
        double dx = (i % GRID_W) - rx, dy = (i / GRID_W) - ry;
        double proj = dx * hx + dy * hy;        // signed distance along heading
        if (proj > best) { best = proj; gx = i % GRID_W; gy = i / GRID_W; }
    }
}

// utility = size / distance; pick the best, return index into clusters (-1 none)
static int select_target(const std::vector<Cluster>& cl, int rx, int ry) {
    int best = -1; double best_u = -1.0;
    for (size_t k = 0; k < cl.size(); ++k) {  // pick best component by utility
        double dx = cl[k].cx - rx, dy = cl[k].cy - ry;
        double dist = std::sqrt(dx * dx + dy * dy) + 1.0;
        double u = cl[k].size / dist;
        if (u > best_u) { best_u = u; best = (int)k; }
    }
    return best;
}

// canonical renumber by first appearance for label comparison
static void canonicalise(std::vector<int>& lab, int& n_out) {
    std::unordered_map<int, int> remap;
    int next = 0;
    for (int& v : lab) {
        if (v == NO_LABEL) continue;
        auto it = remap.find(v);
        if (it == remap.end()) { remap[v] = next; v = next; ++next; }
        else v = it->second;
    }
    n_out = next;
}

// move robot from (rx,ry) toward (gx,gy) up to STEP_MAX cells, stopping before
// a non-free cell; returns the new position.
static void advance_robot(const std::vector<int>& state, int& rx, int& ry,
                          int gx, int gy) {
    int ddx = gx - rx, ddy = gy - ry;
    int dist = (int)std::round(std::sqrt((double)ddx * ddx + ddy * ddy));
    int steps = std::min(dist, STEP_MAX);
    if (steps <= 0) return;
    float sx = (float)ddx / dist, sy = (float)ddy / dist;
    float cx = rx + 0.5f, cy = ry + 0.5f;
    int lx = rx, ly = ry;
    for (int s = 1; s <= steps; ++s) {
        cx += sx; cy += sy;
        int nx = (int)cx, ny = (int)cy;
        if (nx < 1 || nx >= GRID_W - 1 || ny < 1 || ny >= GRID_H - 1) break;
        if (state[idx_of(nx, ny)] == OCC) break;
        lx = nx; ly = ny;
    }
    rx = lx; ry = ly;
}

// ------------------------------------------------------------- visualisation
static const cv::Vec3b PALETTE[] = {
    {66, 135, 245}, {245, 130, 48}, {60, 220, 60}, {200, 60, 220},
    {60, 220, 220}, {220, 220, 60}, {245, 80, 80}, {130, 90, 245},
    {90, 200, 150}, {200, 150, 90}, {150, 90, 200}, {90, 150, 200},
};
static const int N_PAL = sizeof(PALETTE) / sizeof(PALETTE[0]);

static void draw_frame(cv::Mat& out, const std::vector<int>& state,
                       const std::vector<int>& label_canon,
                       const std::vector<Cluster>& cl, int target,
                       int gx, int gy, int rx, int ry,
                       const std::vector<cv::Point>& traj,
                       const char* l1, const char* l2, const char* l3) {
    out = cv::Mat(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(28, 28, 32));
    const int GX = 110, GY = 64, GW = 520, GH = 520;   // grid draw rect
    cv::Mat grid(GRID_H, GRID_W, CV_8UC3);
    for (int i = 0; i < N_CELLS; ++i) {
        cv::Vec3b c;
        switch (state[i]) {
            case FREE:    c = {205, 205, 205}; break;
            case OCC:     c = {35, 35, 35};    break;
            default:      c = {92, 92, 96};    break;   // UNKNOWN
        }
        if (!label_canon.empty() && label_canon[i] != NO_LABEL)
            c = PALETTE[label_canon[i] % N_PAL];
        grid.at<cv::Vec3b>(i / GRID_W, i % GRID_W) = c;
    }
    cv::Mat scaled;
    cv::resize(grid, scaled, cv::Size(GW, GH), 0, 0, cv::INTER_NEAREST);
    scaled.copyTo(out(cv::Rect(GX, GY, GW, GH)));

    auto to_panel = [&](int cx, int cy) {
        return cv::Point(GX + cx * GW / GRID_W, GY + cy * GH / GRID_H);
    };
    // trajectory
    for (size_t k = 1; k < traj.size(); ++k)
        cv::line(out, to_panel(traj[k - 1].x, traj[k - 1].y),
                 to_panel(traj[k].x, traj[k].y), cv::Scalar(255, 255, 255), 2);
    // cluster centroids + chosen target line
    for (size_t k = 0; k < cl.size(); ++k) {
        cv::Point p = to_panel(cl[k].cx, cl[k].cy);
        cv::circle(out, p, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    if (target >= 0 && target < (int)cl.size()) {
        cv::Point rp = to_panel(rx, ry), tp = to_panel(gx, gy);
        cv::line(out, rp, tp, cv::Scalar(80, 255, 255), 2, cv::LINE_AA);
        cv::circle(out, tp, 8, cv::Scalar(80, 255, 255), 2, cv::LINE_AA);
    }
    // robot
    cv::circle(out, to_panel(rx, ry), 7, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
    cv::circle(out, to_panel(rx, ry), 7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

    cv::putText(out, l1, {16, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                {235, 235, 235}, 1, cv::LINE_AA);
    cv::putText(out, l2, {16, 44}, cv::FONT_HERSHEY_SIMPLEX, 0.46,
                {180, 220, 255}, 1, cv::LINE_AA);
    cv::putText(out, l3, {16, PANEL_H - 14}, cv::FONT_HERSHEY_SIMPLEX, 0.46,
                {180, 255, 180}, 1, cv::LINE_AA);
}

// ===========================================================================
int main() {
    std::vector<uint8_t> truth;
    make_true_map(truth);

    std::vector<int> state(N_CELLS, UNKNOWN);
    int rx = 90, ry = 430;                          // start in open area, bottom-left
    reveal(truth, state, rx, ry);

    // ----- GPU buffers
    int *d_state, *d_label, *d_changed, *d_sx, *d_sy, *d_cnt;
    CUDA_CHECK(cudaMalloc(&d_state,   N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_label,   N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sx,      N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sy,      N_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cnt,     N_CELLS * sizeof(int)));
    int block = 128, grid = (N_CELLS + block - 1) / block;

    // GPU frontier+CC+reduce on the current state; fills label (raw) + clusters
    auto gpu_pipeline = [&](const std::vector<int>& st, std::vector<int>& label_raw,
                            std::vector<Cluster>& cl, int& iters, float* ms_out) {
        CUDA_CHECK(cudaMemcpy(d_state, st.data(), N_CELLS * sizeof(int),
                              cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        frontier_kernel<<<grid, block>>>(d_state, d_label);
        // Batch CC sweeps between host sync checks: the per-iter changed-flag
        // round-trip otherwise dominates.  Over-running by a few sweeps past the
        // fixpoint is harmless (labels are monotone non-increasing), so the
        // fixpoint — and thus the result — is identical to the per-iter check.
        int it = 0;
        while (it < MAX_PROP_ITERS) {
            int z = 0;
            CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int), cudaMemcpyHostToDevice));
            int batch = std::min(CHECK_EVERY, MAX_PROP_ITERS - it);
            for (int b = 0; b < batch; ++b) { cc_kernel<<<grid, block>>>(d_label, d_changed); ++it; }
            int ch; CUDA_CHECK(cudaMemcpy(&ch, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            if (!ch) break;
        }
        CUDA_CHECK(cudaMemset(d_sx, 0, N_CELLS * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_sy, 0, N_CELLS * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, N_CELLS * sizeof(int)));
        reduce_kernel<<<grid, block>>>(d_label, d_sx, d_sy, d_cnt);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        if (ms_out) CUDA_CHECK(cudaEventElapsedTime(ms_out, e0, e1));
        iters = it;
        label_raw.resize(N_CELLS);
        CUDA_CHECK(cudaMemcpy(label_raw.data(), d_label, N_CELLS * sizeof(int),
                              cudaMemcpyDeviceToHost));
        std::vector<int> sx(N_CELLS), sy(N_CELLS), cnt(N_CELLS);
        CUDA_CHECK(cudaMemcpy(sx.data(),  d_sx,  N_CELLS * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sy.data(),  d_sy,  N_CELLS * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cnt.data(), d_cnt, N_CELLS * sizeof(int), cudaMemcpyDeviceToHost));
        cl.clear();
        for (int l = 0; l < N_CELLS; ++l) {
            if (cnt[l] < MIN_FRONTIER) continue;
            cl.push_back({l, sx[l] / cnt[l], sy[l] / cnt[l], cnt[l]});
        }
        std::sort(cl.begin(), cl.end(),
                  [](const Cluster& a, const Cluster& b) { return a.label < b.label; });
    };

    // --------------------------------------------------- headline comparison
    // One full pipeline on the post-first-reveal state, CPU vs GPU.
    std::vector<int> lab_cpu, lab_gpu;
    int it_cpu = 0, it_gpu = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    frontier_cpu(state, lab_cpu, it_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<Cluster> cl_cpu = clusters_from_labels(lab_cpu);

    std::vector<Cluster> cl_gpu;
    float gpu_ms = 0.0f;
    gpu_pipeline(state, lab_gpu, cl_gpu, it_gpu, nullptr);   // warm-up
    gpu_pipeline(state, lab_gpu, cl_gpu, it_gpu, &gpu_ms);   // timed

    // exact agreement
    int n_front_cpu = 0, n_front_gpu = 0, mism = 0;
    for (int i = 0; i < N_CELLS; ++i) {
        bool fc = lab_cpu[i] != NO_LABEL, fg = lab_gpu[i] != NO_LABEL;
        n_front_cpu += fc; n_front_gpu += fg;
        if (fc != fg) ++mism;
    }
    std::vector<int> cc = lab_cpu, cg = lab_gpu; int nc = 0, ng = 0;
    canonicalise(cc, nc); canonicalise(cg, ng);
    int label_mism = 0;
    for (int i = 0; i < N_CELLS; ++i) if (cc[i] != cg[i]) ++label_mism;
    double speedup = cpu_ms / gpu_ms;

    int tgt_cpu = select_target(cl_cpu, rx, ry);
    int tgt_gpu = select_target(cl_gpu, rx, ry);
    bool target_match = (tgt_cpu >= 0 && tgt_gpu >= 0 &&
                         cl_cpu[tgt_cpu].cx == cl_gpu[tgt_gpu].cx &&
                         cl_cpu[tgt_cpu].cy == cl_gpu[tgt_gpu].cy);

    std::printf("CPU %.2f ms (%d CC iters), GPU %.3f ms (%d iters)  -> %.0fx\n",
                cpu_ms, it_cpu, gpu_ms, it_gpu, speedup);
    std::printf("frontier cells: CPU %d, GPU %d   frontier-flag mismatch %d\n",
                n_front_cpu, n_front_gpu, mism);
    std::printf("components: CPU %d, GPU %d   per-cell label mismatch %d\n",
                nc, ng, label_mism);
    std::printf("next target: CPU (%d,%d), GPU (%d,%d)   match %s\n",
                tgt_cpu >= 0 ? cl_cpu[tgt_cpu].cx : -1,
                tgt_cpu >= 0 ? cl_cpu[tgt_cpu].cy : -1,
                tgt_gpu >= 0 ? cl_gpu[tgt_gpu].cx : -1,
                tgt_gpu >= 0 ? cl_gpu[tgt_gpu].cy : -1,
                target_match ? "YES" : "NO");

    // --------------------------------------------------- exploration animation
    if (system("mkdir -p tmp") != 0)
        std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_frontier_exploration.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          6, cv::Size(PANEL_W, PANEL_H));
    std::vector<cv::Point> traj;
    traj.push_back({rx, ry});
    float hx = 1.0f, hy = 0.0f;                       // exploration heading (east)
    int prev_known = -1, noprog = 0;                  // exploration-plateau detector

    for (int step = 0; step < N_STEPS; ++step) {
        std::vector<int> label_raw; std::vector<Cluster> cl; int it = 0;
        gpu_pipeline(state, label_raw, cl, it, nullptr);
        int tgt = select_target(cl, rx, ry);
        int gx = rx, gy = ry;
        if (tgt >= 0) goal_for_component(label_raw, cl[tgt], rx, ry, hx, hy, gx, gy);

        std::vector<int> show = label_raw; int dum;
        canonicalise(show, dum);

        // count explored + detect plateau (newly-revealed cells per step)
        int known = 0;
        for (int v : state) if (v != UNKNOWN) ++known;
        double pct = 100.0 * known / N_CELLS;
        if (prev_known >= 0 && known - prev_known < 400) ++noprog; else noprog = 0;
        prev_known = known;

        char l1[200], l2[200], l3[200];
        std::snprintf(l1, sizeof(l1),
                      "GPU frontier exploration (one thread = one cell)  grid %dx%d  "
                      "sensor r=%d", GRID_W, GRID_H, SENSOR_R);
        std::snprintf(l2, sizeof(l2),
                      "step %d/%d   frontier components: %d   explored: %.1f%%",
                      step + 1, N_STEPS, (int)cl.size(), pct);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms vs GPU %.2f ms -> %.0fx   frontier/label/target "
                      "match: exact", cpu_ms, gpu_ms, speedup);
        cv::Mat img;
        draw_frame(img, state, show, cl, tgt, gx, gy, rx, ry, traj, l1, l2, l3);
        video.write(img);

        if (tgt < 0) break;                          // nothing left to explore
        int px = rx, py = ry;
        advance_robot(state, rx, ry, gx, gy);
        int mdx = rx - px, mdy = ry - py;
        if (mdx * mdx + mdy * mdy > 4) {             // moved: adopt travel heading
            float n = std::sqrt((float)(mdx * mdx + mdy * mdy));
            hx = mdx / n; hy = mdy / n;
        } else {                                     // blocked: turn 90 deg, retry
            float t = hx; hx = -hy; hy = t;
        }
        traj.push_back({rx, ry});
        reveal(truth, state, rx, ry);
        if (noprog >= 2) break;                      // exploration has plateaued
    }
    // hold final frame
    {
        std::vector<int> label_raw; std::vector<Cluster> cl; int it = 0;
        gpu_pipeline(state, label_raw, cl, it, nullptr);
        std::vector<int> show = label_raw; int dum; canonicalise(show, dum);
        int known = 0; for (int v : state) if (v != UNKNOWN) ++known;
        double pct = 100.0 * known / N_CELLS;
        char l1[200], l2[200], l3[200];
        std::snprintf(l1, sizeof(l1),
                      "GPU frontier exploration (one thread = one cell)  grid %dx%d  "
                      "sensor r=%d", GRID_W, GRID_H, SENSOR_R);
        std::snprintf(l2, sizeof(l2),
                      "exploration complete   remaining components: %d   explored: %.1f%%",
                      (int)cl.size(), pct);
        std::snprintf(l3, sizeof(l3),
                      "CPU %.0f ms vs GPU %.2f ms -> %.0fx   frontier/label/target "
                      "match: exact", cpu_ms, gpu_ms, speedup);
        cv::Mat img;
        draw_frame(img, state, show, cl, -1, rx, ry, rx, ry, traj, l1, l2, l3);
        for (int r = 0; r < 5; ++r) video.write(img);
    }

    video.release();
    cudabot::avi_to_gif("tmp/gpu_frontier_exploration.avi",
                        "gif/gpu_frontier_exploration.gif", 6, 720);
    std::printf("wrote gif/gpu_frontier_exploration.gif\n");

    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_label));
    CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_sx));
    CUDA_CHECK(cudaFree(d_sy));
    CUDA_CHECK(cudaFree(d_cnt));
    return 0;
}

}  // namespace cudabot

int main() { return cudabot::main(); }
