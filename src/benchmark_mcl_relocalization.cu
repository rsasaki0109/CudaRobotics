// benchmark_mcl_relocalization.cu
//
// Seeded benchmark of Monte Carlo localization with a 2D LiDAR (likelihood
// field) on random indoor maps: how filters recover from a kidnapping and
// localize globally.
//
// Each seed draws one world: a 20 m x 20 m map of 3 x 3 rooms with doors and
// furniture, a drive through it, noisy odometry, and a 360-degree scan per
// step. Every method runs on the same episode, so results pair by seed.
//
// Cells
//   kidnap   the filter starts at the true pose; at step 100 the robot is
//            carried to a random pose at least 5 m away (odometry sees nothing)
//   global   the filter starts with particles spread over the whole map
//
// Methods
//   mcl      plain MCL (no reset)
//   aug      augmented MCL (AMCL): random particles injected when the short-term
//            average likelihood drops below the long-term one
//   er       expansion resetting (Ueda et al. 2004): when the mean per-beam
//            likelihood drops below a threshold, every particle is scattered
//   reloc    GPU relocalization: on the same trigger, the scan is scored at every
//            pose of a coarse grid over the free space (0.2 m x 5 degrees) on
//            the GPU, and part of the particles are replaced by samples around
//            the best poses
//
// Output: one CSV row per (cell, method, seed).

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_check.cuh"

namespace cudabot {

constexpr float PI_F = 3.14159265f;
constexpr int GW = 400, GH = 400;          // map cells
constexpr float RES = 0.05f;               // m per cell
constexpr int NB = 180;                    // scan beams over 360 degrees
constexpr int BEAM_STRIDE = 3;             // the filter uses every 3rd beam
constexpr int NBF = NB / BEAM_STRIDE;
constexpr float MAX_R = 8.0f;
constexpr float SIGMA_HIT = 0.1f;          // likelihood field of the filter
constexpr float SIGMA_COARSE = 0.3f;       // smoother field for the coarse relocalization grid
constexpr float Z_HIT = 0.95f, Z_RAND = 0.05f;
constexpr float DT = 0.1f, SPEED = 0.5f, TURN_RATE = 1.5f;
constexpr float CLEARANCE = 0.3f;          // drivable: farther than this from any obstacle
constexpr int RELOC_HEADINGS = 72;         // 5 degrees
constexpr int RELOC_STRIDE = 4;            // candidate positions every 4 cells (0.2 m)
constexpr int THREADS = 256;

__host__ __device__ inline float wrap_angle(float a) {
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

// ---------------------------------------------------------------------------
// World: map, distance field, likelihood fields, drivable cells
// ---------------------------------------------------------------------------
// The environment: the sensor and how far the world departs from the map.
struct Env {
    std::string name = "open";
    float fov_deg = 360.0f;            // scan field of view, centred ahead
    float max_range = 8.0f;            // beyond this a beam returns nothing
    int clutter = 0;                   // unmapped boxes in the world (moved furniture, bags)
    bool repeat_rooms = false;         // equal rooms with the same furniture layout
};

struct World {
    std::vector<uint8_t> occ;          // 1 = obstacle (the map the filter has)
    std::vector<uint8_t> real;         // the world the scans see: the map plus clutter
    std::vector<float> dist;           // m to the nearest obstacle
    std::vector<float> lf, lf_coarse;  // per-cell log likelihood of a beam endpoint
    std::vector<uint8_t> drive;        // largest connected component with clearance
    std::vector<int> drive_cells;
    std::vector<int> cand_cells;       // relocalization candidates (drivable, every 0.2 m)
};

static void fill_rect(World& w, float x0, float y0, float x1, float y1) {
    int i0 = std::max(0, (int)std::floor(x0 / RES)), i1 = std::min(GW - 1, (int)std::ceil(x1 / RES) - 1);
    int j0 = std::max(0, (int)std::floor(y0 / RES)), j1 = std::min(GH - 1, (int)std::ceil(y1 / RES) - 1);
    for (int j = j0; j <= j1; j++)
        for (int i = i0; i <= i1; i++) w.occ[j * GW + i] = 1;
}

// Felzenszwalb-Huttenlocher 1D squared distance transform.
static void edt_1d(const float* f, int n, float* d, int* v, float* z) {
    int k = 0;
    v[0] = 0; z[0] = -1e20f; z[1] = 1e20f;
    for (int q = 1; q < n; q++) {
        float s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        while (s <= z[k]) {
            k--;
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        }
        k++; v[k] = q; z[k] = s; z[k + 1] = 1e20f;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        d[q] = (q - v[k]) * (float)(q - v[k]) + f[v[k]];
    }
}

static void distance_field(World& w) {
    const int n = std::max(GW, GH);
    std::vector<float> g(GW * GH), f(n), d(n), z(n + 1);
    std::vector<int> v(n);
    for (int i = 0; i < GW * GH; i++) g[i] = w.occ[i] ? 0.0f : 1e20f;
    for (int x = 0; x < GW; x++) {
        for (int y = 0; y < GH; y++) f[y] = g[y * GW + x];
        edt_1d(f.data(), GH, d.data(), v.data(), z.data());
        for (int y = 0; y < GH; y++) g[y * GW + x] = d[y];
    }
    w.dist.assign(GW * GH, 0.0f);
    for (int y = 0; y < GH; y++) {
        for (int x = 0; x < GW; x++) f[x] = g[y * GW + x];
        edt_1d(f.data(), GW, d.data(), v.data(), z.data());
        for (int x = 0; x < GW; x++) w.dist[y * GW + x] = std::sqrt(d[x]) * RES;
    }
}

static World make_world(std::mt19937& rng, const Env& env) {
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    auto uni = [&](float a, float b) { return a + (b - a) * U(rng); };
    World w;
    w.occ.assign(GW * GH, 0);
    const float W = GW * RES, H = GH * RES, T = 0.1f;
    fill_rect(w, 0, 0, W, T); fill_rect(w, 0, H - T, W, H);
    fill_rect(w, 0, 0, T, H); fill_rect(w, W - T, 0, W, H);
    const float jit = env.repeat_rooms ? 0.0f : 1.5f;
    float xs[4] = {0, W / 3 + uni(-jit, jit), 2 * W / 3 + uni(-jit, jit), W};
    float ys[4] = {0, H / 3 + uni(-jit, jit), 2 * H / 3 + uni(-jit, jit), H};
    // internal walls, one segment per room side, each with a 1 m door (or left open)
    for (int i = 1; i <= 2; i++)
        for (int j = 0; j < 3; j++) {
            if (U(rng) < 0.15f) continue;
            float door = uni(ys[j] + 0.4f, ys[j + 1] - 1.4f);
            fill_rect(w, xs[i] - T / 2, ys[j], xs[i] + T / 2, door);
            fill_rect(w, xs[i] - T / 2, door + 1.0f, xs[i] + T / 2, ys[j + 1]);
        }
    for (int j = 1; j <= 2; j++)
        for (int i = 0; i < 3; i++) {
            if (U(rng) < 0.15f) continue;
            float door = uni(xs[i] + 0.4f, xs[i + 1] - 1.4f);
            fill_rect(w, xs[i], ys[j] - T / 2, door, ys[j] + T / 2);
            fill_rect(w, door + 1.0f, ys[j] - T / 2, xs[i + 1], ys[j] + T / 2);
        }
    // furniture (with repeat_rooms, one layout copied into every room)
    std::vector<float> layout;   // x, y, w, h relative to the room origin
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            if (!env.repeat_rooms || (i == 0 && j == 0)) {
                layout.clear();
                int n = (int)(U(rng) * 4.0f);
                if (env.repeat_rooms) n = 2 + (int)(U(rng) * 2.0f);
                for (int k = 0; k < n; k++) {
                    float bw = uni(0.3f, 1.2f), bh = uni(0.3f, 1.2f);
                    float x = uni(0.15f, xs[i + 1] - xs[i] - 0.15f - bw);
                    float y = uni(0.15f, ys[j + 1] - ys[j] - 0.15f - bh);
                    layout.insert(layout.end(), {x, y, bw, bh});
                }
            }
            for (size_t k = 0; k + 3 < layout.size(); k += 4)
                fill_rect(w, xs[i] + layout[k], ys[j] + layout[k + 1], xs[i] + layout[k] + layout[k + 2],
                          ys[j] + layout[k + 1] + layout[k + 3]);
        }
    distance_field(w);
    // clutter the map does not have: small boxes against walls and furniture
    w.real = w.occ;
    {
        std::vector<int> near;
        for (int i = 0; i < GW * GH; i++)
            if (w.dist[i] > 0.1f && w.dist[i] < 0.25f) near.push_back(i);
        std::uniform_int_distribution<int> pick(0, std::max(0, (int)near.size() - 1));
        for (int k = 0; k < env.clutter && !near.empty(); k++) {
            int c = near[pick(rng)];
            float cx = (c % GW + 0.5f) * RES, cy = (c / GW + 0.5f) * RES;
            float bw = uni(0.2f, 0.4f), bh = uni(0.2f, 0.4f);
            int i0 = std::max(0, (int)((cx - bw / 2) / RES)), i1 = std::min(GW - 1, (int)((cx + bw / 2) / RES));
            int j0 = std::max(0, (int)((cy - bh / 2) / RES)), j1 = std::min(GH - 1, (int)((cy + bh / 2) / RES));
            for (int jj = j0; jj <= j1; jj++)
                for (int ii = i0; ii <= i1; ii++) w.real[jj * GW + ii] = 1;
        }
    }
    w.lf.resize(GW * GH); w.lf_coarse.resize(GW * GH);
    for (int i = 0; i < GW * GH; i++) {
        float d = w.dist[i];
        w.lf[i] = std::log(Z_HIT * std::exp(-0.5f * d * d / (SIGMA_HIT * SIGMA_HIT)) + Z_RAND);
        w.lf_coarse[i] = std::log(Z_HIT * std::exp(-0.5f * d * d / (SIGMA_COARSE * SIGMA_COARSE)) + Z_RAND);
    }
    // drivable: the largest 4-connected component of cells with clearance
    std::vector<int> comp(GW * GH, -1);
    int best = -1, best_n = 0, nc = 0;
    for (int s = 0; s < GW * GH; s++) {
        if (comp[s] >= 0 || w.dist[s] <= CLEARANCE) continue;
        int n = 0;
        std::queue<int> q; q.push(s); comp[s] = nc;
        while (!q.empty()) {
            int c = q.front(); q.pop(); n++;
            int cx = c % GW, cy = c / GW;
            const int nx[4] = {cx + 1, cx - 1, cx, cx}, ny[4] = {cy, cy, cy + 1, cy - 1};
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= GW || ny[k] < 0 || ny[k] >= GH) continue;
                int e = ny[k] * GW + nx[k];
                if (comp[e] < 0 && w.dist[e] > CLEARANCE) { comp[e] = nc; q.push(e); }
            }
        }
        if (n > best_n) { best_n = n; best = nc; }
        nc++;
    }
    w.drive.assign(GW * GH, 0);
    for (int i = 0; i < GW * GH; i++)
        if (comp[i] == best) {
            w.drive[i] = 1;
            w.drive_cells.push_back(i);
            if ((i % GW) % RELOC_STRIDE == 0 && (i / GW) % RELOC_STRIDE == 0) w.cand_cells.push_back(i);
        }
    return w;
}

static float raycast(const World& w, float x, float y, float a, float max_range) {
    const float c = std::cos(a), s = std::sin(a);
    for (float r = 0.0f; r < max_range; r += 0.01f) {
        int i = (int)((x + r * c) / RES), j = (int)((y + r * s) / RES);
        if (i < 0 || i >= GW || j < 0 || j >= GH) return MAX_R;
        if (w.real[j * GW + i]) return r;
    }
    return MAX_R;
}

// ---------------------------------------------------------------------------
// Episode: true poses, odometry, scans
// ---------------------------------------------------------------------------
struct Episode {
    int steps = 0, kidnap_step = -1;
    std::vector<float> x, y, th;         // true pose per step
    std::vector<float> ox, oy, oth;      // odometry increment into step t (robot frame of t-1)
    std::vector<float> scan;             // steps x NB ranges (MAX_R = no return)
};

static float cell_x(int c) { return (c % GW + 0.5f) * RES; }
static float cell_y(int c) { return (c / GW + 0.5f) * RES; }

// 8-connected BFS through drivable cells; returns waypoints every 0.2 m.
static std::vector<int> plan_path(const World& w, int s, int g) {
    std::vector<int> parent(GW * GH, -1);
    std::queue<int> q; q.push(s); parent[s] = s;
    while (!q.empty()) {
        int c = q.front(); q.pop();
        if (c == g) break;
        int cx = c % GW, cy = c / GW;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || nx >= GW || ny < 0 || ny >= GH) continue;
                int e = ny * GW + nx;
                if (parent[e] < 0 && w.drive[e]) { parent[e] = c; q.push(e); }
            }
    }
    std::vector<int> path;
    if (parent[g] < 0) return path;
    for (int c = g; c != s; c = parent[c]) path.push_back(c);
    std::reverse(path.begin(), path.end());
    std::vector<int> out;
    for (size_t i = 0; i < path.size(); i += 4) out.push_back(path[i]);
    if (!path.empty() && out.back() != path.back()) out.push_back(path.back());
    return out;
}

static int far_cell(const World& w, std::mt19937& rng, float fx, float fy, float min_d) {
    std::uniform_int_distribution<int> pick(0, (int)w.drive_cells.size() - 1);
    for (int tries = 0; tries < 1000; tries++) {
        int c = w.drive_cells[pick(rng)];
        if (std::hypot(cell_x(c) - fx, cell_y(c) - fy) >= min_d) return c;
    }
    return w.drive_cells[pick(rng)];
}

static Episode make_episode(const World& w, const Env& env, std::mt19937& rng, int steps, int kidnap_step) {
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    Episode e;
    e.steps = steps; e.kidnap_step = kidnap_step;
    std::uniform_int_distribution<int> pick(0, (int)w.drive_cells.size() - 1);
    int cur = w.drive_cells[pick(rng)];
    float x = cell_x(cur), y = cell_y(cur), th = (2.0f * U(rng) - 1.0f) * PI_F;
    std::vector<int> path; size_t wp = 0;
    for (int t = 0; t < steps; t++) {
        float px = x, py = y, pth = th;
        bool kidnapped = false;
        if (t > 0 && t == kidnap_step) {
            int c = far_cell(w, rng, x, y, 5.0f);
            x = cell_x(c); y = cell_y(c); th = (2.0f * U(rng) - 1.0f) * PI_F;
            path.clear(); kidnapped = true;
        } else if (t > 0) {
            // follow the path: move SPEED*DT along it, turn toward the motion
            float left = SPEED * DT;
            while (left > 1e-6f) {
                if (wp >= path.size()) {
                    int sc = (int)(y / RES) * GW + (int)(x / RES);
                    if (!w.drive[sc]) sc = w.drive_cells[pick(rng)];
                    path = plan_path(w, sc, far_cell(w, rng, x, y, 5.0f)); wp = 0;
                    if (path.empty()) break;
                }
                float tx = cell_x(path[wp]), ty = cell_y(path[wp]);
                float d = std::hypot(tx - x, ty - y);
                if (d <= left) { x = tx; y = ty; left -= d; wp++; }
                else { x += (tx - x) / d * left; y += (ty - y) / d * left; left = 0; }
            }
            float mv = std::hypot(x - px, y - py);
            if (mv > 1e-4f) {
                float err = wrap_angle(std::atan2(y - py, x - px) - th);
                th = wrap_angle(th + std::max(-TURN_RATE * DT, std::min(TURN_RATE * DT, err)));
            }
        }
        if (t == 0 || kidnapped) {
            e.ox.push_back(0.0f); e.oy.push_back(0.0f); e.oth.push_back(0.0f);
        } else {
            float dx = x - px, dy = y - py, c = std::cos(pth), s = std::sin(pth);
            float rx = c * dx + s * dy, ry = -s * dx + c * dy, rth = wrap_angle(th - pth);
            float tr = std::hypot(rx, ry);
            e.ox.push_back(rx + N(rng) * (0.05f * tr + 0.002f));
            e.oy.push_back(ry + N(rng) * (0.05f * tr + 0.002f));
            e.oth.push_back(rth + N(rng) * (0.05f * std::fabs(rth) + 0.02f * tr + 0.002f));
        }
        e.x.push_back(x); e.y.push_back(y); e.th.push_back(th);
        for (int b = 0; b < NB; b++) {
            float rel = b * 2.0f * PI_F / NB - PI_F;
            if (std::fabs(rel) > 0.5f * env.fov_deg * PI_F / 180.0f + 1e-4f) { e.scan.push_back(MAX_R); continue; }
            float r = raycast(w, x, y, th + rel, env.max_range);
            if (r < MAX_R) {
                if (U(rng) < 0.05f) r = 0.2f + (r - 0.2f) * U(rng);   // unmodeled obstacle
                else r = std::max(0.05f, r + N(rng) * 0.02f);
            }
            e.scan.push_back(r);
        }
    }
    return e;
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------
__global__ void init_rng_kernel(curandState* s, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &s[i]);
}

// Odometry motion model: the increment (in the robot frame) plus noise.
__global__ void motion_kernel(float* px, float* py, float* pth, curandState* rng, int n,
                              float ox, float oy, float oth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    float tr = sqrtf(ox * ox + oy * oy);
    float rx = ox + curand_normal(&s) * (0.1f * tr + 0.005f);
    float ry = oy + curand_normal(&s) * (0.1f * tr + 0.005f);
    float rth = oth + curand_normal(&s) * (0.1f * fabsf(oth) + 0.05f * tr + 0.005f);
    float c = cosf(pth[i]), sn = sinf(pth[i]);
    px[i] += c * rx - sn * ry;
    py[i] += sn * rx + c * ry;
    pth[i] = wrap_angle(pth[i] + rth);
    rng[i] = s;
}

__device__ inline float beam_loglik(const float* lf, float x, float y, float r, float a) {
    float ex = x + r * cosf(a), ey = y + r * sinf(a);
    int i = (int)floorf(ex / RES), j = (int)floorf(ey / RES);
    if (i < 0 || i >= GW || j < 0 || j >= GH) return logf(Z_RAND);
    return lf[j * GW + i];
}

// Log likelihood of the filter beams for each particle. A particle inside an
// obstacle or off the map gets a large penalty.
__global__ void weight_kernel(const float* px, const float* py, const float* pth, float* loglik, int n,
                              const float* lf, const float* dist, const float* br, const float* ba, int nbf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = px[i], y = py[i], th = pth[i];
    int ci = (int)floorf(x / RES), cj = (int)floorf(y / RES);
    float s = 0.0f;
    if (ci < 0 || ci >= GW || cj < 0 || cj >= GH || dist[cj * GW + ci] < 0.05f) s -= 50.0f;
    for (int b = 0; b < nbf; b++) s += beam_loglik(lf, x, y, br[b], th + ba[b]);
    loglik[i] = s;
}

// Expansion reset: scatter every particle.
__global__ void expand_kernel(float* px, float* py, float* pth, curandState* rng, int n, float sxy, float sth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState s = rng[i];
    px[i] += curand_normal(&s) * sxy;
    py[i] += curand_normal(&s) * sxy;
    pth[i] = wrap_angle(pth[i] + curand_normal(&s) * sth);
    rng[i] = s;
}

// Relocalization grid: one block per candidate position, one thread per
// heading; the block keeps the best heading's score.
__global__ void reloc_kernel(const int* cand, int ncand, const float* lf_coarse, const float* br, const float* ba,
                             int nbf, float* best_score, int* best_heading)
{
    __shared__ float ss[128];
    __shared__ int sh[128];
    int c = blockIdx.x, h = threadIdx.x;
    if (c >= ncand) return;
    float score = -1e30f;
    if (h < RELOC_HEADINGS) {
        int cell = cand[c];
        float x = (cell % GW + 0.5f) * RES, y = (cell / GW + 0.5f) * RES;
        float th = -PI_F + h * (2.0f * PI_F / RELOC_HEADINGS);
        score = 0.0f;
        for (int b = 0; b < nbf; b++) score += beam_loglik(lf_coarse, x, y, br[b], th + ba[b]);
    }
    ss[h] = score; sh[h] = h;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (h < st && ss[h + st] > ss[h]) { ss[h] = ss[h + st]; sh[h] = sh[h + st]; }
        __syncthreads();
    }
    if (h == 0) { best_score[c] = ss[0]; best_heading[c] = sh[0]; }
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------
struct Config {
    int np = 2000;
    float reset_th = 0.45f;     // mean per-beam likelihood below which er / reloc trigger
    float er_sxy = 0.2f, er_sth = 0.2f;
    float reloc_frac = 0.5f;    // share of the particles replaced by relocalization samples
    int reloc_top = 32;
    float inject_prior = 1e-4f; // reloc: prior weight of a relocalization particle against the belief
    int dual_np = 1000;         // reloc_dual: particles of the candidate set
    float dual_exclude = 1.0f;  // reloc_dual: candidates keep this far (m) from the belief's estimate
    int dual_max_age = 30;      // reloc_dual: steps a candidate set may run without winning
    float dual_drop = 20.0f;    // reloc_dual: drop it this far (nats) below the prior
    bool dual_swap = true;      // reloc_dual: keep the old belief as the alternative after a switch
    bool dual_gate = true;      // reloc_dual: switch only while the belief's alpha is below reset_th
    float dual_drift = 2.0f;    // reloc_dual: nats per step the candidates must win by (CUSUM drift)
    float lr_gate = 1.1f;       // reloc_lr: run the grid check when alpha is below this (1.1: every step)
    float lr_margin = 0.05f;    // reloc_lr: reset when the best grid pose beats the belief by this, per beam
    float aug_slow = 0.001f, aug_fast = 0.1f;
};

struct Result {
    std::string env, cell, method; int seed = 0;
    int recovered = 0, recover_steps = -1, final_ok = 0, resets = 0, resets_before = 0;
    float err_before = 0.0f, localized_before = 0.0f, localized_frac = 0.0f, err_after = 0.0f;
    float step_ms = 0.0f, reloc_ms = 0.0f;
};

class Filter {
public:
    Filter(const World& w, const Config& cfg, unsigned long long seed, int n = 0)
        : w_(w), cfg_(cfg), host_rng_(seed) {
        n_ = n > 0 ? n : cfg.np;
        CUDA_CHECK(cudaMalloc(&d_x_, n_ * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_y_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_th_, n_ * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_ll_, n_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rng_, n_ * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_lf_, GW * GH * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_lfc_, GW * GH * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dist_, GW * GH * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_br_, NBF * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_ba_, NBF * sizeof(float)));
        nc_ = (int)w.cand_cells.size();
        CUDA_CHECK(cudaMalloc(&d_cand_, std::max(1, nc_) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cs_, std::max(1, nc_) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ch_, std::max(1, nc_) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_lf_, w.lf.data(), GW * GH * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lfc_, w.lf_coarse.data(), GW * GH * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dist_, w.dist.data(), GW * GH * sizeof(float), cudaMemcpyHostToDevice));
        if (nc_) CUDA_CHECK(cudaMemcpy(d_cand_, w.cand_cells.data(), nc_ * sizeof(int), cudaMemcpyHostToDevice));
        init_rng_kernel<<<(n_ + THREADS - 1) / THREADS, THREADS>>>(d_rng_, seed, n_);
        x_.resize(n_); y_.resize(n_); th_.resize(n_); ll_.resize(n_); wt_.resize(n_); lw_.assign(n_, 0.0f);
    }
    ~Filter() {
        cudaFree(d_x_); cudaFree(d_y_); cudaFree(d_th_); cudaFree(d_ll_); cudaFree(d_rng_);
        cudaFree(d_lf_); cudaFree(d_lfc_); cudaFree(d_dist_); cudaFree(d_br_); cudaFree(d_ba_);
        cudaFree(d_cand_); cudaFree(d_cs_); cudaFree(d_ch_);
    }

    void init_at(float x, float y, float th) {
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < n_; i++) {
            x_[i] = x + N(host_rng_) * 0.1f; y_[i] = y + N(host_rng_) * 0.1f;
            th_[i] = wrap_angle(th + N(host_rng_) * 0.05f);
        }
        upload();
    }
    void init_uniform() {
        for (int i = 0; i < n_; i++) random_particle(i);
        upload();
    }

    // One filter step; method: 0 mcl, 1 aug, 2 er, 3 reloc. Returns true if a reset fired.
    bool step(int method, float ox, float oy, float oth, const float* scan, bool move, float& reloc_ms) {
        if (move)
            motion_kernel<<<(n_ + THREADS - 1) / THREADS, THREADS>>>(d_x_, d_y_, d_th_, d_rng_, n_, ox, oy, oth);
        // filter beams (max-range readings dropped)
        std::vector<float> br, ba;
        for (int b = 0; b < NB; b += BEAM_STRIDE)
            if (scan[b] < MAX_R) { br.push_back(scan[b]); ba.push_back(b * 2.0f * PI_F / NB - PI_F); }
        nbf_ = (int)br.size();
        br_h_ = br; ba_h_ = ba;
        if (nbf_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_br_, br.data(), nbf_ * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ba_, ba.data(), nbf_ * sizeof(float), cudaMemcpyHostToDevice));
        }
        weigh();
        bool reset = false;
        float alpha = mean_beam_likelihood();
        last_alpha = alpha;
        float inject = 0.0f;
        if (method == 1) {
            w_slow_ = w_slow_ < 0 ? alpha : w_slow_ + cfg_.aug_slow * (alpha - w_slow_);
            w_fast_ = w_fast_ < 0 ? alpha : w_fast_ + cfg_.aug_fast * (alpha - w_fast_);
            inject = std::max(0.0f, 1.0f - w_fast_ / w_slow_);
            reset = inject > 0.0f;
        } else if (method == 2 && nbf_ > 0 && alpha < cfg_.reset_th) {
            expand_kernel<<<(n_ + THREADS - 1) / THREADS, THREADS>>>(d_x_, d_y_, d_th_, d_rng_, n_, cfg_.er_sxy, cfg_.er_sth);
            weigh();
            reset = true;
        } else if (method == 3 && nbf_ > 0 && alpha < cfg_.reset_th && nc_ > 0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            grid_search();
            seed_from_grid();
            reloc_ms += std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            grid_runs++;
            weigh();
            reset = true;
        } else if (method == 4 && nbf_ > 0 && nc_ > 0 && alpha < cfg_.lr_gate) {
            // likelihood-ratio trigger: reset only when some pose on the map explains
            // the scan clearly better than the current belief (same coarse field)
            auto t0 = std::chrono::high_resolution_clock::now();
            grid_search();
            last_gap = (grid_best_ - belief_coarse_score()) / nbf_;
            if (last_gap > cfg_.lr_margin) { seed_from_grid(); reset = true; }
            reloc_ms += std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            grid_runs++;
            if (reset) weigh();
        }
        // this scan's marginal likelihood under the belief: log sum w ll / sum w
        {
            double m0 = -1e300, m1 = -1e300, s0 = 0.0, s1 = 0.0;
            for (int i = 0; i < n_; i++) {
                m0 = std::max(m0, (double)lw_[i]);
                m1 = std::max(m1, (double)lw_[i] + ll_[i]);
            }
            for (int i = 0; i < n_; i++) {
                s0 += std::exp(lw_[i] - m0);
                s1 += std::exp((double)lw_[i] + ll_[i] - m1);
            }
            last_logz = (float)((m1 + std::log(s1)) - (m0 + std::log(s0)));
        }
        for (int i = 0; i < n_; i++) lw_[i] += ll_[i];
        download_weights();
        estimate();
        resample(inject);
        return reset;
    }

    float ex = 0, ey = 0, eth = 0;   // pose estimate
    float last_alpha = 0;            // mean per-beam likelihood of the last step
    float last_gap = 0;              // reloc_lr: (best grid pose - belief) per beam, coarse field
    int grid_runs = 0;               // relocalization grid searches run
    float last_logz = 0;             // log marginal likelihood of the last scan under the belief
    bool has_beams() const { return nbf_ > 0 && nc_ > 0; }
    // reloc_dual: spread all particles around the best poses of m's grid search
    // on m's current scan, away from m's estimate: the hypothesis "the robot is
    // somewhere else".
    void init_from_grid(Filter& m) {
        m.grid_search(m.ex, m.ey, cfg_.dual_exclude);
        m.grid_runs++;
        const int k = (int)m.top_.size();
        std::vector<double> p(k);
        for (int q = 0; q < k; q++) p[q] = std::exp((double)m.cs_[m.top_[q]] - m.cs_[m.top_[0]]);
        std::discrete_distribution<int> pick(p.begin(), p.end());
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (int i = 0; i < n_; i++) {
            int c = m.top_[pick(host_rng_)];
            x_[i] = cell_x(w_.cand_cells[c]) + N(host_rng_) * 0.1f;
            y_[i] = cell_y(w_.cand_cells[c]) + N(host_rng_) * 0.1f;
            th_[i] = wrap_angle(-PI_F + m.ch_[c] * (2.0f * PI_F / RELOC_HEADINGS) + N(host_rng_) * 0.05f);
        }
        std::fill(lw_.begin(), lw_.end(), 0.0f);
        upload();
    }
    // reloc_dual: take over another filter's belief (resampled to this size).
    struct Snapshot { std::vector<float> x, y, th, lw; float ex, ey, eth; };
    Snapshot snapshot() const { return Snapshot{x_, y_, th_, lw_, ex, ey, eth}; }
    void adopt(const Filter& c) { take(c.snapshot()); }
    void take(const Snapshot& c) {
        const int cn = (int)c.x.size();
        double mx = *std::max_element(c.lw.begin(), c.lw.end());
        std::vector<double> cw(cn);
        double sum = 0.0;
        for (int i = 0; i < cn; i++) { cw[i] = std::exp(c.lw[i] - mx); sum += cw[i]; }
        std::uniform_real_distribution<double> U(0.0, 1.0);
        double r = U(host_rng_) / n_, acc = cw[0] / sum;
        int j = 0;
        for (int i = 0; i < n_; i++) {
            double u = r + (double)i / n_;
            while (u > acc && j < cn - 1) acc += cw[++j] / sum;
            x_[i] = c.x[j]; y_[i] = c.y[j]; th_[i] = c.th[j];
        }
        std::fill(lw_.begin(), lw_.end(), 0.0f);
        upload();
        ex = c.ex; ey = c.ey; eth = c.eth;
    }

private:
    void upload() {
        CUDA_CHECK(cudaMemcpy(d_x_, x_.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_, y_.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_th_, th_.data(), n_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    void random_particle(int i) {
        std::uniform_int_distribution<int> pick(0, (int)w_.drive_cells.size() - 1);
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        int c = w_.drive_cells[pick(host_rng_)];
        x_[i] = cell_x(c) + (U(host_rng_) - 0.5f) * RES;
        y_[i] = cell_y(c) + (U(host_rng_) - 0.5f) * RES;
        th_[i] = (2.0f * U(host_rng_) - 1.0f) * PI_F;
    }
    void weigh() {
        weight_kernel<<<(n_ + THREADS - 1) / THREADS, THREADS>>>(d_x_, d_y_, d_th_, d_ll_, n_, d_lf_, d_dist_,
                                                                 d_br_, d_ba_, nbf_);
        CUDA_CHECK(cudaMemcpy(ll_.data(), d_ll_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
    // mean over particles of the per-beam (geometric mean) likelihood
    float mean_beam_likelihood() const {
        if (nbf_ == 0) return 1.0f;
        double s = 0.0;
        for (int i = 0; i < n_; i++) s += std::exp(std::max(-50.0f, ll_[i]) / nbf_);
        return (float)(s / n_);
    }
    void download_weights() {
        CUDA_CHECK(cudaMemcpy(x_.data(), d_x_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y_.data(), d_y_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(th_.data(), d_th_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        float mx = *std::max_element(lw_.begin(), lw_.end());
        double s = 0.0;
        for (int i = 0; i < n_; i++) { wt_[i] = std::exp(lw_[i] - mx); s += wt_[i]; }
        for (int i = 0; i < n_; i++) wt_[i] = (float)(wt_[i] / s);
    }
    // weighted mean of the particles near the most likely one
    void estimate() {
        int b = (int)(std::max_element(wt_.begin(), wt_.end()) - wt_.begin());
        double sx = 0, sy = 0, sc = 0, ss = 0, sw = 0;
        for (int i = 0; i < n_; i++) {
            if (std::hypot(x_[i] - x_[b], y_[i] - y_[b]) > 1.0f || std::fabs(wrap_angle(th_[i] - th_[b])) > 0.5f) continue;
            sx += wt_[i] * x_[i]; sy += wt_[i] * y_[i];
            sc += wt_[i] * std::cos(th_[i]); ss += wt_[i] * std::sin(th_[i]); sw += wt_[i];
        }
        ex = (float)(sx / sw); ey = (float)(sy / sw); eth = (float)std::atan2(ss, sc);
    }
    void resample(float inject) {
        double s2 = 0.0;
        for (int i = 0; i < n_; i++) s2 += (double)wt_[i] * wt_[i];
        float neff = (float)(1.0 / s2);
        if (neff >= 0.5f * n_ && inject <= 0.0f) return;
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        std::vector<float> nx(n_), ny(n_), nth(n_);
        float r = U(host_rng_) / n_, c = wt_[0];
        int j = 0;
        for (int i = 0; i < n_; i++) {
            float u = r + (float)i / n_;
            while (u > c && j < n_ - 1) c += wt_[++j];
            nx[i] = x_[j]; ny[i] = y_[j]; nth[i] = th_[j];
        }
        x_.swap(nx); y_.swap(ny); th_.swap(nth);
        std::fill(lw_.begin(), lw_.end(), 0.0f);
        if (inject > 0.0f)
            for (int i = 0; i < n_; i++)
                if (U(host_rng_) < inject) random_particle(i);
        upload();
    }
    // Score every candidate pose on the GPU (best heading per position) and keep
    // the best positions.
    // With exclude_r > 0, positions within exclude_r of (exclude_x, exclude_y) are left out.
    void grid_search(float exclude_x = 0.0f, float exclude_y = 0.0f, float exclude_r = -1.0f) {
        reloc_kernel<<<nc_, 128>>>(d_cand_, nc_, d_lfc_, d_br_, d_ba_, nbf_, d_cs_, d_ch_);
        cs_.resize(nc_); ch_.resize(nc_);
        CUDA_CHECK(cudaMemcpy(cs_.data(), d_cs_, nc_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ch_.data(), d_ch_, nc_ * sizeof(int), cudaMemcpyDeviceToHost));
        if (exclude_r > 0.0f)
            for (int c = 0; c < nc_; c++)
                if (std::hypot(cell_x(w_.cand_cells[c]) - exclude_x, cell_y(w_.cand_cells[c]) - exclude_y) < exclude_r)
                    cs_[c] = -1e30f;
        int m = std::min(cfg_.reloc_top, nc_);
        top_.resize(nc_);
        std::iota(top_.begin(), top_.end(), 0);
        std::partial_sort(top_.begin(), top_.begin() + m, top_.end(), [&](int a, int b) { return cs_[a] > cs_[b]; });
        top_.resize(m);
        grid_best_ = cs_[top_[0]];
    }
    // Coarse-field score of the current belief: the best of its 10 most likely particles.
    float belief_coarse_score() {
        CUDA_CHECK(cudaMemcpy(x_.data(), d_x_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y_.data(), d_y_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(th_.data(), d_th_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<int> o(n_);
        std::iota(o.begin(), o.end(), 0);
        int k = std::min(10, n_);
        std::partial_sort(o.begin(), o.begin() + k, o.end(), [&](int a, int b) { return ll_[a] > ll_[b]; });
        float best = -1e30f;
        for (int q = 0; q < k; q++) {
            int i = o[q];
            float sc = 0.0f;
            for (int b = 0; b < nbf_; b++) {
                float a = th_[i] + ba_h_[b];
                int gi = (int)std::floor((x_[i] + br_h_[b] * std::cos(a)) / RES);
                int gj = (int)std::floor((y_[i] + br_h_[b] * std::sin(a)) / RES);
                sc += (gi < 0 || gi >= GW || gj < 0 || gj >= GH) ? std::log(Z_RAND) : w_.lf_coarse[gj * GW + gi];
            }
            best = std::max(best, sc);
        }
        return best;
    }
    // Replace the least likely particles by samples around the best grid poses.
    void seed_from_grid() {
        const int m = (int)top_.size();
        std::vector<double> p(m);
        for (int k = 0; k < m; k++) p[k] = std::exp((double)cs_[top_[k]] - cs_[top_[0]]);
        std::discrete_distribution<int> pick(p.begin(), p.end());
        std::normal_distribution<float> N(0.0f, 1.0f);
        CUDA_CHECK(cudaMemcpy(x_.data(), d_x_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y_.data(), d_y_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(th_.data(), d_th_, n_ * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<int> order(n_);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return lw_[a] + ll_[a] < lw_[b] + ll_[b]; });
        int nr = (int)(cfg_.reloc_frac * n_);
        // the new particles enter with the kept particles' mean weight times the
        // prior that the robot was moved
        double mx = -1e30;
        for (int k = nr; k < n_; k++) mx = std::max(mx, (double)lw_[order[k]]);
        double se = 0.0;
        for (int k = nr; k < n_; k++) se += std::exp(lw_[order[k]] - mx);
        const float base = (float)(mx + std::log(se / std::max(1, n_ - nr))) + std::log(cfg_.inject_prior);
        for (int k = 0; k < nr; k++) {
            int i = order[k], c = top_[pick(host_rng_)];
            lw_[i] = base;
            x_[i] = cell_x(w_.cand_cells[c]) + N(host_rng_) * 0.1f;
            y_[i] = cell_y(w_.cand_cells[c]) + N(host_rng_) * 0.1f;
            th_[i] = wrap_angle(-PI_F + ch_[c] * (2.0f * PI_F / RELOC_HEADINGS) + N(host_rng_) * 0.05f);
        }
        upload();
    }

    const World& w_;
    Config cfg_;
    std::mt19937 host_rng_;
    int n_ = 0, nc_ = 0, nbf_ = 0;
    float w_slow_ = -1.0f, w_fast_ = -1.0f;
    std::vector<float> x_, y_, th_, ll_, wt_;
    std::vector<float> lw_;                       // log weight carried since the last resampling
    std::vector<float> br_h_, ba_h_;              // this step's filter beams
    std::vector<float> cs_; std::vector<int> ch_, top_;   // grid scores, headings, best positions
    float grid_best_ = 0.0f;
    float *d_x_ = nullptr, *d_y_ = nullptr, *d_th_ = nullptr, *d_ll_ = nullptr;
    float *d_lf_ = nullptr, *d_lfc_ = nullptr, *d_dist_ = nullptr, *d_br_ = nullptr, *d_ba_ = nullptr;
    float *d_cs_ = nullptr; int *d_ch_ = nullptr, *d_cand_ = nullptr;
    curandState* d_rng_ = nullptr;
};

static const char* METHOD_NAMES[] = {"mcl", "aug", "er", "reloc", "reloc_lr", "reloc_dual"};
constexpr int N_METHODS = 6;

static bool g_trace = false;

static Result run_episode(const World& w, const Episode& e, const Config& cfg, const std::string& cell, int method,
                          int seed)
{
    Result r; r.cell = cell; r.method = METHOD_NAMES[method]; r.seed = seed;
    Filter f(w, cfg, 7919ull * seed + 101ull * method + 17ull);
    const bool global = cell == "global";
    if (global) f.init_uniform(); else f.init_at(e.x[0], e.y[0], e.th[0]);
    const int start = global ? 0 : e.kidnap_step;
    int run = 0, loc_after = 0, n_before = 0;
    float err_after_sum = 0.0f;
    float reloc_ms = 0.0f;
    // reloc_dual: a second particle set around the grid's best poses runs beside the
    // belief; it replaces the belief once the log odds of "the robot was moved"
    // (starting at the kidnap prior, plus each scan's log marginal likelihood ratio)
    // turn positive, and is dropped when it keeps losing.
    const bool dual = method == 5;
    std::unique_ptr<Filter> cand;
    if (dual) cand.reset(new Filter(w, cfg, 7919ull * seed + 101ull * method + 29ull, cfg.dual_np));
    bool alive = false;
    float lam = 0.0f, dummy_ms = 0.0f;
    int age = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < e.steps; t++) {
        bool reset = f.step(dual ? 0 : method, e.ox[t], e.oy[t], e.oth[t], &e.scan[t * NB], t > 0, reloc_ms);
        if (dual) {
            bool switched = false;
            if (alive) {
                cand->step(0, e.ox[t], e.oy[t], e.oth[t], &e.scan[t * NB], t > 0, dummy_ms);
                lam += cand->last_logz - f.last_logz - cfg.dual_drift;
                age++;
                // switch only while the belief itself fails to explain the scan
                if (lam > 0.0f && (!cfg.dual_gate || f.last_alpha < cfg.reset_th)) {
                    switched = reset = true;
                    if (cfg.dual_swap) {
                        // the old belief stays on as the alternative: the switch can be undone
                        Filter::Snapshot old = f.snapshot();
                        f.adopt(*cand);
                        cand->take(old);
                        lam = std::log(cfg.inject_prior); age = 0;
                    } else {
                        f.adopt(*cand); alive = false;
                    }
                }
                else if (age >= cfg.dual_max_age || lam < std::log(cfg.inject_prior) - cfg.dual_drop) alive = false;
            }
            if (!alive && !switched && f.last_alpha < cfg.reset_th && f.has_beams()) {
                auto g0 = std::chrono::high_resolution_clock::now();
                cand->init_from_grid(f);
                reloc_ms += std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - g0).count();
                alive = true; lam = std::log(cfg.inject_prior); age = 0;
            }
        }
        float pe = std::hypot(f.ex - e.x[t], f.ey - e.y[t]);
        float ae = std::fabs(wrap_angle(f.eth - e.th[t]));
        bool loc = pe < 0.3f && ae < 0.2f;
        if (g_trace && t % 5 == 0)
            std::printf("  trace %s %s t %d alpha %.3f gap %.3f err %.2f %.2f reset %d dual %d lam %.1f\n", cell.c_str(),
                        METHOD_NAMES[method], t, f.last_alpha, f.last_gap, pe, ae, (int)reset, (int)alive, lam);
        if (reset) { r.resets++; if (t < start) r.resets_before++; }
        if (t < start) { r.err_before += pe; r.localized_before += loc; n_before++; continue; }
        if (loc) { loc_after++; err_after_sum += pe; }
        run = loc ? run + 1 : 0;
        if (!r.recovered && run >= 10) { r.recovered = 1; r.recover_steps = t - 9 - start; }
        if (t == e.steps - 1) r.final_ok = loc;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    r.step_ms = ms / e.steps;
    r.reloc_ms = f.grid_runs > 0 ? reloc_ms / f.grid_runs : 0.0f;
    r.err_before = n_before ? r.err_before / n_before : 0.0f;
    r.localized_before = n_before ? r.localized_before / n_before : 1.0f;
    r.localized_frac = (float)loc_after / (e.steps - start);
    r.err_after = loc_after ? err_after_sum / loc_after : -1.0f;
    return r;
}

}  // namespace cudabot

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(t);
    return out;
}

int main(int argc, char** argv) {
    using namespace cudabot;
    int seed_count = 10, seed_offset = 0, steps = 400, kidnap_step = 100;
    std::string csv_path, methods_arg = "mcl,aug,er,reloc,reloc_lr,reloc_dual", cells_arg = "kidnap,global";
    Config cfg;
    Env env;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--seed-count") seed_count = std::atoi(next().c_str());
        else if (a == "--seed-offset") seed_offset = std::atoi(next().c_str());
        else if (a == "--steps") steps = std::atoi(next().c_str());
        else if (a == "--kidnap-step") kidnap_step = std::atoi(next().c_str());
        else if (a == "--methods") methods_arg = next();
        else if (a == "--cells") cells_arg = next();
        else if (a == "--csv") csv_path = next();
        else if (a == "--trace") g_trace = true;
        else if (a == "--env") {
            std::string v = next();
            if (v == "hard") { env.name = "hard"; env.fov_deg = 270.0f; env.max_range = 5.0f; env.clutter = 25; env.repeat_rooms = true; }
            else if (v != "open") { std::fprintf(stderr, "unknown env %s\n", v.c_str()); return 1; }
        }
        else if (a == "--fov-deg") { env.fov_deg = (float)std::atof(next().c_str()); env.name = "custom"; }
        else if (a == "--max-range") { env.max_range = std::min(MAX_R, (float)std::atof(next().c_str())); env.name = "custom"; }
        else if (a == "--clutter") { env.clutter = std::atoi(next().c_str()); env.name = "custom"; }
        else if (a == "--repeat-rooms") { env.repeat_rooms = std::atoi(next().c_str()) != 0; env.name = "custom"; }
        else if (a == "--particles") cfg.np = std::atoi(next().c_str());
        else if (a == "--reset-th") cfg.reset_th = (float)std::atof(next().c_str());
        else if (a == "--er-sxy") cfg.er_sxy = (float)std::atof(next().c_str());
        else if (a == "--er-sth") cfg.er_sth = (float)std::atof(next().c_str());
        else if (a == "--reloc-frac") cfg.reloc_frac = (float)std::atof(next().c_str());
        else if (a == "--reloc-top") cfg.reloc_top = std::atoi(next().c_str());
        else if (a == "--inject-prior") cfg.inject_prior = (float)std::atof(next().c_str());
        else if (a == "--dual-exclude") cfg.dual_exclude = (float)std::atof(next().c_str());
        else if (a == "--dual-swap") cfg.dual_swap = std::atoi(next().c_str()) != 0;
        else if (a == "--dual-gate") cfg.dual_gate = std::atoi(next().c_str()) != 0;
        else if (a == "--dual-drift") cfg.dual_drift = (float)std::atof(next().c_str());
        else if (a == "--dual-np") cfg.dual_np = std::atoi(next().c_str());
        else if (a == "--dual-max-age") cfg.dual_max_age = std::atoi(next().c_str());
        else if (a == "--dual-drop") cfg.dual_drop = (float)std::atof(next().c_str());
        else if (a == "--lr-gate") cfg.lr_gate = (float)std::atof(next().c_str());
        else if (a == "--lr-margin") cfg.lr_margin = (float)std::atof(next().c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    std::vector<int> methods;
    for (auto& m : split(methods_arg))
        for (int k = 0; k < N_METHODS; k++) if (m == METHOD_NAMES[k]) methods.push_back(k);
    std::vector<std::string> cells = split(cells_arg);

    std::vector<Result> rows;
    for (int s = seed_offset; s < seed_offset + seed_count; s++) {
        std::mt19937 wrng(1000003u * (unsigned)s + 11u);
        World w = make_world(wrng, env);
        for (auto& cell : cells) {
            std::mt19937 erng(2000003u * (unsigned)s + (cell == "global" ? 7u : 3u));
            Episode e = make_episode(w, env, erng, cell == "global" ? std::min(steps, 300) : steps,
                                     cell == "global" ? -1 : kidnap_step);
            for (int m : methods) {
                Result r = run_episode(w, e, cfg, cell, m, s);
                r.env = env.name;
                std::printf("[%s] %-5s seed %d: recovered %d in %d steps, localized %.2f, final %d, resets %d (%d before), "
                            "err before %.3f, step %.2f ms, reloc %.2f ms\n",
                            cell.c_str(), r.method.c_str(), s, r.recovered, r.recover_steps, r.localized_frac,
                            r.final_ok, r.resets, r.resets_before, r.err_before, r.step_ms, r.reloc_ms);
                rows.push_back(r);
            }
        }
    }
    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        out << "env,cell,method,seed,recovered,recover_steps,localized_frac,final_ok,resets,resets_before,err_before,localized_before,"
               "err_after,step_ms,reloc_ms\n";
        for (auto& r : rows)
            out << r.env << "," << r.cell << "," << r.method << "," << r.seed << "," << r.recovered << "," << r.recover_steps << ","
                << r.localized_frac << "," << r.final_ok << "," << r.resets << "," << r.resets_before << ","
                << r.err_before << "," << r.localized_before << "," << r.err_after << "," << r.step_ms << "," << r.reloc_ms << "\n";
    }
    // summary
    for (auto& cell : cells)
        for (int m : methods) {
            int n = 0, rec = 0, fin = 0, rb = 0; float st = 0, lf = 0, eb = 0, lb = 0, gm = 0;
            for (auto& r : rows)
                if (r.cell == cell && r.method == METHOD_NAMES[m]) {
                    n++; rec += r.recovered; fin += r.final_ok; rb += r.resets_before > 0;
                    if (r.recovered) st += r.recover_steps;
                    lf += r.localized_frac; eb += r.err_before; lb += r.localized_before; gm += r.reloc_ms;
                }
            if (!n) continue;
            std::printf("%-6s %-5s recovered %d/%d (mean %.1f steps), final %d/%d, localized %.2f, "
                        "runs with resets before kidnap %d, err before %.3f, localized before %.2f, grid %.2f ms\n",
                        cell.c_str(), METHOD_NAMES[m], rec, n, rec ? st / rec : 0.0f, fin, n, lf / n, rb, eb / n, lb / n,
                        gm / n);
        }
    return 0;
}
