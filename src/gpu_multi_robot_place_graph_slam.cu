// gpu_multi_robot_place_graph_slam.cu
//
// Multi-robot place-recognition pose-graph SLAM on the GPU.
//
// The demo creates three drifting robot odometry chains over a shared route.
// A CUDA all-pairs descriptor matcher proposes inter-robot place-recognition
// edges, then a CUDA edge-projection pose-graph optimizer pulls the local robot
// graphs into one shared map.
//
// Output: gif/gpu_multi_robot_place_graph_slam.gif

#include <cuda_runtime.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.cuh"

namespace cudabot {

constexpr int N_ROBOTS = 3;
constexpr int N_PER_ROBOT = 82;
constexpr int N_NODES = N_ROBOTS * N_PER_ROBOT;
constexpr int N_PLACES = 126;
constexpr int DESC_DIM = 16;
constexpr int OPT_ITERS = 180;
constexpr int FRAME_W = 1140;
constexpr int FRAME_H = 470;
constexpr int PANEL_W = FRAME_W / 3;
constexpr int VIDEO_FPS = 12;

struct Pose {
    float x, y, th;
};

struct Node {
    int robot;
    int place;
    Pose gt;
    Pose initial;
};

struct Edge {
    int i, j;
    float dx, dy, dth;
    float w;
    int place_edge;
};

struct Match {
    int i, j;
    float score;
};

struct Bounds {
    float minx, maxx, miny, maxy;
};

static inline float wrap_angle(float a) {
    while (a > static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}

static Pose place_pose(int place) {
    float u = 2.0f * static_cast<float>(M_PI) * static_cast<float>(place) / static_cast<float>(N_PLACES);
    float x = 12.0f * std::cos(u) + 2.5f * std::cos(2.3f * u);
    float y = 8.0f * std::sin(u) + 2.0f * std::sin(1.7f * u);
    float dx = -12.0f * std::sin(u) - 2.5f * 2.3f * std::sin(2.3f * u);
    float dy = 8.0f * std::cos(u) + 2.0f * 1.7f * std::cos(1.7f * u);
    return {x, y, std::atan2(dy, dx)};
}

static Pose relative_world(const Pose& a, const Pose& b) {
    return {b.x - a.x, b.y - a.y, wrap_angle(b.th - a.th)};
}

static Pose drift_pose(const Pose& p, int robot, int k, std::mt19937& rng) {
    std::normal_distribution<float> nxy(0.0f, 0.045f);
    std::normal_distribution<float> nth(0.0f, 0.012f);
    const float rot[N_ROBOTS] = {0.0f, 0.44f, -0.38f};
    const float tx[N_ROBOTS] = {0.0f, 7.0f, -6.0f};
    const float ty[N_ROBOTS] = {0.0f, -5.0f, 6.5f};
    const float drift_x[N_ROBOTS] = {0.018f, -0.035f, 0.028f};
    const float drift_y[N_ROBOTS] = {-0.010f, 0.030f, -0.024f};
    float c = std::cos(rot[robot]);
    float s = std::sin(rot[robot]);
    float x = c * p.x - s * p.y + tx[robot] + drift_x[robot] * k + nxy(rng);
    float y = s * p.x + c * p.y + ty[robot] + drift_y[robot] * k + nxy(rng);
    float th = wrap_angle(p.th + rot[robot] + 0.004f * k * (robot == 0 ? 0.4f : (robot == 1 ? -1.0f : 0.8f)) + nth(rng));
    if (robot == 0 && k == 0) return p;
    return {x, y, th};
}

static void descriptor_for_place(int place, float* dst) {
    float norm = 0.0f;
    for (int d = 0; d < DESC_DIM; ++d) {
        float f = static_cast<float>(d + 1);
        float p = static_cast<float>(place);
        float v = std::sin(0.173f * f * p) + 0.65f * std::cos(0.097f * (f + 2.0f) * p) +
                  0.35f * std::sin(0.311f * f * static_cast<float>((place * 7) % N_PLACES));
        dst[d] = v;
        norm += v * v;
    }
    norm = std::sqrt(std::max(norm, 1.0e-8f));
    for (int d = 0; d < DESC_DIM; ++d) dst[d] /= norm;
}

static void make_nodes(std::vector<Node>& nodes, std::vector<float>& descriptors) {
    nodes.clear();
    nodes.reserve(N_NODES);
    descriptors.assign(N_NODES * DESC_DIM, 0.0f);

    std::mt19937 rng(11);
    const int offsets[N_ROBOTS] = {0, 31, 62};
    for (int r = 0; r < N_ROBOTS; ++r) {
        for (int k = 0; k < N_PER_ROBOT; ++k) {
            int idx = r * N_PER_ROBOT + k;
            int place = (offsets[r] + k) % N_PLACES;
            Pose gt = place_pose(place);
            Pose initial = drift_pose(gt, r, k, rng);
            nodes.push_back({r, place, gt, initial});
            descriptor_for_place(place, &descriptors[idx * DESC_DIM]);
        }
    }
}

__global__ void descriptor_match_kernel(int n,
                                        const int* __restrict__ robot,
                                        const float* __restrict__ desc,
                                        float* __restrict__ scores) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    int i = idx / n;
    int j = idx - i * n;
    if (i >= j || robot[i] == robot[j]) {
        scores[idx] = 0.0f;
        return;
    }
    float dist2 = 0.0f;
    for (int d = 0; d < DESC_DIM; ++d) {
        float diff = desc[i * DESC_DIM + d] - desc[j * DESC_DIM + d];
        dist2 += diff * diff;
    }
    scores[idx] = expf(-8.0f * dist2);
}

__global__ void project_edges_kernel(int n_edges,
                                     const int* __restrict__ ei,
                                     const int* __restrict__ ej,
                                     const float* __restrict__ ez,
                                     const float* __restrict__ ew,
                                     const float* __restrict__ poses,
                                     float* __restrict__ accum,
                                     float* __restrict__ weights) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int i = ei[e];
    int j = ej[e];
    float dx = ez[3 * e + 0];
    float dy = ez[3 * e + 1];
    float dth = ez[3 * e + 2];
    float w = ew[e];

    float xi = poses[3 * i + 0];
    float yi = poses[3 * i + 1];
    float ti = poses[3 * i + 2];
    float xj = poses[3 * j + 0];
    float yj = poses[3 * j + 1];
    float tj = poses[3 * j + 2];

    float rx = (xj - xi) - dx;
    float ry = (yj - yi) - dy;
    float rt = tj - ti - dth;
    while (rt > M_PI) rt -= 2.0f * M_PI;
    while (rt < -M_PI) rt += 2.0f * M_PI;

    atomicAdd(&accum[3 * i + 0],  w * rx);
    atomicAdd(&accum[3 * i + 1],  w * ry);
    atomicAdd(&accum[3 * i + 2],  w * rt);
    atomicAdd(&accum[3 * j + 0], -w * rx);
    atomicAdd(&accum[3 * j + 1], -w * ry);
    atomicAdd(&accum[3 * j + 2], -w * rt);
    atomicAdd(&weights[i], w);
    atomicAdd(&weights[j], w);
}

__global__ void apply_pose_delta_kernel(int n,
                                        float* __restrict__ poses,
                                        const float* __restrict__ accum,
                                        const float* __restrict__ weights,
                                        float alpha,
                                        float anchor_x,
                                        float anchor_y,
                                        float anchor_th) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0) {
        poses[0] = anchor_x;
        poses[1] = anchor_y;
        poses[2] = anchor_th;
        return;
    }
    float inv = 1.0f / fmaxf(weights[i], 1.0e-4f);
    poses[3 * i + 0] += alpha * accum[3 * i + 0] * inv;
    poses[3 * i + 1] += alpha * accum[3 * i + 1] * inv;
    poses[3 * i + 2] += alpha * accum[3 * i + 2] * inv;
    while (poses[3 * i + 2] > M_PI) poses[3 * i + 2] -= 2.0f * M_PI;
    while (poses[3 * i + 2] < -M_PI) poses[3 * i + 2] += 2.0f * M_PI;
}

static std::vector<Match> run_place_recognition(const std::vector<Node>& nodes,
                                                const std::vector<float>& descriptors,
                                                std::vector<float>& score_matrix) {
    std::vector<int> robots(N_NODES);
    for (int i = 0; i < N_NODES; ++i) robots[i] = nodes[i].robot;

    int* d_robot = nullptr;
    float* d_desc = nullptr;
    float* d_scores = nullptr;
    CUDA_CHECK(cudaMalloc(&d_robot, N_NODES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_desc, N_NODES * DESC_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, N_NODES * N_NODES * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_robot, robots.data(), N_NODES * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_desc, descriptors.data(), N_NODES * DESC_DIM * sizeof(float), cudaMemcpyHostToDevice));

    int total = N_NODES * N_NODES;
    descriptor_match_kernel<<<(total + 255) / 256, 256>>>(N_NODES, d_robot, d_desc, d_scores);
    CUDA_CHECK(cudaGetLastError());
    score_matrix.assign(total, 0.0f);
    CUDA_CHECK(cudaMemcpy(score_matrix.data(), d_scores, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_robot));
    CUDA_CHECK(cudaFree(d_desc));
    CUDA_CHECK(cudaFree(d_scores));

    std::vector<Match> candidates;
    for (int i = 0; i < N_NODES; ++i) {
        for (int j = i + 1; j < N_NODES; ++j) {
            float s = score_matrix[i * N_NODES + j];
            if (s > 0.985f) candidates.push_back({i, j, s});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Match& a, const Match& b) {
        return a.score > b.score;
    });

    std::vector<unsigned char> used(N_NODES, 0u);
    std::vector<Match> matches;
    for (const Match& m : candidates) {
        if (used[m.i] || used[m.j]) continue;
        matches.push_back(m);
        used[m.i] = used[m.j] = 1u;
        if (static_cast<int>(matches.size()) >= 42) break;
    }
    return matches;
}

static void make_edges(const std::vector<Node>& nodes,
                       const std::vector<Match>& matches,
                       std::vector<Edge>& edges) {
    edges.clear();
    edges.reserve(N_ROBOTS * (N_PER_ROBOT - 1) + matches.size());
    for (int r = 0; r < N_ROBOTS; ++r) {
        for (int k = 0; k + 1 < N_PER_ROBOT; ++k) {
            int i = r * N_PER_ROBOT + k;
            int j = i + 1;
            Pose z = relative_world(nodes[i].gt, nodes[j].gt);
            edges.push_back({i, j, z.x, z.y, z.th, 0.85f, 0});
        }
    }
    for (const Match& m : matches) {
        Pose z = relative_world(nodes[m.i].gt, nodes[m.j].gt);
        edges.push_back({m.i, m.j, z.x, z.y, z.th, 4.0f, 1});
    }
}

static float rmse_xy(const std::vector<float>& poses, const std::vector<Node>& nodes) {
    double acc = 0.0;
    for (int i = 0; i < N_NODES; ++i) {
        double dx = poses[3 * i + 0] - nodes[i].gt.x;
        double dy = poses[3 * i + 1] - nodes[i].gt.y;
        acc += dx * dx + dy * dy;
    }
    return static_cast<float>(std::sqrt(acc / N_NODES));
}

static std::vector<std::vector<float>> optimize_graph(const std::vector<Node>& nodes,
                                                      const std::vector<Edge>& edges,
                                                      std::vector<float>& final_poses,
                                                      float& gpu_ms) {
    std::vector<float> poses(N_NODES * 3);
    for (int i = 0; i < N_NODES; ++i) {
        poses[3 * i + 0] = nodes[i].initial.x;
        poses[3 * i + 1] = nodes[i].initial.y;
        poses[3 * i + 2] = nodes[i].initial.th;
    }

    std::vector<int> ei(edges.size()), ej(edges.size());
    std::vector<float> ez(edges.size() * 3), ew(edges.size());
    for (size_t e = 0; e < edges.size(); ++e) {
        ei[e] = edges[e].i;
        ej[e] = edges[e].j;
        ez[3 * e + 0] = edges[e].dx;
        ez[3 * e + 1] = edges[e].dy;
        ez[3 * e + 2] = edges[e].dth;
        ew[e] = edges[e].w;
    }

    int *d_ei = nullptr, *d_ej = nullptr;
    float *d_ez = nullptr, *d_ew = nullptr, *d_poses = nullptr, *d_accum = nullptr, *d_weights = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ei, ei.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ej, ej.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ez, ez.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ew, ew.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_poses, poses.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accum, poses.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, N_NODES * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ei, ei.data(), ei.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ej, ej.data(), ej.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ez, ez.data(), ez.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ew, ew.data(), ew.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_poses, poses.data(), poses.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<std::vector<float>> snapshots;
    snapshots.push_back(poses);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int edge_blocks = (static_cast<int>(edges.size()) + 255) / 256;
    int node_blocks = (N_NODES + 255) / 256;
    for (int it = 0; it < OPT_ITERS; ++it) {
        CUDA_CHECK(cudaMemset(d_accum, 0, poses.size() * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_weights, 0, N_NODES * sizeof(float)));
        project_edges_kernel<<<edge_blocks, 256>>>(static_cast<int>(edges.size()), d_ei, d_ej, d_ez, d_ew,
                                                   d_poses, d_accum, d_weights);
        apply_pose_delta_kernel<<<node_blocks, 256>>>(N_NODES, d_poses, d_accum, d_weights, 0.42f,
                                                      nodes[0].gt.x, nodes[0].gt.y, nodes[0].gt.th);
        CUDA_CHECK(cudaGetLastError());
        if ((it + 1) % 6 == 0 || it + 1 == OPT_ITERS) {
            CUDA_CHECK(cudaMemcpy(poses.data(), d_poses, poses.size() * sizeof(float), cudaMemcpyDeviceToHost));
            snapshots.push_back(poses);
        }
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));
    CUDA_CHECK(cudaMemcpy(final_poses.data(), d_poses, poses.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_ei));
    CUDA_CHECK(cudaFree(d_ej));
    CUDA_CHECK(cudaFree(d_ez));
    CUDA_CHECK(cudaFree(d_ew));
    CUDA_CHECK(cudaFree(d_poses));
    CUDA_CHECK(cudaFree(d_accum));
    CUDA_CHECK(cudaFree(d_weights));
    return snapshots;
}

static cv::Scalar robot_color(int robot) {
    static const cv::Scalar colors[N_ROBOTS] = {
        cv::Scalar(60, 95, 225),
        cv::Scalar(45, 165, 80),
        cv::Scalar(215, 115, 35),
    };
    return colors[robot % N_ROBOTS];
}

static Bounds compute_bounds(const std::vector<Node>& nodes, const std::vector<float>& optimized) {
    Bounds b{1e9f, -1e9f, 1e9f, -1e9f};
    auto add = [&](float x, float y) {
        b.minx = std::min(b.minx, x);
        b.maxx = std::max(b.maxx, x);
        b.miny = std::min(b.miny, y);
        b.maxy = std::max(b.maxy, y);
    };
    for (int i = 0; i < N_NODES; ++i) {
        add(nodes[i].gt.x, nodes[i].gt.y);
        add(nodes[i].initial.x, nodes[i].initial.y);
        add(optimized[3 * i + 0], optimized[3 * i + 1]);
    }
    float pad = 2.0f;
    b.minx -= pad; b.maxx += pad; b.miny -= pad; b.maxy += pad;
    return b;
}

static cv::Point to_px(const Bounds& b, int panel_x, float x, float y) {
    constexpr int margin = 34;
    float sx = static_cast<float>(PANEL_W - 2 * margin) / std::max(1.0e-4f, b.maxx - b.minx);
    float sy = static_cast<float>(FRAME_H - 2 * margin) / std::max(1.0e-4f, b.maxy - b.miny);
    float s = std::min(sx, sy);
    int px = panel_x + margin + static_cast<int>((x - b.minx) * s);
    int py = margin + static_cast<int>((b.maxy - y) * s);
    return cv::Point(px, py);
}

static void draw_pose_set(cv::Mat& img,
                          const Bounds& b,
                          int panel_x,
                          const std::vector<Node>& nodes,
                          const std::vector<float>& poses,
                          bool draw_gt,
                          int upto = N_PER_ROBOT) {
    if (draw_gt) {
        for (int k = 1; k < N_PLACES; ++k) {
            Pose a = place_pose(k - 1);
            Pose c = place_pose(k);
            cv::line(img, to_px(b, panel_x, a.x, a.y), to_px(b, panel_x, c.x, c.y),
                     cv::Scalar(205, 210, 214), 1, cv::LINE_AA);
        }
    }
    for (int r = 0; r < N_ROBOTS; ++r) {
        cv::Scalar col = robot_color(r);
        for (int k = 1; k < upto; ++k) {
            int i0 = r * N_PER_ROBOT + k - 1;
            int i1 = r * N_PER_ROBOT + k;
            cv::Point p0 = to_px(b, panel_x, poses[3 * i0 + 0], poses[3 * i0 + 1]);
            cv::Point p1 = to_px(b, panel_x, poses[3 * i1 + 0], poses[3 * i1 + 1]);
            cv::line(img, p0, p1, col, 2, cv::LINE_AA);
        }
        for (int k = 0; k < upto; k += 8) {
            int i = r * N_PER_ROBOT + k;
            cv::circle(img, to_px(b, panel_x, poses[3 * i + 0], poses[3 * i + 1]), 3, col, -1, cv::LINE_AA);
        }
    }
}

static std::vector<float> initial_pose_vector(const std::vector<Node>& nodes) {
    std::vector<float> poses(N_NODES * 3);
    for (int i = 0; i < N_NODES; ++i) {
        poses[3 * i + 0] = nodes[i].initial.x;
        poses[3 * i + 1] = nodes[i].initial.y;
        poses[3 * i + 2] = nodes[i].initial.th;
    }
    return poses;
}

static void draw_matches(cv::Mat& img,
                         const Bounds& b,
                         int panel_x,
                         const std::vector<float>& poses,
                         const std::vector<Match>& matches,
                         int reveal) {
    int n = std::min(reveal, static_cast<int>(matches.size()));
    for (int m = 0; m < n; ++m) {
        int i = matches[m].i;
        int j = matches[m].j;
        cv::Point pi = to_px(b, panel_x, poses[3 * i + 0], poses[3 * i + 1]);
        cv::Point pj = to_px(b, panel_x, poses[3 * j + 0], poses[3 * j + 1]);
        cv::line(img, pi, pj, cv::Scalar(30, 190, 210), 1, cv::LINE_AA);
        cv::circle(img, pi, 4, cv::Scalar(20, 150, 190), -1, cv::LINE_AA);
        cv::circle(img, pj, 4, cv::Scalar(20, 150, 190), -1, cv::LINE_AA);
    }
}

static void panel_title(cv::Mat& img, int x, const std::string& title, const std::string& subtitle) {
    cv::rectangle(img, cv::Rect(x, 0, PANEL_W, 52), cv::Scalar(250, 251, 252), -1);
    cv::putText(img, title, cv::Point(x + 14, 22), cv::FONT_HERSHEY_SIMPLEX, 0.54,
                cv::Scalar(24, 30, 38), 2, cv::LINE_AA);
    cv::putText(img, subtitle, cv::Point(x + 14, 43), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(84, 98, 112), 1, cv::LINE_AA);
}

static void render_video(const std::vector<Node>& nodes,
                         const std::vector<Match>& matches,
                         const std::vector<std::vector<float>>& snapshots,
                         const std::vector<float>& final_poses,
                         float rmse_init,
                         float rmse_final,
                         float gpu_ms) {
    int mkdir_rc = std::system("mkdir -p gif tmp/gpu_multi_robot_place_graph_slam_frames");
    if (mkdir_rc != 0) std::fprintf(stderr, "mkdir failed (%d)\n", mkdir_rc);
    Bounds bounds = compute_bounds(nodes, final_poses);
    std::vector<float> initial = initial_pose_vector(nodes);

    int frames = static_cast<int>(snapshots.size());
    int frame_id = 0;
    for (int f = 0; f < frames + 18; ++f) {
        int reveal = std::min(static_cast<int>(matches.size()), 2 + f * 2);
        cv::Mat img(FRAME_H, FRAME_W, CV_8UC3, cv::Scalar(244, 247, 249));

        for (int p = 1; p < 3; ++p) {
            cv::line(img, cv::Point(p * PANEL_W, 0), cv::Point(p * PANEL_W, FRAME_H),
                     cv::Scalar(218, 224, 230), 1, cv::LINE_AA);
        }

        panel_title(img, 0, "raw robot odometry", cv::format("three local graphs, RMSE %.2f m", rmse_init));
        panel_title(img, PANEL_W, "GPU place recognition", cv::format("%d inter-robot matches from %dx%d scores", static_cast<int>(matches.size()), N_NODES, N_NODES));
        panel_title(img, PANEL_W * 2, "optimized shared graph", cv::format("RMSE %.2f m, %.2f ms GPU", rmse_final, gpu_ms));

        draw_pose_set(img, bounds, 0, nodes, initial, true);
        draw_pose_set(img, bounds, PANEL_W, nodes, initial, true);
        draw_matches(img, bounds, PANEL_W, initial, matches, reveal);
        draw_pose_set(img, bounds, PANEL_W * 2, nodes, final_poses, true);
        draw_matches(img, bounds, PANEL_W * 2, final_poses, matches, std::min(18, static_cast<int>(matches.size())));

        cv::putText(img, "descriptor all-pairs -> inter-robot loop edges -> pose graph projection",
                    cv::Point(18, FRAME_H - 16), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(45, 55, 65), 1, cv::LINE_AA);
        char path[256];
        std::snprintf(path, sizeof(path), "tmp/gpu_multi_robot_place_graph_slam_frames/frame_%03d.png", frame_id++);
        cv::imwrite(path, img);
    }
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -framerate %d -i tmp/gpu_multi_robot_place_graph_slam_frames/frame_%%03d.png "
                  "-vf \"fps=%d,scale=980:-1:flags=lanczos,split[a][b];"
                  "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" "
                  "gif/gpu_multi_robot_place_graph_slam.gif 2>/dev/null",
                  VIDEO_FPS, VIDEO_FPS);
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d) for gpu_multi_robot_place_graph_slam.gif\n", rc);
}

int run_demo() {
    std::vector<Node> nodes;
    std::vector<float> descriptors;
    make_nodes(nodes, descriptors);

    std::vector<float> score_matrix;
    std::vector<Match> matches = run_place_recognition(nodes, descriptors, score_matrix);
    std::vector<Edge> edges;
    make_edges(nodes, matches, edges);

    std::vector<float> initial = initial_pose_vector(nodes);
    std::vector<float> final_poses(N_NODES * 3, 0.0f);
    float gpu_ms = 0.0f;
    std::vector<std::vector<float>> snapshots = optimize_graph(nodes, edges, final_poses, gpu_ms);

    float init_rmse = rmse_xy(initial, nodes);
    float final_rmse = rmse_xy(final_poses, nodes);
    render_video(nodes, matches, snapshots, final_poses, init_rmse, final_rmse, gpu_ms);

    int exact_place_matches = 0;
    for (const Match& m : matches) {
        if (nodes[m.i].place == nodes[m.j].place) ++exact_place_matches;
    }
    std::printf("GPU multi-robot place-graph SLAM\n");
    std::printf("nodes=%d descriptors=%d scores=%d edges=%zu place_edges=%zu exact_place_matches=%d/%zu\n",
                N_NODES, DESC_DIM, N_NODES * N_NODES, edges.size(), matches.size(),
                exact_place_matches, matches.size());
    std::printf("RMSE %.3f m -> %.3f m, GPU optimize %.2f ms (%d iterations)\n",
                init_rmse, final_rmse, gpu_ms, OPT_ITERS);
    std::printf("Wrote gif/gpu_multi_robot_place_graph_slam.gif\n");
    return final_rmse < init_rmse ? 0 : 1;
}

}  // namespace cudabot

int main() {
    return cudabot::run_demo();
}
