/*************************************************************************
    RRT-Connect (Bidirectional RRT) - CUDA parallelized
    Two trees grow simultaneously and connect
 ************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cfloat>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#define EXPAND_DIS 0.5f
#define GOAL_SAMPLE_RATE 5
#define MAX_NODES 5000

struct Node {
    float x, y;
    int parent;
};

// Parallel nearest neighbor search
__global__ void find_nearest_kernel(const float* nx, const float* ny, int n,
                                     float qx, float qy,
                                     float* block_min_dist, int* block_min_idx) {
    extern __shared__ char smem[];
    float* sdist = (float*)smem;
    int* sidx = (int*)(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;

    float best = FLT_MAX;
    int bestj = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float dx = nx[i] - qx, dy = ny[i] - qy;
        float d = dx * dx + dy * dy;
        if (d < best) { best = d; bestj = i; }
    }
    sdist[tid] = best;
    sidx[tid] = bestj;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdist[tid + s] < sdist[tid]) {
            sdist[tid] = sdist[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_min_dist[blockIdx.x] = sdist[0];
        block_min_idx[blockIdx.x] = sidx[0];
    }
}

bool collision_check(float x, float y,
                     const std::vector<std::vector<float>>& obs) {
    for (auto& o : obs) {
        float dx = o[0] - x, dy = o[1] - y;
        if (sqrtf(dx * dx + dy * dy) <= o[2]) return false;
    }
    return true;
}

bool line_collision(float x1, float y1, float x2, float y2,
                    const std::vector<std::vector<float>>& obs) {
    int steps = (int)(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)) / 0.1f) + 1;
    for (int s = 0; s <= steps; s++) {
        float t = (float)s / steps;
        float x = x1 + t * (x2 - x1), y = y1 + t * (y2 - y1);
        if (!collision_check(x, y, obs)) return false;
    }
    return true;
}

int gpu_find_nearest(float* d_nx, float* d_ny, int n, float qx, float qy,
                     float* d_bdist, int* d_bidx) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    size_t smem = threads * (sizeof(float) + sizeof(int));
    find_nearest_kernel<<<blocks, threads, smem>>>(d_nx, d_ny, n, qx, qy, d_bdist, d_bidx);

    std::vector<float> h_bd(blocks);
    std::vector<int> h_bi(blocks);
    CUDA_CHECK(cudaMemcpy(h_bd.data(), d_bdist, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bi.data(), d_bidx, blocks * sizeof(int), cudaMemcpyDeviceToHost));

    float best = FLT_MAX;
    int bestj = 0;
    for (int i = 0; i < blocks; i++) {
        if (h_bd[i] < best) { best = h_bd[i]; bestj = h_bi[i]; }
    }
    return bestj;
}

int main() {
    std::vector<std::vector<float>> obs = {{5,5,1},{3,6,2},{3,8,2},{3,10,2},{7,5,2},{9,5,2}};
    float sx = 0, sy = 0, gx = 6, gy = 9;
    float rand_min = -2, rand_max = 15;

    std::vector<Node> tree_a, tree_b;
    tree_a.push_back({sx, sy, -1});
    tree_b.push_back({gx, gy, -1});

    // Device arrays for both trees
    float *d_ax, *d_ay, *d_bx, *d_by;
    CUDA_CHECK(cudaMalloc(&d_ax, MAX_NODES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ay, MAX_NODES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bx, MAX_NODES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_by, MAX_NODES * sizeof(float)));

    float tmp;
    tmp = sx; CUDA_CHECK(cudaMemcpy(d_ax, &tmp, sizeof(float), cudaMemcpyHostToDevice));
    tmp = sy; CUDA_CHECK(cudaMemcpy(d_ay, &tmp, sizeof(float), cudaMemcpyHostToDevice));
    tmp = gx; CUDA_CHECK(cudaMemcpy(d_bx, &tmp, sizeof(float), cudaMemcpyHostToDevice));
    tmp = gy; CUDA_CHECK(cudaMemcpy(d_by, &tmp, sizeof(float), cudaMemcpyHostToDevice));

    float *d_bdist; int *d_bidx;
    CUDA_CHECK(cudaMalloc(&d_bdist, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bidx, 256 * sizeof(int)));

    std::mt19937 gen(42);
    std::uniform_int_distribution<> goal_dis(0, 100);
    std::uniform_real_distribution<float> area_dis(rand_min, rand_max);

    int img_size = 17, img_reso = 50;
    cv::namedWindow("rrt_connect", cv::WINDOW_NORMAL);
    cv::Mat bg(img_size * img_reso, img_size * img_reso, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<cv::Mat> frames;

    // Draw obstacles and start/goal
    auto to_px = [&](float v) { return (int)((v - rand_min) * img_reso); };
    cv::circle(bg, cv::Point(to_px(sx), to_px(sy)), 20, cv::Scalar(0, 0, 255), -1);
    cv::circle(bg, cv::Point(to_px(gx), to_px(gy)), 20, cv::Scalar(255, 0, 0), -1);
    for (auto& o : obs)
        cv::circle(bg, cv::Point(to_px(o[0]), to_px(o[1])), (int)(o[2] * img_reso), cv::Scalar(0, 0, 0), -1);

    bool connected = false;
    int connect_a = -1, connect_b = -1;
    bool swapped = false;

    std::cout << "RRT-Connect with CUDA" << std::endl;

    for (int iter = 0; iter < 5000 && !connected; iter++) {
        // Pointers to "active" and "other" tree
        auto& ta = swapped ? tree_b : tree_a;
        auto& tb = swapped ? tree_a : tree_b;
        float* d_tax = swapped ? d_bx : d_ax;
        float* d_tay = swapped ? d_by : d_ay;
        float* d_tbx = swapped ? d_ax : d_bx;
        float* d_tby = swapped ? d_ay : d_by;

        // Random sample
        float rx, ry;
        if (goal_dis(gen) > GOAL_SAMPLE_RATE) {
            rx = area_dis(gen); ry = area_dis(gen);
        } else {
            rx = tb[0].x; ry = tb[0].y;
        }

        // Extend tree_a toward random
        int near_a = gpu_find_nearest(d_tax, d_tay, (int)ta.size(), rx, ry, d_bdist, d_bidx);
        float dx = rx - ta[near_a].x, dy = ry - ta[near_a].y;
        float dist = sqrtf(dx * dx + dy * dy);
        float nx = ta[near_a].x + EXPAND_DIS * dx / (dist + 1e-6f);
        float ny = ta[near_a].y + EXPAND_DIS * dy / (dist + 1e-6f);

        if (!collision_check(nx, ny, obs) || !line_collision(ta[near_a].x, ta[near_a].y, nx, ny, obs))
            { swapped = !swapped; continue; }

        int new_a_idx = (int)ta.size();
        ta.push_back({nx, ny, near_a});
        float tnx = nx, tny = ny;
        CUDA_CHECK(cudaMemcpy(d_tax + new_a_idx, &tnx, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tay + new_a_idx, &tny, sizeof(float), cudaMemcpyHostToDevice));

        // Draw tree_a edge
        cv::Scalar col_a = swapped ? cv::Scalar(200, 200, 0) : cv::Scalar(0, 200, 0);
        cv::line(bg, cv::Point(to_px(ta[near_a].x), to_px(ta[near_a].y)),
                 cv::Point(to_px(nx), to_px(ny)), col_a, 2);

        // CONNECT: tree_b tries to reach new_a
        for (int c = 0; c < 100; c++) {
            int near_b = gpu_find_nearest(d_tbx, d_tby, (int)tb.size(), nx, ny, d_bdist, d_bidx);
            float dx2 = nx - tb[near_b].x, dy2 = ny - tb[near_b].y;
            float dist2 = sqrtf(dx2 * dx2 + dy2 * dy2);

            if (dist2 < EXPAND_DIS) {
                // Connected!
                int new_b_idx = (int)tb.size();
                tb.push_back({nx, ny, near_b});
                float tbx = nx, tby = ny;
                CUDA_CHECK(cudaMemcpy(d_tbx + new_b_idx, &tbx, sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_tby + new_b_idx, &tby, sizeof(float), cudaMemcpyHostToDevice));

                connect_a = new_a_idx;
                connect_b = new_b_idx;
                connected = true;
                break;
            }

            float bx = tb[near_b].x + EXPAND_DIS * dx2 / (dist2 + 1e-6f);
            float by = tb[near_b].y + EXPAND_DIS * dy2 / (dist2 + 1e-6f);

            if (!collision_check(bx, by, obs) || !line_collision(tb[near_b].x, tb[near_b].y, bx, by, obs))
                break;

            int new_b_idx = (int)tb.size();
            tb.push_back({bx, by, near_b});
            float tbx = bx, tby = by;
            CUDA_CHECK(cudaMemcpy(d_tbx + new_b_idx, &tbx, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tby + new_b_idx, &tby, sizeof(float), cudaMemcpyHostToDevice));

            cv::Scalar col_b = swapped ? cv::Scalar(0, 200, 0) : cv::Scalar(200, 200, 0);
            cv::line(bg, cv::Point(to_px(tb[near_b].x), to_px(tb[near_b].y)),
                     cv::Point(to_px(bx), to_px(by)), col_b, 2);
        }

        if (iter % 3 == 0) {
            frames.push_back(bg.clone());
            cv::imshow("rrt_connect", bg);
            cv::waitKey(1);
        }
        swapped = !swapped;
    }

    if (connected) {
        std::cout << "Connected! Drawing path..." << std::endl;

        // Trace path through tree_a
        auto& ta_final = swapped ? tree_a : tree_a; // connect indices relative to unswapped
        auto& tb_final = swapped ? tree_b : tree_b;

        // Path from tree_a root to connection
        std::vector<cv::Point> path;
        int idx = connect_a;
        while (idx >= 0) {
            path.push_back(cv::Point(to_px(tree_a[idx].x), to_px(tree_a[idx].y)));
            idx = tree_a[idx].parent;
        }
        std::reverse(path.begin(), path.end());

        // Path from connection to tree_b root
        idx = connect_b;
        while (idx >= 0) {
            path.push_back(cv::Point(to_px(tree_b[idx].x), to_px(tree_b[idx].y)));
            idx = tree_b[idx].parent;
        }

        for (int i = 0; i < (int)path.size() - 1; i++) {
            cv::line(bg, path[i], path[i + 1], cv::Scalar(255, 0, 255), 6);
        }
    }

    for (int i = 0; i < 30; i++) { frames.push_back(bg.clone()); }
    cv::imshow("rrt_connect", bg);
    cv::waitKey(1);

    // Write video from collected frames
    if (!frames.empty()) {
        cv::VideoWriter vid("gif/rrt_connect.avi",
                            cv::VideoWriter::fourcc('X','V','I','D'), 30,
                            cv::Size(frames[0].cols, frames[0].rows));
        for (auto& f : frames) vid.write(f);
        vid.release();
    }
    std::cout << "Video saved to gif/rrt_connect.avi" << std::endl;

    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_bx); cudaFree(d_by);
    cudaFree(d_bdist); cudaFree(d_bidx);
    return 0;
}
