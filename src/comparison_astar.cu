/*************************************************************************
    A* Search: CPU vs CUDA side-by-side comparison GIF generator
    Left panel:  CPU (triple-nested loop obstacle map construction)
    Right panel: CUDA (GPU-parallel obstacle map construction)
    The A* search itself is sequential on both sides.
    Timing compares obstacle map construction only.
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Node
// -------------------------------------------------------------------------
class Node {
public:
    int x;
    int y;
    float sum_cost;
    Node* p_node;

    Node(int x_, int y_, float sum_cost_ = 0, Node* p_node_ = NULL)
        : x(x_), y(y_), sum_cost(sum_cost_), p_node(p_node_) {}
};

// -------------------------------------------------------------------------
// CUDA kernel: obstacle map construction (1 thread per grid cell)
// -------------------------------------------------------------------------
__global__ void calc_obstacle_map_kernel(
    int* d_obmap,
    const int* d_ox,
    const int* d_oy,
    int n_obstacles,
    int xwidth,
    int ywidth,
    int min_ox,
    int min_oy,
    float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= xwidth || j >= ywidth)
        return;

    int x = i + min_ox;
    int y = j + min_oy;
    float threshold_sq = threshold * threshold;

    int blocked = 0;
    for (int k = 0; k < n_obstacles; k++) {
        float dx = (float)(d_ox[k] - x);
        float dy = (float)(d_oy[k] - y);
        if (dx * dx + dy * dy <= threshold_sq) {
            blocked = 1;
            break;
        }
    }

    d_obmap[i * ywidth + j] = blocked;
}

// -------------------------------------------------------------------------
// CPU obstacle map construction (triple-nested loop)
// -------------------------------------------------------------------------
std::vector<std::vector<int>> calc_obstacle_map_cpu(
    std::vector<int>& ox, std::vector<int>& oy,
    int min_ox, int max_ox,
    int min_oy, int max_oy,
    float reso, float vr,
    cv::Mat& img, int img_reso)
{
    int xwidth = max_ox - min_ox;
    int ywidth = max_oy - min_oy;

    std::vector<std::vector<int>> obmap(xwidth, vector<int>(ywidth, 0));

    for (int i = 0; i < xwidth; i++) {
        int x = i + min_ox;
        for (int j = 0; j < ywidth; j++) {
            int y = j + min_oy;
            for (int k = 0; k < (int)ox.size(); k++) {
                float d = std::sqrt(std::pow((float)(ox[k] - x), 2) + std::pow((float)(oy[k] - y), 2));
                if (d <= vr / reso) {
                    obmap[i][j] = 1;
                    cv::rectangle(img,
                                  cv::Point(i * img_reso + 1, j * img_reso + 1),
                                  cv::Point((i + 1) * img_reso, (j + 1) * img_reso),
                                  cv::Scalar(0, 0, 0), -1);
                    break;
                }
            }
        }
    }
    return obmap;
}

// -------------------------------------------------------------------------
// CUDA obstacle map construction (GPU kernel + host wrapper)
// -------------------------------------------------------------------------
std::vector<std::vector<int>> calc_obstacle_map_cuda(
    std::vector<int>& ox, std::vector<int>& oy,
    int min_ox, int max_ox,
    int min_oy, int max_oy,
    float reso, float vr,
    cv::Mat& img, int img_reso)
{
    int xwidth = max_ox - min_ox;
    int ywidth = max_oy - min_oy;
    int n_obstacles = (int)ox.size();
    float threshold = vr / reso;

    int* d_ox = nullptr;
    int* d_oy = nullptr;
    int* d_obmap = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ox, n_obstacles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_oy, n_obstacles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_obmap, xwidth * ywidth * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), n_obstacles * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), n_obstacles * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((xwidth + blockDim.x - 1) / blockDim.x,
                 (ywidth + blockDim.y - 1) / blockDim.y);

    calc_obstacle_map_kernel<<<gridDim, blockDim>>>(
        d_obmap, d_ox, d_oy,
        n_obstacles, xwidth, ywidth,
        min_ox, min_oy, threshold);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_obmap_flat(xwidth * ywidth);
    CUDA_CHECK(cudaMemcpy(h_obmap_flat.data(), d_obmap,
                          xwidth * ywidth * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));
    CUDA_CHECK(cudaFree(d_obmap));

    std::vector<std::vector<int>> obmap(xwidth, vector<int>(ywidth, 0));
    for (int i = 0; i < xwidth; i++) {
        for (int j = 0; j < ywidth; j++) {
            obmap[i][j] = h_obmap_flat[i * ywidth + j];
            if (obmap[i][j]) {
                cv::rectangle(img,
                              cv::Point(i * img_reso + 1, j * img_reso + 1),
                              cv::Point((i + 1) * img_reso, (j + 1) * img_reso),
                              cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    return obmap;
}

// -------------------------------------------------------------------------
// A* helper functions
// -------------------------------------------------------------------------
void calc_final_path(Node* goal, float reso, cv::Mat& img, float img_reso) {
    Node* node = goal;
    while (node->p_node != NULL) {
        node = node->p_node;
        cv::rectangle(img,
                      cv::Point(node->x * img_reso + 1, node->y * img_reso + 1),
                      cv::Point((node->x + 1) * img_reso, (node->y + 1) * img_reso),
                      cv::Scalar(255, 0, 0), -1);
    }
}

bool verify_node(Node* node,
                 vector<vector<int>>& obmap,
                 int min_ox, int max_ox,
                 int min_oy, int max_oy) {
    if (node->x < min_ox || node->y < min_oy ||
        node->x >= max_ox || node->y >= max_oy) {
        return false;
    }
    if (obmap[node->x - min_ox][node->y - min_oy])
        return false;
    return true;
}

float calc_heuristic(Node* n1, Node* n2, float w = 1.0f) {
    return w * std::sqrt((float)((n1->x - n2->x) * (n1->x - n2->x) +
                                 (n1->y - n2->y) * (n1->y - n2->y)));
}

std::vector<Node> get_motion_model() {
    return {Node(1, 0, 1),
            Node(0, 1, 1),
            Node(-1, 0, 1),
            Node(0, -1, 1),
            Node(-1, -1, std::sqrt(2.0f)),
            Node(-1, 1, std::sqrt(2.0f)),
            Node(1, -1, std::sqrt(2.0f)),
            Node(1, 1, std::sqrt(2.0f))};
}

// -------------------------------------------------------------------------
// A* search (CPU, same for both sides - returns explored nodes for animation)
// -------------------------------------------------------------------------
struct SearchState {
    Node* nstart;
    Node* ngoal;
    std::vector<std::vector<int>> visit_map;
    std::vector<std::vector<float>> path_cost;
    std::priority_queue<Node*, std::vector<Node*>,
                        std::function<bool(const Node*, const Node*)>> pq;
    std::vector<Node> motion;
    bool finished;
    int min_ox, max_ox, min_oy, max_oy;
    std::vector<std::vector<int>> obmap;

    SearchState(Node* start, Node* goal,
                int xw, int yw,
                int mnx, int mxx, int mny, int mxy,
                std::vector<std::vector<int>>& ob)
        : nstart(start), ngoal(goal),
          visit_map(xw, vector<int>(yw, 0)),
          path_cost(xw, vector<float>(yw, std::numeric_limits<float>::max())),
          pq([](const Node* l, const Node* r) { return l->sum_cost > r->sum_cost; }),
          motion(get_motion_model()),
          finished(false),
          min_ox(mnx), max_ox(mxx), min_oy(mny), max_oy(mxy),
          obmap(ob)
    {
        path_cost[nstart->x][nstart->y] = 0;
        pq.push(nstart);
    }

    // Advance one step: process one node from the priority queue
    // Returns list of newly explored nodes for drawing
    std::vector<std::pair<int,int>> step() {
        std::vector<std::pair<int,int>> explored;
        if (finished || pq.empty()) return explored;

        // Pop nodes until we find an unvisited one
        Node* node = nullptr;
        while (!pq.empty()) {
            node = pq.top();
            if (visit_map[node->x - min_ox][node->y - min_oy] == 1) {
                pq.pop();
                delete node;
                node = nullptr;
                continue;
            } else {
                pq.pop();
                visit_map[node->x - min_ox][node->y - min_oy] = 1;
                break;
            }
        }
        if (!node) return explored;

        if (node->x == ngoal->x && node->y == ngoal->y) {
            ngoal->sum_cost = node->sum_cost;
            ngoal->p_node = node;
            finished = true;
            return explored;
        }

        for (int i = 0; i < (int)motion.size(); i++) {
            Node* new_node = new Node(
                node->x + motion[i].x,
                node->y + motion[i].y,
                path_cost[node->x][node->y] + motion[i].sum_cost + calc_heuristic(ngoal, node),
                node);

            if (!verify_node(new_node, obmap, min_ox, max_ox, min_oy, max_oy)) {
                delete new_node;
                continue;
            }

            if (visit_map[new_node->x - min_ox][new_node->y - min_oy]) {
                delete new_node;
                continue;
            }

            if (path_cost[node->x][node->y] + motion[i].sum_cost < path_cost[new_node->x][new_node->y]) {
                path_cost[new_node->x][new_node->y] = path_cost[node->x][node->y] + motion[i].sum_cost;
                pq.push(new_node);
                explored.push_back({new_node->x, new_node->y});
            } else {
                delete new_node;
            }
        }

        return explored;
    }
};

// -------------------------------------------------------------------------
// Setup obstacles (same as original a_star.cpp)
// -------------------------------------------------------------------------
void setup_obstacles(vector<float>& ox, vector<float>& oy) {
    for (float i = 0; i < 60; i++) { ox.push_back(i); oy.push_back(60.0f); }
    for (float i = 0; i < 60; i++) { ox.push_back(60.0f); oy.push_back(i); }
    for (float i = 0; i < 61; i++) { ox.push_back(i); oy.push_back(60.0f); }
    for (float i = 0; i < 61; i++) { ox.push_back(0.0f); oy.push_back(i); }
    for (float i = 0; i < 40; i++) { ox.push_back(20.0f); oy.push_back(i); }
    for (float i = 0; i < 40; i++) { ox.push_back(40.0f); oy.push_back(60.0f - i); }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main() {
    float sx = 10.0f, sy = 10.0f;
    float gx = 50.0f, gy = 50.0f;
    float grid_size = 1.0f, robot_size = 1.0f;

    vector<float> ox_, oy_;
    setup_obstacles(ox_, oy_);

    // Convert to grid coordinates
    vector<int> ox, oy;
    int min_ox = std::numeric_limits<int>::max();
    int max_ox = std::numeric_limits<int>::min();
    int min_oy = std::numeric_limits<int>::max();
    int max_oy = std::numeric_limits<int>::min();

    for (float iox : ox_) {
        int map_x = (int)std::round(iox / grid_size);
        ox.push_back(map_x);
        min_ox = std::min(map_x, min_ox);
        max_ox = std::max(map_x, max_ox);
    }
    for (float ioy : oy_) {
        int map_y = (int)std::round(ioy / grid_size);
        oy.push_back(map_y);
        min_oy = std::min(map_y, min_oy);
        max_oy = std::max(map_y, max_oy);
    }

    int xwidth = max_ox - min_ox;
    int ywidth = max_oy - min_oy;
    int img_reso = 5;
    int panel_w = img_reso * xwidth;   // 300
    int panel_h = img_reso * ywidth;   // 300

    // Create two background images
    cv::Mat bg_cpu(panel_w, panel_h, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat bg_cuda(panel_w, panel_h, CV_8UC3, cv::Scalar(255, 255, 255));

    int sx_grid = (int)std::round(sx / grid_size);
    int sy_grid = (int)std::round(sy / grid_size);
    int gx_grid = (int)std::round(gx / grid_size);
    int gy_grid = (int)std::round(gy / grid_size);

    // Draw start/goal on both
    for (auto* bg : {&bg_cpu, &bg_cuda}) {
        cv::rectangle(*bg,
                      cv::Point(sx_grid * img_reso + 1, sy_grid * img_reso + 1),
                      cv::Point((sx_grid + 1) * img_reso, (sy_grid + 1) * img_reso),
                      cv::Scalar(255, 0, 0), -1);
        cv::rectangle(*bg,
                      cv::Point(gx_grid * img_reso + 1, gy_grid * img_reso + 1),
                      cv::Point((gx_grid + 1) * img_reso, (gy_grid + 1) * img_reso),
                      cv::Scalar(0, 0, 255), -1);
    }

    // ========== Measure CPU obstacle map construction ==========
    auto t0_cpu = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> obmap_cpu = calc_obstacle_map_cpu(
        ox, oy, min_ox, max_ox, min_oy, max_oy,
        grid_size, robot_size, bg_cpu, img_reso);
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    double cpu_obmap_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();

    // ========== Measure CUDA obstacle map construction ==========
    // Warm up GPU
    CUDA_CHECK(cudaFree(0));

    auto t0_cuda = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> obmap_cuda = calc_obstacle_map_cuda(
        ox, oy, min_ox, max_ox, min_oy, max_oy,
        grid_size, robot_size, bg_cuda, img_reso);
    auto t1_cuda = std::chrono::high_resolution_clock::now();
    double cuda_obmap_ms = std::chrono::duration<double, std::milli>(t1_cuda - t0_cuda).count();

    printf("Obstacle Map Construction:\n");
    printf("  CPU:  %.2f ms\n", cpu_obmap_ms);
    printf("  CUDA: %.2f ms\n", cuda_obmap_ms);
    printf("  Speedup: %.1fx\n", cpu_obmap_ms / cuda_obmap_ms);

    // Create search states for both (identical A* search)
    Node* nstart_cpu = new Node(sx_grid, sy_grid, 0.0f);
    Node* ngoal_cpu = new Node(gx_grid, gy_grid, 0.0f);
    Node* nstart_cuda = new Node(sx_grid, sy_grid, 0.0f);
    Node* ngoal_cuda = new Node(gx_grid, gy_grid, 0.0f);

    SearchState cpu_search(nstart_cpu, ngoal_cpu, xwidth, ywidth,
                           min_ox, max_ox, min_oy, max_oy, obmap_cpu);
    SearchState cuda_search(nstart_cuda, ngoal_cuda, xwidth, ywidth,
                            min_ox, max_ox, min_oy, max_oy, obmap_cuda);

    // Video setup
    cv::VideoWriter video("gif/comparison_astar.avi",
                          cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
                          cv::Size(panel_h * 2, panel_w));

    int steps_per_frame = 8;  // advance multiple steps per video frame
    int frame_count = 0;

    while (!cpu_search.finished || !cuda_search.finished) {
        // Advance both searches in lockstep
        for (int s = 0; s < steps_per_frame; s++) {
            auto cpu_explored = cpu_search.step();
            for (auto& p : cpu_explored) {
                cv::rectangle(bg_cpu,
                              cv::Point(p.first * img_reso + 1, p.second * img_reso + 1),
                              cv::Point((p.first + 1) * img_reso, (p.second + 1) * img_reso),
                              cv::Scalar(0, 255, 0));
            }

            auto cuda_explored = cuda_search.step();
            for (auto& p : cuda_explored) {
                cv::rectangle(bg_cuda,
                              cv::Point(p.first * img_reso + 1, p.second * img_reso + 1),
                              cv::Point((p.first + 1) * img_reso, (p.second + 1) * img_reso),
                              cv::Scalar(0, 255, 0));
            }
        }

        // Compose frame with labels
        cv::Mat left = bg_cpu.clone();
        cv::Mat right = bg_cuda.clone();
        char buf[128];

        // CPU labels
        cv::putText(left, "CPU", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Obstacle Map: %.2f ms", cpu_obmap_ms);
        cv::putText(left, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        if (cpu_search.finished) {
            cv::putText(left, "GOAL REACHED", cv::Point(10, 62),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 180, 0), 1);
        }

        // CUDA labels
        cv::putText(right, "CUDA", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Obstacle Map: %.2f ms", cuda_obmap_ms);
        cv::putText(right, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        if (cuda_search.finished) {
            cv::putText(right, "GOAL REACHED", cv::Point(10, 62),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 180, 0), 1);
        }

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        video.write(combined);
        frame_count++;
    }

    // Draw final paths
    calc_final_path(ngoal_cpu, grid_size, bg_cpu, img_reso);
    calc_final_path(ngoal_cuda, grid_size, bg_cuda, img_reso);

    // Write final frames (hold for 2 seconds)
    {
        cv::Mat left = bg_cpu.clone();
        cv::Mat right = bg_cuda.clone();
        char buf[128];

        cv::putText(left, "CPU", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Obstacle Map: %.2f ms", cpu_obmap_ms);
        cv::putText(left, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        cv::putText(left, "GOAL REACHED", cv::Point(10, 62),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 180, 0), 1);

        cv::putText(right, "CUDA", cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        snprintf(buf, sizeof(buf), "Obstacle Map: %.2f ms", cuda_obmap_ms);
        cv::putText(right, buf, cv::Point(10, 42),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 200), 1);
        cv::putText(right, "GOAL REACHED", cv::Point(10, 62),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 180, 0), 1);

        cv::Mat combined;
        cv::hconcat(left, right, combined);
        for (int f = 0; f < 60; f++) video.write(combined);
    }

    video.release();
    std::cout << "Video saved to gif/comparison_astar.avi (" << frame_count << " frames)" << std::endl;

    // Convert to gif
    system("ffmpeg -y -i gif/comparison_astar.avi "
           "-vf 'fps=15,scale=600:-1:flags=lanczos' -loop 0 "
           "gif/comparison_astar.gif 2>/dev/null");
    std::cout << "GIF saved to gif/comparison_astar.gif" << std::endl;

    delete ngoal_cpu;
    delete nstart_cpu;
    delete ngoal_cuda;
    delete nstart_cuda;

    return 0;
}
