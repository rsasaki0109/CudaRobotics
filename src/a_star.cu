/*************************************************************************
    > File Name: a_star.cu
    > CUDA-parallelized A* path planning
    > Based on original C++ implementation by TAI Lei
    > GPU kernel parallelizes obstacle map computation:
    >   each thread handles one grid cell (i,j)
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

using namespace std;

// -------------------------------------------------------------------------
// Node (CPU-side, used by A* search)
// -------------------------------------------------------------------------
class Node {
public:
    int x;
    int y;
    float sum_cost;
    Node* p_node;

    Node(int x_, int y_, float sum_cost_ = 0, Node* p_node_ = NULL)
        : x(x_), y(y_), sum_cost(sum_cost_), p_node(p_node_) {};
};

// -------------------------------------------------------------------------
// CUDA kernel: compute obstacle map in parallel
// Each thread handles one grid cell (i, j).
// It checks the distance to every obstacle and marks the cell as blocked
// if any obstacle is within the inflated radius (vr / reso).
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
    float threshold)  // vr / reso, pre-computed on host
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
// Host wrapper: launches CUDA kernel, copies result back, draws on image
// -------------------------------------------------------------------------
std::vector<std::vector<int>> calc_obstacle_map(
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

    // Allocate device memory
    int* d_ox = nullptr;
    int* d_oy = nullptr;
    int* d_obmap = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ox, n_obstacles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_oy, n_obstacles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_obmap, xwidth * ywidth * sizeof(int)));

    // Copy obstacle coordinates to device
    CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), n_obstacles * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), n_obstacles * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with 2D grid
    dim3 blockDim(16, 16);
    dim3 gridDim((xwidth + blockDim.x - 1) / blockDim.x,
                 (ywidth + blockDim.y - 1) / blockDim.y);

    calc_obstacle_map_kernel<<<gridDim, blockDim>>>(
        d_obmap, d_ox, d_oy,
        n_obstacles, xwidth, ywidth,
        min_ox, min_oy, threshold);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host (flat array)
    std::vector<int> h_obmap_flat(xwidth * ywidth);
    CUDA_CHECK(cudaMemcpy(h_obmap_flat.data(), d_obmap,
                          xwidth * ywidth * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_ox));
    CUDA_CHECK(cudaFree(d_oy));
    CUDA_CHECK(cudaFree(d_obmap));

    // Convert flat array to 2D vector and draw obstacles on image
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
// A* helper functions (CPU, same as original)
// -------------------------------------------------------------------------
std::vector<std::vector<float>> calc_final_path(
    Node* goal, float reso, cv::Mat& img, float img_reso)
{
    std::vector<float> rx;
    std::vector<float> ry;
    Node* node = goal;
    while (node->p_node != NULL) {
        node = node->p_node;
        rx.push_back(node->x * reso);
        ry.push_back(node->y * reso);
        cv::rectangle(img,
                      cv::Point(node->x * img_reso + 1, node->y * img_reso + 1),
                      cv::Point((node->x + 1) * img_reso, (node->y + 1) * img_reso),
                      cv::Scalar(255, 0, 0), -1);
    }
    return {rx, ry};
}

bool verify_node(Node* node,
                 vector<vector<int>>& obmap,
                 int min_ox, int max_ox,
                 int min_oy, int max_oy)
{
    if (node->x < min_ox || node->y < min_oy ||
        node->x >= max_ox || node->y >= max_oy) {
        return false;
    }
    if (obmap[node->x - min_ox][node->y - min_oy])
        return false;
    return true;
}

float calc_heristic(Node* n1, Node* n2, float w = 1.0f)
{
    return w * std::sqrt((float)((n1->x - n2->x) * (n1->x - n2->x) +
                                 (n1->y - n2->y) * (n1->y - n2->y)));
}

std::vector<Node> get_motion_model()
{
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
// A* planning (CPU search loop, GPU-accelerated obstacle map)
// -------------------------------------------------------------------------
void a_star_planning(float sx, float sy,
                     float gx, float gy,
                     vector<float> ox_, vector<float> oy_,
                     float reso, float rr)
{
    Node* nstart = new Node((int)std::round(sx / reso), (int)std::round(sy / reso), 0.0f);
    Node* ngoal = new Node((int)std::round(gx / reso), (int)std::round(gy / reso), 0.0f);

    vector<int> ox;
    vector<int> oy;

    int min_ox = std::numeric_limits<int>::max();
    int max_ox = std::numeric_limits<int>::min();
    int min_oy = std::numeric_limits<int>::max();
    int max_oy = std::numeric_limits<int>::min();

    for (float iox : ox_) {
        int map_x = (int)std::round(iox * 1.0f / reso);
        ox.push_back(map_x);
        min_ox = std::min(map_x, min_ox);
        max_ox = std::max(map_x, max_ox);
    }

    for (float ioy : oy_) {
        int map_y = (int)std::round(ioy * 1.0f / reso);
        oy.push_back(map_y);
        min_oy = std::min(map_y, min_oy);
        max_oy = std::max(map_y, max_oy);
    }

    int xwidth = max_ox - min_ox;
    int ywidth = max_oy - min_oy;

    // Visualization
    cv::namedWindow("astar", cv::WINDOW_NORMAL);
    int count = 0;
    int img_reso = 5;
    cv::Mat bg(img_reso * xwidth,
               img_reso * ywidth,
               CV_8UC3,
               cv::Scalar(255, 255, 255));
    cv::VideoWriter video("gif/astar.avi", cv::VideoWriter::fourcc('X','V','I','D'), 30, cv::Size(bg.cols, bg.rows));

    cv::rectangle(bg,
                  cv::Point(nstart->x * img_reso + 1, nstart->y * img_reso + 1),
                  cv::Point((nstart->x + 1) * img_reso, (nstart->y + 1) * img_reso),
                  cv::Scalar(255, 0, 0), -1);
    cv::rectangle(bg,
                  cv::Point(ngoal->x * img_reso + 1, ngoal->y * img_reso + 1),
                  cv::Point((ngoal->x + 1) * img_reso, (ngoal->y + 1) * img_reso),
                  cv::Scalar(0, 0, 255), -1);

    std::vector<std::vector<int>> visit_map(xwidth, vector<int>(ywidth, 0));
    std::vector<std::vector<float>> path_cost(xwidth, vector<float>(ywidth, std::numeric_limits<float>::max()));

    path_cost[nstart->x][nstart->y] = 0;

    // GPU-accelerated obstacle map computation
    std::vector<std::vector<int>> obmap = calc_obstacle_map(
        ox, oy,
        min_ox, max_ox,
        min_oy, max_oy,
        reso, rr,
        bg, img_reso);

    // A* search loop (CPU, sequential with priority queue)
    auto cmp = [](const Node* left, const Node* right) {
        return left->sum_cost > right->sum_cost;
    };
    std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> pq(cmp);

    pq.push(nstart);
    std::vector<Node> motion = get_motion_model();

    while (true) {
        Node* node = pq.top();

        if (visit_map[node->x - min_ox][node->y - min_oy] == 1) {
            pq.pop();
            delete node;
            continue;
        } else {
            pq.pop();
            visit_map[node->x - min_ox][node->y - min_oy] = 1;
        }

        if (node->x == ngoal->x && node->y == ngoal->y) {
            ngoal->sum_cost = node->sum_cost;
            ngoal->p_node = node;
            break;
        }

        for (int i = 0; i < (int)motion.size(); i++) {
            Node* new_node = new Node(
                node->x + motion[i].x,
                node->y + motion[i].y,
                path_cost[node->x][node->y] + motion[i].sum_cost + calc_heristic(ngoal, node),
                node);

            if (!verify_node(new_node, obmap, min_ox, max_ox, min_oy, max_oy)) {
                delete new_node;
                continue;
            }

            if (visit_map[new_node->x - min_ox][new_node->y - min_oy]) {
                delete new_node;
                continue;
            }

            cv::rectangle(bg,
                          cv::Point(new_node->x * img_reso + 1, new_node->y * img_reso + 1),
                          cv::Point((new_node->x + 1) * img_reso, (new_node->y + 1) * img_reso),
                          cv::Scalar(0, 255, 0));

            count++;
            cv::imshow("astar", bg);
            video.write(bg);
            cv::waitKey(5);

            if (path_cost[node->x][node->y] + motion[i].sum_cost < path_cost[new_node->x][new_node->y]) {
                path_cost[new_node->x][new_node->y] = path_cost[node->x][node->y] + motion[i].sum_cost;
                pq.push(new_node);
            }
        }
    }

    calc_final_path(ngoal, reso, bg, img_reso);
    delete ngoal;
    delete nstart;

    cv::imshow("astar", bg);
    video.write(bg);
    video.release();
    std::cout << "Video saved to videos/astar.avi" << std::endl;
    cv::waitKey(0);
}

// -------------------------------------------------------------------------
// main – identical obstacle/start/goal setup as the original
// -------------------------------------------------------------------------
int main()
{
    float sx = 10.0f;
    float sy = 10.0f;
    float gx = 50.0f;
    float gy = 50.0f;

    float grid_size = 1.0f;
    float robot_size = 1.0f;

    vector<float> ox;
    vector<float> oy;

    // add edges
    for (float i = 0; i < 60; i++) {
        ox.push_back(i);
        oy.push_back(60.0f);
    }
    for (float i = 0; i < 60; i++) {
        ox.push_back(60.0f);
        oy.push_back(i);
    }
    for (float i = 0; i < 61; i++) {
        ox.push_back(i);
        oy.push_back(60.0f);
    }
    for (float i = 0; i < 61; i++) {
        ox.push_back(0.0f);
        oy.push_back(i);
    }
    for (float i = 0; i < 40; i++) {
        ox.push_back(20.0f);
        oy.push_back(i);
    }
    for (float i = 0; i < 40; i++) {
        ox.push_back(40.0f);
        oy.push_back(60.0f - i);
    }

    a_star_planning(sx, sy, gx, gy, ox, oy, grid_size, robot_size);
    return 0;
}
