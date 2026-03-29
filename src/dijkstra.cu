/*************************************************************************
    > File Name: dijkstra.cu
    > CUDA-parallelized Dijkstra's Algorithm
    > Based on original C++ implementation by TAI Lei
    > Obstacle map computation parallelized on GPU (one thread per grid cell)
    > Dijkstra search loop remains on CPU (priority queue is inherently sequential)
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

using namespace std;

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// -------------------------------------------------------------------------
// Node (CPU side only)
// -------------------------------------------------------------------------
class Node {
public:
  int x;
  int y;
  float cost;
  Node* p_node;

  Node(int x_, int y_, float cost_, Node* p_node_ = NULL)
    : x(x_), y(y_), cost(cost_), p_node(p_node_) {};
};

// -------------------------------------------------------------------------
// CUDA kernel: compute obstacle map
// Each thread handles one grid cell (i, j). It checks distance to every
// obstacle and sets obmap[i * ywidth + j] = 1 if any obstacle is within range.
// -------------------------------------------------------------------------
__global__ void calc_obstacle_map_kernel(
    int* obmap,
    const int* d_ox, const int* d_oy, int num_obs,
    int min_ox, int min_oy,
    int xwidth, int ywidth,
    float vr_over_reso)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = xwidth * ywidth;
  if (idx >= total) return;

  int i = idx / ywidth;
  int j = idx % ywidth;

  int x = i + min_ox;
  int y = j + min_oy;

  int is_obstacle = 0;
  for (int k = 0; k < num_obs; k++) {
    float dx = (float)(d_ox[k] - x);
    float dy = (float)(d_oy[k] - y);
    float d = sqrtf(dx * dx + dy * dy);
    if (d <= vr_over_reso) {
      is_obstacle = 1;
      break;
    }
  }

  obmap[i * ywidth + j] = is_obstacle;
}

// -------------------------------------------------------------------------
// calc_obstacle_map: launches CUDA kernel, copies result back to host
// Also draws obstacle cells on the OpenCV image.
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
  int total = xwidth * ywidth;
  int num_obs = (int)ox.size();
  float vr_over_reso = vr / reso;

  // Host flat obstacle map
  std::vector<int> h_obmap(total, 0);

  // Allocate device memory
  int *d_obmap, *d_ox, *d_oy;
  CUDA_CHECK(cudaMalloc(&d_obmap, total * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ox, num_obs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_oy, num_obs * sizeof(int)));

  // Copy obstacle coordinates to device
  CUDA_CHECK(cudaMemcpy(d_ox, ox.data(), num_obs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_oy, oy.data(), num_obs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_obmap, 0, total * sizeof(int)));

  // Launch kernel
  int blockSize = 256;
  int gridSize = (total + blockSize - 1) / blockSize;
  calc_obstacle_map_kernel<<<gridSize, blockSize>>>(
      d_obmap, d_ox, d_oy, num_obs,
      min_ox, min_oy, xwidth, ywidth, vr_over_reso);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_obmap.data(), d_obmap, total * sizeof(int), cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK(cudaFree(d_obmap));
  CUDA_CHECK(cudaFree(d_ox));
  CUDA_CHECK(cudaFree(d_oy));

  // Convert flat array to 2D vector and draw obstacles on image
  std::vector<std::vector<int>> obmap(xwidth, vector<int>(ywidth, 0));
  for (int i = 0; i < xwidth; i++) {
    for (int j = 0; j < ywidth; j++) {
      obmap[i][j] = h_obmap[i * ywidth + j];
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
// calc_final_path: trace back from goal to start
// -------------------------------------------------------------------------
std::vector<std::vector<float>> calc_final_path(Node* goal, float reso, cv::Mat& img, float img_reso) {
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

// -------------------------------------------------------------------------
// verify_node: bounds + obstacle check
// -------------------------------------------------------------------------
bool verify_node(Node* node,
                 vector<vector<int>>& obmap,
                 int min_ox, int max_ox,
                 int min_oy, int max_oy) {
  if (node->x < min_ox || node->y < min_oy || node->x >= max_ox || node->y >= max_oy) {
    return false;
  }
  if (obmap[node->x - min_ox][node->y - min_oy]) return false;
  return true;
}

// -------------------------------------------------------------------------
// Motion model (8-connected grid)
// -------------------------------------------------------------------------
std::vector<Node> get_motion_model() {
  return {Node(1,   0,  1),
          Node(0,   1,  1),
          Node(-1,  0,  1),
          Node(0,  -1,  1),
          Node(-1, -1,  std::sqrt(2.0f)),
          Node(-1,  1,  std::sqrt(2.0f)),
          Node(1,  -1,  std::sqrt(2.0f)),
          Node(1,   1,  std::sqrt(2.0f))};
}

// -------------------------------------------------------------------------
// Dijkstra planning (CPU search with GPU-computed obstacle map)
// -------------------------------------------------------------------------
void dijkstra_planning(float sx, float sy,
                       float gx, float gy,
                       vector<float> ox_, vector<float> oy_,
                       float reso, float rr) {

  Node* nstart = new Node((int)std::round(sx / reso), (int)std::round(sy / reso), 0.0f);
  Node* ngoal  = new Node((int)std::round(gx / reso), (int)std::round(gy / reso), 0.0f);

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

  // Visualization setup
  cv::namedWindow("dijkstra", cv::WINDOW_NORMAL);
  int count = 0;
  int img_reso = 5;
  cv::Mat bg(img_reso * xwidth,
             img_reso * ywidth,
             CV_8UC3,
             cv::Scalar(255, 255, 255));

  cv::rectangle(bg,
                cv::Point(nstart->x * img_reso + 1, nstart->y * img_reso + 1),
                cv::Point((nstart->x + 1) * img_reso, (nstart->y + 1) * img_reso),
                cv::Scalar(255, 0, 0), -1);
  cv::rectangle(bg,
                cv::Point(ngoal->x * img_reso + 1, ngoal->y * img_reso + 1),
                cv::Point((ngoal->x + 1) * img_reso, (ngoal->y + 1) * img_reso),
                cv::Scalar(0, 0, 255), -1);

  std::vector<std::vector<int>> visit_map(xwidth, vector<int>(ywidth, 0));

  // GPU-accelerated obstacle map computation
  std::vector<std::vector<int>> obmap = calc_obstacle_map(
      ox, oy,
      min_ox, max_ox,
      min_oy, max_oy,
      reso, rr,
      bg, img_reso);

  // Priority queue for Dijkstra (CPU-based, sequential)
  auto cmp = [](const Node* left, const Node* right) { return left->cost > right->cost; };
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
      ngoal->cost = node->cost;
      ngoal->p_node = node;
      break;
    }

    for (int i = 0; i < (int)motion.size(); i++) {
      Node* new_node = new Node(
          node->x + motion[i].x,
          node->y + motion[i].y,
          node->cost + motion[i].cost,
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
      cv::imshow("dijkstra", bg);
      cv::waitKey(5);

      pq.push(new_node);
    }
  }

  calc_final_path(ngoal, reso, bg, img_reso);
  delete ngoal;
  delete nstart;

  cv::imshow("dijkstra", bg);
  cv::waitKey(5);
}

// -------------------------------------------------------------------------
// main: same setup as original (obstacles, start, goal)
// -------------------------------------------------------------------------
int main() {
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

  dijkstra_planning(sx, sy, gx, gy, ox, oy, grid_size, robot_size);
  return 0;
}
