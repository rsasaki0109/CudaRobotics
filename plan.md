# CudaRobotics — Novel Research & Development Plan

## このドキュメントについて

Codex や AI コーディングエージェントが**このファイルだけ読めば実装を開始できる**ように、
全てのデータ構造・関数シグネチャ・カーネル仕様・テストケース・ビルド手順を記載する。

---

## リポジトリ現状

- パス: ``
- ビルド: `cmake -B build && cmake --build build -j$(nproc)`
- CUDA 12.0, OpenCV 4.6, Eigen3, Ubuntu 22.04
- C++14 / CUDA 14
- 既存: 38+ アルゴリズム、60+ バイナリ、50+ GIF
- 全 `.cu` ファイルは `src/` 直下に flat に配置
- GIF は `gif/` に保存、GitHub Pages (`rsasaki0109.github.io/CudaRobotics/`) で配信
- CMakeLists.txt にターゲット追加でビルド
- VideoWriter は XVID codec + avi → ffmpeg で gif 変換

## 共通規約

```cpp
// 全ファイル共通のエラーチェックマクロ
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
```

- Eigen はデバイスコードで使用禁止（ホスト側のみ OK）
- `__device__` 関数内の行列演算は全て inline で実装
- cuRAND でデバイス側乱数生成
- OpenCV で可視化、`cv::VideoWriter` で avi 保存
- ffmpeg で gif 変換: `system("ffmpeg -y -i input.avi -vf 'fps=15,scale=400:-1' -loop 0 output.gif 2>/dev/null");`
- 絶対パス使用: `gif/xxx.avi`

## ブランチ戦略

```
master (安定版、現在の38+アルゴリズム)
│
├── feat/common-foundation        ← まず最初にこれ。完了後 master へ merge
│     実装: autodiff_engine.cuh, gpu_mlp.cuh
│
├── feat/differentiable-mppi      ← common-foundation merge 後に作成
├── feat/neuroevolution
├── feat/neural-sdf-nav           ← common-foundation merge 後に作成
├── feat/cuda-pointcloud
├── feat/swarm-optimization
└── feat/mini-isaac-gym           ← common-foundation merge 後に作成
```

各ブランチは master から切る。common-foundation が必要なプロジェクトは merge 後に開始。

---

## Project 0: 共通基盤 (Common Foundation)

**ブランチ**: `feat/common-foundation`
**ファイル**:
- `include/autodiff_engine.cuh`
- `include/gpu_mlp.cuh`
- `src/test_autodiff.cu` (テスト)
- `src/test_gpu_mlp.cu` (テスト)

### 0-A: Dual Number 自動微分エンジン

**ファイル**: `include/autodiff_engine.cuh`

```cpp
#pragma once
#include <cmath>
#include <cuda_runtime.h>

namespace cudabot {

// Forward-mode autodiff via dual numbers
// DualNumber.val = f(x), DualNumber.deriv = f'(x)
template <typename T = float>
struct DualNumber {
    T val;
    T deriv;

    __host__ __device__ DualNumber() : val(0), deriv(0) {}
    __host__ __device__ DualNumber(T v, T d = 0) : val(v), deriv(d) {}

    // 変数を作る: x = DualNumber(value, 1.0) で ∂/∂x を追跡
    __host__ __device__ static DualNumber variable(T v) { return DualNumber(v, T(1)); }
    __host__ __device__ static DualNumber constant(T v) { return DualNumber(v, T(0)); }
};

// 算術演算子
template <typename T>
__host__ __device__ DualNumber<T> operator+(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val + b.val, a.deriv + b.deriv};
}
template <typename T>
__host__ __device__ DualNumber<T> operator-(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val - b.val, a.deriv - b.deriv};
}
template <typename T>
__host__ __device__ DualNumber<T> operator*(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val * b.val, a.val * b.deriv + a.deriv * b.val};
}
template <typename T>
__host__ __device__ DualNumber<T> operator/(const DualNumber<T>& a, const DualNumber<T>& b) {
    return {a.val / b.val, (a.deriv * b.val - a.val * b.deriv) / (b.val * b.val)};
}

// 比較演算子（val のみで比較）
template <typename T>
__host__ __device__ bool operator<(const DualNumber<T>& a, const DualNumber<T>& b) { return a.val < b.val; }
template <typename T>
__host__ __device__ bool operator>(const DualNumber<T>& a, const DualNumber<T>& b) { return a.val > b.val; }
template <typename T>
__host__ __device__ bool operator<=(const DualNumber<T>& a, const DualNumber<T>& b) { return a.val <= b.val; }

// スカラーとの演算
template <typename T>
__host__ __device__ DualNumber<T> operator*(T s, const DualNumber<T>& a) { return {s * a.val, s * a.deriv}; }
template <typename T>
__host__ __device__ DualNumber<T> operator*(const DualNumber<T>& a, T s) { return {a.val * s, a.deriv * s}; }
template <typename T>
__host__ __device__ DualNumber<T> operator+(const DualNumber<T>& a, T s) { return {a.val + s, a.deriv}; }
template <typename T>
__host__ __device__ DualNumber<T> operator-(const DualNumber<T>& a, T s) { return {a.val - s, a.deriv}; }

// 数学関数
template <typename T>
__host__ __device__ DualNumber<T> sin(const DualNumber<T>& a) {
    return {std::sin(a.val), a.deriv * std::cos(a.val)};
}
template <typename T>
__host__ __device__ DualNumber<T> cos(const DualNumber<T>& a) {
    return {std::cos(a.val), -a.deriv * std::sin(a.val)};
}
template <typename T>
__host__ __device__ DualNumber<T> tan(const DualNumber<T>& a) {
    T c = std::cos(a.val);
    return {std::tan(a.val), a.deriv / (c * c)};
}
template <typename T>
__host__ __device__ DualNumber<T> sqrt(const DualNumber<T>& a) {
    T s = std::sqrt(a.val);
    return {s, a.deriv / (T(2) * s + T(1e-10))};
}
template <typename T>
__host__ __device__ DualNumber<T> exp(const DualNumber<T>& a) {
    T e = std::exp(a.val);
    return {e, a.deriv * e};
}
template <typename T>
__host__ __device__ DualNumber<T> log(const DualNumber<T>& a) {
    return {std::log(a.val), a.deriv / a.val};
}
template <typename T>
__host__ __device__ DualNumber<T> atan2(const DualNumber<T>& y, const DualNumber<T>& x) {
    T denom = x.val * x.val + y.val * y.val + T(1e-10);
    return {std::atan2(y.val, x.val), (x.val * y.deriv - y.val * x.deriv) / denom};
}
template <typename T>
__host__ __device__ DualNumber<T> abs(const DualNumber<T>& a) {
    return {std::abs(a.val), a.val >= 0 ? a.deriv : -a.deriv};
}
// clamp (val ベースで clamp、微分は範囲内のみ伝播)
template <typename T>
__host__ __device__ DualNumber<T> clamp(const DualNumber<T>& a, T lo, T hi) {
    if (a.val < lo) return {lo, T(0)};
    if (a.val > hi) return {hi, T(0)};
    return a;
}

using Dualf = DualNumber<float>;
using Duald = DualNumber<double>;

// ヤコビアン計算ヘルパー
// f: R^n -> R^m の関数に対して、1変数ずつ DualNumber で微分を取る
// 使い方: 各入力変数を順番に variable() にして forward pass を実行

} // namespace cudabot
```

**テストファイル**: `src/test_autodiff.cu`
- `f(x) = sin(x) * x^2` の微分を x=1.0 で検証
- 数値微分 `(f(x+h) - f(x-h)) / (2h)` との一致を確認 (h=1e-5)
- Bicycle dynamics のヤコビアン ∂f/∂x を検証
- GPU カーネル内での使用テスト
- 全テスト pass で "ALL TESTS PASSED" を出力
- CMakeLists.txt: `add_executable(test_autodiff src/test_autodiff.cu)`

### 0-B: GPU MLP 推論/学習エンジン

**ファイル**: `include/gpu_mlp.cuh`

```cpp
#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

namespace cudabot {

// 固定アーキテクチャ MLP
// Layer sizes は template parameter ではなく、コンストラクタで指定
// 重みは1次元 float 配列に flatten して GPU メモリに格納
//
// メモリレイアウト:
//   weights[]: [W0 (in*h0), b0 (h0), W1 (h0*h1), b1 (h1), ..., Wn (h_{n-1}*out), bn (out)]
//
// Forward pass: 1スレッドが1入力ベクトルに対して全レイヤーを計算
// 中間活性化は thread-local array (スタック) に格納

struct MLPConfig {
    int n_layers;          // 隠れ層の数
    int input_dim;
    int output_dim;
    int hidden_dim;        // 全隠れ層同一サイズ（簡略化）
    int total_weights;     // 全重み+バイアスの総数

    // 計算: input_dim*hidden_dim + hidden_dim  (第1層)
    //      + (n_layers-1) * (hidden_dim*hidden_dim + hidden_dim)  (中間層)
    //      + hidden_dim*output_dim + output_dim  (出力層)
};

// デバイス側 forward pass
// weights: GPU メモリ上の重み配列
// input: 入力ベクトル（thread-local）
// output: 出力ベクトル（thread-local）
// scratch: 中間バッファ（thread-local、サイズ = hidden_dim * 2）
// activation: 0=ReLU, 1=tanh, 2=sigmoid
__device__ void mlp_forward(
    const float* weights,
    const float* input, int input_dim,
    float* output, int output_dim,
    int hidden_dim, int n_layers,
    float* scratch,  // サイズ hidden_dim * 2
    int activation = 0  // 0=ReLU
);

// ホスト側 MLP 管理クラス
class GpuMLP {
public:
    GpuMLP(int input_dim, int hidden_dim, int n_layers, int output_dim);
    ~GpuMLP();

    void init_random(unsigned long long seed = 42);  // Xavier 初期化
    void load_weights(const std::vector<float>& weights);  // ホストから転送
    std::vector<float> get_weights() const;  // デバイスから取得
    float* device_weights() const { return d_weights_; }
    MLPConfig config() const { return config_; }

    // バッチ推論（N 入力を同時に推論）
    // d_input: [N x input_dim], d_output: [N x output_dim]
    void forward_batch(const float* d_input, float* d_output, int N);

    // SGD 学習ステップ
    // d_input: [batch x input_dim], d_target: [batch x output_dim]
    // loss を返す
    float train_step(const float* d_input, const float* d_target,
                     int batch_size, float lr = 0.001f);

private:
    MLPConfig config_;
    float* d_weights_;
    curandState* d_rng_;
};

} // namespace cudabot
```

**テストファイル**: `src/test_gpu_mlp.cu`
- XOR 問題の学習（2→4→1 MLP, 1000 ステップで loss < 0.01）
- SDF 学習テスト（円の SDF を MLP で近似）
- バッチ推論速度テスト（100万点を推論、時間計測）
- CMakeLists.txt: `add_executable(test_gpu_mlp src/test_gpu_mlp.cu)`

### 0-C: CMakeLists.txt 追加

```cmake
##############################
# Common Foundation Tests
##############################
add_executable(test_autodiff src/test_autodiff.cu)
add_executable(test_gpu_mlp src/test_gpu_mlp.cu)
```

### 受け入れ基準
- `test_autodiff`: "ALL TESTS PASSED" を出力
- `test_gpu_mlp`: XOR loss < 0.01, SDF loss < 0.05
- 他の既存ターゲットのビルドが壊れないこと

---

## Project 1: Differentiable MPPI

**ブランチ**: `feat/differentiable-mppi`
**前提**: common-foundation が master に merge 済み

### ファイル構成

```
src/diff_mppi.cu              # メイン: Differentiable MPPI コントローラ
src/comparison_diff_mppi.cu   # 比較: 標準MPPI vs Diff-MPPI
include/diff_dynamics.cuh     # 微分可能 Bicycle/Quadrotor dynamics
include/diff_cost.cuh         # 微分可能コスト関数
```

### diff_dynamics.cuh

```cpp
#pragma once
#include "autodiff_engine.cuh"

namespace cudabot {

// Bicycle model: state = (x, y, theta, v), control = (accel, steer)
// x' = x + v*cos(theta)*dt
// y' = y + v*sin(theta)*dt
// theta' = theta + v/L*tan(steer)*dt
// v' = v + accel*dt

struct BicycleParams {
    float L = 2.5f;       // wheelbase
    float max_speed = 5.0f;
    float max_steer = 0.5f;
    float dt = 0.05f;
};

// 通常の forward pass（float 版）
__device__ void bicycle_step(
    float& x, float& y, float& theta, float& v,
    float accel, float steer,
    const BicycleParams& p);

// 微分可能な forward pass（DualNumber 版）
// 入力の一つを DualNumber::variable() にすると、出力の .deriv にその偏微分が入る
__device__ void bicycle_step_diff(
    Dualf& x, Dualf& y, Dualf& theta, Dualf& v,
    Dualf accel, Dualf steer,
    const BicycleParams& p);

// ヤコビアン計算: ∂[x',y',theta',v']/∂[x,y,theta,v,accel,steer]
// 6回の forward pass（各入力変数を variable にして）で 4x6 ヤコビアンを構築
__device__ void bicycle_jacobian(
    float x, float y, float theta, float v,
    float accel, float steer,
    const BicycleParams& p,
    float J[4][6]  // output: 4 state dims x 6 input dims (4 state + 2 control)
);

} // namespace cudabot
```

### diff_cost.cuh

```cpp
#pragma once
#include "autodiff_engine.cuh"

namespace cudabot {

struct CostParams {
    float goal_x = 45.0f, goal_y = 45.0f;
    float goal_weight = 5.0f;
    float control_weight = 0.1f;
    float obs_weight = 10.0f;
    float obs_influence = 5.0f;  // 障害物影響半径
};

struct Obstacle { float x, y, r; };

// Smooth obstacle cost（微分可能）
// log-barrier: -log(d - r) if d < influence, else 0
// d = sqrt((x-ox)^2 + (y-oy)^2)
__device__ Dualf obstacle_cost_diff(
    Dualf px, Dualf py,
    const Obstacle* obs, int n_obs,
    float influence);

// Goal cost
__device__ Dualf goal_cost_diff(Dualf px, Dualf py, float gx, float gy, float weight);

// Control cost
__device__ Dualf control_cost_diff(Dualf accel, Dualf steer, float weight);

// Total stage cost
__device__ Dualf stage_cost_diff(
    Dualf px, Dualf py, Dualf accel, Dualf steer,
    const Obstacle* obs, int n_obs,
    const CostParams& cp);

} // namespace cudabot
```

### diff_mppi.cu — カーネル仕様

```cpp
// カーネル1: 標準 MPPI ロールアウト（既存 mppi.cu と同じ）
__global__ void mppi_rollout_kernel(
    float sx, float sy, float stheta, float sv,   // 現在状態
    const float* nominal_a, const float* nominal_s, // [T] nominal controls
    float* costs,           // [K] output costs
    float* noisy_a,         // [K*T] output noisy controls
    float* noisy_s,         // [K*T] output noisy controls
    curandState* rng,
    const Obstacle* obs, int n_obs,
    int K, int T, float dt, float lambda,
    float sigma_a, float sigma_s);

// カーネル2: 勾配計算
// nominal trajectory に沿って、各タイムステップの ∂J/∂u_t を計算
// chain rule: ∂J/∂u_t = ∂c_t/∂u_t + ∂c_{t+1}/∂x_{t+1} * ∂x_{t+1}/∂u_t + ...
// 実装: backward pass（T → 0 方向に propagate）
__global__ void compute_gradient_kernel(
    float sx, float sy, float stheta, float sv,
    const float* nominal_a, const float* nominal_s, // [T]
    float* grad_a, float* grad_s,  // [T] output gradients
    const Obstacle* obs, int n_obs,
    int T, float dt,
    const CostParams& cp, const BicycleParams& bp);
// 注意: このカーネルは1スレッドで T ステップを逆伝播する
// もしくは T スレッドで各ステップを担当し、sequential に backward
// 実装上は1スレッドで十分（T=30 は小さい）

// カーネル3: MPPI 重み計算 + 制御更新（既存と同じ）
__global__ void compute_weights_kernel(float* costs, float* weights, int K);
__global__ void update_controls_kernel(
    const float* weights, const float* noisy_a, const float* noisy_s,
    float* nominal_a, float* nominal_s, int K, int T);

// カーネル4: 勾配ステップ
// nominal_a[t] -= alpha * grad_a[t]
// nominal_s[t] -= alpha * grad_s[t]
__global__ void gradient_step_kernel(
    float* nominal_a, float* nominal_s,
    const float* grad_a, const float* grad_s,
    float alpha, int T);
```

### メインループ（diff_mppi.cu の main()）

```
各制御ステップで:
1. mppi_rollout_kernel <<<(K+255)/256, 256>>>  (4096 サンプル)
2. compute_weights_kernel <<<1, 256>>>
3. update_controls_kernel <<<1, T>>>
4. compute_gradient_kernel <<<1, 1>>>  (1スレッドで backward)
5. gradient_step_kernel <<<1, T>>>  (alpha = 0.01)
6. 最初の制御を適用、horizon をシフト
```

### テストシナリオ

- 環境: 50x50m、10個の円形障害物（mppi.cu と同じ配置）
- Start: (5, 5, 0, 0), Goal: (45, 45)
- K=4096, T=30, dt=0.05, lambda=1.0
- sigma_a=1.0, sigma_s=0.3
- gradient alpha=0.01, 各ステップで1回の勾配更新

### 比較 (comparison_diff_mppi.cu)

- 左パネル: 標準 MPPI（サンプリングのみ）
- 右パネル: Differentiable MPPI（サンプリング + 勾配）
- 表示: コストの収束グラフ（右上にオーバーレイ）
- メトリクス: ゴール到達までのステップ数、累積コスト
- 800x400 (400 per side)、gif 出力

### 可視化

- 800x800 ウィンドウ "diff_mppi"
- サンプル軌道 200本をコスト色付き（green=low, red=high）
- Nominal trajectory: 太い青線
- 勾配方向: nominal 各点から矢印（オプション）
- 障害物: 黒丸、ゴール: 青丸、ロボット: 赤丸+方向矢印

### CMakeLists.txt 追加

```cmake
add_executable(diff_mppi src/diff_mppi.cu)
target_link_libraries(diff_mppi ${OpenCV_LIBS})

add_executable(comparison_diff_mppi src/comparison_diff_mppi.cu)
target_link_libraries(comparison_diff_mppi ${OpenCV_LIBS})
```

### 受け入れ基準

- Diff-MPPI が標準 MPPI より少ないステップでゴール到達（同一環境で）
- 勾配が数値微分と一致（相対誤差 < 1%）
- ビルド成功、GIF 生成、60/60 既存テスト通過

---

## Project 2: GPU-Parallel Neuroevolution

**ブランチ**: `feat/neuroevolution`

### ファイル構成

```
src/neuroevo.cu               # メイン: Cart-Pole + Car Racing
src/comparison_neuroevo.cu    # CPU vs GPU 進化速度比較
include/gpu_neural_net.cuh    # 固定トポロジー NN (device forward pass)
include/gpu_environments.cuh  # 環境定義 (Cart-Pole, Car Racing)
include/gpu_genetic.cuh       # 遺伝的操作 (selection, crossover, mutation)
```

### gpu_neural_net.cuh — NN 仕様

```cpp
// 固定アーキテクチャ: input → 32 → 16 → output
// 活性化: tanh (隠れ層), linear (出力層)
// 重みの総数: input*32 + 32 + 32*16 + 16 + 16*output + output
//
// Cart-Pole: input=4 (x, x_dot, theta, theta_dot), output=1 (force direction)
//   重み数 = 4*32 + 32 + 32*16 + 16 + 16*1 + 1 = 128+32+512+16+16+1 = 705
//
// Car Racing: input=7 (5 lidar + v + theta_error), output=2 (accel, steer)
//   重み数 = 7*32 + 32 + 32*16 + 16 + 16*2 + 2 = 224+32+512+16+32+2 = 818

#define NN_HIDDEN1 32
#define NN_HIDDEN2 16
#define MAX_WEIGHTS 1024

// 1スレッドが1ネットワークの forward pass を実行
__device__ void nn_forward(
    const float* weights,  // この個体の重み配列
    const float* input,    // [input_dim]
    float* output,         // [output_dim]
    int input_dim, int output_dim);
```

### gpu_environments.cuh — 環境仕様

```cpp
// Cart-Pole 環境（OpenAI Gym 互換パラメータ）
struct CartPoleEnv {
    float x, x_dot, theta, theta_dot;
    int steps;
    bool done;

    __device__ void reset() {
        x = 0; x_dot = 0; theta = 0.05f; theta_dot = 0;
        steps = 0; done = false;
    }
    __device__ void step(float action) {
        // action: -1.0 or +1.0 (force direction)
        float force = action > 0 ? 10.0f : -10.0f;
        // physics: standard cart-pole equations
        // gravity=9.8, masscart=1.0, masspole=0.1, length=0.5
        // ... (Euler integration, dt=0.02)
        steps++;
        if (fabsf(x) > 2.4f || fabsf(theta) > 12.0f * M_PI / 180.0f || steps >= 200)
            done = true;
    }
    __device__ float fitness() { return (float)steps; }  // max 200
};

// Car Racing 環境
struct CarRacingEnv {
    float x, y, theta, v;
    float track_x[100], track_y[100];  // トラック中心線
    int n_track;
    int steps;
    bool done;
    float total_reward;

    __device__ void reset();
    __device__ void step(float accel, float steer);
    __device__ void get_observation(float* obs);  // 5 lidar + v + theta_error
    __device__ float fitness() { return total_reward; }
};
```

### gpu_genetic.cuh — 遺伝的操作

```cpp
// カーネル: 適応度評価
// 1スレッド = 1個体: NN forward + 環境シミュレーション全エピソード
__global__ void evaluate_fitness_kernel(
    const float* population_weights,  // [POP_SIZE * n_weights]
    float* fitness,                   // [POP_SIZE]
    curandState* rng,
    int pop_size, int n_weights,
    int input_dim, int output_dim,
    int env_type  // 0=CartPole, 1=CarRacing
);

// カーネル: トーナメント選択 + 交叉 + 突然変異
// 1スレッド = 1新個体を生成
__global__ void reproduce_kernel(
    const float* old_population,   // [POP_SIZE * n_weights]
    const float* fitness,          // [POP_SIZE]
    float* new_population,         // [POP_SIZE * n_weights]
    curandState* rng,
    int pop_size, int n_weights,
    int tournament_size,    // 5
    float crossover_rate,   // 0.8
    float mutation_rate,    // 0.1
    float mutation_sigma,   // 0.1
    int elite_count         // 10
);
```

### メインループ (neuroevo.cu)

```
POP_SIZE = 4096
N_GENERATIONS = 500

for gen in 0..N_GENERATIONS:
    evaluate_fitness_kernel <<<(POP_SIZE+255)/256, 256>>>
    cudaDeviceSynchronize()
    // ホスト側: 最良適応度を読み取り、表示
    reproduce_kernel <<<(POP_SIZE+255)/256, 256>>>
    // swap population buffers
    // OpenCV で最良個体のリプレイを表示 (毎10世代)
```

### 可視化

- 左半分: 最良個体の Cart-Pole リプレイ（台車 + 棒のアニメーション）
- 右半分: 適応度グラフ（世代 vs 平均/最良適応度）
- 800x600、gif 出力

### 受け入れ基準

- Cart-Pole: 100世代以内に適応度 200（完全解決）
- 4096 個体の1世代評価が 10ms 以内（GPU）
- CPU 版（sequential）との速度比較で 100x+ 高速化

---

## Project 3: Neural Implicit Map Navigation

**ブランチ**: `feat/neural-sdf-nav`
**前提**: common-foundation が master に merge 済み

### ファイル構成

```
src/neural_sdf.cu              # SDF学習 + 推論デモ
src/sdf_potential_field.cu     # SDF ポテンシャルフィールドナビゲーション
src/sdf_mppi.cu                # SDF + MPPI
src/comparison_sdf_nav.cu      # Grid map vs Neural SDF 比較
```

### neural_sdf.cu

```cpp
// GPU MLP で 2D SDF を学習・推論
// 入力: (x, y) → 出力: signed distance
//
// 学習データ:
//   - 障害物表面上の点 → distance = 0
//   - 障害物内部の点 → distance < 0
//   - 自由空間の点 → distance > 0
//   - 真の SDF 値は最近傍障害物表面への距離（符号付き）
//
// 障害物: 円形 + 長方形の組み合わせ（複雑な形状）
//   - 3個の円: (15,15,5), (30,10,3), (40,35,4)
//   - 2個の壁: (10,25)-(10,40), (25,20)-(40,20)
//
// 学習:
//   - 64x64 グリッドから真の SDF を計算
//   - ランダムサンプル 10000点で学習
//   - MLP: 2→64→64→64→1 (ReLU + linear output)
//   - SGD, lr=0.001, batch_size=256, epochs=500
//   - 損失: MSE(predicted_sdf, true_sdf)
//
// 可視化:
//   - 左: 真の SDF (heatmap)
//   - 右: 学習した Neural SDF (heatmap)
//   - 800x400、gif でゼロ等高線のアニメーション
```

### sdf_potential_field.cu

```cpp
// Neural SDF を使ったポテンシャルフィールドナビゲーション
//
// カーネル: compute_potential_kernel
//   各グリッドセル (100x100) で:
//   1. MLP forward pass で SDF(x,y) を評価
//   2. SDF 勾配を数値微分で計算: (SDF(x+h,y)-SDF(x-h,y))/(2h)
//   3. attractive potential = 0.5 * KP * dist_to_goal
//   4. repulsive potential = f(SDF): SDF が小さいほど大きい反発
//   5. total potential = attractive + repulsive
//
// ナビゲーション: gradient descent on potential field (CPU)
//
// 比較ポイント:
//   - 従来: 円形障害物の解析的距離計算
//   - Neural SDF: MLP で任意形状の障害物に対応
//   - Neural SDF は複雑な形状（凹形状含む）に対応可能
```

### sdf_mppi.cu

```cpp
// MPPI のコスト関数に Neural SDF を組み込む
//
// rollout_kernel の各タイムステップで:
//   1. MLP forward pass → SDF(x,y)
//   2. SDF < 0 → 衝突、大きなペナルティ
//   3. 0 < SDF < margin → smooth barrier cost
//   4. SDF > margin → コスト 0
//
// K=4096 サンプル × T=30 ステップ → 122,880 回の MLP 推論
// MLP の重みは __constant__ memory に格納
//
// 可視化: MPPI サンプル軌道 + SDF の等高線表示
```

### 受け入れ基準

- SDF 学習: 500 epochs で MSE < 0.01
- SDF ナビゲーション: ゴール到達
- SDF MPPI: 従来の circle-based MPPI と同等以上の性能
- 任意形状の障害物（L字型、コの字型）に対応できることを可視化で確認

---

## Project 4: CudaPointCloud

**ブランチ**: `feat/cuda-pointcloud`

### ファイル構成

```
include/cuda_pointcloud.cuh    # GPU 点群データ構造
src/voxel_grid_filter.cu       # ダウンサンプリング
src/statistical_filter.cu      # 外れ値除去
src/normal_estimation.cu       # 法線推定
src/gicp.cu                    # Generalized ICP
src/ransac_plane.cu            # RANSAC 平面検出
src/benchmark_pointcloud.cu    # 速度比較
```

### cuda_pointcloud.cuh — データ構造

```cpp
namespace cudabot {

struct PointXYZ { float x, y, z; };
struct PointNormal { float x, y, z, nx, ny, nz; };

class CudaPointCloud {
public:
    CudaPointCloud();
    ~CudaPointCloud();

    void upload(const std::vector<PointXYZ>& points);
    void download(std::vector<PointXYZ>& points) const;

    int size() const { return n_; }
    float* d_x() { return d_x_; }  // SoA accessors
    float* d_y() { return d_y_; }
    float* d_z() { return d_z_; }

private:
    float *d_x_, *d_y_, *d_z_;
    int n_, capacity_;
};

// Voxel Grid Filter
CudaPointCloud voxel_grid_filter(const CudaPointCloud& input, float leaf_size);

// Statistical Outlier Removal
CudaPointCloud statistical_outlier_removal(const CudaPointCloud& input, int k, float std_mul);

// Normal Estimation
void estimate_normals(const CudaPointCloud& input, float* d_nx, float* d_ny, float* d_nz, int k);

// GICP
void gicp_align(const CudaPointCloud& source, const CudaPointCloud& target,
                float* R, float* t, int max_iter = 50, float tolerance = 1e-4f);

// RANSAC plane detection
void ransac_plane(const CudaPointCloud& input, float* plane_coeffs, // [a,b,c,d]
                  float distance_threshold = 0.01f, int max_iterations = 1000);

} // namespace cudabot
```

### カーネル仕様

```cpp
// Voxel Grid: 3D ハッシュグリッドでダウンサンプリング
// 1スレッド/点: 各点のボクセルインデックスを計算、atomic add で重心を累積
__global__ void voxel_assign_kernel(
    const float* x, const float* y, const float* z, int n,
    float leaf_size, float min_x, float min_y, float min_z,
    int* voxel_ids,  // [n] output: 各点のボクセル ID
    float* voxel_sum_x, float* voxel_sum_y, float* voxel_sum_z,
    int* voxel_count,  // [n_voxels]
    int grid_x, int grid_y, int grid_z);

// Normal Estimation: k-NN → PCA
// 1スレッド/点: k 近傍を見つけ、共分散行列を計算、最小固有値の固有ベクトル = 法線
__global__ void estimate_normals_kernel(
    const float* x, const float* y, const float* z, int n,
    float* nx, float* ny, float* nz,
    int k);  // k=20 typical

// GICP: point-to-plane ICP
// 1スレッド/source点: nearest neighbor → 法線方向の距離で最適化
__global__ void gicp_find_correspondences_kernel(
    const float* src_x, const float* src_y, const float* src_z, int n_src,
    const float* tgt_x, const float* tgt_y, const float* tgt_z, int n_tgt,
    const float* tgt_nx, const float* tgt_ny, const float* tgt_nz,
    int* correspondences,  // [n_src] index into target
    float* distances);     // [n_src]

// RANSAC: 1スレッド/iteration: ランダムに3点選んで平面フィット、inlier 数をカウント
__global__ void ransac_plane_kernel(
    const float* x, const float* y, const float* z, int n,
    float dist_threshold,
    curandState* rng,
    float* plane_coeffs,  // [max_iter * 4]
    int* inlier_counts,   // [max_iter]
    int max_iter);
```

### テストデータ

- 直方体の部屋（8m x 6m x 3m）の壁面から 10000 点サンプル
- ノイズ: ガウシアン sigma=0.01m
- 外れ値: 5% のランダム点を追加
- Source 点群: 30度回転 + (1,0.5,0.2) 平行移動

### ベンチマーク

- 点群サイズ: 10K, 50K, 100K, 500K
- 各操作（VoxelGrid, Normal, ICP, RANSAC）の CPU vs GPU 速度比較
- CPU 版は PCL 風の sequential 実装

### 受け入れ基準

- GICP: 回転誤差 < 0.1度、並進誤差 < 0.01m
- RANSAC: 正しい平面を検出（inlier ratio > 90%）
- Voxel Grid: PCL と同等の出力（点数の差 < 5%）
- GPU が CPU 比 10x+ 高速

---

## Project 5: CUDA Swarm Optimization

**ブランチ**: `feat/swarm-optimization`

### ファイル構成

```
src/pso.cu                     # Particle Swarm Optimization
src/differential_evolution.cu  # Differential Evolution
src/cma_es.cu                  # CMA-ES
src/aco_tsp.cu                 # Ant Colony Optimization for TSP
src/comparison_swarm.cu        # 全手法の比較
include/benchmark_functions.cuh # Rastrigin, Rosenbrock, Ackley, Schwefel
```

### benchmark_functions.cuh

```cpp
// 全関数は __device__ で、任意次元に対応
// D = 問題の次元数

__device__ float rastrigin(const float* x, int D) {
    float sum = 10.0f * D;
    for (int i = 0; i < D; i++)
        sum += x[i]*x[i] - 10.0f * cosf(2.0f * M_PI * x[i]);
    return sum;
}  // 最適値: f(0,...,0) = 0, 範囲: [-5.12, 5.12]

__device__ float rosenbrock(const float* x, int D) {
    float sum = 0;
    for (int i = 0; i < D-1; i++)
        sum += 100.0f * powf(x[i+1]-x[i]*x[i], 2) + powf(1-x[i], 2);
    return sum;
}  // 最適値: f(1,...,1) = 0, 範囲: [-5, 10]

__device__ float ackley(const float* x, int D) {
    float sum1=0, sum2=0;
    for (int i = 0; i < D; i++) { sum1 += x[i]*x[i]; sum2 += cosf(2*M_PI*x[i]); }
    return -20*expf(-0.2f*sqrtf(sum1/D)) - expf(sum2/D) + 20 + M_E;
}  // 最適値: f(0,...,0) = 0, 範囲: [-5, 5]

__device__ float schwefel(const float* x, int D) {
    float sum = 418.9829f * D;
    for (int i = 0; i < D; i++)
        sum -= x[i] * sinf(sqrtf(fabsf(x[i])));
    return sum;
}  // 最適値: f(420.9687,...) ≈ 0, 範囲: [-500, 500]
```

### PSO カーネル仕様

```cpp
#define PSO_N 100000  // 粒子数
#define PSO_D 30      // 次元

__global__ void pso_evaluate_kernel(
    const float* positions,   // [N * D]
    float* fitness,           // [N]
    int N, int D, int func_id);

__global__ void pso_update_kernel(
    float* positions,     // [N * D]
    float* velocities,    // [N * D]
    float* p_best_pos,    // [N * D] 個体最良位置
    float* p_best_fit,    // [N] 個体最良適応度
    const float* g_best_pos,  // [D] 全体最良位置
    float w,   // 慣性重み (0.9 → 0.4 linear decay)
    float c1,  // 認知係数 (2.0)
    float c2,  // 社会係数 (2.0)
    curandState* rng,
    int N, int D);
```

### 受け入れ基準

- Rastrigin D=30: PSO が f < 1.0 に到達（1000世代以内）
- 10万粒子の評価が 1ms 以内
- 4手法の比較グラフを含む gif 生成

---

## Project 6: MiniIsaacGym (軽量 Differentiable Simulator)

**ブランチ**: `feat/mini-isaac-gym`
**前提**: common-foundation が master に merge 済み

### ファイル構成

```
src/mini_isaac.cu              # メイン: 2D 物理 + 並列環境
src/mini_isaac_rl.cu           # 方策勾配法での制御学習
src/comparison_mini_isaac.cu   # PyBullet 的 CPU sim vs GPU 並列 sim 速度比較
include/rigid_body_2d.cuh      # 2D 剛体
include/contact_2d.cuh         # 接触検出・応答
include/diff_physics_2d.cuh    # 微分可能物理（autodiff_engine 使用）
include/parallel_env.cuh       # 並列環境 API
```

### parallel_env.cuh — API 仕様

```cpp
namespace cudabot {

// 並列環境の統一 API
// 4096 環境を同時にシミュレーション

struct EnvConfig {
    int n_envs = 4096;
    float dt = 0.01f;
    float gravity = -9.81f;
};

class ParallelEnv {
public:
    ParallelEnv(EnvConfig config);
    ~ParallelEnv();

    // 全環境をリセット
    void reset_all();

    // 制御入力を適用して1ステップ進める
    // d_actions: [n_envs * action_dim] (GPU memory)
    // d_observations: [n_envs * obs_dim] (GPU memory, output)
    // d_rewards: [n_envs] (GPU memory, output)
    // d_dones: [n_envs] (GPU memory, output)
    void step(const float* d_actions, float* d_observations,
              float* d_rewards, int* d_dones);

    // 微分可能バージョン（DualNumber 使用）
    // d_observations の各要素に対する d_actions の勾配を計算
    void step_diff(const float* d_actions, float* d_observations,
                   float* d_rewards, float* d_grad_obs_wrt_actions);

    int obs_dim() const;
    int action_dim() const;
};

} // namespace cudabot
```

### 環境: Cart-Pole (物理ベース)

```cpp
// カーネル: 1スレッド = 1環境
__global__ void cartpole_step_kernel(
    float* cart_x,        // [N]
    float* cart_v,        // [N]
    float* pole_theta,    // [N]
    float* pole_omega,    // [N]
    const float* forces,  // [N] 制御入力
    float* observations,  // [N * 4] 出力
    float* rewards,       // [N] 出力
    int* dones,           // [N] 出力
    float dt, float gravity,
    int N);
```

### 環境: PushBox

```cpp
// ロボット（円）が箱（正方形）を目標位置に押すタスク
// 2D 物理: ロボット-箱の接触、箱-壁の接触
// 観測: ロボット位置、箱位置、目標位置 (8 dim)
// 行動: ロボットの力 (2 dim: fx, fy)
```

### RL 統合 (mini_isaac_rl.cu)

```cpp
// REINFORCE (方策勾配法) を GPU で実行
//
// 方策: 小さなNN (8→32→16→2, tanh)
// 重みは GPU メモリに格納
//
// 学習ループ:
// 1. reset_all()
// 2. for t in 0..200:
//      actions = policy(observations)  [GPU NN forward]
//      step(actions, observations, rewards, dones)
//      store log_probs and rewards
// 3. compute returns (discounted cumulative rewards) [GPU]
// 4. compute policy gradient: ∇J = E[∇log π(a|s) * R] [GPU]
// 5. update weights: w -= lr * ∇J
//
// 全て GPU で完結。CPU-GPU 転送はログ出力時のみ。
```

### 受け入れ基準

- 4096 環境の Cart-Pole を同時に 1ms/step 以下でシミュレーション
- REINFORCE で Cart-Pole を 200 世代以内に解決
- PushBox タスクで箱を目標位置に押せるようになる
- 微分可能版: 数値微分との一致（相対誤差 < 1%）

---

## 実装順序（依存関係考慮）

```
Phase 0: feat/common-foundation
  ├── autodiff_engine.cuh  ← Project 1, 3, 6 が依存
  └── gpu_mlp.cuh          ← Project 3 が依存
        │
        ▼ (master に merge)
        │
  ┌─────┼─────┬─────┬─────┬─────┐
  ▼     ▼     ▼     ▼     ▼     ▼
  P1    P2    P3    P4    P5    P6
  Diff  Neuro SDF   Point Swarm Mini
  MPPI  Evo   Nav   Cloud Opt   Isaac
        │                       │
        └───── P6 は P2 の環境を再利用可能 ─────┘
```

P2 (Neuroevolution) と P4 (CudaPointCloud) と P5 (Swarm) は独立。
P1, P3, P6 は common-foundation に依存。
P6 は P2 の Cart-Pole 環境を物理ベースに拡張したもの。

## 各プロジェクトの完了チェックリスト

- [ ] ソースコード (.cu + .cuh)
- [ ] CMakeLists.txt にターゲット追加
- [ ] ビルド成功（既存ターゲットも含めて）
- [ ] GIF 生成（通常版 + 比較版）
- [ ] gh-pages に GIF アップロード
- [ ] README に説明追加
- [ ] 受け入れ基準を全て満たす
- [ ] master に merge、push
