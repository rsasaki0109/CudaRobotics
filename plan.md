# CudaRobotics — Codex 引き継ぎドキュメント

## このドキュメントの目的

Codex（または他の AI コーディングエージェント）が **このファイルだけ読めば全作業を完了できる** ように、
現状・未完了タスク・実装仕様・ビルド手順・受け入れ基準を全て記載する。

---

## 1. リポジトリ概要

- **パス**: ``
- **GitHub**: `rsasaki0109/CudaRobotics`
- **言語**: CUDA C++ (C++14 / CUDA 14)
- **ビルド**: `cmake -B build && cmake --build build -j$(nproc)`
- **環境**: Ubuntu 22.04, CUDA 12.0, OpenCV 4.6, Eigen3
- **GIF 配信**: GitHub Pages (`rsasaki0109.github.io/CudaRobotics/`)、`gh-pages` ブランチ
- **現在のブランチ**: `master`（全作業は master 上で行う。ブランチ分けは不要に変更）

---

## 2. 現在の状態

### 2-A: 完成済み（ビルド確認済み、push 済み）

84 個の CMake ターゲット。以下のアルゴリズムが実装済み:

**Localization**: EKF, PF, FastSLAM, AMCL, emcl2, PFoE
**SLAM**: Graph SLAM
**Registration**: ICP, NDT, GICP (新規、未ビルド確認)
**Path Planning**: A*, Dijkstra, RRT, RRT*, Informed RRT*, RRT* RS, RRT-Connect, 3D RRT*, DWA, Frenet, State Lattice, Potential Field, 3D PF, PRM, Voronoi, Hybrid A*, STOMP, Value Iteration, MPPI
**Multi-agent**: Multi-robot Planner, ORCA, Social Force
**Mapping**: Occupancy Grid
**Path Tracking**: LQR (2種)
**Benchmark**: PF, DWA, RRT
**Comparison GIFs**: 20+ の CPU vs CUDA 比較

### 2-B: 新規作成済み（未ビルド確認、未コミット）

以下のファイルが master 上に存在するが、まだ `git add` / `git commit` / `git push` されていない:

```
# 共通基盤 (Project 0)
include/autodiff_engine.cuh    # Dual Number 自動微分エンジン
include/gpu_mlp.cuh            # GPU MLP 推論/学習エンジン
src/test_autodiff.cu           # autodiff テスト
src/test_gpu_mlp.cu            # GPU MLP テスト

# Neuroevolution (Project 2)
include/gpu_neural_net.cuh     # GPU NN (input→32→16→output)
include/gpu_environments.cuh   # Cart-Pole 環境
include/gpu_genetic.cuh        # 遺伝的操作カーネル
src/neuroevo.cu                # メイン: 4096個体を同時進化
src/comparison_neuroevo.cu     # CPU 100 vs GPU 4096 比較

# CudaPointCloud (Project 4)
include/cuda_pointcloud.cuh    # GPU 点群データ構造
src/voxel_grid_filter.cu       # ボクセルグリッドダウンサンプリング
src/statistical_filter.cu      # 統計的外れ値除去
src/normal_estimation.cu       # 法線推定 (PCA)
src/gicp.cu                    # Generalized ICP
src/ransac_plane.cu            # RANSAC 平面検出
src/benchmark_pointcloud.cu    # PCL vs GPU ベンチマーク

# Swarm Optimization (Project 5)
include/benchmark_functions.cuh # Rastrigin, Rosenbrock, Ackley, Schwefel
src/pso.cu                     # Particle Swarm Optimization (N=100K)
src/differential_evolution.cu  # DE/rand/1/bin (N=10K)
src/cma_es.cu                  # CMA-ES (lambda=4096)
src/aco_tsp.cu                 # ACO for TSP (4096蟻, 50都市)
src/comparison_swarm.cu        # 全手法比較
```

### 2-C: 未実装 (Project 1, 3, 6)

以下の3プロジェクトはまだソースコードが存在しない:

- **Project 1: Differentiable MPPI** — MPPI + 自動微分で勾配も使う制御
- **Project 3: Neural SDF Navigation** — MLP で SDF を表現し経路計画
- **Project 6: MiniIsaacGym** — 軽量 differentiable simulator

---

## 3. Codex がやるべきタスク（優先順位順）

### Task 1: 未コミットファイルのビルド確認とコミット

```bash
cd .
cmake -B build
cmake --build build -j$(nproc)
```

ビルドエラーがあれば修正する。よくあるエラー:
- `__device__` 関数内で `std::sin` → `sinf` に変更
- CMakeLists.txt のターゲット重複 → 重複を削除
- 不足している `#include` → 追加
- `--expt-relaxed-constexpr` が必要な場合 → `target_compile_options` 追加

ビルド成功後:
```bash
git add -A
git commit -m "Add P0 (autodiff+MLP), P2 (neuroevolution), P4 (pointcloud), P5 (swarm optimization)"
git push origin master
```

### Task 2: テスト実行

```bash
./bin/test_autodiff    # "ALL TESTS PASSED" を確認
./bin/test_gpu_mlp     # XOR loss<0.01, SDF loss<0.05 を確認
./bin/benchmark_pointcloud  # 速度比較テーブル出力を確認
```

テストが失敗したら修正してコミット。

### Task 3: GIF 生成

新規バイナリを実行して GIF を生成:

```bash
cd bin
for b in neuroevo comparison_neuroevo pso_cuda aco_tsp comparison_swarm; do
  timeout 120 ./$b
done
```

生成された gif/ 以下の avi を gif に変換:
```bash
cd ../gif
for f in *.avi; do
  name="${f%.avi}"
  ffmpeg -y -i "$f" -vf "fps=15,scale=400:-1" -loop 0 "${name}.gif" 2>/dev/null
done
rm -f *.avi
```

### Task 4: gh-pages 更新

```bash
git stash
git checkout gh-pages
git checkout master -- gif/*.gif
cp gif/*.gif .
git add *.gif gif/*.gif
git commit -m "Update GIFs"
git push origin gh-pages
git checkout master
git stash pop
```

### Task 5: Project 1 — Differentiable MPPI を実装

#### 概要
既存の MPPI (src/mppi.cu) を拡張し、勾配ベース最適化を追加する。
Dual Number 自動微分 (include/autodiff_engine.cuh) を使って dynamics と cost の勾配を計算。

#### 作成するファイル

**include/diff_dynamics.cuh**:
```cpp
#pragma once
#include "autodiff_engine.cuh"

namespace cudabot {

struct BicycleParams {
    float L = 2.5f;
    float max_speed = 5.0f;
    float max_steer = 0.5f;
    float dt = 0.05f;
};

// 通常 forward pass
__device__ void bicycle_step(
    float& x, float& y, float& theta, float& v,
    float accel, float steer, const BicycleParams& p)
{
    v += accel * p.dt;
    if (v > p.max_speed) v = p.max_speed;
    if (v < 0) v = 0;
    theta += v / p.L * tanf(steer) * p.dt;
    x += v * cosf(theta) * p.dt;
    y += v * sinf(theta) * p.dt;
}

// Dual Number 版（微分可能）
__device__ void bicycle_step_diff(
    Dualf& x, Dualf& y, Dualf& theta, Dualf& v,
    Dualf accel, Dualf steer, const BicycleParams& p)
{
    v = v + accel * p.dt;
    v = clamp(v, 0.0f, p.max_speed);
    theta = theta + v / Dualf::constant(p.L) * cudabot::tan(steer) * p.dt;
    x = x + v * cudabot::cos(theta) * p.dt;
    y = y + v * cudabot::sin(theta) * p.dt;
}

// ヤコビアン: 6回の forward pass で 4x6 行列を構築
__device__ void bicycle_jacobian(
    float x, float y, float theta, float v,
    float accel, float steer, const BicycleParams& p,
    float J[4][6])
{
    float inputs[6] = {x, y, theta, v, accel, steer};
    for (int col = 0; col < 6; col++) {
        Dualf dx = (col==0) ? Dualf::variable(x) : Dualf::constant(x);
        Dualf dy = (col==1) ? Dualf::variable(y) : Dualf::constant(y);
        Dualf dth = (col==2) ? Dualf::variable(theta) : Dualf::constant(theta);
        Dualf dv = (col==3) ? Dualf::variable(v) : Dualf::constant(v);
        Dualf da = (col==4) ? Dualf::variable(accel) : Dualf::constant(accel);
        Dualf ds = (col==5) ? Dualf::variable(steer) : Dualf::constant(steer);
        bicycle_step_diff(dx, dy, dth, dv, da, ds, p);
        J[0][col] = dx.deriv;
        J[1][col] = dy.deriv;
        J[2][col] = dth.deriv;
        J[3][col] = dv.deriv;
    }
}

} // namespace cudabot
```

**include/diff_cost.cuh**:
```cpp
#pragma once
#include "autodiff_engine.cuh"

namespace cudabot {

struct Obstacle { float x, y, r; };

struct CostParams {
    float goal_x = 45.0f, goal_y = 45.0f;
    float goal_weight = 5.0f;
    float control_weight = 0.1f;
    float obs_weight = 10.0f;
    float obs_influence = 5.0f;
};

// Smooth obstacle cost (微分可能、log-barrier)
__device__ Dualf obstacle_cost_diff(
    Dualf px, Dualf py,
    const Obstacle* obs, int n_obs, float influence)
{
    Dualf cost = Dualf::constant(0.0f);
    for (int i = 0; i < n_obs; i++) {
        Dualf dx = px - Dualf::constant(obs[i].x);
        Dualf dy = py - Dualf::constant(obs[i].y);
        Dualf d = cudabot::sqrt(dx * dx + dy * dy) - Dualf::constant(obs[i].r);
        // smooth barrier: if d < influence, cost += weight / d^2
        if (d.val < influence && d.val > 0.1f) {
            cost = cost + Dualf::constant(1.0f) / (d * d);
        } else if (d.val <= 0.1f) {
            cost = cost + Dualf::constant(1000.0f);
        }
    }
    return cost;
}

__device__ Dualf goal_cost_diff(Dualf px, Dualf py, float gx, float gy, float w) {
    Dualf dx = px - Dualf::constant(gx);
    Dualf dy = py - Dualf::constant(gy);
    return Dualf::constant(w) * cudabot::sqrt(dx * dx + dy * dy + Dualf::constant(0.01f));
}

__device__ Dualf control_cost_diff(Dualf a, Dualf s, float w) {
    return Dualf::constant(w) * (a * a + s * s);
}

} // namespace cudabot
```

**src/diff_mppi.cu**:

カーネル構成:
1. `mppi_rollout_kernel` — 既存 MPPI と同じ（4096サンプル並列ロールアウト）
2. `compute_gradient_kernel` — 1スレッド、T→0 方向に backward propagation
   - 各タイムステップ t で:
     - bicycle_jacobian() で ∂f/∂u_t を計算
     - stage_cost の ∂c/∂u_t を Dual Number で計算
     - chain rule: dJ/du_t = dc_t/du_t + (∂f/∂u_t)^T * dJ/dx_{t+1}
3. `compute_weights_kernel` — softmin（既存と同じ）
4. `update_controls_kernel` — 重み付き平均（既存と同じ）
5. `gradient_step_kernel` — `u[t] -= alpha * grad[t]`（T スレッド）

メインループ:
```
各ステップで:
1. mppi_rollout_kernel (4096 samples)
2. compute_weights_kernel + update_controls_kernel (MPPI 更新)
3. compute_gradient_kernel (勾配計算)
4. gradient_step_kernel (alpha=0.01)
5. apply u[0], shift horizon
```

テスト環境: 50x50m、10個の円形障害物（src/mppi.cu と同じ配置）
Start: (5, 5, 0, 0), Goal: (45, 45)

可視化: 800x800、サンプル軌道を200本表示（コスト色付き green→red）、
nominal trajectory を太い青線、勾配方向を矢印で表示（オプション）

**src/comparison_diff_mppi.cu**:
- 左: 標準 MPPI（サンプリングのみ）
- 右: Differentiable MPPI（サンプリング + 勾配）
- 各ステップのコストを右上にグラフ表示
- 800x400、gif 出力

CMakeLists.txt に追加:
```cmake
add_executable(diff_mppi src/diff_mppi.cu)
target_link_libraries(diff_mppi ${OpenCV_LIBS})

add_executable(comparison_diff_mppi src/comparison_diff_mppi.cu)
target_link_libraries(comparison_diff_mppi ${OpenCV_LIBS})
```

受け入れ基準:
- Diff-MPPI が標準 MPPI より少ないステップでゴール到達
- 勾配が数値微分と一致（相対誤差 < 5%）
- comparison gif が視覚的に差がわかる

### Task 6: Project 3 — Neural SDF Navigation を実装

#### 作成するファイル

**src/neural_sdf.cu**:
- gpu_mlp.cuh の GpuMLP を使って 2D SDF を学習
- 障害物: 円3個 + 壁2個の組み合わせ
- 学習データ: 64x64 グリッドから真の SDF を計算、10000 サンプルで学習
- MLP: 2→64→64→64→1 (ReLU, linear output)
- SGD, lr=0.001, batch=256, epochs=500
- 可視化: 左=真の SDF (heatmap)、右=学習した SDF (heatmap)
- gif 出力: gif/neural_sdf.gif

**src/sdf_potential_field.cu**:
- 学習済み Neural SDF でポテンシャルフィールドナビゲーション
- カーネル: compute_sdf_potential_kernel (100x100 グリッド、1スレッド/セル)
  - MLP forward で SDF(x,y) を評価
  - SDF 勾配を数値微分 (h=0.01)
  - attractive + repulsive potential
- Gradient descent on CPU
- gif 出力: gif/sdf_potential_field.gif

**src/sdf_mppi.cu**:
- MPPI のコスト関数に Neural SDF を組み込む
- rollout_kernel 内で MLP forward → SDF 値をコスト化
- MLP 重みは `__constant__` memory
- K=4096, T=30
- gif 出力: gif/sdf_mppi.gif

**src/comparison_sdf_nav.cu**:
- 左: 従来の circle obstacle MPPI
- 右: Neural SDF MPPI
- 複雑な形状（L字型障害物）で Neural SDF の優位性を示す
- gif 出力: gif/comparison_sdf_nav.gif

CMakeLists.txt に追加:
```cmake
add_executable(neural_sdf src/neural_sdf.cu)
target_link_libraries(neural_sdf ${OpenCV_LIBS})

add_executable(sdf_potential_field src/sdf_potential_field.cu)
target_link_libraries(sdf_potential_field ${OpenCV_LIBS})

add_executable(sdf_mppi src/sdf_mppi.cu)
target_link_libraries(sdf_mppi ${OpenCV_LIBS})

add_executable(comparison_sdf_nav src/comparison_sdf_nav.cu)
target_link_libraries(comparison_sdf_nav ${OpenCV_LIBS})
```

### Task 7: Project 6 — MiniIsaacGym を実装

#### 作成するファイル

**include/rigid_body_2d.cuh**:
```cpp
struct RigidBody2D {
    float x, y, angle;      // position + orientation
    float vx, vy, omega;    // velocity + angular velocity
    float mass, inertia;
    float radius;            // collision radius (circle approx)
};

__device__ void integrate(RigidBody2D& body, float fx, float fy, float torque, float dt) {
    body.vx += fx / body.mass * dt;
    body.vy += (fy / body.mass + gravity) * dt;
    body.omega += torque / body.inertia * dt;
    body.x += body.vx * dt;
    body.y += body.vy * dt;
    body.angle += body.omega * dt;
}
```

**include/contact_2d.cuh**:
```cpp
// 接触検出: 円-円、円-壁
// 接触力: penalty method (spring + damper)
// k_spring = 10000, k_damper = 100
__device__ void compute_contact_force(
    const RigidBody2D& a, const RigidBody2D& b,
    float& fx, float& fy);

__device__ void compute_wall_force(
    const RigidBody2D& body, float wall_y,
    float& fx, float& fy);
```

**include/parallel_env.cuh**:
```cpp
class ParallelEnv {
public:
    ParallelEnv(int n_envs, int env_type); // 0=CartPole, 1=PushBox
    ~ParallelEnv();
    void reset_all();
    void step(const float* d_actions, float* d_obs, float* d_rewards, int* d_dones);
    int obs_dim() const;
    int action_dim() const;
private:
    // device arrays for all environments' states
    float* d_states_; // [n_envs * state_dim]
    int n_envs_;
};
```

**src/mini_isaac.cu**:
- 4096 環境の Cart-Pole を同時シミュレーション
- カーネル: cartpole_step_kernel (1スレッド=1環境)
- OpenCV: 代表的な1環境の cart-pole アニメーション + 全環境の報酬ヒストグラム
- gif 出力: gif/mini_isaac.gif

**src/mini_isaac_rl.cu**:
- REINFORCE (方策勾配法) を GPU で実行
- 方策 NN: 4→32→16→1 (tanh)
- 全て GPU で完結: シミュレーション → 報酬 → 勾配 → 更新
- 学習曲線を表示（世代 vs 平均報酬）
- Cart-Pole を200世代以内に解決
- gif 出力: gif/mini_isaac_rl.gif

CMakeLists.txt に追加:
```cmake
add_executable(mini_isaac src/mini_isaac.cu)
target_link_libraries(mini_isaac ${OpenCV_LIBS})

add_executable(mini_isaac_rl src/mini_isaac_rl.cu)
target_link_libraries(mini_isaac_rl ${OpenCV_LIBS})
```

### Task 8: README 最終更新

全プロジェクトの GIF と説明を README に追加:

- Novel Research セクションを新設
- 各プロジェクトの GIF + 1-2行の説明
- GitHub Pages URL で画像参照
- ベンチマーク結果（あれば）

### Task 9: 最終 push

```bash
git add -A
git commit -m "Add 6 novel research projects: Diff-MPPI, Neuroevolution, Neural SDF, CudaPointCloud, Swarm, MiniIsaacGym"
git push origin master

# gh-pages 更新
git stash
git checkout gh-pages
git checkout master -- gif/*.gif
cp gif/*.gif .
git add -A
git commit -m "Final GIF update"
git push origin gh-pages
git checkout master
git stash pop
```

---

## 4. 共通規約（全ファイル共通）

### エラーチェックマクロ
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
```

### コーディング規約
- Eigen はデバイスコードで使用禁止（ホスト側のみ OK）
- `__device__` 内の数学関数は `sinf`, `cosf`, `sqrtf`, `expf`, `tanf` を使用
- `std::sin` 等は `__host__` 関数のみ
- cuRAND でデバイス側乱数生成
- 行列演算は全て inline 実装（2x2, 3x3, 4x4）

### 可視化規約
- OpenCV で可視化
- VideoWriter: XVID codec (`cv::VideoWriter::fourcc('X','V','I','D')`)
- 絶対パス: `gif/xxx.avi`
- ffmpeg 変換: `system("ffmpeg -y -i input.avi -vf 'fps=15,scale=400:-1' -loop 0 output.gif 2>/dev/null");`
- ウィンドウ名は各アルゴリズム名

### CMakeLists.txt 規約
- `add_executable(name src/name.cu)` + `target_link_libraries(name ${OpenCV_LIBS})`
- Eigen 使用時は `target_compile_options(name PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)`
- 重複ターゲット名がないか確認

### Git 規約
- Co-Authored-By は付けない
- コミットメッセージは英語
- force push しない

---

## 5. トラブルシューティング

### ビルドエラー: `__device__` 内で `std::sin` が使えない
→ `sinf`, `cosf` 等に置き換え

### ビルドエラー: Eigen の `__host__ __device__` 警告
→ `--expt-relaxed-constexpr` を追加。警告は無視して OK

### VideoWriter が 0 byte のファイルを生成する
→ GStreamer backend の問題。XVID codec + 絶対パスで解決
→ それでもダメなら `std::vector<cv::Mat> frames;` でフレーム収集 → ループ後に VideoWriter

### CMake "target already exists"
→ 重複した `add_executable` がないか `grep -n "add_executable(targetname" CMakeLists.txt` で確認

### gh-pages ブランチ切り替えで master のファイルが消える
→ `git stash` してから checkout。完了後 `git stash pop`
→ 新規ファイルは `git add` してから stash

### テストで数値微分と一致しない
→ h=1e-4 程度を使用。相対誤差 5% 以内なら OK

---

## 6. 完了条件

全てのタスク (1-9) が完了し、以下を満たすこと:

- [ ] `cmake --build build -j$(nproc)` がエラーなしで完了
- [ ] `test_autodiff` が "ALL TESTS PASSED" を出力
- [ ] `test_gpu_mlp` が XOR loss < 0.01 を出力
- [ ] 全新規バイナリが `timeout 5 ./binary` でクラッシュしない
- [ ] gif/ に全新規アルゴリズムの GIF が存在
- [ ] README に全プロジェクトの説明と GIF リンク
- [ ] `git push origin master` 完了
- [ ] gh-pages に GIF がアップロード済み
