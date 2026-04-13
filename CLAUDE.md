# CudaRobotics

GPU加速ロボティクスアルゴリズムライブラリ（CUDA + C++）。
PythonRobotics / CppRobotics のGPU並列化拡張に加え、Diff-MPPI等の研究拡張を含む。

## ビルド

```bash
cd build && cmake .. && make -j$(nproc)
```

### 要件
- CMake >= 3.18
- CUDA Toolkit >= 12.0
- C++14 / CUDA C++14
- Eigen3, OpenCV >= 4.5
- NVIDIA GPU (compute capability >= 3.5)

### 出力先
- 実行バイナリ: `bin/`
- 静的ライブラリ: `lib/`

## プロジェクト構成

```
src/           # ソースコード（87 .cu + 14 .cpp）
include/       # ヘッダ（.h = CPU, .cuh = CUDA）
core/          # 安定インターフェース（Python dataclass契約）
experiments/   # 設計探索（Python、3問題×3バリアント）
scripts/       # 分析・可視化スクリプト（Python）
docs/          # 生成ドキュメント（experiments.md, decisions.md等）
paper/         # 研究論文ドラフト
ros2_ws/       # ROS2ワークスペース（particle_filter_node, dwa_node）
```

## コーディング規約

### CUDA (.cu)
- namespace: `cudabot`
- カーネル構成: 定数定義 → `__global__` カーネル → cuRAND初期化 → メインロールアウト → リダクション → 可視化(OpenCV) → main
- 並列パターン: 1スレッド = 1サンプル軌道 / 1パーティクル / 1候補
- 静的障害物は `__constant__` メモリに配置
- cuRAND: スレッドごとにランダムステート初期化
- Autodiff: テンプレートベースの forward-mode dual number（`.val`, `.deriv`）
- GPU MLP: フラット化重み配列によるスレッドローカル forward pass
- CUDAコンパイルオプション: `--expt-relaxed-constexpr`

### C++ (.cpp)
- CPU参照実装（CUDAバージョンとの比較用）

### Python
- frozen dataclass でインターフェース定義
- CSV駆動ベンチマーク → Markdownレポート生成

## 主要カテゴリ

- **Localization**: EKF, Particle Filter, FastSLAM, AMCL, emcl2, PFoE
- **Path Planning**: RRT/RRT*, A*/Dijkstra, DWA, MPPI, STOMP, Hybrid A*, PRM
- **Research**: Diff-MPPI（5バリアント）, Neural SDF, Neuroevolution, MiniIsaacGym, PSO/DE/CMA-ES
- **PointCloud**: Voxel/Statistical filtering, Normal estimation, GICP, RANSAC
- **Mapping**: Occupancy Grid

## テスト・ベンチマーク

```bash
# Autodiff検証
./bin/test_autodiff

# GPU MLP検証
./bin/test_gpu_mlp

# Diff-MPPIベンチマーク
./bin/benchmark_diff_mppi

# CPU vs CUDA比較
./bin/comparison_*
```

## ROS2ビルド（オプション）

```bash
cd ros2_ws
colcon build --packages-select cuda_robotics
source install/setup.bash
ros2 launch cuda_robotics cuda_pf.launch.py
```

## 設計原則

- `core/` は最小限の契約のみ。具体実装は `experiments/` に留める
- 1問題につき3バリアント（functional, OOP, pipeline）を並行開発
- プロセス状態は `docs/` に外部化（CSV → Markdown自動生成）
- 合意が明確になるまでバリアントを core に昇格しない
