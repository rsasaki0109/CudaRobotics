# CudaRobotics — 開発スナップショット（アドバイザー相談用）

最終更新: 2026-05-19。直近 1 日でローカルプランナの比較フレームワークと
Hybrid A* ファミリーが揃ったので、次の方向性について意見を求めたい。

---

## 1. プロジェクト概要

- リポジトリ: <https://github.com/rsasaki0109/CudaRobotics>
- 言語: CUDA / C++14、Python (分析・可視化)
- 中心物: GPU 加速ロボティクスアルゴリズム集
  - **PythonRobotics / CppRobotics の CUDA 移植** (path planning, localization, SLAM, etc.)
  - **Diff-MPPI**: MPPI + autodiff 制御勾配補正（5 バリアント、論文ドラフトあり）
  - **その他**: Neural SDF, Neuroevolution, MiniIsaacGym, CudaPointCloud, Swarm
- ベンチマーク基盤: `bin/benchmark_diff_mppi`
  - 12 planners × 3 dynamic scenarios × 5 speed-scales × 2 radius-scales × 4 seeds の sweep を回せる
  - シナリオは自転車モデル + 静的/動的障害物 (`dynamic_crossing` / `dynamic_pincer` / `dynamic_slalom`)
  - success = `goal 到達 ∧ 衝突なし`

---

## 2. 直近の作業 (PR #4–#11)

順に 1 日でまとめて入れた。

| PR | 内容 | 状態 |
|---|---|---|
| #4 | local planner comparison report（DWA tuning + STOMP M projection 改善） | merged |
| #5 | hybrid_astar_pp (planner_kind=3): 静的 Hybrid A* + Pure Pursuit ベースライン | merged |
| #6 | hybrid_astar_dwa (kind=4): 静的 Hybrid A* + path-follow DWA local | merged |
| #7 | experiments/horizon_selection に 3 paradigm 整備（CI fix） | merged |
| #8 | hybrid_astar_dyn_pp (kind=5): 動的予測 Hybrid A* + Pure Pursuit | merged |
| #9 | hybrid_astar_mppi (kind=6): 静的 Hybrid A* + path-follow MPPI local | merged |
| #10 | README に Hybrid A* family + cross-comparison 追記 | merged |
| #11 | README を 622→355 行に短縮 | merged |

---

## 3. 現状の定量結果

12 planners × 30 cells (3 scenarios × 5 speed × 2 radius) × 4 seeds, K=4096。

### 全体（30 cells）
| planner             | family    | solved | mean coll | mean ms |
|---------------------|-----------|-------:|----------:|--------:|
| hybrid_astar_dwa    | Hybrid-A* |  30/30 |      0.00 |    0.07 |
| dwa_med / dwa_fine  | DWA       |  30/30 |      0.00 |    0.06 |
| hybrid_astar_mppi   | Hybrid-A* |  29/30 |      0.00 |    0.56 |
| dwa_fast            | DWA       |  28/30 |      0.60 |    0.06 |
| diff_mppi_3_early8  | Diff-MPPI |  23/30 |      0.56 |    0.66 |
| diff_mppi_3         | Diff-MPPI |  21/30 |      0.00 |    0.66 |
| hybrid_astar_pp     | Hybrid-A* |  21/30 |      6.23 |    0.02 |
| hybrid_astar_dyn_pp | Hybrid-A* |  20/30 |      6.37 |    0.02 |
| stomp_3_smooth      | STOMP     |  18/30 |      0.00 |    1.41 |
| stomp_2             | STOMP     |  18/30 |      0.00 |    0.95 |
| stomp_1             | STOMP     |  17/30 |      0.00 |    0.49 |

### Hard half (`dyn_speed_scale ≥ 1.5`, 12 cells)
| planner             | hard solved | mean coll |
|---------------------|------------:|----------:|
| hybrid_astar_dwa    |       12/12 |      0.00 |
| dwa_med / dwa_fine  |       12/12 |      0.00 |
| hybrid_astar_mppi   |       11/12 |      0.00 |
| dwa_fast            |       10/12 |      1.50 |
| stomp_3_smooth      |        6/12 |      0.00 |
| diff_mppi_3_early8  |        5/12 |      1.40 |
| hybrid_astar_pp     |        3/12 |     15.58 |
| hybrid_astar_dyn_pp |        2/12 |     15.92 |

### Hybrid A* マトリクスの paradigm 解釈

| 変種 | global plan | local controller | hard 結果 | 結論 |
|---|---|---|---|---|
| `pp` | 静的（dyn 無視） | pure pursuit | 3/12, 15.6 coll | naive ベースライン |
| `dyn_pp` | 動的予測 + safety inflation | pure pursuit | 2/12, 15.9 coll | brittle: search の constant-speed 仮定 vs sim の accel-from-rest による ~1s タイミングずれが致命的 |
| `dwa` | 静的（dyn 無視） | DWA argmin | 12/12, 0 coll | global path を local が dyn-aware で追従 → 完全解決 |
| `mppi` | 静的（dyn 無視） | MPPI sampling | 11/12, 0 coll | 同上、ただし sampling 平均化で goal pull が薄まるため w_terminal=50 が必要 |

**main finding**: 「global plan + local reactive controller」の hybrid pattern は paradigm-agnostic（DWA / MPPI どちらでも閉じる）。一方、global plan 単体に dyn 予測を入れるだけでは linearised prediction の timing 誤差で逆に悪化する。

---

## 4. 設計プロセス（process state）

- `core/`: 最小 interface のみ
- `experiments/`: 3 paradigm (functional / OOP / pipeline) の使い捨て variant 群
  - 現在の concrete problems: `planner_selection`, `fixture_promotion`,
    `time_budget_selection`, `horizon_selection` (各 3 variant)
- `docs/`: CSV → markdown 自動生成（experiments.md, decisions.md など）
- `paper/`: 論文ドラフト群（Diff-MPPI 主、Hybrid A* は未着手）
- スナップショット / 退行検出 / promotion watch を script で自動化

---

## 5. アーキテクチャ寸描

```
benchmark_diff_mppi (CUDA, 3000+ lines)
├── PlannerVariant (planner_kind=0..6)
│   ├── 0: legacy MPPI / Diff-MPPI / feedback
│   ├── 1: DWA grid kernel (argmin over 9 accel × 13 steer)
│   ├── 2: STOMP (cost-weighted noise + M=(RᵀR)⁻¹ projection)
│   ├── 3: hybrid_astar_pp (CPU search + pure pursuit)
│   ├── 4: hybrid_astar_dwa (CPU search + path-follow DWA kernel)
│   ├── 5: hybrid_astar_dyn_pp (dyn-aware CPU search + pure pursuit)
│   └── 6: hybrid_astar_mppi (CPU search + path-follow MPPI rollout)
└── 共通インフラ: cuRAND、d_obstacles_bench / d_dynamic_obstacles_bench
                   (__constant__ memory)
```

Hybrid A* は CPU 専用、ヘッダ単体 (`include/hybrid_astar_pp.h`, 380 行)。
- Lattice key: (ix, iy, itheta) 3D、`std::priority_queue` + `std::unordered_map`
- Motion primitives: 7 steering choices × 8 sub-steps の bicycle integration
- 1 plan 呼び出しあたり 3-27 ms（CPU）、エピソード開始時に 1 回のみ

---

## 6. 今後の候補（こちらの判断材料）

### A. GPU port of Hybrid A*
- 動機: 学習価値、ライブラリ的価値
- 課題: 現状 planning は per-episode 1 回（3-27 ms）でボトルネックではない
- 効果が活きるのは「周期 replan」の文脈
- 見積もり: 4-6 時間。Priority queue の GPU 化が難しい → batched wavefront expansion 案が有力

### B. hybrid_astar_replan_dwa（periodic replan）
- 動機: hybrid_astar_dyn_pp の brittleness（2/12）を replan で解消できるか
- 想定: 20 sim step ごとに replan → 13 replans/episode × 5ms = 65ms 追加
  - per-step avg_ms が 0.07 → 0.32（許容範囲）
- ただし `hybrid_astar_dwa` が既に 12/12 0-coll なので「悪い結果を救う」用途
  - hybrid_astar_pp の場合に replan を入れたら何 cells 上がるか、は新情報

### C. paper/hybrid_astar_paper.md ドラフト
- 動機: paradigm completion の結果は publishable
- コード変更ゼロ、docs を整理して paper/ に転載
- Diff-MPPI 論文系の流れと統合できる

### D. TEB / CHOMP / Lattice planner 追加
- 動機: 比較表をさらに厚くする
- 課題: 各 planner の実装に半日〜数日

### E. uncertainty-aware DWA
- 動機: `uncertain_*` シナリオでの DWA 改善
- 効果: マイナー、しかも uncertain_* は seed 摂動で再現性が低い

### F. ROS2 統合
- 動機: 既に `ros2_ws/` があるので、新 planner を ROS2 ノード化
- 課題: 実機 sensor とつなぐと scope が広がる

### G. Diff-MPPI 論文への取り込み
- `hybrid_astar_mppi` を Diff-MPPI ブランチに置いて「global path + Diff-MPPI local」も評価
- 投稿戦略次第

---

## 7. 相談したいこと

1. **次に取り組む価値が高いのはどれか？**（A-G、または別の方向）
   - 直近のロードマップは Diff-MPPI 系の論文化と、ローカルプランナ比較の研究的価値の両立。
   - 個人プロジェクトで GPU/ロボティクスの勉強と OSS 貢献も兼ねている。
2. **paradigm-completion の結果はどう公開すべきか？**
   - 既存 Diff-MPPI 論文に統合するか、別 paper として書くか
   - blog post としてだけ書くオプションも
3. **Hybrid A* マトリクスをさらに価値化する方法**
   - 例: 別ベンチ（Carla, TurtleBot3 sim）への移植、ROS2 統合のデモ動画
4. **dyn-aware Hybrid A* の brittleness をどう扱うか**
   - negative result としてそのまま docs に残す（現状）
   - 4D lattice / replan を入れて救う
   - そもそも追わない

ご助言ください。
