# CudaRobotics 引き継ぎドキュメント

最終更新: 2026-04-14 JST

このファイルは、Claude / Codex / 別のコーディングエージェントにそのまま渡すための handoff ドキュメントです。
前回の plan.md (2026-04-04) から大幅に進展しているため全面更新しています。

---

## 0. まず結論

このリポジトリは 3 本の流れで構成されています。

1. **CudaRobotics 本体** — 87+ CUDA ロボティクスアルゴリズム（完成済）
2. **Diff-MPPI 研究ライン** — 論文投稿準備がかなり進んだ状態
3. **experiment-first 開発プロセス** — 基盤完成済（improvement track）

**現在の最重要タスク**: Diff-MPPI 論文の ICRA/IROS 投稿準備の最終仕上げ

---

## 1. Git / リポジトリ状態

- パス: `.`
- 主要ブランチ: `master`
- 公開ブランチ: `gh-pages`
- HEAD: `f653952` `Update README and paper with 8-baseline benchmark results`
- clean 状態で push 済み

### 直近のコミット履歴（2026-04-13〜14 セッション）

```
f653952 Update README and paper with 8-baseline benchmark results
3faf840 Tune feedback_mppi_paper baseline and update paper with full results
dcc30fd Add paper-faithful Feedback-MPPI baseline and multi-param time tuning
f10ea37 Add Step-MPPI baseline results to paper
0779ad7 Add paper toolchain, CTest, Step-MPPI baseline, and Docker improvements
```

---

## 2. このセッションで完了したこと

### 2.1 論文基盤

- **LaTeX 論文**: `paper/latex/diff_mppi.tex` (IEEE RA-L 形式, 4ページ PDF)
  - IEEEtran.cls / .bst はローカルダウンロード済（.gitignore 対象）
  - 図5枚: fig_pareto, fig_mechanism, fig_7dof, fig_ablation, fig_scenarios
  - 参考文献: `references.bib` (11本)
  - ビルド: `cd paper/latex && make`

- **図表自動生成**: `scripts/generate_paper_figures.py` (715行)
  - 全5図を論文品質 PDF で出力
  - `python3 scripts/generate_paper_figures.py --csv build/benchmark_full_final.csv`

### 2.2 新ベースライン

- **step_mppi** (Step-MPPI inspired)
  - cost-weighted EMA で sampling bias をオンライン学習
  - 結果: dynamic_slalom で dist=14.25, success=0.00 — vanilla MPPI と同等
  - 「learned sampling だけでは dynamic obstacle は解けない」ことを実証

- **feedback_mppi_paper** (Paper-faithful Feedback-MPPI)
  - feedback_mode=9: covariance-regression + LQR blend, replan_stride=1
  - regularization=0.15, cov_blend=0.80, lqr_blend=0.35
  - 結果: dynamic_slalom で dist=11.74, success=0.00
  - feedback_mppi_fused(10.28) より若干悪いが、同じく fail

### 2.3 インフラ

- **CTest 統合**: `enable_testing()`, gpu/python/cpu 3ラベル
  - `ctest --label-regex python` (CI), `ctest --label-regex gpu` (ローカル)
  - 新テスト: `src/test_host_math.cpp` (CPU-only)

- **Docker 改善**: `Dockerfile` + `docker-entrypoint.sh` + `docker-compose.yml`
  - `docker compose run cudarobotics benchmark`
  - `docker compose run cudarobotics test`

- **multi-param time tuning**: `scripts/tune_diff_mppi_time_targets.py --multi-param`
  - K だけでなく feedback_gain_scale, grad_steps, alpha, mlp_lr も探索
  - CLI override flags を全5ベンチマークバイナリに追加

- **CLAUDE.md**: プロジェクト説明書（Claude Code 用）
- **CI 更新**: `.github/workflows/build.yml` を CTest ベースに移行

### 2.4 GitHub Pages

- `gh-pages` ブランチに Step-MPPI + ablation セクション追加 (`782c93a`)
- baseline count を 7 → 8 に更新

---

## 3. Diff-MPPI 最新ベンチマーク結果

### 3.1 dynamic_slalom（全 baseline 比較, K=1024, 4-seed 平均）

| プランナー | success | final_dist | avg_ms | 種別 |
|---|---|---|---|---|
| **diff_mppi_3** | **1.00** | **1.91** | **0.29** | hybrid |
| feedback_mppi_fused | 0.00 | 10.28 | 17.45 | feedback (strongest) |
| feedback_mppi_paper | 0.00 | 11.74 | 8.75 | feedback (faithful) |
| feedback_mppi_ref | 0.00 | 11.87 | 0.92 | feedback |
| step_mppi | 0.00 | 14.25 | 0.19 | learned sampling |
| mppi | 0.00 | 14.23 | 0.18 | vanilla |

### 3.2 exact matched-time @ 1.0ms

| プランナー | K (tuned) | actual_ms | success | final_dist |
|---|---|---|---|---|
| **diff_mppi_3** | 6156 | 0.99 | 1.00 | 1.91 |
| feedback_mppi_fused | 128 | 1.90 | 0.50 | 6.11 |
| feedback_mppi_paper | 128 | 1.00 | 0.50 | 6.78 |
| feedback_mppi_ref | 1067 | 1.00 | 0.50 | 6.89 |
| step_mppi | 7467 | 0.99 | 0.00 | 8.54 |
| mppi | 6834 | 1.00 | 0.00 | 8.56 |

注: success=0.50 は dynamic_crossing で成功, dynamic_slalom で失敗の平均

### 3.3 dynamic_slalom @ 1.0ms

| プランナー | K | ms | final_dist |
|---|---|---|---|
| diff_mppi_3 | 5966 | 0.988 | 1.90 |
| mppi | 7167 | 0.991 | 14.15 |
| feedback_mppi_fused | 128 | 1.882 | 10.33 |

### 3.4 7-DOF manipulator (7dof_dynamic_avoid, K=512)

| プランナー | success | final_dist | avg_ms |
|---|---|---|---|
| diff_mppi_3 | 1.00 | 0.090 | 0.84 |
| feedback_mppi_ref | 0.75 | 0.283 | 4.01 |
| mppi | 0.25 | 0.635 | 0.39 |

### 3.5 中心的主張

> dynamic_slalom で **8 つの non-hybrid baseline が全て success=0.00**（K=128〜8192 の全域で）。
> diff_mppi_3 だけが success=1.00。
> gradient refinement は sampling 改善（step_mppi）でも feedback 強化（8 variants）でも代替不可能。

---

## 4. 論文の現状

### 4.1 ファイル構成

- `paper/diff_mppi_paper.md` — Markdown 版（source of truth）
- `paper/latex/diff_mppi.tex` — LaTeX 版（IEEE RA-L 形式）
- `paper/latex/references.bib` — 参考文献 11 本
- `paper/figures/` — 図 5 枚（PDF）
- `paper/diff_mppi_submission_draft.md` — 初期 submission draft（参考用）
- `paper/icra_iros_gap_list.md` — venue 向け gap 分析
- `paper/cudarobotics_systems_paper.md` — CudaRobotics systems paper draft

### 4.2 ICRA/IROS gap list 対応状況

| Gap | 状態 | 備考 |
|---|---|---|
| Tier 1 #1: Literature-faithful baseline | **done** | feedback_mppi_paper (mode 9), step_mppi |
| Tier 1 #2: Higher-fidelity domain | **done** | 7-DOF manipulator benchmark |
| Tier 1 #3: Time-tuning protocol | **done** | exact-time + multi-param tuning |
| Tier 2 #4: Uncertainty | 既存 | uncertain_crossing/slalom follow-up |
| Tier 2 #5: Mechanism analysis | 既存 | gradient freshness analysis |
| Tier 3 #6: Hardware demo | 未着手 | — |
| Tier 3 #7: Standardized benchmark | 未着手 | MuJoCo 未導入 |

### 4.3 Minimum Submission Bar チェック

gap list の "Minimum Submission Bar" に対して:

1. ✅ Current static benchmark
2. ✅ Current two dynamic tasks (dynamic_crossing, dynamic_slalom)
3. ✅ grad_only_3 ablation
4. ✅ literature-faithful baseline（feedback_mppi_paper + step_mppi 追加）
5. ✅ exact matched-time tuning（1.0ms target 実行済）
6. ✅ one higher-fidelity experiment（7-DOF manipulator）

**6/6 全項目を満たしている。**

### 4.4 論文の推定評価

- workshop / demo / tech report: **strong**
- ICRA/IROS full paper: **borderline accept** → **weak accept** に改善
  - 8 baseline（previously 6）
  - faithful baseline 追加
  - step_mppi ablation でメカニズム解明
  - exact-time tuning 実施済み

---

## 5. 次にやるべきこと（優先順）

### 5.1 論文を本当に submit するなら

#### 優先度 A: すぐやれる

1. **論文テキスト最終推敲**
   - LaTeX の Introduction, Related Work を最新の baseline 数に合わせて最終調整
   - Limitations セクションを更新（step_mppi, feedback_mppi_paper の結果を反映）
   - Conclusion を更新

2. **main figure / main table の最終固定**
   - fig_pareto を matched-time データ（build/exact_time_1ms.csv）で再生成
   - Table I を exact-time 結果に更新

3. **Abstract の数字を最終確認**
   - "eight strong non-hybrid baselines" に合わせる
   - 7-DOF の数字を最新に

#### 優先度 B: 価値は高いが工数がかかる

4. **MuJoCo 標準ベンチマーク**
   - Codex 用プロンプトが `codex_tasks.md` Task 1 にある
   - まず `pip3 install mujoco` でインストール
   - Reacher か InvertedPendulum で 1 ドメイン追加
   - reviewer の "custom benchmark only" 批判に直接対処

5. **multi-param time tuning 完全実行**
   - `--multi-param` フラグで feedback_gain_scale, grad_steps, alpha も探索
   - 現在は K-only tuning だけ実行済み

#### 優先度 C: nice-to-have

6. **GitHub Pages の図追加**
   - fig_pareto 等の PDF → PNG 変換して gh-pages に配置
7. **systems paper (cudarobotics_systems_paper.md) の LaTeX 化**
8. **video / animation 生成** — comparison_diff_mppi で AVI 出力

---

## 6. 重要ファイルマップ

### ベンチマーク

| ファイル | 内容 |
|---|---|
| `src/benchmark_diff_mppi.cu` | 2D 動的ナビゲーション（main suite, ~2200行） |
| `src/benchmark_diff_mppi_manipulator_7dof.cu` | 7-DOF アーム |
| `src/benchmark_diff_mppi_dynamic_bicycle.cu` | dynamic bicycle |
| `src/benchmark_diff_mppi_manipulator.cu` | 2-link planar arm |
| `src/benchmark_diff_mppi_cartpole.cu` | CartPole |

### スクリプト

| ファイル | 内容 |
|---|---|
| `scripts/tune_diff_mppi_time_targets.py` | exact-time tuning（binary search） |
| `scripts/generate_paper_figures.py` | 論文図5枚一括生成 |
| `scripts/summarize_diff_mppi.py` | CSV → テキストサマリ |
| `scripts/plot_diff_mppi.py` | 基本プロット |
| `scripts/plot_diff_mppi_mechanism.py` | gradient freshness |
| `scripts/plot_pareto_frontier.py` | Pareto frontier |

### 論文

| ファイル | 内容 |
|---|---|
| `paper/latex/diff_mppi.tex` | IEEE RA-L LaTeX (4ページ) |
| `paper/latex/references.bib` | BibTeX 11本 |
| `paper/diff_mppi_paper.md` | Markdown source of truth |
| `paper/icra_iros_gap_list.md` | Gap 分析 |

### 生成物（build/）

| ファイル | 内容 |
|---|---|
| `build/benchmark_full_final.csv` | 全 baseline × 全シナリオ最新結果 |
| `build/exact_time_1ms.csv` | matched-time @ 1.0ms 結果 |
| `build/exact_time_1ms_summary.md` | 同サマリ |
| `paper/figures/fig_*.pdf` | 論文用図 |

---

## 7. ベースライン一覧

現在 benchmark_diff_mppi.cu に登録されている全 planner variant:

| 名前 | 種別 | 特徴 |
|---|---|---|
| mppi | vanilla | sampling-only |
| feedback_mppi | feedback | nominal linearization |
| feedback_mppi_ref | feedback | released current-action gain |
| feedback_mppi_release | feedback | released weighting |
| feedback_mppi_sens | feedback | rollout sensitivity |
| feedback_mppi_cov | feedback | covariance regression |
| feedback_mppi_fused | feedback | cov + LQR blend (strongest non-hybrid) |
| feedback_mppi_hf | feedback | two-rate, full-horizon cov+LQR |
| feedback_mppi_faithful | feedback | two-rate, current-action-only |
| feedback_mppi_paper | feedback | mode 9, cov+LQR, replan_stride=1 |
| step_mppi | learned | cost-weighted EMA sampling bias |
| grad_only_3 | gradient | gradient-only ablation |
| diff_mppi_1 | hybrid | MPPI + 1 grad step |
| diff_mppi_3 | hybrid | MPPI + 3 grad steps (proposed) |
| diff_mppi_adaptive | hybrid | adaptive gradient skip |

---

## 8. 再現コマンド集

### ビルド

```bash
cmake -B build && cmake --build build -j$(nproc)
```

### テスト

```bash
cd build && ctest --output-on-failure  # 全テスト
ctest --label-regex python             # Python のみ
ctest --label-regex gpu                # GPU テスト
```

### ベンチマーク（フル比較）

```bash
./bin/benchmark_diff_mppi \
  --planners mppi,feedback_mppi_fused,feedback_mppi_paper,feedback_mppi_ref,diff_mppi_3,step_mppi \
  --scenarios dynamic_slalom,dynamic_crossing \
  --k-values 256,1024,4096 --seed-count 4 \
  --csv build/benchmark_full_final.csv
```

### exact-time tuning

```bash
python3 scripts/tune_diff_mppi_time_targets.py \
  --preset dynamic_nav \
  --planners mppi,diff_mppi_3,feedback_mppi_fused,feedback_mppi_ref,step_mppi \
  --time-targets 1.0 --seed-count 4 \
  --csv-out build/exact_time_1ms.csv \
  --summary-out build/exact_time_1ms_summary.md
```

### multi-param tuning（新機能）

```bash
python3 scripts/tune_diff_mppi_time_targets.py \
  --preset dynamic_nav --multi-param \
  --time-targets 1.0 --seed-count 2
```

### 図表生成

```bash
python3 scripts/generate_paper_figures.py \
  --csv build/benchmark_full_final.csv --out-dir paper/figures/
```

### LaTeX ビルド

```bash
cd paper/latex && make
```

### Docker

```bash
docker compose build
docker compose run cudarobotics benchmark
docker compose run cudarobotics test
```

---

## 9. Codex への引き継ぎ用プロンプト集

`codex_tasks.md` に 6 タスクの Codex プロンプトがある。残りは:

- Task 1 (MuJoCo): 未実装、プロンプトあり
- Task 2 (Step-MPPI): **実装済み**
- Task 3 (LaTeX): **実装済み**
- Task 4 (CTest): **実装済み**
- Task 5 (Dockerfile): **実装済み**
- Task 6 (図表生成): **実装済み**

---

## 10. やらなくていいこと

- README / Pages をゼロから作り直すこと（成果サマリは入っている）
- baseline proxy をさらに増やすこと（8 baseline で十分）
- experiment-first workflow の大幅拡張（基盤完成済み）
- paper draft を無理に main task にすること（コード側の準備は完了）
- 古い plan.md の「未実装」扱いの項目を追いかけること（全て実装済み）

---

## 11. もし最初の 30 分でやるなら

```bash
git log --oneline -n 10
cat paper/icra_iros_gap_list.md | head -50
cat build/exact_time_1ms_summary.md
python3 scripts/generate_paper_figures.py --csv build/benchmark_full_final.csv
cd paper/latex && make && open diff_mppi.pdf
```

これで論文の全体像と最新データが把握できる。
