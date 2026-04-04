# CudaRobotics 引き継ぎドキュメント

最終更新: 2026-04-04 JST

このファイルは、Claude あるいは別のコーディングエージェントにそのまま渡すための、現状整理と次アクションの handoff です。
前の `plan.md` はかなり古く、未実装扱いだったものがすでに大量に実装・push 済みなので、内容を全面更新しています。

---

## 0. まず結論

この repo はもう「未完成の実装集」ではなく、かなり大きく 3 本の流れに分かれています。

1. **CudaRobotics 本体**
   - 既存の CUDA robotics 実装群
   - CPU vs CUDA の多数の GIF
   - README / GitHub Pages で外向けに見せられる状態

2. **Diff-MPPI 研究ライン**
   - いま一番論文の核になりうるライン
   - dynamic navigation / uncertainty / CartPole / dynamic bicycle / planar manipulator まで follow-up あり
   - baseline gap はかなり縮まったが、まだ「paper-faithful reproduction」までは行っていない

3. **experiment-first 開発プロセス**
   - `core/` と `experiments/` を分けた「experiment -> convergence」型の開発フロー
   - docs / history / regression / convergence / next-actions / helper-promotion まで一通り回る
   - これは repo の開発方法そのものを支える基盤で、もう独立した成果物になっている

現時点での方針は次の通りです。

- **論文本文を今すぐ詰めるのは最優先ではない**
- **README と GitHub Pages には成果サマリが反映済み**
- **次に大きく効くのは、Diff-MPPI の stronger benchmark か paper-faithful baseline reproduction**
- **experiment-first 側は主要目標を達成済みで、今は improvement track**

---

## 1. 現在の Git / リポジトリ状態

- リポジトリパス: `.`
- 主要ブランチ: `master`
- 公開用ブランチ: `gh-pages`
- 現在の `master` HEAD: `4d596e5` `Summarize research results in README`
- 直前の主要 commit:
  - `16128df` `Add Diff-MPPI submission draft`
  - `3c090db` `Add manipulator feedback baseline follow-up`
  - `461c172` `Add release-weighting Diff-MPPI baseline`
  - `d121598` `Add covariance exact-time baseline for Diff-MPPI`
  - `11aac81` `Add exact-time gap-closure presets for Diff-MPPI`
- `gh-pages` 側の最新:
  - `9a3d948` `Add research results landing page`

この handoff を書いている時点で:

- `git status --short` は clean
- `master` / `gh-pages` とも push 済み

---

## 2. 今この repo で「終わっていること」

### 2.1 既存 CUDA robotics 群

古典ロボティクスの CUDA 実装群は repo の元々の中核で、README 冒頭の CPU vs CUDA 比較 GIF 群もこの系統です。

主なカテゴリ:

- Localization:
  - EKF, PF, FastSLAM, AMCL, emcl2, PFoE
- Path planning / navigation:
  - A*, Dijkstra, RRT, RRT*, DWA, Frenet, Voronoi, Potential Field, MPPI など
- Mapping / tracking / multi-agent:
  - Occupancy Grid, LQR, multi-robot planner など

この部分は「未実装」ではありません。README と GitHub Pages の既存 GIF 群で外向けに見せられる状態です。

### 2.2 研究拡張ライン

README の `Novel Research Extensions` に出ている以下は、すでに実装済みです。

- autodiff + GPU MLP 基盤
- Diff-MPPI
- Neural SDF Navigation
- GPU neuroevolution
- MiniIsaacGym
- CudaPointCloud
- Swarm optimization

### 2.3 公開面

外向けの成果サマリは、もう最低限できています。

- README:
  - `Research Results Snapshot` を追加済み
  - 論文ドラフトではなく「今 repo に何があって何が出ているか」を先に見せる形
- GitHub Pages:
  - `https://rsasaki0109.github.io/CudaRobotics/`
  - `index.html` を新設済み
  - Diff-MPPI, SDF, MiniIsaac, PointCloud, swarm の結果サマリと GIF を表示

つまり、次の担当者は「まず repo の成果を表に出す」作業から始める必要はありません。

---

## 3. 研究ラインの現状整理

### 3.1 最重要ライン: Diff-MPPI

これが現在もっとも論文主題に近いラインです。

中身は「vanilla MPPI の sampling update の後に、短い autodiff refinement を足す lightweight hybrid controller」です。

広く言って「Diff-MPPI 全体が新しい」と主張するのは厳しいですが、次の狭い主張はかなり defend しやすくなっています。

> lightweight CUDA MPPI + short autodiff refinement は、
> strong non-hybrid feedback baselines と比べても、
> hard dynamic-obstacle tasks において、
> matched-time budget 下で better compute-quality tradeoff を示す

### 3.2 Diff-MPPI で今ある benchmark 群

#### 主 benchmark

- `src/benchmark_diff_mppi.cu`
  - main dynamic navigation suite
  - fixed-budget
  - cap-based wall-clock
  - equal-time target
  - exact matched-time tuning と接続

#### follow-up 群

- `src/benchmark_diff_mppi_cartpole.cu`
  - CartPole outside-domain pilot
- `src/benchmark_diff_mppi_dynamic_bicycle.cu`
  - steering lag / drag を入れた higher-order mobile dynamics
- `src/benchmark_diff_mppi_manipulator.cu`
  - planar 2-link manipulator obstacle-avoidance pilot

### 3.3 Diff-MPPI で今ある baseline 群

現時点では baseline はかなり増えています。README の記述もこの状態に合わせて更新済みです。

main suite には少なくとも以下があります。

- `mppi`
- `feedback_mppi`
  - nominal-linearization 系の強化 baseline
- `feedback_mppi_ref`
  - released gain に寄せた current-action feedback proxy
- `feedback_mppi_release`
  - released weighting まで寄せた proxy
- `feedback_mppi_sens`
  - rollout-sensitivity 系
- `feedback_mppi_cov`
  - covariance-regression 系
- `feedback_mppi_hf`
  - low-rate replan / high-frequency feedback 実行系
- `feedback_mppi_fused`
  - heavier fused feedback baseline
- `grad_only_3`
  - hybrid の中の gradient-only ablation
- `diff_mppi_1`
- `diff_mppi_3`

重要なのは、

- baseline gap は前よりかなり狭い
- ただし **paper-faithful reproduction ではまだない**

という点です。

### 3.4 Diff-MPPI の現在の strongest talking points

README と Pages で今押している数字はこのあたりです。

#### dynamic navigation

README の現在値:

- `dynamic_slalom` matched `1.0 ms`
  - `mppi`: unsuccessful, final distance 約 `14.12`
  - `feedback_mppi_ref`: 約 `11.90`
  - `diff_mppi_3`: successful, 約 `1.95`

この task がいちばん diagnostic です。
easy task の `dynamic_crossing` は strong feedback baseline でもかなり詰められる一方、hard task の `dynamic_slalom` は hybrid がまだ一番強い、という構図になっています。

#### manipulator pilot

README の現在値:

- `arm_static_shelf`, `K=256`
  - `mppi`: `success=0.00`, final distance `0.23`
  - `feedback_mppi_ref`: `success=1.00`, `0.15`
  - `feedback_mppi_cov`: `success=1.00`, `0.15`

manipulator では「hybrid が圧勝」というより、「vanilla MPPI を強い feedback baseline が明確に上回る outside-domain pilot がある」という位置づけです。
これは reviewer の「2D kinematic nav だけ」という批判を弱める材料にはなりますが、main result の核は still dynamic navigation 側です。

### 3.5 mechanism analysis

trace ベースの mechanism 分析も入っています。

- `scripts/plot_diff_mppi_mechanism.py`
- trace 出力は `benchmark_diff_mppi` の `--trace-csv`

現在の重要ポイント:

- `dynamic_slalom @ K=1024`
- `diff_mppi_1` と `diff_mppi_3` は early-horizon 側の correction が強い
- late-horizon 側はほぼ小さい

つまり、

> autodiff stage は sampled plan 全体を大きく置き換えるのではなく、
> 実際に直近で実行される control を前寄りに sharpen している

という説明を支える材料になっています。

### 3.6 uncertainty follow-up

`paper/diff_mppi_uncertainty_followup.md` にまとまっています。

これは nominal obstacle model のまま plan して、実行側だけ obstacle の time offset / speed scale / lateral offset を seed ごとにずらす mild mismatch study です。

現状の読み方:

- `uncertain_crossing`
  - feedback baseline でもかなり回復する
- `uncertain_slalom`
  - hybrid の優位が残る

これは rebuttal や limitations 対応には効きますが、ここを main story にしすぎる必要はありません。

### 3.7 outside-domain pilots

現在の outside-domain 系:

- CartPole
- dynamic bicycle
- planar manipulator

この順に価値があります。

- CartPole:
  - 一番弱い pilot
  - 「完全に 2D nav しかない」ではない、程度
- dynamic bicycle:
  - higher-order mobile dynamics として useful
  - strong feedback baseline が efficiency competitor になる点が面白い
- planar manipulator:
  - 一番 reviewer に返しやすい outside-domain pilot
  - ただし standardized benchmark ではない

結論としては:

- outside-domain evidence は **ある**
- しかし venue-level の strongest gap を fully close はしていない

---

## 4. Diff-MPPI の文書群マップ

Diff-MPPI の文書は散らばっているので、次の担当者はまずこれを見ればよいです。

### 主な文書

- `paper/diff_mppi_results.md`
  - 初期の results draft
- `paper/diff_mppi_novelty_followup.md`
  - baseline 強化と exact-time の follow-up をまとめた主ノート
- `paper/diff_mppi_uncertainty_followup.md`
  - uncertainty follow-up
- `paper/diff_mppi_cartpole_followup.md`
  - CartPole pilot
- `paper/diff_mppi_dynamic_bicycle_followup.md`
  - dynamic bicycle pilot
- `paper/diff_mppi_manipulator_followup.md`
  - planar manipulator pilot
- `paper/icra_iros_gap_list.md`
  - venue-level の gap analysis
- `paper/diff_mppi_submission_draft.md`
  - submission 向けに主張を細くした draft

### 実務的な読み順

もし Claude が `Diff-MPPI` ラインを続けるなら、この順がよいです。

1. `readme.md`
2. `paper/diff_mppi_submission_draft.md`
3. `paper/diff_mppi_novelty_followup.md`
4. `paper/icra_iros_gap_list.md`
5. `paper/diff_mppi_manipulator_followup.md`

理由:

- README で現状の outward-facing summary を掴む
- submission draft で「今どこを主張として切っているか」を掴む
- novelty follow-up で raw evidence を把握する
- gap list で reviewer 目線の弱点を確認する
- manipulator follow-up で outside-domain の strongest pilot を確認する

---

## 5. いま論文を書かなくてよい、という前提での優先順位

ユーザー意向として、今は「まず paper を完成させる」より、成果整理・引き継ぎ・将来の continuation をしやすくすることが重要です。

その前提だと、優先度は次の順です。

### 優先度 A

- README / Pages / handoff docs の整合
- 再現コマンドと生成物の整理
- 次の担当者が迷わない状態にする

これは今回の `plan.md` 更新もその一部です。

### 優先度 B

Diff-MPPI を続けるなら:

- paper-faithful に近い baseline reproduction
- stronger public benchmark
- main figure / main table の固定

### 優先度 C

experiment-first workflow の extension

- 4つ目の problem を足す
- helper promotion を進める
- history / convergence をさらに使う

このラインは「完成済みの基盤の improvement」です。すぐやらなくてもよいです。

---

## 6. experiment-first 開発フローの現状

これはユーザーから明確に要求されて導入したプロセスで、今はかなり整っています。

### 6.1 目的

設計先行ではなく、

> experiment -> convergence

で進めること。

つまり、

- 最初から完璧な抽象を置かない
- まず concrete variants を複数作る
- 同一 interface / 同一 input / 同一 metrics で比べる
- 共通部分が見えたら最小抽象だけ残す

という方針です。

### 6.2 今ある problem 群

現在は少なくとも 3 つの concrete problem が回っています。

1. `planner_selection`
2. `time_budget_selection`
3. `fixture_promotion`

各 problem に対して、3 variants あります。

典型的には:

- functional
- oop
- pipeline

### 6.3 重要ディレクトリ

- `core/`
  - 最小 interface だけを置く
- `experiments/`
  - discardable concrete variants
- `docs/experiments.md`
  - 現在の比較結果
- `docs/decisions.md`
  - 採用 / 保留 / 不採用の理由
- `docs/interfaces.md`
  - 現在の最小 interface
- `docs/experiments_history.md`
  - snapshot 履歴
- `docs/convergence.md`
  - leader streak / convergence signal
- `docs/next_actions.md`
  - 現時点での action recommendation
- `docs/helper_promotion.md`
  - shared helper の watchlist

### 6.4 主要スクリプト

- `scripts/run_design_experiments.py`
  - comparison 実行
- `scripts/refresh_design_docs.py`
  - docs 更新
- `scripts/snapshot_design_experiments.py`
  - history snapshot 生成
- `scripts/compare_design_snapshots.py`
  - snapshot delta 比較
- `scripts/check_design_regressions.py`
  - regression guard
- `scripts/render_design_convergence.py`
  - convergence doc 生成
- `scripts/render_design_actions.py`
  - next-actions doc 生成
- `scripts/render_helper_promotion.py`
  - helper promotion watchlist 生成
- `scripts/design_doctor.py`
  - 入口 1 本にまとめた maintenance command
- `scripts/validate_design_workflow.py`
  - whole workflow validation
- `scripts/scaffold_design_problem.py`
  - 新 problem scaffold
- `scripts/check_scaffold_design_problem.py`
  - scaffold self-check

### 6.5 外部化された policy

- `experiments/history/policy.json`
  - regression policy
- `experiments/history/actions_policy.json`
  - action recommendation policy
- `experiments/history/helper_policy.json`
  - helper promotion policy
- `experiments/data/manifest.json`
  - tracked fixture set

### 6.6 現状の評価

このラインは「まだ作り途中」ではなく、主要目標を達成しています。

今の状態:

- 3 problems で回る
- docs 自動生成される
- history が残る
- regression が見られる
- convergence / next_actions / helper promotion まで出る
- design doctor で一括更新できる

つまり、

> “設計が進化し続ける場所”

という目標に対して、かなり近い状態です。

### 6.7 これ以上このラインでやるなら

優先度は下がりますが、やるとしたら:

- 4つ目の concrete problem
- helper promotion の実際の昇格
- history 可視化の強化

ただし、Diff-MPPI の論文価値に直結するのはこのラインではありません。

---

## 7. README / GitHub Pages の現状

### 7.1 README

`readme.md` は最近整理済みです。

重要箇所:

- `Research Results Snapshot`
  - 主な成果の短いまとめ
- `Diff-MPPI experiment workflow`
  - benchmark / summarize / plot / exact-time tuning / mechanism / uncertainty / outside-domain follow-up の導線
- `Experiment-First Development`
  - process 側の導線

つまり README は今、

- repo の顔
- 再現導線
- 成果サマリ

を兼ねています。

### 7.2 GitHub Pages

`gh-pages` には `index.html` を追加済みで、結果サマリ landing page があります。

URL:

- `https://rsasaki0109.github.io/CudaRobotics/`

載せている内容:

- hero summary
- Diff-MPPI の current takeaways
- manipulator / dynamic nav の数字
- Neural SDF, MiniIsaac, neuroevolution, swarm
- point-cloud speedup table

いま Pages 側で直す必要がある urgent issue はありません。

---

## 8. 今後 Claude が最初に見るべきソース / 生成物

### 8.1 Diff-MPPI のコード

- `src/benchmark_diff_mppi.cu`
- `src/benchmark_diff_mppi_manipulator.cu`
- `src/benchmark_diff_mppi_dynamic_bicycle.cu`
- `src/diff_mppi.cu`
- `src/comparison_diff_mppi.cu`

### 8.2 Diff-MPPI のスクリプト

- `scripts/summarize_diff_mppi.py`
- `scripts/plot_diff_mppi.py`
- `scripts/plot_diff_mppi_mechanism.py`
- `scripts/tune_diff_mppi_time_targets.py`

### 8.3 主な build 生成物

代表的な summary 類:

- `build/benchmark_diff_mppi_exact_time_summary.md`
- `build/benchmark_diff_mppi_exact_time_ref_summary.md`
- `build/benchmark_diff_mppi_exact_time_cov_summary.md`
- `build/benchmark_diff_mppi_exact_time_hf_summary.md`
- `build/benchmark_diff_mppi_exact_time_fused_summary.md`
- `build/benchmark_diff_mppi_manipulator_summary.md`
- `build/benchmark_diff_mppi_manipulator_exact_time_summary.md`

代表的な plot:

- `build/plots/diff_mppi_final_distance_vs_time_cap.png`
- `build/plots/diff_mppi_final_distance_vs_equal_time.png`
- `build/plots_mechanism/dynamic_slalom_correction_vs_horizon.png`

これらは README や paper notes の記述と対応しています。

### 8.4 experiment-first 側

- `scripts/design_doctor.py`
- `scripts/validate_design_workflow.py`
- `docs/experiments.md`
- `docs/experiments_history.md`
- `docs/convergence.md`
- `docs/next_actions.md`
- `docs/helper_promotion.md`

---

## 9. 現時点での「新規性」と「論文性」の判断

これは repo 内でかなり議論してきたので、判断を固定しておきます。

### 9.1 repo / artifact として

かなり強いです。

理由:

- CUDA robotics 実装の量
- GPU learning / point-cloud / swarm まで広い
- GIF / README / Pages が整っている
- experiment-first process 自体も成果物になっている

### 9.2 Diff-MPPI 論文として

**strong accept ではない**

現時点の評価は概ね:

- workshop / demo / artifact / tech report:
  - 強い
- main-track:
  - `weak accept` から `borderline accept` くらい

### 9.3 strong accept に届いていない理由

主な gap:

1. baseline はかなり近づいたが、まだ paper-faithful reproduction ではない
2. outside-domain は増えたが、public standardized benchmark / high-fidelity robotics domain ではない
3. 原理的新規性は狭く、systems / control empirical contribution として読むのが自然

### 9.4 ただし以前よりかなり良くなった点

- exact matched-time tuning がある
- strong non-hybrid baselines が多い
- mechanism analysis がある
- uncertainty がある
- outside-domain pilots が複数ある
- manipulator pilot まである

つまり、今は

> reject 級ではないが、still strong accept 級でもない

という位置です。

---

## 10. もし Claude がこの後続けるなら、何をやるべきか

### 10.1 パターン A: 論文性を本当に上げに行く場合

優先順位:

1. **paper-faithful baseline reproduction**
   - 近い proxy を増やすのではなく、closest baseline を 1 本本気で再現する
2. **stronger public benchmark**
   - 7-DOF / Isaac / MuJoCo / public manipulation benchmark 系
3. **main table / main figure 固定**
   - dynamic nav
   - one stronger domain
   - one mechanism figure

この 3 つが main-track 的には一番効きます。

### 10.2 パターン B: repo をさらに見せやすくする場合

優先順位:

1. README の表現調整
2. Pages のスクリーンショット / 図追加
3. benchmark summary から HTML を自動生成

ただし、ここは論文の acceptability を劇的には上げません。

### 10.3 パターン C: experiment-first をさらに伸ばす場合

優先順位:

1. 4つ目の concrete problem
2. helper promotion の実行
3. richer history visualization

これは process 研究としては面白いですが、Diff-MPPI 本体の venue-level gap を埋める話ではありません。

---

## 11. 今この repo で「やらなくていいこと」

次の担当者が無駄に時間を使わないために明記します。

- もう一度「全部未実装扱い」から見直すこと
  - 古い `plan.md` がそういう状態だっただけで、今は違う
- README / Pages をゼロから作り直すこと
  - もう成果サマリは入っている
- experiment-first workflow を「まだ未完成」と誤認して土台ばかり触ること
  - 今は improvement track
- baseline proxy をさらに無限に増やすこと
  - 価値は逓減している
- paper draft を無理に main task にすること
  - ユーザーは現時点でそこを最優先にしていない

---

## 12. 再現コマンド集

### 12.1 基本ビルド

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

### 12.2 Diff-MPPI main benchmark

```bash
./bin/benchmark_diff_mppi --quick
python3 scripts/summarize_diff_mppi.py --csv build/benchmark_diff_mppi.csv
python3 scripts/plot_diff_mppi.py --csv build/benchmark_diff_mppi.csv --out-dir build/plots
```

### 12.3 exact matched-time

```bash
python3 scripts/tune_diff_mppi_time_targets.py --preset dynamic_nav
```

### 12.4 uncertainty follow-up

```bash
python3 scripts/tune_diff_mppi_time_targets.py --preset uncertain_dynamic_nav
```

### 12.5 dynamic bicycle

```bash
./bin/benchmark_diff_mppi_dynamic_bicycle --csv build/benchmark_diff_mppi_dynamic_bicycle.csv
python3 scripts/tune_diff_mppi_time_targets.py --preset dynamic_bicycle
```

### 12.6 manipulator pilot

```bash
./bin/benchmark_diff_mppi_manipulator --seed-count 4 --k-values 256,512 --csv build/benchmark_diff_mppi_manipulator.csv
python3 scripts/tune_diff_mppi_time_targets.py --preset manipulator_pilot
```

### 12.7 design workflow maintenance

```bash
python3 scripts/design_doctor.py
python3 scripts/validate_design_workflow.py
python3 scripts/check_design_regressions.py
```

---

## 13. handoff 用の短い一言まとめ

Claude 向けに一言で言うと、今の repo はこうです。

> CudaRobotics はすでにかなり完成している。
> 今の主戦場は Diff-MPPI の narrow-but-defensible research line で、
> README / GitHub Pages で成果公開は済み、
> experiment-first 開発フローも基盤として完成済み。
> 次に本当に価値があるのは、
> 「もっと paper-faithful な baseline」か
> 「もっと強い benchmark」
> のどちらかを 1 本きちんと入れること。

---

## 14. もし最初の 30 分でやるなら

本当に次担当がすぐ入るなら、この順がよいです。

1. `git log --oneline -n 20`
2. `sed -n '1,260p' readme.md`
3. `sed -n '1,260p' paper/diff_mppi_submission_draft.md`
4. `sed -n '1,260p' paper/icra_iros_gap_list.md`
5. `sed -n '1,260p' paper/diff_mppi_novelty_followup.md`
6. 必要なら `./bin/benchmark_diff_mppi --quick`

これで、

- 何があるか
- 何を主張しているか
- 何が弱点か
- 次にどこを埋めるか

まで一通り把握できます。

