# Codex Task Prompts for CudaRobotics

各タスクを順番に Codex に投げる。1タスク完了 → 手元でビルド確認 → 次のタスクへ。

---

## Task 1: MuJoCo ベンチマーク追加

### Codex プロンプト

```
You are working on the CudaRobotics project — a GPU-accelerated C++/CUDA robotics algorithm library.
The project has a Diff-MPPI research line that benchmarks a hybrid MPPI controller against various baselines.

Current benchmark files follow a consistent pattern:
- src/benchmark_diff_mppi.cu (2D dynamic navigation, ~2141 lines)
- src/benchmark_diff_mppi_manipulator_7dof.cu (7-DOF arm, ~1310 lines)

The project's biggest paper weakness is "no standard benchmark" — all environments are custom.

YOUR TASK: Add a MuJoCo-based benchmark for Diff-MPPI using the MuJoCo C API.

### Requirements

1. Create `src/benchmark_diff_mppi_mujoco.cu`:
   - Use MuJoCo C API (mujoco/mujoco.h) for dynamics simulation
   - Target domain: `InvertedPendulum-v4` or `Reacher-v4` (simple enough for MPPI)
   - The MPPI rollouts should run on GPU (CUDA kernels for parallel cost evaluation)
   - MuJoCo stepping can run on CPU (host-side) since MuJoCo is CPU-only
   - Architecture: GPU parallel cost evaluation + CPU MuJoCo dynamics stepping
   - Include at minimum these planner variants: mppi, feedback_mppi_ref, diff_mppi_3
   - CSV output format must match existing benchmarks:
     scenario,planner,k_samples,seed,success,final_distance,avg_control_ms,steps

2. Update `CMakeLists.txt`:
   - Add conditional MuJoCo support:
     ```cmake
     find_package(mujoco QUIET)
     if(mujoco_FOUND)
       add_executable(benchmark_diff_mppi_mujoco src/benchmark_diff_mppi_mujoco.cu)
       target_link_libraries(benchmark_diff_mppi_mujoco ${OpenCV_LIBS} mujoco::mujoco)
       target_compile_options(benchmark_diff_mppi_mujoco PRIVATE
         $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
     endif()
     ```
   - Place this in a new "# MuJoCo Benchmarks" section at the end

3. Add `mujoco_models/` directory with the XML model file for the chosen task
   (InvertedPendulum or Reacher), based on the standard Gymnasium/MuJoCo XML.

### Architecture Pattern

Follow the existing benchmark pattern in benchmark_diff_mppi_manipulator_7dof.cu:
- Struct-based scenario/planner variant configuration
- __constant__ memory for obstacle parameters
- cuRAND per-thread RNG initialization
- Wall-clock timing with chrono::high_resolution_clock
- CSV writing per-episode
- namespace cudabot

### Key Constraint
MuJoCo is CPU-only. The integration pattern should be:
- MPPI sampling: GPU kernels evaluate cost over K parallel rollouts
- Dynamics: CPU-side mj_step() calls (or a simplified analytical model on GPU
  that approximates the MuJoCo dynamics for rollout, with MuJoCo used for
  ground-truth stepping)
- This hybrid CPU/GPU pattern is acceptable — document it clearly in comments

### Do NOT:
- Modify any existing benchmark files
- Add MuJoCo as a hard dependency (keep it optional via find_package QUIET)
- Change the project's C++ or CUDA standard
```

### 手元確認
```bash
# MuJoCo インストール（未導入の場合）
pip3 install mujoco
# または C ライブラリ直接:
# wget https://github.com/google-deepmind/mujoco/releases/download/3.1.3/mujoco-3.1.3-linux-x86_64.tar.gz

# ビルド確認
cd build && cmake .. && make -j$(nproc) benchmark_diff_mppi_mujoco
```

---

## Task 2: Step-MPPI ベースライン実装

### Codex プロンプト

```
You are working on the CudaRobotics project's Diff-MPPI benchmark suite.
The main benchmark file is src/benchmark_diff_mppi.cu (~2141 lines).

It already contains these planner variants (defined as PlannerVariant structs):
- mppi (vanilla MPPI)
- feedback_mppi, feedback_mppi_ref, feedback_mppi_sens, feedback_mppi_cov,
  feedback_mppi_fused, feedback_mppi_hf, feedback_mppi_release
- grad_only_3, diff_mppi_1, diff_mppi_3

The paper's gap list says the biggest weakness is lacking a "literature-faithful"
baseline. Step-MPPI (arxiv 2604.01539, 2026) is the most relevant recent work:
it learns a neural sampling distribution to achieve multi-step foresight.

YOUR TASK: Add a simplified Step-MPPI baseline to the existing benchmark.

### Requirements

1. In `src/benchmark_diff_mppi.cu`, add a new PlannerVariant `step_mppi`:
   - Core idea: instead of sampling from a fixed Gaussian around the nominal,
     use a small MLP (from include/gpu_mlp.cuh) to predict a better sampling
     distribution conditioned on the current state
   - The MLP takes current state [x, y, theta, v] as input and outputs
     per-timestep mean shifts for the control distribution
   - Online adaptation: after each MPPI update, do one gradient step on the MLP
     to minimize the weighted cost of the sampled trajectories
   - This is a simplified version — the full Step-MPPI uses offline RL pretraining,
     but for a fair baseline comparison, online-only adaptation is sufficient

2. Add necessary fields to PlannerVariant struct:
   - bool use_learned_sampling = false
   - int mlp_hidden_size = 32
   - float mlp_lr = 0.001f

3. Integration points:
   - Before the MPPI sampling kernel, if use_learned_sampling is true,
     run the MLP forward pass to get per-timestep mean shifts
   - Add these shifts to the nominal control before adding noise
   - After the MPPI weighted update, compute a simple policy gradient step
     using the trajectory costs as signal
   - The MLP weights live in GPU global memory, initialized once per episode

4. Register the variant in the default variant list alongside existing planners.

### Key Constraints
- Use the existing include/gpu_mlp.cuh for the MLP (it already supports
  GPU forward/backward passes with flattened weight arrays)
- Keep the same CSV output format — no new columns
- The step_mppi variant must participate in the same exact-time comparison
  framework as all other variants
- Do NOT modify any other planner variant's behavior

### Pattern Reference
Look at how feedback_mppi_ref is implemented:
- It adds feedback gains AFTER the sampling step
- step_mppi should add learned bias BEFORE the sampling step
- Both are single-pass modifications to the control pipeline
```

### 手元確認
```bash
cd build && cmake .. && make -j$(nproc) benchmark_diff_mppi
./bin/benchmark_diff_mppi --planners step_mppi --scenarios dynamic_slalom --k-values 256,512 --seed-count 2
```

---

## Task 3: 論文 LaTeX 化

### Codex プロンプト

```
You are working on the CudaRobotics project. The file paper/diff_mppi_paper.md
contains a complete Diff-MPPI research paper draft in IEEE RA-L format (152 lines
of markdown). It has: Abstract, Introduction, Related Work, Method, Experimental
Setup, Results, Discussion, and References.

YOUR TASK: Convert this paper to IEEE RA-L LaTeX format.

### Requirements

1. Create directory `paper/latex/`

2. Create `paper/latex/diff_mppi.tex`:
   - Use IEEEtran document class: \documentclass[journal]{IEEEtran}
   - Convert ALL content from paper/diff_mppi_paper.md faithfully
   - Section numbering: \section{Introduction}, \section{Related Work}, etc.
   - Math: convert inline formulas to $...$ and display formulas to \begin{equation}
   - Tables: use \begin{table} with \begin{tabular} for all data tables
   - Figures: use \begin{figure} with \includegraphics placeholders pointing to
     ../figures/ directory (these will be generated separately)
   - Algorithm block: use \begin{algorithm} with algorithmic package for the
     controller pseudocode in Section III
   - References: create paper/latex/references.bib with all cited works
     (extract from the markdown [1]-[N] citations and create proper BibTeX entries)

3. Create `paper/latex/Makefile`:
   ```makefile
   all: diff_mppi.pdf

   diff_mppi.pdf: diff_mppi.tex references.bib
   	pdflatex diff_mppi
   	bibtex diff_mppi
   	pdflatex diff_mppi
   	pdflatex diff_mppi

   clean:
   	rm -f *.aux *.bbl *.blg *.log *.out *.pdf
   ```

4. Figure placeholders to include (with \includegraphics):
   - fig:architecture — controller block diagram
   - fig:pareto — Pareto frontier (time vs final_distance)
   - fig:mechanism — gradient correction vs horizon
   - fig:7dof — 7-DOF manipulation results
   - fig:scenarios — dynamic_crossing and dynamic_slalom visualizations

5. Create `paper/latex/.gitignore`:
   ```
   *.aux
   *.bbl
   *.blg
   *.log
   *.out
   *.pdf
   *.synctex.gz
   ```

### Key Constraints
- Faithful conversion — do not rewrite or add content
- All 11+ references in the markdown must appear in references.bib
- Use IEEEtran.bst for bibliography style
- Paper should compile with: pdflatex + bibtex + pdflatex + pdflatex
- Do NOT include IEEEtran.cls itself (it's in standard TeX distributions)
```

### 手元確認
```bash
cd paper/latex && make
# PDF が生成されることを確認
```

---

## Task 4: CTest 統合

### Codex プロンプト

```
You are working on the CudaRobotics project. The build system is CMake 3.18+
with CUDA and C++ support. The main CMakeLists.txt is at the project root.

Current test binaries exist but are NOT integrated with CTest:
- test_autodiff (validates forward-mode dual-number derivatives)
- test_gpu_mlp (MLP forward/backward pass validation)

Python validation scripts exist:
- scripts/validate_design_workflow.py
- scripts/check_design_regressions.py
- scripts/check_scaffold_design_problem.py

CI exists at .github/workflows/build.yml but only runs build + Python scripts.

YOUR TASK: Integrate CTest and add test infrastructure.

### Requirements

1. Update `CMakeLists.txt`:
   - Add `enable_testing()` after the project() declaration
   - Add CTest registration for existing test binaries:
     ```cmake
     add_test(NAME test_autodiff COMMAND test_autodiff)
     add_test(NAME test_gpu_mlp COMMAND test_gpu_mlp)
     ```
   - Set test properties:
     ```cmake
     set_tests_properties(test_autodiff test_gpu_mlp PROPERTIES
       TIMEOUT 60
       LABELS "gpu")
     ```
   - Add Python tests:
     ```cmake
     add_test(NAME validate_design_workflow
       COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/validate_design_workflow.py)
     add_test(NAME check_design_regressions
       COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/check_design_regressions.py)
     add_test(NAME check_scaffold
       COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/check_scaffold_design_problem.py)
     set_tests_properties(validate_design_workflow check_design_regressions check_scaffold
       PROPERTIES LABELS "python")
     ```

2. Update `.github/workflows/build.yml`:
   - Replace the three separate Python script steps with a single CTest step:
     ```yaml
     - name: Run tests
       run: |
         cd build
         ctest --output-on-failure --label-regex python -j$(nproc)
     ```
   - Note: GPU tests (label "gpu") cannot run in CI without GPU,
     so only run "python" labeled tests
   - Add a comment explaining this

3. Create `src/test_host_math.cpp`:
   - A lightweight CPU-only test that validates host-side utility functions
   - Test csv_reader.h: parse a small inline CSV string
   - Test cpprobotics_types.h: basic type constructions
   - Return 0 on success, 1 on failure (no test framework needed, just assert)
   - Register in CMakeLists.txt with label "cpu"

### Key Constraints
- Do NOT remove or change any existing build targets
- Do NOT change existing test binary behavior
- Keep the existing Python script steps as CTest tests (don't delete functionality)
- The GPU tests should be labeled so CI can skip them
- EXECUTABLE_OUTPUT_PATH is already set to ${PROJECT_SOURCE_DIR}/bin,
  so test commands must use the correct path
```

### 手元確認
```bash
cd build && cmake .. && make -j$(nproc)
ctest --output-on-failure               # 全テスト
ctest --label-regex python              # Python のみ
ctest --label-regex gpu                 # GPU テスト（要GPU）
```

---

## Task 5: Dockerfile 改善

### Codex プロンプト

```
You are working on the CudaRobotics project. The existing Dockerfile at the
project root uses a two-stage build (nvidia/cuda:12.0.0-devel → runtime).

Current issues:
- Runtime stage only copies bin/ — no Python scripts, no experiment data
- No docker-compose.yml for easy GPU passthrough
- ENTRYPOINT is just /bin/bash — no structured benchmark execution
- Missing Python dependencies for analysis scripts

YOUR TASK: Improve the Docker setup for reproducible benchmarking.

### Requirements

1. Update `Dockerfile`:
   - Keep the two-stage build (builder + runtime)
   - In runtime stage, also install Python 3 and matplotlib:
     ```dockerfile
     RUN apt-get update && apt-get install -y --no-install-recommends \
         python3 python3-pip \
         libopencv-core4.5d libopencv-highgui4.5d \
         libopencv-imgproc4.5d libopencv-imgcodecs4.5d \
         && pip3 install --no-cache-dir matplotlib \
         && rm -rf /var/lib/apt/lists/*
     ```
   - Copy scripts and experiment data into runtime:
     ```dockerfile
     COPY --from=builder /app/bin/ bin/
     COPY scripts/ scripts/
     COPY experiments/ experiments/
     COPY core/ core/
     COPY lookuptable.csv .
     ```
   - Add a helper entrypoint script:
     ```dockerfile
     COPY docker-entrypoint.sh /app/
     RUN chmod +x /app/docker-entrypoint.sh
     ENTRYPOINT ["/app/docker-entrypoint.sh"]
     CMD ["bash"]
     ```

2. Create `docker-entrypoint.sh`:
   ```bash
   #!/bin/bash
   set -e

   case "${1}" in
     benchmark)
       echo "Running Diff-MPPI benchmark..."
       ./bin/benchmark_diff_mppi --quick \
         --csv build/benchmark_diff_mppi.csv
       python3 scripts/summarize_diff_mppi.py \
         --csv build/benchmark_diff_mppi.csv
       ;;
     benchmark-7dof)
       echo "Running 7-DOF manipulator benchmark..."
       ./bin/benchmark_diff_mppi_manipulator_7dof \
         --seed-count 4 --csv build/benchmark_7dof.csv
       ;;
     test)
       echo "Running tests..."
       ./bin/test_autodiff && ./bin/test_gpu_mlp
       python3 scripts/validate_design_workflow.py
       python3 scripts/check_design_regressions.py
       ;;
     bash)
       exec /bin/bash "${@:2}"
       ;;
     *)
       exec "$@"
       ;;
   esac
   ```

3. Create `docker-compose.yml`:
   ```yaml
   services:
     cudarobotics:
       build: .
       runtime: nvidia
       environment:
         - NVIDIA_VISIBLE_DEVICES=all
         - NVIDIA_DRIVER_CAPABILITIES=compute,utility
       volumes:
         - ./build:/app/build
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

4. Update `.gitignore` if needed (docker-compose.override.yml).

### Key Constraints
- Keep backward compatibility — existing `docker build` must still work
- Base images stay as nvidia/cuda:12.0.0 (match CI)
- Do NOT add large files or datasets to the Docker image
- The build output directory should be mountable as a volume
```

### 手元確認
```bash
docker compose build
docker compose run cudarobotics benchmark
docker compose run cudarobotics test
docker compose run cudarobotics bash
```

---

## Task 6: 論文図表自動生成スクリプト

### Codex プロンプト

```
You are working on the CudaRobotics project. The project has several plotting
scripts in scripts/:
- plot_diff_mppi.py (basic time-cap and equal-time plots)
- plot_diff_mppi_mechanism.py (correction vs horizon analysis)
- plot_pareto_frontier.py (Pareto frontier for dynamic_slalom)
- plot_paper_main_figure.py (main paper figure)

These scripts work but are fragmented. For paper submission, we need a single
script that generates ALL paper figures in publication quality.

YOUR TASK: Create a unified paper figure generation script.

### Requirements

1. Create `scripts/generate_paper_figures.py`:
   - Single entry point for all paper figures
   - Output directory: paper/figures/ (create if needed)
   - Output format: PDF (vector graphics for LaTeX)
   - Use matplotlib with publication settings:
     - Font size 8pt (IEEE column width)
     - Figure width: 3.5 inches (single column) or 7.0 inches (double column)
     - Font family: serif (Times-like, matching IEEEtran)
     - Tight layout with minimal whitespace

2. Generate these figures:

   a) `fig_pareto.pdf` — Pareto Frontier (MAIN FIGURE, double-column)
      - X-axis: wall-clock time per step (ms)
      - Y-axis: final distance to goal
      - One subplot per scenario (dynamic_crossing, dynamic_slalom)
      - All planner variants as different markers/colors
      - Highlight the "non-hybrid ceiling" — draw a horizontal band showing
        the best non-hybrid result vs diff_mppi breakthrough
      - Input: build/benchmark_diff_mppi_exact_time_full.csv
        (fall back to build/benchmark_diff_mppi.csv)

   b) `fig_mechanism.pdf` — Gradient Correction Analysis (single-column)
      - Top: correction magnitude vs episode step (time series)
      - Bottom: correction magnitude vs horizon position (aggregated)
      - Show diff_mppi_1 and diff_mppi_3 as separate lines
      - Mark obstacle encounter region with shaded background
      - Input: build/trace_dynamic_slalom_*.csv

   c) `fig_7dof.pdf` — 7-DOF Results (single-column)
      - Bar chart: success rate by planner variant
      - Grouped by scenario (7dof_shelf_reach, 7dof_dynamic_avoid)
      - Error bars from multi-seed runs
      - Time annotation on each bar
      - Input: build/benchmark_diff_mppi_manipulator_7dof.csv
        (fall back to build/benchmark_7dof.csv)

   d) `fig_ablation.pdf` — Ablation Study (single-column)
      - Compare: mppi, grad_only_3, diff_mppi_1, diff_mppi_3
      - Bar chart of final distance on dynamic_slalom
      - Shows that sampling alone fails, gradient alone fails,
        but hybrid succeeds
      - Input: same CSV as fig_pareto

   e) `fig_scenarios.pdf` — Scenario Visualization (double-column)
      - 2x1 grid showing dynamic_crossing and dynamic_slalom
      - Plot: obstacles (circles), start/goal markers, example trajectories
      - Use fixed coordinates from the benchmark scenarios
      - This is a schematic — no CSV input needed, hardcode the geometry

3. CLI interface:
   ```
   python3 scripts/generate_paper_figures.py [--csv PATH] [--trace-dir PATH] [--out-dir PATH]
   ```
   - Default --csv: build/benchmark_diff_mppi_exact_time_full.csv
   - Default --trace-dir: build/
   - Default --out-dir: paper/figures/

4. Add `paper/figures/.gitignore`:
   ```
   *.pdf
   ```

### Style Constants (define at top of script)
```python
COLORS = {
    "mppi": "#1f77b4",
    "feedback_mppi_ref": "#ff7f0e",
    "feedback_mppi_cov": "#2ca02c",
    "feedback_mppi_fused": "#d62728",
    "diff_mppi_1": "#9467bd",
    "diff_mppi_3": "#e377c2",
    "grad_only_3": "#7f7f7f",
    "step_mppi": "#bcbd22",
}
MARKERS = {
    "mppi": "o", "feedback_mppi_ref": "s", "feedback_mppi_cov": "^",
    "feedback_mppi_fused": "D", "diff_mppi_1": "P", "diff_mppi_3": "*",
    "grad_only_3": "x", "step_mppi": "v",
}
```

### Key Constraints
- Must work even if some CSV files are missing (skip that figure, print warning)
- Do NOT modify existing plot scripts
- Use only matplotlib (no seaborn, plotly, etc.)
- All text in figures must be in English
- Legend should use clean planner names (e.g., "MPPI", "Feedback-MPPI (ref)",
  "Diff-MPPI-3", etc.)
```

### 手元確認
```bash
python3 scripts/generate_paper_figures.py
ls paper/figures/  # fig_pareto.pdf, fig_mechanism.pdf, etc.
```
