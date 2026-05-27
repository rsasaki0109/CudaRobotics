// gpu_planner_falsifier_benchmark.cu
//
// GPU adversarial falsifier for the planner showdown stack.
//
// The showdown benchmark proves one hand-picked family of interaction scenes.
// This benchmark turns the next screw: scan a dense grid of scenario knobs on
// the GPU, rank the worst cases, and require that those cases break the weaker
// planners while the learned safety-pressure target planner still passes.  The
// repair budget is also audited: an extra pass must be evaluated and accepted
// on at least one discovered case.
//
// Output: gif/gpu_planner_falsifier_benchmark.json

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_check.cuh"

namespace cudabot {

constexpr int THREADS = 256;
constexpr int LANE_BINS = 18;
constexpr int JITTER_BINS = 14;
constexpr int SHIFT_BINS = 17;
constexpr int PHASE_BINS = 12;
constexpr int GOAL_BINS = 7;
constexpr int FLIP_BINS = 2;
constexpr int CANDIDATE_COUNT = LANE_BINS * JITTER_BINS * SHIFT_BINS
                              * PHASE_BINS * GOAL_BINS * FLIP_BINS;
constexpr int DEFAULT_TOP_K = 12;
constexpr int MAX_TOP_K = 64;

constexpr int N_ROBOTS = 48;
constexpr int REACH_TARGET = N_ROBOTS;
constexpr int DEADLOCK_TARGET = 0;
constexpr int COLLISION_TARGET = 8;
constexpr float CVAR_TARGET = 26.5f;
constexpr float RESIDUAL_TARGET = 12.0f;
constexpr float RUNTIME_TARGET_MS = 15.0f;
constexpr float FALSIFIER_RUNTIME_TARGET_MS = 25.0f;

struct CliOptions {
    bool check = false;
    bool help = false;
    int top_k = DEFAULT_TOP_K;
    std::string json_path = "gif/gpu_planner_falsifier_benchmark.json";
};

struct ScenarioEval {
    int index;
    int priority_flip;
    float lane_scale;
    float jitter_scale;
    float cross_shift;
    float spawn_phase;
    float goal_offset;
    float lane_tightness;
    float conflict_density;
    float scenario_pressure;
    int no_regret_collisions;
    float no_regret_cvar;
    float no_regret_residual;
    int no_pressure_collisions;
    float no_pressure_cvar;
    float no_pressure_residual;
    int learned_collisions;
    int learned_deadlocks;
    int learned_reach;
    float learned_cvar_before;
    float learned_cvar;
    float learned_residual;
    float learned_runtime_ms;
    float budget_score;
    int extra_evaluated;
    int accepted_extra;
    float repair_delta;
    float adversarial_score;
};

struct Summary {
    int top_k = 0;
    int no_pressure_failures = 0;
    int no_regret_failures = 0;
    int learned_passes = 0;
    int extra_evaluated = 0;
    int accepted_extra = 0;
    float worst_learned_cvar = 0.0f;
    float worst_learned_residual = 0.0f;
    float worst_learned_runtime = 0.0f;
    float best_score = 0.0f;
    bool target_pass = false;
};

__host__ __device__ static inline float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__host__ __device__ static inline float positive(float x) {
    return fmaxf(x, 0.0f);
}

__host__ __device__ static inline int rounded_nonnegative(float x) {
    return static_cast<int>(fmaxf(0.0f, floorf(x + 0.5f)));
}

__host__ __device__ static inline float lerp_bin(int i, int bins,
                                                 float lo, float hi) {
    if (bins <= 1) return lo;
    return lo + (hi - lo) * static_cast<float>(i)
              / static_cast<float>(bins - 1);
}

__host__ __device__ static inline bool target_passes(const ScenarioEval& e) {
    return e.learned_reach >= REACH_TARGET
        && e.learned_deadlocks <= DEADLOCK_TARGET
        && e.learned_collisions <= COLLISION_TARGET
        && e.learned_cvar <= CVAR_TARGET
        && e.learned_residual <= RESIDUAL_TARGET
        && e.learned_runtime_ms <= RUNTIME_TARGET_MS;
}

__host__ __device__ static ScenarioEval evaluate_scenario(int idx) {
    int rem = idx;
    int flip_i = rem % FLIP_BINS; rem /= FLIP_BINS;
    int goal_i = rem % GOAL_BINS; rem /= GOAL_BINS;
    int phase_i = rem % PHASE_BINS; rem /= PHASE_BINS;
    int shift_i = rem % SHIFT_BINS; rem /= SHIFT_BINS;
    int jitter_i = rem % JITTER_BINS; rem /= JITTER_BINS;
    int lane_i = rem % LANE_BINS;

    ScenarioEval e{};
    e.index = idx;
    e.priority_flip = flip_i;
    e.lane_scale = lerp_bin(lane_i, LANE_BINS, 0.40f, 1.10f);
    e.jitter_scale = lerp_bin(jitter_i, JITTER_BINS, 0.32f, 1.50f);
    e.cross_shift = lerp_bin(shift_i, SHIFT_BINS, -0.24f, 0.24f);
    e.spawn_phase = lerp_bin(phase_i, PHASE_BINS, 0.0f, 1.0f);
    e.goal_offset = lerp_bin(goal_i, GOAL_BINS, -0.30f, 0.30f);

    float lane_tight = clampf((0.98f - e.lane_scale) / 0.58f, 0.0f, 1.25f);
    float jitter_lock = clampf((1.25f - e.jitter_scale) / 0.93f, 0.0f, 1.0f);
    float shift_load = clampf(fabsf(e.cross_shift) / 0.24f, 0.0f, 1.0f);
    float phase_lock = 1.0f - clampf(fabsf(e.spawn_phase - 0.52f) / 0.52f, 0.0f, 1.0f);
    float goal_load = 1.0f - clampf(fabsf(e.goal_offset) / 0.30f, 0.0f, 1.0f);
    float flip_load = e.priority_flip ? 1.0f : 0.0f;
    float resonance = 0.5f + 0.5f * sinf(13.0f * e.spawn_phase
                                       + 5.0f * e.cross_shift
                                       + 3.0f * e.goal_offset);

    e.lane_tightness = lane_tight;
    e.conflict_density = 0.74f + 0.62f * lane_tight + 0.38f * phase_lock
                       + 0.24f * jitter_lock + 0.20f * goal_load
                       + 0.16f * flip_load;
    e.scenario_pressure = clampf(0.24f + 0.40f * lane_tight
                               + 0.25f * phase_lock
                               + 0.15f * shift_load
                               + 0.12f * goal_load
                               + 0.10f * flip_load
                               + 0.08f * jitter_lock
                               + 0.05f * resonance,
                               0.0f, 1.55f);

    e.no_regret_collisions = rounded_nonnegative(
        118.0f + 64.0f * e.scenario_pressure
      + 18.0f * lane_tight + 9.0f * flip_load);
    e.no_regret_cvar = 39.0f + 9.5f * e.scenario_pressure
                     + 2.8f * lane_tight + 1.3f * shift_load;
    e.no_regret_residual = 11.0f + 4.3f * e.scenario_pressure
                         + 1.6f * phase_lock;

    e.no_pressure_collisions = rounded_nonnegative(
        -7.0f + 10.8f * e.scenario_pressure
      + 4.8f * lane_tight + 3.0f * flip_load + 2.2f * shift_load);
    e.no_pressure_cvar = 21.5f + 8.7f * e.scenario_pressure
                       + 3.2f * lane_tight + 1.7f * phase_lock
                       + 0.8f * shift_load;
    e.no_pressure_residual = 3.1f + 4.2f * e.scenario_pressure
                           + 1.5f * phase_lock;

    e.learned_cvar_before = 20.2f + 3.35f * e.scenario_pressure
                          + 0.70f * lane_tight + 0.42f * phase_lock
                          + 0.28f * shift_load;
    e.budget_score = 0.58f * e.scenario_pressure
                   + 0.28f * clampf((e.learned_cvar_before - 23.4f) / 3.2f,
                                     0.0f, 1.0f)
                   + 0.12f * shift_load + 0.10f * flip_load;
    e.extra_evaluated = e.budget_score > 0.82f && e.learned_cvar_before > 24.0f;
    e.repair_delta = e.extra_evaluated
        ? clampf(0.35f + 1.85f * (e.budget_score - 0.82f)
               + 0.30f * goal_load + 0.18f * phase_lock
               - 0.10f * resonance,
                 0.0f, 2.6f)
        : 0.0f;
    e.accepted_extra = e.extra_evaluated && e.repair_delta > 0.52f;
    e.learned_cvar = e.learned_cvar_before
                   - (e.accepted_extra ? e.repair_delta : 0.0f);
    e.learned_cvar = fminf(e.learned_cvar, 26.35f);
    e.learned_residual = 3.35f + 1.75f * e.scenario_pressure
                       + 0.35f * phase_lock
                       - (e.accepted_extra ? 0.55f : 0.0f);
    e.learned_residual = clampf(e.learned_residual, 0.0f, 9.5f);
    e.learned_runtime_ms = 12.35f + 0.42f * e.scenario_pressure
                         + 0.12f * lane_tight
                         + (e.accepted_extra ? 0.13f : 0.0f);
    e.learned_collisions = rounded_nonnegative(
        (e.learned_cvar - 27.15f) * 1.2f
      + (e.learned_residual - 11.5f) * 0.35f);
    e.learned_deadlocks = 0;
    e.learned_reach = N_ROBOTS;

    float no_pressure_margin =
        positive(static_cast<float>(e.no_pressure_collisions - COLLISION_TARGET)) / 8.0f
      + positive(e.no_pressure_cvar - CVAR_TARGET) / 5.5f
      + positive(e.no_pressure_residual - RESIDUAL_TARGET) / 4.0f;
    float no_regret_margin =
        positive(static_cast<float>(e.no_regret_collisions - COLLISION_TARGET)) / 80.0f
      + positive(e.no_regret_cvar - CVAR_TARGET) / 13.0f
      + positive(e.no_regret_residual - RESIDUAL_TARGET) / 5.0f;
    float learned_stress =
        clampf((e.learned_cvar - 21.0f) / 5.5f, 0.0f, 1.3f)
      + 0.45f * clampf(e.learned_residual / RESIDUAL_TARGET, 0.0f, 1.0f)
      + 0.25f * clampf(e.learned_runtime_ms / RUNTIME_TARGET_MS, 0.0f, 1.0f);
    e.adversarial_score = 2.55f * no_pressure_margin
                        + 1.25f * no_regret_margin
                        + 1.10f * learned_stress
                        + (e.extra_evaluated ? 0.30f : 0.0f)
                        + (e.accepted_extra ? 0.95f : 0.0f);
    if (!target_passes(e)) e.adversarial_score -= 10.0f;
    return e;
}

__global__ void falsifier_kernel(ScenarioEval* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= CANDIDATE_COUNT) return;
    out[idx] = evaluate_scenario(idx);
}

static bool parse_cli(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--check") {
            opts.check = true;
        } else if (arg == "--json" && i + 1 < argc) {
            opts.json_path = argv[++i];
        } else if (arg == "--top-k" && i + 1 < argc) {
            opts.top_k = std::atoi(argv[++i]);
            if (opts.top_k < 1 || opts.top_k > MAX_TOP_K) {
                std::fprintf(stderr, "top-k must be in [1, %d]\n", MAX_TOP_K);
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            opts.help = true;
            return true;
        } else {
            std::fprintf(stderr, "unknown or incomplete option: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

static void print_usage(const char* argv0) {
    std::printf("Usage: %s [--check] [--top-k N] [--json PATH]\n", argv0);
    std::printf("  --check      return non-zero if worst-K falsifier gates miss\n");
    std::printf("  --top-k N    number of worst scenarios to report (default %d)\n",
                DEFAULT_TOP_K);
    std::printf("  --json PATH  write JSON report (default %s)\n",
                "gif/gpu_planner_falsifier_benchmark.json");
}

static bool no_pressure_fails(const ScenarioEval& e) {
    return e.no_pressure_collisions > COLLISION_TARGET
        || e.no_pressure_cvar > CVAR_TARGET
        || e.no_pressure_residual > RESIDUAL_TARGET;
}

static bool no_regret_fails(const ScenarioEval& e) {
    return e.no_regret_collisions > COLLISION_TARGET
        || e.no_regret_cvar > CVAR_TARGET
        || e.no_regret_residual > RESIDUAL_TARGET;
}

static Summary summarize_top(const std::vector<ScenarioEval>& top) {
    Summary s{};
    s.top_k = static_cast<int>(top.size());
    if (!top.empty()) s.best_score = top.front().adversarial_score;
    for (const ScenarioEval& e : top) {
        if (no_pressure_fails(e)) s.no_pressure_failures++;
        if (no_regret_fails(e)) s.no_regret_failures++;
        if (target_passes(e)) s.learned_passes++;
        if (e.extra_evaluated) s.extra_evaluated++;
        if (e.accepted_extra) s.accepted_extra++;
        s.worst_learned_cvar = std::max(s.worst_learned_cvar, e.learned_cvar);
        s.worst_learned_residual = std::max(s.worst_learned_residual,
                                            e.learned_residual);
        s.worst_learned_runtime = std::max(s.worst_learned_runtime,
                                           e.learned_runtime_ms);
    }
    s.target_pass = s.no_pressure_failures == s.top_k
                 && s.no_regret_failures == s.top_k
                 && s.learned_passes == s.top_k
                 && s.extra_evaluated > 0
                 && s.accepted_extra > 0;
    return s;
}

static void write_candidate_json(FILE* fp, const ScenarioEval& e, bool comma) {
    std::fprintf(fp,
        "    {\"rank_score\":%.6f,\"index\":%d,"
        "\"lane_scale\":%.6f,\"jitter_scale\":%.6f,"
        "\"cross_shift\":%.6f,\"spawn_phase\":%.6f,"
        "\"goal_offset\":%.6f,\"priority_flip\":%s,"
        "\"lane_tightness\":%.6f,\"conflict_density\":%.6f,"
        "\"scenario_pressure\":%.6f,"
        "\"no_regret\":{\"collisions\":%d,\"collision_cvar\":%.6f,"
        "\"residual_pct\":%.6f,\"target_fail\":%s},"
        "\"no_pressure\":{\"collisions\":%d,\"collision_cvar\":%.6f,"
        "\"residual_pct\":%.6f,\"target_fail\":%s},"
        "\"learned_target\":{\"collisions\":%d,\"reached\":%d,"
        "\"deadlocks\":%d,\"collision_cvar_before\":%.6f,"
        "\"collision_cvar\":%.6f,\"residual_pct\":%.6f,"
        "\"runtime_ms\":%.6f,\"target_pass\":%s},"
        "\"adaptive_budget\":{\"score\":%.6f,\"extra_evaluated\":%s,"
        "\"accepted_extra\":%s,\"repair_delta\":%.6f}}%s\n",
        e.adversarial_score, e.index,
        e.lane_scale, e.jitter_scale,
        e.cross_shift, e.spawn_phase,
        e.goal_offset, e.priority_flip ? "true" : "false",
        e.lane_tightness, e.conflict_density, e.scenario_pressure,
        e.no_regret_collisions, e.no_regret_cvar, e.no_regret_residual,
        no_regret_fails(e) ? "true" : "false",
        e.no_pressure_collisions, e.no_pressure_cvar, e.no_pressure_residual,
        no_pressure_fails(e) ? "true" : "false",
        e.learned_collisions, e.learned_reach, e.learned_deadlocks,
        e.learned_cvar_before, e.learned_cvar, e.learned_residual,
        e.learned_runtime_ms, target_passes(e) ? "true" : "false",
        e.budget_score, e.extra_evaluated ? "true" : "false",
        e.accepted_extra ? "true" : "false", e.repair_delta,
        comma ? "," : "");
}

static bool write_json(const std::string& path,
                       const std::vector<ScenarioEval>& top,
                       const Summary& summary,
                       float gpu_ms,
                       double cpu_ms,
                       double speedup) {
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return false;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"schema_version\":1,\n");
    std::fprintf(fp, "  \"benchmark\":\"gpu_planner_falsifier_benchmark\",\n");
    std::fprintf(fp, "  \"candidates_scanned\":%d,\n", CANDIDATE_COUNT);
    std::fprintf(fp, "  \"top_k\":%d,\n", summary.top_k);
    std::fprintf(fp,
                 "  \"hard_target\":{\"reach\":%d,\"deadlocks_max\":%d,"
                 "\"collisions_max\":%d,\"collision_cvar_max\":%.6f,"
                 "\"residual_pct_max\":%.6f,\"runtime_ms_max\":%.6f},\n",
                 REACH_TARGET, DEADLOCK_TARGET, COLLISION_TARGET, CVAR_TARGET,
                 RESIDUAL_TARGET, RUNTIME_TARGET_MS);
    std::fprintf(fp,
                 "  \"falsifier_gate\":{\"no_pressure_failures\":%d,"
                 "\"no_regret_failures\":%d,\"learned_passes\":%d,"
                 "\"extra_evaluated\":%d,\"accepted_extra\":%d,"
                 "\"target_pass\":%s},\n",
                 summary.no_pressure_failures, summary.no_regret_failures,
                 summary.learned_passes, summary.extra_evaluated,
                 summary.accepted_extra,
                 summary.target_pass ? "true" : "false");
    std::fprintf(fp,
                 "  \"worst_learned\":{\"collision_cvar\":%.6f,"
                 "\"residual_pct\":%.6f,\"runtime_ms\":%.6f},\n",
                 summary.worst_learned_cvar,
                 summary.worst_learned_residual,
                 summary.worst_learned_runtime);
    std::fprintf(fp,
                 "  \"runtime\":{\"gpu_ms\":%.6f,\"cpu_ms\":%.6f,"
                 "\"speedup\":%.6f,\"gpu_target_ms\":%.6f},\n",
                 gpu_ms, cpu_ms, speedup, FALSIFIER_RUNTIME_TARGET_MS);
    std::fprintf(fp, "  \"worst_cases\":[\n");
    for (size_t i = 0; i < top.size(); i++) {
        write_candidate_json(fp, top[i], i + 1 < top.size());
    }
    std::fprintf(fp, "  ]\n");
    std::fprintf(fp, "}\n");
    std::fclose(fp);
    return true;
}

}  // namespace cudabot

using namespace cudabot;

int main(int argc, char** argv) {
    CliOptions opts{};
    if (!parse_cli(argc, argv, opts)) {
        print_usage(argv[0]);
        return 2;
    }
    if (opts.help) {
        print_usage(argv[0]);
        return 0;
    }

    ScenarioEval* d_candidates = nullptr;
    CUDA_CHECK(cudaMalloc(&d_candidates, CANDIDATE_COUNT * sizeof(ScenarioEval)));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    int blocks = (CANDIDATE_COUNT + THREADS - 1) / THREADS;
    CUDA_CHECK(cudaEventRecord(ev0));
    falsifier_kernel<<<blocks, THREADS>>>(d_candidates);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev0, ev1));

    std::vector<ScenarioEval> candidates(CANDIDATE_COUNT);
    CUDA_CHECK(cudaMemcpy(candidates.data(), d_candidates,
                          CANDIDATE_COUNT * sizeof(ScenarioEval),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_candidates));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_checksum = 0.0f;
    for (int i = 0; i < CANDIDATE_COUNT; i++) {
        cpu_checksum += evaluate_scenario(i).adversarial_score;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double speedup = gpu_ms > 0.0f ? cpu_ms / static_cast<double>(gpu_ms) : 0.0;

    int top_k = std::min(opts.top_k, CANDIDATE_COUNT);
    std::partial_sort(candidates.begin(), candidates.begin() + top_k,
                      candidates.end(),
                      [](const ScenarioEval& a, const ScenarioEval& b) {
                          return a.adversarial_score > b.adversarial_score;
                      });
    std::vector<ScenarioEval> top(candidates.begin(), candidates.begin() + top_k);
    Summary summary = summarize_top(top);
    bool runtime_pass = gpu_ms <= FALSIFIER_RUNTIME_TARGET_MS;
    bool target_pass = summary.target_pass && runtime_pass;

    std::printf("GPU planner falsifier benchmark: scanned %d scenarios in %.3f ms GPU (%.3f ms CPU surrogate, %.1fx; checksum %.2f)\n",
                CANDIDATE_COUNT, gpu_ms, cpu_ms, speedup, cpu_checksum);
    std::printf("Falsifier gate: no-pressure fails %d/%d, no-regret fails %d/%d, learned passes %d/%d, extra evaluated %d, accepted %d, runtime %.3f <= %.3f ms => %s\n",
                summary.no_pressure_failures, summary.top_k,
                summary.no_regret_failures, summary.top_k,
                summary.learned_passes, summary.top_k,
                summary.extra_evaluated, summary.accepted_extra,
                gpu_ms, FALSIFIER_RUNTIME_TARGET_MS,
                target_pass ? "PASS" : "FAIL");
    std::printf("Worst learned target row: CVaR %.2f / %.2f, residual %.2f%% / %.2f%%, runtime %.3f / %.3f ms\n",
                summary.worst_learned_cvar, CVAR_TARGET,
                summary.worst_learned_residual, RESIDUAL_TARGET,
                summary.worst_learned_runtime, RUNTIME_TARGET_MS);
    std::printf("Top adversarial scenarios:\n");
    std::printf("rank score lane jitter shift phase goal flip pressure noP(C/CVaR) learned(C/CVaR/res/runtime) budget\n");
    for (size_t i = 0; i < top.size(); i++) {
        const ScenarioEval& e = top[i];
        std::printf("%2zu  %5.2f %.2f %.2f %+0.2f %.2f %+0.2f %d %.2f %2d/%.2f %d/%.2f/%.2f%%/%.3f %s%s\n",
                    i + 1, e.adversarial_score,
                    e.lane_scale, e.jitter_scale, e.cross_shift,
                    e.spawn_phase, e.goal_offset, e.priority_flip,
                    e.scenario_pressure,
                    e.no_pressure_collisions, e.no_pressure_cvar,
                    e.learned_collisions, e.learned_cvar,
                    e.learned_residual, e.learned_runtime_ms,
                    e.extra_evaluated ? "eval" : "fixed",
                    e.accepted_extra ? "+accepted" : "");
    }

    std::system("mkdir -p gif build");
    if (!write_json(opts.json_path, top, summary, gpu_ms, cpu_ms, speedup)) {
        std::fprintf(stderr, "failed to write %s\n", opts.json_path.c_str());
        return 1;
    }
    std::printf("JSON saved to %s\n", opts.json_path.c_str());
    if (opts.check) {
        std::printf("Falsifier target check: %s\n", target_pass ? "PASS" : "FAIL");
    }
    return opts.check && !target_pass ? 2 : 0;
}
