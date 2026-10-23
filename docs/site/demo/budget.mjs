import {Planner, scenes, cpuCosts} from './planner.mjs';
import {WORKLOAD_COUNTS, MISSION_SEEDS, runMission} from './workloads.mjs?v=budget-1';
import {compareMppiPlans} from './plan-benchmark.mjs';
import {summarize} from '../assets/benchmark.mjs';

export function selectBudgetCounts(calibrations, budgetMs, gpuAvailable) {
  if (!Number.isFinite(budgetMs) || budgetMs <= 0 || budgetMs > 1000) throw new Error('Invalid planning budget');
  return Object.fromEntries(['cpu', 'gpu'].map(backend => {
    const eligibleCounts = calibrations.filter(({count, report}) => {
      if (!WORKLOAD_COUNTS.includes(count) || (backend === 'gpu' && !gpuAvailable)) return false;
      if (gpuAvailable && (!report.accuracy?.matches || !report.commandAccuracy?.matches)) return false;
      const duration = report[backend]?.p90Ms;
      return Number.isFinite(duration) && duration > 0 && duration <= budgetMs;
    }).map(entry => entry.count);
    const count = eligibleCounts.length ? Math.max(...eligibleCounts) : null;
    return [backend, {count, eligibleCounts, reason: count ? 'Largest tested count with complete-plan P90 within budget.'
      : backend === 'gpu' && !gpuAvailable ? 'WebGPU unavailable; no synthetic count or mission.'
        : 'No verified tested count fits the complete-plan P90 budget; no mission run.'}];
  }));
}

export async function compareBudget({sceneId, temperature = 3, budgetMs = 20}, {gpuEvaluate = null, cpuEvaluate = cpuCosts,
  calibrate = compareMppiPlans, warmups = 3, repetitions = 15, maxSteps = 600,
  shouldCancel = () => false, onProgress = () => {}, now = () => performance.now(),
  yieldControl = () => new Promise(resolve => setTimeout(resolve, 0))} = {}) {
  if (!Object.hasOwn(scenes, sceneId) || !Number.isFinite(temperature) || temperature < .5 || temperature > 10
    || !Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 600) throw new Error('Invalid budget experiment settings');
  selectBudgetCounts([], budgetMs, !!gpuEvaluate);
  const check = () => {if (shouldCancel()) throw new DOMException('Cancelled', 'AbortError');};
  const scene = structuredClone(scenes[sceneId]);
  const target = scene.waypoints?.[0] ?? scene.goal;
  const calibrations = [];
  for (const count of WORKLOAD_COUNTS) {
    check();
    const report = await calibrate({state: scene.start, goal: target, obstacles: scene.obstacles, count, temperature},
      {gpuEvaluate, cpuEvaluate, warmups, repetitions, now, yieldControl, shouldCancel,
        onProgress: progress => onProgress({...progress, phase: 'calibration', calibrationPhase: progress.phase, count})});
    check(); calibrations.push({count, report});
  }
  const selections = selectBudgetCounts(calibrations, budgetMs, !!gpuEvaluate);
  const results = {cpu: selections.cpu.count ? [] : null, gpu: selections.gpu.count ? [] : null};
  for (const [index, seed] of MISSION_SEEDS.entries()) {
    const order = index % 2 ? ['gpu', 'cpu'] : ['cpu', 'gpu'];
    for (const backend of order) {
      const count = selections[backend].count;
      if (!count) continue;
      check(); await yieldControl(); check();
      const evaluate = backend === 'gpu' ? gpuEvaluate : cpuEvaluate;
      const controls = new Planner(seed).sample(scene.start, target, count);
      await evaluate(scene.start, target, scene.obstacles, controls, count);
      check();
      onProgress({phase: 'mission', backend, count, seed, steps: 0, maxSteps});
      const trial = await runMission({scene, count, seed, temperature, maxSteps}, {evaluate, budgetMs, now, yieldControl, shouldCancel,
        onProgress: progress => onProgress({phase: 'mission', backend, count, ...progress})});
      check(); results[backend].push(trial);
    }
  }
  check();
  const summaries = Object.fromEntries(Object.entries(results).map(([backend, trials]) => {
    if (!trials) return [backend, null];
    const samples = trials.flatMap(trial => trial.planning.timing?.samplesMs ?? []);
    return [backend, {count: selections[backend].count, trials: trials.length,
      reached: trials.filter(trial => trial.status === 'reached').length,
      safeStops: trials.filter(trial => trial.status === 'safe_stop').length,
      deadlineStops: trials.filter(trial => trial.status === 'deadline_stop').length,
      timeouts: trials.filter(trial => trial.status === 'timeout').length,
      deadlineMisses: trials.reduce((sum, trial) => sum + trial.planning.deadlineMisses, 0),
      meanPathLength: trials.reduce((sum, trial) => sum + trial.pathLength, 0) / trials.length,
      minimumClearance: Math.min(...trials.map(trial => trial.minimumClearance)), planning: samples.length ? summarize(samples) : null}];
  }));
  return {schema: 'cudarobotics.mppi-budget.v1', sceneId, scene, temperature, budgetMs, warmups, repetitions, maxSteps,
    seeds: [...MISSION_SEEDS], calibrations, selections, results, summaries,
    policy: 'cold-start complete-plan calibration at the built-in initial pose and first guide target; largest tested count whose P90 fits; mismatched calibration counts excluded from both backends; one untimed evaluator warmup per mission; independent warm-start missions at selected fixed counts',
    timing: 'mission plans time sampling, rollout costs and CPU softmin/update, including allocations and GPU transfers; checks, motion, drawing, between-step yields and setup excluded; actual plan durations and overruns retained',
    deadlinePolicy: 'soft deadline, not preemption or real-time scheduling: an over-budget evaluator finishes, then its command is discarded and the mission stops before motion; simulation advances DT only on accepted commands',
    limitations: 'P90 calibration is not a deadline guarantee for later states or a continuous maximum count; counts may differ, so candidate sets differ despite matching seeds/policy; three deterministic trials do not establish reliability or that GPU improves motion; path length sums the two guarded model substep segments, not continuous ground truth; no fitting count means untested, not a failed mission'};
}
