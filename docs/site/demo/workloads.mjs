import {mppiSnapshot} from './benchmark.mjs';
import {summarizeWorkloads} from './workload-insights.mjs';
import {Planner, scenes, cpuCosts, safeAdvance, advance, clearance, HORIZON, DT, COLLISION} from './planner.mjs';

export const WORKLOAD_COUNTS = Object.freeze([128, 512, 2048, 8192]);
export const MISSION_SEEDS = Object.freeze([311, 312, 313]);
export const workloadMatches = report => report.accuracy?.matches && (report.commandAccuracy?.matches ?? true);
const yieldFrame = () => new Promise(resolve => setTimeout(resolve, 0));
const checkCancellation = shouldCancel => {if (shouldCancel()) throw new DOMException('Cancelled', 'AbortError');};

export async function compareWorkloads(source, {createJob, shouldCancel = () => false, onProgress = () => {}}) {
  const snapshot = structuredClone(source);
  snapshot.measurement ??= 'kernel';
  if (!['kernel', 'plan'].includes(snapshot.measurement)) throw new Error('Unknown workload measurement');
  const results = [];
  for (const count of WORKLOAD_COUNTS) {
    checkCancellation(shouldCancel);
    let job;
    try {
      const input = mppiSnapshot(snapshot.state, snapshot.goal, snapshot.obstacles, count);
      input.measurement = snapshot.measurement; input.temperature = snapshot.temperature ?? 3;
      job = await createJob(input);
      checkCancellation(shouldCancel);
      const report = await job.run({shouldCancel, onProgress: progress => onProgress({count, ...progress})});
      checkCancellation(shouldCancel);
      results.push({count, modelSteps: count * HORIZON, report});
    } finally {job?.destroy();}
  }
  checkCancellation(shouldCancel);
  const completeAgreement = results.every(entry => workloadMatches(entry.report));
  const report = {schema: 'cudarobotics.mppi-workloads.v1', source: snapshot, counts: [...WORKLOAD_COUNTS], results,
    firstTestedGpuWin: completeAgreement ? results.find(entry => entry.report.speedup > 1)?.count ?? null : null,
    interpretation: 'first tested count with a GPU median win, not an interpolated crossover or a guarantee; suppressed if any workload lacks agreement',
    measurement: snapshot.measurement,
    scope: `fixed current state and local target; fresh seed 311 at each count; matched controls within each CPU/GPU pair, not nested samples across counts; ${snapshot.measurement === 'plan' ? 'cold-start sampling + costs + CPU softmin/update' : 'rollout costs only'}; not closed-loop mission success`};
  return {...report, insights: summarizeWorkloads(report)};
}

export async function runMission({scene, count, seed, temperature = 3, maxSteps = 600},
  {evaluate = cpuCosts, shouldCancel = () => false, yieldControl = yieldFrame, onProgress = () => {}} = {}) {
  if (!scene || !Number.isInteger(count) || !WORKLOAD_COUNTS.includes(count) || !Number.isInteger(seed)
    || !Number.isFinite(temperature) || temperature < .5 || temperature > 10
    || !Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 600) throw new Error('Invalid mission settings');
  const planner = new Planner(seed);
  let state = [...scene.start];
  const waypoints = (scene.waypoints ?? []).map(point => [...point]);
  const path = [[...state]];
  let minimumClearance = clearance(state[0], state[1], scene.obstacles);
  let status = minimumClearance > 0 ? 'timeout' : 'unsafe_start';
  let steps = 0;
  let validRollouts = 0;
  let evaluatedRollouts = 0;
  for (; status === 'timeout' && steps < maxSteps;) {
    checkCancellation(shouldCancel); await yieldControl(); checkCancellation(shouldCancel);
    if (Math.hypot(state[0] - scene.goal[0], state[1] - scene.goal[1]) < .3) {status = 'reached'; break;}
    while (waypoints.length && Math.hypot(state[0] - waypoints[0][0], state[1] - waypoints[0][1]) < .65) waypoints.shift();
    const target = waypoints[0] ?? scene.goal;
    const controls = planner.sample(state, target, count);
    const costs = await evaluate(state, target, scene.obstacles, controls, count);
    checkCancellation(shouldCancel);
    if (costs.length !== count || !costs.every(cost => Number.isFinite(cost) && cost >= 0)) throw new Error('Invalid mission rollout costs');
    const command = planner.update(costs, controls, temperature);
    validRollouts += command.valid; evaluatedRollouts += count;
    const next = safeAdvance(state, command.velocity, command.angular, scene.obstacles);
    if (!command.valid || !next) {status = 'safe_stop'; break;}
    const half = advance(state, command.velocity, command.angular, DT / 2);
    minimumClearance = Math.min(minimumClearance, clearance(half[0], half[1], scene.obstacles), clearance(next[0], next[1], scene.obstacles));
    state = next; steps++; path.push([...state]);
    if (Math.hypot(state[0] - scene.goal[0], state[1] - scene.goal[1]) < .3) status = 'reached';
    onProgress({seed, steps, maxSteps});
  }
  checkCancellation(shouldCancel);
  return {seed, count, status, steps, simulationSeconds: steps * DT, goalDistance: Math.hypot(state[0] - scene.goal[0], state[1] - scene.goal[1]),
    minimumClearance, validRollouts, evaluatedRollouts, path};
}

export async function compareMissions({sceneId, count, temperature}, {gpuEvaluate = null, shouldCancel = () => false,
  onProgress = () => {}, ...options} = {}) {
  if (!Object.hasOwn(scenes, sceneId)) throw new Error('Unknown mission scene');
  const scene = structuredClone(scenes[sceneId]);
  const results = {cpu: [], gpu: gpuEvaluate ? [] : null};
  for (const [index, seed] of MISSION_SEEDS.entries()) {
    const order = gpuEvaluate ? index % 2 ? ['gpu', 'cpu'] : ['cpu', 'gpu'] : ['cpu'];
    for (const backend of order) {
      const result = await runMission({scene, count, seed, temperature}, {...options, shouldCancel,
        ...(backend === 'gpu' ? {evaluate: gpuEvaluate} : {}),
        onProgress: progress => onProgress({backend, ...progress})});
      results[backend].push(result);
    }
  }
  checkCancellation(shouldCancel);
  const summaries = Object.fromEntries(Object.entries(results).map(([backend, trials]) => [backend, trials ? {
    trials: trials.length, reached: trials.filter(trial => trial.status === 'reached').length,
    safeStops: trials.filter(trial => trial.status === 'safe_stop').length, timeouts: trials.filter(trial => trial.status === 'timeout').length,
    minimumClearance: Math.min(...trials.map(trial => trial.minimumClearance)),
  } : null]));
  return {schema: 'cudarobotics.mppi-missions.v1', sceneId, scene, count, temperature, seeds: [...MISSION_SEEDS], maxSteps: 600,
    results, summaries, scope: 'built-in scene from its initial pose, not the edited live goal; independent closed-loop CPU/GPU runs with identical seeds and policy; floating-point differences can change later controls and paths',
    limitations: 'three deterministic trials, not a statistical reliability estimate; guarded motion stops before unsafe commands, so safe stops are failures, not collision-free successes; no mission speedup inferred from these untimed trials; CPU mission costs run on the main thread, yielding between steps'};
}
