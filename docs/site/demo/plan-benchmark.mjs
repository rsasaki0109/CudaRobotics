import {Planner, cpuCosts, HORIZON, DT, COLLISION} from './planner.mjs';
import {summarize} from '../assets/benchmark.mjs';

export async function compareMppiPlans(source, {gpuEvaluate = null, cpuEvaluate = cpuCosts, warmups = 3, repetitions = 15,
  now = () => performance.now(), yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
  shouldCancel = () => false, onProgress = () => {}} = {}) {
  const input = structuredClone(source);
  const temperature = input.temperature ?? 3;
  if (!Number.isInteger(input.count) || input.count < 1 || input.count > 8192 || !Number.isFinite(temperature)
    || temperature < .5 || temperature > 10 || !Number.isInteger(warmups) || warmups < 0 || warmups > 20
    || !Number.isInteger(repetitions) || repetitions < 1 || repetitions > 100) throw new Error('Invalid plan benchmark settings');
  const samples = {cpu: {total: [], sampling: [], rollout: [], reduction: []}, gpu: {total: [], sampling: [], rollout: [], reduction: []}};
  const plans = {cpu: [], gpu: gpuEvaluate ? [] : null};
  const accuracy = gpuEvaluate ? {matches: true, maxAbsoluteError: 0, mismatchedCellComparisons: 0,
    classificationMismatches: 0, checkedCellComparisons: 0, absoluteTolerance: .001, relativeTolerance: .001} : null;
  const commandAccuracy = gpuEvaluate ? {matches: true, checkedPairs: 0, mismatchedPairs: 0, validCountMismatches: 0,
    maxControlError: 0, absoluteTolerance: .001} : null;
  const check = () => {if (shouldCancel()) throw new DOMException('Cancelled', 'AbortError');};
  for (let iteration = 0; iteration < warmups + repetitions; iteration++) {
    const outputs = {};
    const order = !gpuEvaluate ? ['cpu'] : iteration % 2 ? ['gpu', 'cpu'] : ['cpu', 'gpu'];
    for (const backend of order) {
      check(); await yieldControl(); check();
      const start = now();
      const planner = new Planner(311);
      const controls = planner.sample(input.state, input.goal, input.count);
      const sampled = now();
      const costs = await (backend === 'gpu' ? gpuEvaluate : cpuEvaluate)(input.state, input.goal, input.obstacles, controls, input.count);
      const evaluated = now();
      const command = planner.update(costs, controls, temperature);
      const end = now();
      check();
      const timing = {total: end - start, sampling: sampled - start, rollout: evaluated - sampled, reduction: end - evaluated};
      if (!Object.values(timing).every(value => Number.isFinite(value) && value >= 0)) throw new Error('Invalid plan benchmark clock');
      if (costs.length !== input.count || !costs.every(value => Number.isFinite(value) && value >= 0)
        || !command.average.every(Number.isFinite)) throw new Error('Invalid plan benchmark output');
      outputs[backend] = {costs, command};
      if (iteration >= warmups) {
        for (const stage of Object.keys(timing)) samples[backend][stage].push(timing[stage]);
        plans[backend].push({valid: command.valid, best: command.best, minimum: command.minimum,
          velocity: command.velocity, angular: command.angular, average: Array.from(command.average)});
      }
      onProgress({backend, phase: iteration < warmups ? 'warmup' : 'measurement',
        iteration: iteration < warmups ? iteration + 1 : iteration - warmups + 1, total: iteration < warmups ? warmups : repetitions});
      check();
    }
    if (gpuEvaluate && iteration >= warmups) {
      for (let index = 0; index < input.count; index++) {
        const expected = outputs.cpu.costs[index];
        const actual = outputs.gpu.costs[index];
        const error = Math.abs(expected - actual);
        const classificationMismatch = (expected >= COLLISION) !== (actual >= COLLISION);
        accuracy.maxAbsoluteError = Math.max(accuracy.maxAbsoluteError, error);
        if (classificationMismatch) accuracy.classificationMismatches++;
        if (classificationMismatch || error > Math.max(.001, .001 * Math.abs(expected))) accuracy.mismatchedCellComparisons++;
        accuracy.checkedCellComparisons++;
      }
      const validMatches = outputs.cpu.command.valid === outputs.gpu.command.valid;
      let controlMatches = true;
      for (let index = 0; index < HORIZON * 2; index++) {
        const error = Math.abs(outputs.cpu.command.average[index] - outputs.gpu.command.average[index]);
        commandAccuracy.maxControlError = Math.max(commandAccuracy.maxControlError, error);
        if (error > commandAccuracy.absoluteTolerance) controlMatches = false;
      }
      commandAccuracy.checkedPairs++;
      if (!validMatches) commandAccuracy.validCountMismatches++;
      if (!validMatches || !controlMatches) commandAccuracy.mismatchedPairs++;
    }
  }
  check();
  const stages = Object.fromEntries(['cpu', 'gpu'].map(backend => [backend, backend === 'gpu' && !gpuEvaluate ? null
    : Object.fromEntries(Object.entries(samples[backend]).map(([stage, values]) => [stage, summarize(values)]))]));
  if (accuracy) accuracy.matches = accuracy.mismatchedCellComparisons === 0;
  if (commandAccuracy) commandAccuracy.matches = commandAccuracy.mismatchedPairs === 0;
  const cpu = stages.cpu.total;
  const gpu = stages.gpu?.total ?? null;
  return {schema: 'cudarobotics.mppi-plan-benchmark.v1', demo: 'mppi', warmups, repetitions, cpu, gpu, stages, plans, accuracy, commandAccuracy,
    speedup: accuracy?.matches && commandAccuracy.matches && cpu.medianMs > 0 && gpu.medianMs > 0 ? cpu.medianMs / gpu.medianMs : null,
    input: {state: input.state, goal: input.goal, obstacles: input.obstacles, count: input.count, temperature, seed: 311, horizon: HORIZON, dt: DT},
    order: 'alternating CPU-first / GPU-first pairs; yielding before every call',
    scope: 'one fixed-input cold-start plan: fresh seed-311 planner with zero nominal controls on every call; sampling + rollout evaluation + CPU softmin/update; not a closed-loop mission or warm-start latency',
    cpuExecution: 'main-thread JavaScript for sampling, costs and softmin; not a worker round trip',
    timing: 'per-call wall time including CPU allocations, GPU uploads/synchronization/readback, sampling and softmin; GPU setup, validation, agreement checks, drawing, motion and command safety guard excluded; stage medians need not sum to the total median',
    limitations: 'all rollout costs and collision classes plus all 64 weighted control values and valid counts are checked; control tolerance is 0.001 in m/s or rad/s; safe commands and arrival are tested separately by mission trials; WebGPU/JavaScript, not native CUDA/C++'};
}
