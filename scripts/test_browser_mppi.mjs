import test from 'node:test';
import assert from 'node:assert/strict';
import {Planner, scenes, HORIZON, COLLISION, clearance, safeAdvance, cpuCosts, randomGenerator} from '../docs/site/demo/planner.mjs';
import {WORKLOAD_COUNTS, MISSION_SEEDS, compareWorkloads, runMission, compareMissions} from '../docs/site/demo/workloads.mjs';
import {compareMppi} from '../docs/site/demo/benchmark.mjs';
import {compareMppiPlans} from '../docs/site/demo/plan-benchmark.mjs';
import {workloadInsight, summarizeWorkloads} from '../docs/site/demo/workload-insights.mjs';
import {summarize} from '../docs/site/assets/benchmark.mjs';
import {selectBudgetCounts, compareBudget} from '../docs/site/demo/budget.mjs';

test('seeded sampling is deterministic and respects command limits', () => {
  const first = new Planner().sample(scenes.gap.start, scenes.gap.goal, 128);
  assert.deepEqual(first, new Planner().sample(scenes.gap.start, scenes.gap.goal, 128));
  for (let index = 0; index < first.length; index += 2) {
    assert.ok(first[index] >= 0 && first[index] <= 1.6 + 1e-6);
    assert.ok(Math.abs(first[index + 1]) <= 2.4 + 1e-6);
  }
  const random = randomGenerator();
  for (let index = 0; index < 100; index++) assert.ok(random() > 0 && random() < 1);
});

test('disk clearance, boundaries and collision stops', () => {
  assert.ok(clearance(5, 5, scenes.gap.obstacles) > 0);
  assert.ok(clearance(5, 2, scenes.gap.obstacles) < 0);
  assert.ok(clearance(0.1, 5, []) < 0);
  assert.equal(safeAdvance([4.5, 2, 0], 1.6, 0, scenes.gap.obstacles), null);
  const controls = new Float32Array(HORIZON * 2).fill(0);
  for (let index = 0; index < controls.length; index += 2) controls[index] = 1.6;
  assert.equal(cpuCosts([4.5, 2, 0], [9, 2], scenes.gap.obstacles, controls, 1)[0], COLLISION);
});

test('softmin is stable, rejects invalid samples and shifts warm start', () => {
  const planner = new Planner();
  const controls = new Float32Array(HORIZON * 2 * 3);
  controls.fill(0.5, 0, HORIZON * 2);
  controls.fill(1, HORIZON * 2, HORIZON * 4);
  const result = planner.update(new Float32Array([10000, 10000, COLLISION]), controls, 0.5);
  assert.equal(result.valid, 2);
  assert.equal(result.velocity, 0.75);
  assert.equal(planner.nominal[0], 0.75);
  const stopped = planner.update(new Float32Array([NaN, Infinity, COLLISION]), controls, 3);
  assert.equal(stopped.valid, 0);
  assert.equal(stopped.velocity, 0);
  assert.equal(stopped.angular, 0);
});

for (const [name, scene] of Object.entries(scenes)) {
  test(`closed-loop ${name} reaches goal without collision`, () => {
    const planner = new Planner();
    let state = [...scene.start];
    const waypoints = [...(scene.waypoints ?? [])];
    for (let step = 0; step < 600; step++) {
      while (waypoints.length && Math.hypot(state[0] - waypoints[0][0], state[1] - waypoints[0][1]) < 0.65) waypoints.shift();
      const target = waypoints[0] ?? scene.goal;
      const controls = planner.sample(state, target, 128);
      const result = planner.update(cpuCosts(state, target, scene.obstacles, controls, 128), controls, 3);
      assert.ok(result.valid > 0, `no valid rollout at step ${step}`);
      const next = safeAdvance(state, result.velocity, result.angular, scene.obstacles);
      assert.ok(next, `collision at step ${step}`);
      state = next;
      if (Math.hypot(state[0] - scene.goal[0], state[1] - scene.goal[1]) < 0.3) return;
    }
    assert.fail(`goal not reached: ${state}`);
  });
}

test('app initializes CPU fallback and supports start, pause, goal validation and reset', async () => {
  const identifiers = ['world', 'backend', 'scene', 'engine', 'count', 'temperature',
    'temperature-value', 'goal-x', 'goal-y', 'set-goal', 'run', 'reset', 'status',
    'latency', 'valid', 'distance', 'time', 'compare', 'export-comparison', 'comparison-status',
    'cpu-median', 'gpu-median', 'cpu-range', 'gpu-range', 'cpu-bar', 'gpu-bar', 'speedup', 'accuracy',
    'compare-workloads', 'compare-missions', 'export-workloads', 'export-missions', 'cancel-workloads', 'mission-count',
    'workload-status', 'workload-crossover', 'workload-scope', 'workload-method', 'workload-takeaway', 'mission-cpu', 'mission-gpu',
    'budget-ms', 'compare-budget', 'cancel-budget', 'export-budget', 'budget-calibration', 'budget-status',
    ...['cpu', 'gpu'].flatMap(backend => ['status', 'canvas', 'trials'].map(field => `budget-${backend}-${field}`)),
    ...WORKLOAD_COUNTS.flatMap(count => ['cpu', 'gpu', 'ratio', 'accuracy', 'stages', 'cpu-bar', 'gpu-bar',
      'verdict', 'effect', 'share', 'sampling-share', 'rollout-share', 'reduction-share'].map(field => `workload-${count}-${field}`))];
  const elements = Object.fromEntries(identifiers.map(id => [id, {
    id, value: '', textContent: '', listeners: {}, style: {},
    addEventListener(name, listener) { this.listeners[name] = listener; },
  }]));
  Object.assign(elements.scene, {value: 'gap'});
  Object.assign(elements.engine, {value: 'auto'});
  Object.assign(elements.count, {value: '128'});
  Object.assign(elements.temperature, {value: '3'});
  elements['mission-count'].value = '128';
  elements['workload-scope'].value = 'kernel';
  elements['budget-ms'].value = '20';
  elements.world.getContext = () => new Proxy({}, {get: () => () => {}});
  for (const backend of ['cpu', 'gpu']) elements[`budget-${backend}-canvas`].getContext = elements.world.getContext;
  let frame;
  globalThis.document = {hidden: false, querySelectorAll: () => Object.values(elements)};
  globalThis.requestAnimationFrame = callback => { frame = callback; };
  try {
    await import('../docs/site/demo/app.mjs');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(elements.backend.textContent, 'CPU / JavaScript');
    assert.equal(elements.run.disabled, false);
    assert.match(elements.status.textContent, /CPU fallback/);
    await elements.compare.listeners.click();
    assert.match(elements['cpu-median'].textContent, /ms$/);
    assert.equal(elements['gpu-median'].textContent, 'Unavailable');
    assert.equal(elements.speedup.textContent, '—');
    assert.equal(elements['export-comparison'].disabled, false);
    const sweep = elements['compare-workloads'].listeners.click();
    assert.equal(elements.compare.disabled, true);
    assert.equal(elements.run.disabled, true);
    elements['cancel-workloads'].listeners.click();
    await sweep;
    assert.equal(elements['export-workloads'].disabled, true);
    assert.equal(elements.compare.disabled, false);
    assert.match(elements['workload-status'].textContent, /Cancelled/);
    const budget = elements['compare-budget'].listeners.click();
    assert.equal(elements.compare.disabled, true);
    elements['cancel-budget'].listeners.click(); await budget;
    assert.equal(elements['export-budget'].disabled, true);
    assert.equal(elements.compare.disabled, false);
    assert.match(elements['budget-status'].textContent, /Cancelled/);
    elements.run.listeners.click();
    await frame();
    assert.equal(elements.time.textContent, '0.1 s');
    assert.equal(elements.run.textContent, 'Pause');
    elements.run.listeners.click();
    await frame();
    assert.equal(elements.time.textContent, '0.1 s');
    elements['goal-x'].valueAsNumber = 5;
    elements['goal-y'].valueAsNumber = 2;
    elements['set-goal'].listeners.click();
    assert.match(elements.status.textContent, /free space/);
    elements['goal-y'].valueAsNumber = 5;
    elements['set-goal'].listeners.click();
    assert.match(elements.status.textContent, /New goal set/);
    elements.reset.listeners.click();
    assert.equal(elements['export-comparison'].disabled, true);
    assert.equal(elements.time.textContent, '0.0 s');
    assert.equal(elements.run.textContent, 'Start');
    elements.scene.value = 'slalom';
    elements.scene.listeners.change();
    elements.run.listeners.click();
    for (let step = 0; step < 600 && elements.run.textContent === 'Pause'; step++) await frame();
    assert.match(elements.status.textContent, /Goal reached/);
  } finally {
    delete globalThis.document;
    delete globalThis.requestAnimationFrame;
  }
});

test('workload sweep uses four matched fixed snapshots, raw timing and an observed GPU win', async () => {
  const source = {state: [...scenes.gap.start], goal: [...scenes.gap.goal], obstacles: structuredClone(scenes.gap.obstacles)};
  const before = structuredClone(source);
  const destroyed = [];
  const report = await compareWorkloads(source, {createJob: async input => {
    assert.equal(input.controls.length, input.count * HORIZON * 2);
    assert.deepEqual(input.state, before.state);
    let clock = 0;
    return {run: options => compareMppi(input, {...options, warmups: 1, repetitions: 3, yieldControl: async () => {}, now: () => clock,
      cpuEvaluate: () => {clock += input.count < 512 ? 1 : 4; return new Float32Array(input.count).fill(1);},
      gpuEvaluate: () => {clock += 2; return new Float32Array(input.count).fill(1);}}), destroy: () => destroyed.push(input.count)};
  }});
  assert.deepEqual(source, before);
  assert.deepEqual(destroyed, WORKLOAD_COUNTS);
  assert.equal(report.firstTestedGpuWin, 512);
  for (const {count, report: measured} of report.results) {
    assert.equal(measured.cpu.samplesMs.length, 3);
    assert.equal(measured.accuracy.checkedCellComparisons, count * 3);
    assert.equal(measured.accuracy.matches, true);
  }
});

test('unavailable or mismatched workload outputs suppress crossover and cleanup runs on cancellation', async () => {
  const source = {state: scenes.open.start, goal: scenes.open.goal, obstacles: []};
  for (const mismatch of [false, true]) {
    const report = await compareWorkloads(source, {createJob: async input => ({
      run: options => compareMppi(input, {...options, warmups: 0, repetitions: 1, yieldControl: async () => {},
        cpuEvaluate: () => new Float32Array(input.count).fill(1),
        gpuEvaluate: mismatch ? () => new Float32Array(input.count).fill(2) : null}), destroy() {},
    })});
    assert.equal(report.firstTestedGpuWin, null);
    for (const entry of report.results) assert.equal(entry.report.speedup, null);
  }
  let cancelled = false;
  let destroyed = 0;
  await assert.rejects(compareWorkloads(source, {shouldCancel: () => cancelled, createJob: async () => {
    cancelled = true; return {destroy() {destroyed++;}};
  }}), {name: 'AbortError'});
  assert.equal(destroyed, 1);
});

test('three-seed mission quality reports actual guarded arrivals and complete paths', async () => {
  for (const sceneId of Object.keys(scenes)) {
    const report = await compareMissions({sceneId, count: 128, temperature: 3}, {yieldControl: async () => {}});
    assert.equal(report.summaries.cpu.trials, 3);
    assert.equal(report.summaries.cpu.reached, 3);
    assert.equal(report.summaries.gpu, null);
    assert.deepEqual(report.seeds, MISSION_SEEDS);
    for (const trial of report.results.cpu) {
      assert.equal(trial.status, 'reached');
      assert.equal(trial.path.length, trial.steps + 1);
      assert.ok(trial.minimumClearance > 0);
      assert.ok(trial.goalDistance < .3);
      assert.ok(trial.validRollouts <= trial.evaluatedRollouts);
    }
  }
});

test('paired mission trials use independent seeded planners, not shared state', async () => {
  const report = await compareMissions({sceneId: 'gap', count: 128, temperature: 3}, {gpuEvaluate: cpuCosts, yieldControl: async () => {}});
  assert.equal(report.summaries.gpu.reached, 3);
  for (const first of report.results.cpu) {
    const second = report.results.gpu.find(trial => trial.seed === first.seed);
    assert.deepEqual(second, first);
  }
});

test('mission safe stops, timeouts, bad costs and cancellation cannot masquerade as arrivals', async () => {
  const settings = {scene: scenes.open, count: 128, seed: 311};
  const stopped = await runMission(settings, {evaluate: () => new Float32Array(128).fill(COLLISION), yieldControl: async () => {}});
  assert.equal(stopped.status, 'safe_stop');
  assert.equal(stopped.steps, 0);
  const timeout = await runMission({...settings, maxSteps: 1}, {yieldControl: async () => {}});
  assert.equal(timeout.status, 'timeout');
  await assert.rejects(runMission(settings, {evaluate: () => new Float32Array(128).fill(NaN), yieldControl: async () => {}}), /Invalid mission rollout costs/);
  await assert.rejects(runMission(settings, {shouldCancel: () => true}), {name: 'AbortError'});
  await assert.rejects(runMission({...settings, count: 999}), /Invalid mission settings/);
  await assert.rejects(compareMissions({sceneId: 'unknown', count: 128, temperature: 3}), /Unknown mission scene/);
});

test('complete plan timing includes fresh sampling and softmin with identical seeded inputs', async () => {
  const input = {state: [1, 8, 0], goal: [9, 2], obstacles: [], count: 128, temperature: 3};
  const before = structuredClone(input);
  const controlsSeen = [];
  let clock = 0;
  const report = await compareMppiPlans(input, {warmups: 1, repetitions: 3, now: () => clock++, yieldControl: async () => {},
    cpuEvaluate: (...args) => {controlsSeen.push([...args[3]]); return cpuCosts(...args);}, gpuEvaluate: cpuCosts});
  assert.equal(report.schema, 'cudarobotics.mppi-plan-benchmark.v1');
  assert.deepEqual(input, before);
  assert.equal(report.cpu.medianMs, 3);
  assert.equal(report.gpu.medianMs, 3);
  assert.equal(report.speedup, 1);
  for (const backend of ['cpu', 'gpu']) {
    for (const stage of ['sampling', 'rollout', 'reduction']) assert.equal(report.stages[backend][stage].medianMs, 1);
    assert.equal(report.plans[backend].length, 3);
    assert.equal(report.plans[backend][0].average.length, 64);
  }
  assert.ok(report.plans.cpu[0].average.some(value => value < 0));
  for (const controls of controlsSeen) assert.deepEqual(controls, controlsSeen[0]);
  assert.deepEqual(report.plans.cpu, report.plans.gpu);
  assert.equal(report.accuracy.matches, true);
  assert.equal(report.commandAccuracy.matches, true);
  assert.equal(report.commandAccuracy.checkedPairs, 3);
});

test('matching costs within tolerance can still change weighted controls and suppress a plan ratio', async () => {
  const input = {state: scenes.gap.start, goal: scenes.gap.goal, obstacles: scenes.gap.obstacles, count: 128, temperature: .5};
  let clock = 0;
  const report = await compareMppiPlans(input, {warmups: 0, repetitions: 1, now: () => clock++, yieldControl: async () => {},
    cpuEvaluate: () => new Float32Array(128).fill(100),
    gpuEvaluate: () => {const costs = new Float32Array(128).fill(100); costs.fill(99.9, 0, 64); return costs;}});
  assert.equal(report.accuracy.matches, true);
  assert.equal(report.commandAccuracy.matches, false);
  assert.ok(report.commandAccuracy.maxControlError > .001);
  assert.equal(report.speedup, null);
});

test('complete plans classify invalid rollouts and do not invent GPU timings or timer ratios', async () => {
  const input = {state: scenes.open.start, goal: scenes.open.goal, obstacles: [], count: 128};
  const options = {warmups: 0, repetitions: 1, now: () => 0, yieldControl: async () => {}};
  const cpuOnly = await compareMppiPlans(input, options);
  assert.equal(cpuOnly.gpu, null);
  assert.equal(cpuOnly.stages.gpu, null);
  assert.equal(cpuOnly.plans.gpu, null);
  assert.equal(cpuOnly.commandAccuracy, null);
  assert.equal(cpuOnly.speedup, null);
  const blocked = await compareMppiPlans(input, {...options,
    cpuEvaluate: () => new Float32Array(128).fill(COLLISION), gpuEvaluate: () => new Float32Array(128).fill(COLLISION)});
  assert.equal(blocked.commandAccuracy.matches, true);
  assert.equal(blocked.plans.cpu[0].valid, 0);
  assert.equal(blocked.plans.cpu[0].velocity, 0);
  assert.equal(blocked.speedup, null);
  const mismatch = await compareMppiPlans(input, {...options,
    cpuEvaluate: () => new Float32Array(128).fill(COLLISION), gpuEvaluate: () => new Float32Array(128).fill(COLLISION - .125)});
  assert.equal(mismatch.accuracy.matches, false);
  assert.equal(mismatch.commandAccuracy.validCountMismatches, 1);
  assert.equal(mismatch.speedup, null);
});

test('plan cancellation, invalid output and settings discard the entire report', async () => {
  const input = {state: scenes.open.start, goal: scenes.open.goal, obstacles: [], count: 128};
  const options = {warmups: 0, repetitions: 1, yieldControl: async () => {}};
  await assert.rejects(compareMppiPlans(input, {...options, shouldCancel: () => true}), {name: 'AbortError'});
  await assert.rejects(compareMppiPlans(input, {...options, cpuEvaluate: () => new Float32Array(128).fill(NaN)}), /Invalid plan benchmark output/);
  await assert.rejects(compareMppiPlans({...input, temperature: 0}, options), /Invalid plan benchmark settings/);
  let clock = 5;
  await assert.rejects(compareMppiPlans(input, {...options, now: () => clock--}), /Invalid plan benchmark clock/);
});

test('plan sweeps export scope and suppress crossover when only the commands disagree', async () => {
  const source = {state: scenes.open.start, goal: scenes.open.goal, obstacles: [], measurement: 'plan', temperature: 4};
  const report = await compareWorkloads(source, {createJob: async input => {
    assert.equal(input.measurement, 'plan'); assert.equal(input.temperature, 4);
    return {run: async () => ({accuracy: {matches: true}, commandAccuracy: {matches: input.count !== 512}, speedup: input.count === 512 ? null : 2}), destroy() {}};
  }});
  assert.equal(report.measurement, 'plan');
  assert.equal(report.firstTestedGpuWin, null);
  assert.match(report.scope, /cold-start/);
  await assert.rejects(compareWorkloads({...source, measurement: 'unknown'}, {}), /Unknown workload measurement/);
});

function insightMeasurement() {
  const gpu = summarize([10, 20, 100]);
  return {cpu: summarize([40, 60, 200]), gpu, accuracy: {matches: true}, commandAccuracy: {matches: true}, stages: {
    cpu: {rollout: summarize([36, 54, 180])},
    gpu: {total: gpu, sampling: summarize([8, 1, 20]), rollout: summarize([1, 18, 79]), reduction: summarize([1, 1, 1])},
  }};
}

test('insights explain total versus rollout ratios and use summed raw stage shares, not median ratios', () => {
  const measured = insightMeasurement();
  const before = structuredClone(measured);
  const result = workloadInsight(measured, 512, 'plan');
  assert.equal(result.winner, 'gpu');
  assert.equal(result.ratio, 3);
  assert.equal(result.rolloutRatio, 3);
  assert.equal(result.gpuPathShares.rollout, 98 / 130);
  assert.notEqual(result.gpuPathShares.rollout, measured.stages.gpu.rollout.medianMs / measured.gpu.medianMs);
  assert.ok(Math.abs(Object.values(result.gpuPathShares).reduce((total, share) => total + share, 0) - 1) < 1e-12);
  assert.match(result.verdict, /GPU 3.00× faster by measured median/);
  assert.match(result.effect, /CPU sampling \+ softmin/);
  assert.deepEqual(measured, before);
});

test('insights retain CPU wins, exact ties and descriptive overlap warnings', () => {
  const measured = insightMeasurement();
  measured.cpu = summarize([5, 5, 5]); measured.gpu = summarize([10, 10, 10]);
  let result = workloadInsight(measured, 128, 'kernel');
  assert.equal(result.winner, 'cpu');
  assert.match(result.verdict, /CPU 2.00× faster/);
  assert.equal(result.gpuPathShares, null);
  assert.equal(result.rolloutRatio, null);
  assert.match(result.effect, /does not establish whole-plan speed/);
  measured.cpu = summarize([10, 10, 10]);
  result = workloadInsight(measured, 128, 'plan');
  assert.equal(result.winner, 'tie'); assert.match(result.verdict, /equal measured medians/);
  measured.cpu = summarize([4, 10, 16]); measured.gpu = summarize([3, 9, 17]);
  result = workloadInsight(measured, 128, 'plan');
  assert.equal(result.rangesOverlap, true);
  assert.match(result.effect, /not a statistically established winner/);
});

test('insights never infer a winner or GPU share from missing GPUs, mismatches or unresolved timers', () => {
  for (const variation of [
    measured => {measured.gpu = null;},
    measured => {measured.accuracy.matches = false;},
    measured => {measured.commandAccuracy.matches = false;},
    measured => {delete measured.commandAccuracy;},
    measured => {measured.gpu.medianMs = 0;},
    measured => {measured.cpu.medianMs = NaN;},
    measured => {measured.cpu.medianMs = 1e-310;},
  ]) {
    const measured = insightMeasurement(); variation(measured);
    const result = workloadInsight(measured, 512, 'plan');
    assert.notEqual(result.status, 'measured');
    assert.equal(result.winner, null); assert.equal(result.ratio, null); assert.equal(result.gpuPathShares, null);
  }
});

test('stage shares reject inconsistent, absent, non-finite and zero-time raw samples', () => {
  for (const variation of [
    measured => {measured.stages.gpu.sampling.samplesMs[0] = 999;},
    measured => {measured.stages.gpu.rollout.samplesMs.pop();},
    measured => {measured.stages.gpu.reduction.samplesMs[0] = NaN;},
    measured => {delete measured.stages;},
    measured => {for (const stage of Object.values(measured.stages.gpu)) stage.samplesMs = [0, 0, 0];},
  ]) {
    const measured = insightMeasurement(); variation(measured);
    assert.equal(workloadInsight(measured, 512, 'plan').gpuPathShares, null);
  }
});

test('sweep conclusions count only supported per-count winners and preserve exportable caveats', () => {
  const gpuWin = insightMeasurement();
  const cpuWin = insightMeasurement(); cpuWin.cpu = summarize([1, 1, 1]);
  const mismatch = insightMeasurement(); mismatch.commandAccuracy.matches = false;
  const insights = summarizeWorkloads({measurement: 'plan', results: [
    {count: 128, report: cpuWin}, {count: 512, report: gpuWin}, {count: 2048, report: mismatch},
  ]});
  assert.deepEqual(insights.gpuWins, [512]); assert.deepEqual(insights.cpuWins, [128]);
  assert.match(insights.conclusion, /Some workloads cannot support a conclusion/);
  assert.match(insights.shareMethod, /not isolated GPU kernel time or utilization/);
  assert.deepEqual(JSON.parse(JSON.stringify(insights)), insights);
  const absent = summarizeWorkloads({measurement: 'kernel', results: [{count: 128, report: {...cpuWin, gpu: null}}]});
  assert.equal(absent.conclusion, 'CPU measurements only. No GPU/CPU conclusion.');
});

function budgetCalibration(count, cpuMs, gpuMs = null) {
  return {count, report: {cpu: {medianMs: 1, p90Ms: cpuMs}, gpu: gpuMs === null ? null : {medianMs: 1, p90Ms: gpuMs},
    accuracy: gpuMs === null ? null : {matches: true}, commandAccuracy: gpuMs === null ? null : {matches: true}}};
}

test('budget selection uses P90, largest tested verified counts and no synthetic minimum', () => {
  const rows = [budgetCalibration(128, 4, 7), budgetCalibration(512, 12, 9), budgetCalibration(2048, 50, 15), budgetCalibration(8192, 200, 25)];
  let selected = selectBudgetCounts(rows, 20, true);
  assert.equal(selected.cpu.count, 512); assert.equal(selected.gpu.count, 2048);
  rows[2].report.commandAccuracy.matches = false;
  selected = selectBudgetCounts(rows, 20, true);
  assert.equal(selected.gpu.count, 512);
  assert.equal(selectBudgetCounts(rows, 1, true).cpu.count, null);
  assert.equal(selectBudgetCounts(rows, 20, false).gpu.count, null);
  assert.equal(selectBudgetCounts([budgetCalibration(128, 0), budgetCalibration(999, 1)], 20, false).cpu.count, null);
  for (const budget of [0, -1, NaN, 1001]) assert.throws(() => selectBudgetCounts(rows, budget, true), /Invalid planning budget/);
});

test('late mission plans are timed but discarded before any motion, including overrun samples', async () => {
  let clock = 0;
  const trial = await runMission({scene: scenes.gap, count: 128, seed: 311}, {budgetMs: 5, now: () => clock,
    yieldControl: async () => {}, evaluate: (...args) => {clock += 6; return cpuCosts(...args);}});
  assert.equal(trial.status, 'deadline_stop'); assert.equal(trial.steps, 0);
  assert.deepEqual(trial.path, [scenes.gap.start]); assert.equal(trial.pathLength, 0);
  assert.deepEqual(trial.planning.timing.samplesMs, [6]); assert.equal(trial.planning.deadlineMisses, 1);
  assert.ok(trial.validRollouts > 0);
});

test('plans exactly on budget advance guarded substeps and record path distance, not wall-time motion', async () => {
  let clock = 0;
  const trial = await runMission({scene: scenes.gap, count: 128, seed: 311, maxSteps: 1}, {budgetMs: 5, now: () => clock,
    yieldControl: async () => {}, evaluate: (...args) => {clock += 5; return cpuCosts(...args);}});
  assert.equal(trial.status, 'timeout'); assert.equal(trial.steps, 1); assert.equal(trial.planning.deadlineMisses, 0);
  assert.equal(trial.simulationSeconds, .1); assert.ok(trial.pathLength > 0); assert.ok(trial.minimumClearance > 0);
  assert.ok(trial.pathLength >= Math.hypot(trial.path[1][0] - trial.path[0][0], trial.path[1][1] - trial.path[0][1]) - 1e-12);
  clock = 10;
  await assert.rejects(runMission({scene: scenes.gap, count: 128, seed: 311}, {budgetMs: 5, now: () => clock--, yieldControl: async () => {}}), /Invalid mission benchmark clock/);
});

test('budget experiment calibrates fixed initial targets then independently runs three seeds at selected counts', async () => {
  let clock = 0;
  const calls = {cpu: [], gpu: []};
  const progress = [];
  const evaluate = backend => (...args) => {calls[backend].push(args[4]); clock += backend === 'cpu' ? 2 : 3; return cpuCosts(...args);};
  const report = await compareBudget({sceneId: 'slalom', temperature: 3, budgetMs: 10}, {
    maxSteps: 1, now: () => clock, yieldControl: async () => {}, cpuEvaluate: evaluate('cpu'), gpuEvaluate: evaluate('gpu'),
    calibrate: async (input, options) => {
      assert.deepEqual(input.state, scenes.slalom.start); assert.deepEqual(input.goal, scenes.slalom.waypoints[0]);
      options.onProgress({phase: 'measurement', backend: 'cpu', iteration: 1, total: 1});
      return budgetCalibration(input.count, input.count === 128 ? 4 : 40, input.count === 512 ? 5 : 40).report;
    }, onProgress: value => progress.push(value),
  });
  assert.equal(report.selections.cpu.count, 128); assert.equal(report.selections.gpu.count, 512);
  assert.deepEqual(calls.cpu, Array(6).fill(128)); assert.deepEqual(calls.gpu, Array(6).fill(512));
  assert.equal(report.summaries.cpu.timeouts, 3); assert.equal(report.summaries.gpu.timeouts, 3);
  assert.deepEqual(report.summaries.cpu.planning.samplesMs, [2, 2, 2]);
  assert.deepEqual(report.summaries.gpu.planning.samplesMs, [3, 3, 3]);
  assert.equal(report.summaries.cpu.deadlineMisses, 0);
  assert.ok(progress.some(value => value.phase === 'calibration' && value.calibrationPhase === 'measurement'));
  assert.deepEqual(report.seeds, [311, 312, 313]);
  assert.equal(JSON.parse(JSON.stringify(report)).schema, 'cudarobotics.mppi-budget.v1');
});

test('budget experiments distinguish no fitting count and absent GPUs from failed missions', async () => {
  for (const fits of [false, true]) {
    let clock = 0;
    const report = await compareBudget({sceneId: 'gap', budgetMs: 10}, {maxSteps: 1, yieldControl: async () => {}, now: () => clock,
      cpuEvaluate: (...args) => {assert.ok(fits); clock++; return cpuCosts(...args);},
      calibrate: async input => budgetCalibration(input.count, fits && input.count === 128 ? 5 : 30).report});
    assert.equal(report.selections.cpu.count, fits ? 128 : null);
    assert.equal(report.results.cpu === null, !fits); assert.equal(report.summaries.cpu === null, !fits);
    assert.equal(report.results.gpu, null); assert.equal(report.summaries.gpu, null);
  }
});

test('budget trials can still reach all goals; fitting calibration does not silently disable deadline enforcement', async () => {
  let clock = 0;
  const evaluate = (...args) => {clock++; return cpuCosts(...args);};
  const report = await compareBudget({sceneId: 'gap', budgetMs: 10}, {yieldControl: async () => {}, now: () => clock,
    cpuEvaluate: evaluate, gpuEvaluate: evaluate,
    calibrate: async input => budgetCalibration(input.count, input.count === 128 ? 5 : 30, input.count === 128 ? 5 : 30).report});
  assert.equal(report.summaries.cpu.reached, 3); assert.equal(report.summaries.gpu.reached, 3);
  assert.equal(report.summaries.cpu.deadlineStops, 0); assert.ok(report.summaries.cpu.meanPathLength > 0);
  let calls = 0;
  const late = await compareBudget({sceneId: 'gap', budgetMs: 10}, {yieldControl: async () => {}, now: () => clock,
    cpuEvaluate: (...args) => {calls++; clock += 11; return cpuCosts(...args);},
    calibrate: async input => budgetCalibration(input.count, input.count === 128 ? 5 : 30).report});
  assert.equal(late.summaries.cpu.deadlineStops, 3); assert.equal(late.summaries.cpu.deadlineMisses, 3);
  assert.equal(late.summaries.cpu.meanPathLength, 0); assert.equal(calls, 6);
});

test('budget calibration or mission cancellation never exports a partial experiment', async () => {
  let cancelled = false;
  await assert.rejects(compareBudget({sceneId: 'gap'}, {shouldCancel: () => cancelled,
    calibrate: async input => {cancelled = true; return budgetCalibration(input.count, 5).report;}}), {name: 'AbortError'});
  cancelled = false;
  let calls = 0;
  await assert.rejects(compareBudget({sceneId: 'gap'}, {shouldCancel: () => cancelled, yieldControl: async () => {},
    calibrate: async input => budgetCalibration(input.count, input.count === 128 ? 5 : 30).report,
    cpuEvaluate: (...args) => {calls++; if(calls === 2) cancelled = true; return cpuCosts(...args);}}), {name: 'AbortError'});
  await assert.rejects(compareBudget({sceneId: 'unknown'}), /Invalid budget experiment settings/);
});
