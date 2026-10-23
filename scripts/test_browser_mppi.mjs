import test from 'node:test';
import assert from 'node:assert/strict';
import {Planner, scenes, HORIZON, COLLISION, clearance, safeAdvance, cpuCosts, randomGenerator} from '../docs/site/demo/planner.mjs';
import {WORKLOAD_COUNTS, MISSION_SEEDS, compareWorkloads, runMission, compareMissions} from '../docs/site/demo/workloads.mjs';
import {compareMppi} from '../docs/site/demo/benchmark.mjs';

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
    'workload-status', 'workload-crossover', 'mission-cpu', 'mission-gpu',
    ...WORKLOAD_COUNTS.flatMap(count => ['cpu', 'gpu', 'ratio', 'accuracy', 'cpu-bar', 'gpu-bar'].map(field => `workload-${count}-${field}`))];
  const elements = Object.fromEntries(identifiers.map(id => [id, {
    id, value: '', textContent: '', listeners: {}, style: {},
    addEventListener(name, listener) { this.listeners[name] = listener; },
  }]));
  Object.assign(elements.scene, {value: 'gap'});
  Object.assign(elements.engine, {value: 'auto'});
  Object.assign(elements.count, {value: '128'});
  Object.assign(elements.temperature, {value: '3'});
  elements['mission-count'].value = '128';
  elements.world.getContext = () => new Proxy({}, {get: () => () => {}});
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
