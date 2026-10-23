import test from 'node:test';
import assert from 'node:assert/strict';
import {distanceField, preset, plan, mapDocument} from '../docs/site/navigation/model.mjs';
import {distanceFieldEdt} from '../docs/site/navigation/edt.mjs';
import {SCALE_GRIDS} from '../docs/site/navigation/scaling.mjs';
import {compareImplementations} from '../docs/site/navigation/implementations.mjs';
import {GpuField, createPipelines} from '../docs/site/navigation/gpu.mjs';

function scenario() {
  const source = preset('empty');
  return mapDocument(source.occupancy, source.start, source.goal, source.radius, source.margin);
}
const fastOptions = {grid: SCALE_GRIDS[0], warmups: 0, repetitions: 1, yieldControl: async () => {}};

test('three-way trials balance all positions, exclude warmups and preserve the source', async () => {
  const input = scenario();
  const original = structuredClone(input);
  let clock = 0;
  const evaluate = duration => (occupancy, grid) => { clock += duration; return distanceFieldEdt(occupancy, grid); };
  const report = await compareImplementations(input, {...fastOptions, warmups: 3, repetitions: 15,
    now: () => clock, cpuEvaluate: evaluate(1), gpuReferenceEvaluate: evaluate(5), gpuEdtEvaluate: evaluate(2),
    yieldControl: async () => { input.start[0] = 15; input.occupancy.fill(1); },
    planRoute: (...args) => { clock += 3; return plan(...args); }});
  assert.deepEqual(report.source, original);
  assert.equal(report.orders.length, 18);
  for (const id of ['cpuEdt', 'gpuReference', 'gpuEdt']) {
    assert.equal(report.backends[id].total.samplesMs.length, 15);
    assert.equal(report.backends[id].routes.length, 15);
    for (let position = 0; position < 3; position++) assert.equal(report.orders.slice(3).filter(order => order[position] === id).length, 5);
  }
  assert.equal(report.backends.cpuEdt.total.medianMs, 4);
  assert.equal(report.backends.gpuReference.total.medianMs, 8);
  assert.equal(report.backends.gpuEdt.total.medianMs, 5);
  assert.equal(report.comparisons['gpuReference:gpuEdt'].totalRatio, 1.6);
  assert.equal(report.comparisons['gpuReference:gpuEdt'].fieldRatio, 2.5);
  for (const comparison of Object.values(report.comparisons)) {
    assert.equal(comparison.fieldAccuracy.matches, true);
    assert.equal(comparison.routeAccuracy.matches, true);
    assert.equal(comparison.fieldAccuracy.checkedCellComparisons, 64 * 48 * 15);
  }
});

test('missing GPUs and zero-resolution clocks never produce fabricated ratios', async () => {
  const cpuOnly = await compareImplementations(scenario(), fastOptions);
  assert.equal(cpuOnly.backends.gpuReference, null); assert.equal(cpuOnly.backends.gpuEdt, null);
  assert.ok(Object.values(cpuOnly.comparisons).every(value => value === null));
  const partial = await compareImplementations(scenario(), {...fastOptions, repetitions: 2,
    gpuEdtEvaluate: distanceFieldEdt, now: () => 0});
  assert.deepEqual(partial.orders, [['cpuEdt', 'gpuEdt'], ['gpuEdt', 'cpuEdt']]);
  assert.equal(partial.backends.gpuReference, null);
  assert.equal(partial.comparisons['cpuEdt:gpuEdt'].fieldAccuracy.matches, true);
  assert.equal(partial.comparisons['cpuEdt:gpuEdt'].totalRatio, null);
  assert.equal(partial.comparisons['cpuEdt:gpuEdt'].fieldRatio, null);
});

test('a bad field suppresses only affected pairings and still checks GPU versus GPU', async () => {
  const report = await compareImplementations(scenario(), {...fastOptions, gpuReferenceEvaluate: distanceFieldEdt,
    gpuEdtEvaluate: (occupancy, grid) => { const field = distanceFieldEdt(occupancy, grid); field[0] += .01; return field; }});
  assert.equal(report.comparisons['cpuEdt:gpuReference'].fieldAccuracy.matches, true);
  for (const key of ['cpuEdt:gpuEdt', 'gpuReference:gpuEdt']) {
    assert.equal(report.comparisons[key].fieldAccuracy.matches, false);
    assert.equal(report.comparisons[key].totalRatio, null);
    assert.equal(report.comparisons[key].fieldRatio, null);
  }
});

test('route disagreement suppresses ratios even when the fields match within tolerance', async () => {
  const report = await compareImplementations(scenario(), {...fastOptions, gpuReferenceEvaluate: distanceFieldEdt,
    gpuEdtEvaluate: (occupancy, grid) => { const field = distanceFieldEdt(occupancy, grid); field[0] += 1e-7; return field; },
    planRoute: (field, ...args) => field[0] > .125 ? {status: 'no_path', path: [], length: 0, minimumClearance: null}
      : plan(field, ...args)});
  assert.equal(report.comparisons['cpuEdt:gpuEdt'].fieldAccuracy.matches, true);
  assert.equal(report.comparisons['cpuEdt:gpuEdt'].routeAccuracy.matches, false);
  assert.equal(report.comparisons['cpuEdt:gpuEdt'].totalRatio, null);
});

test('cancellation, invalid data and unsupported shader sizes cannot save a result', async () => {
  let cancelled = false;
  await assert.rejects(compareImplementations(scenario(), {...fastOptions, shouldCancel: () => cancelled,
    onProgress: () => { cancelled = true; }}), {name: 'AbortError'});
  await assert.rejects(compareImplementations(scenario(), {...fastOptions, repetitions: 0}));
  await assert.rejects(compareImplementations(scenario(), {...fastOptions, cpuEvaluate: () => new Float32Array(1)}), /Invalid cpuEdt distance field/);
  await assert.rejects(compareImplementations(scenario(), {...fastOptions, planRoute: () => null}), /Invalid CPUEDT route/);
  let clock = 0;
  await assert.rejects(compareImplementations(scenario(), {...fastOptions, now: () => --clock}), /Invalid implementation benchmark clock/);
  await assert.rejects(createPipelines({}, '', SCALE_GRIDS[0], 'unknown'), /Unknown GPU/);
  await assert.rejects(createPipelines({}, '', {width: 257, height: 192}, 'edt'), /1 to 256/);
  await assert.rejects(GpuField.create(SCALE_GRIDS[0], 'unknown'), /Unknown GPU/);
  const data = preset('doorway');
  const blocked = await compareImplementations(mapDocument(data.occupancy, data.start, data.goal, .75, .12), {
    ...fastOptions, gpuReferenceEvaluate: distanceField, gpuEdtEvaluate: distanceFieldEdt,
  });
  assert.ok(Object.values(blocked.backends).every(backend => backend.routes[0].status === 'no_path'));
  assert.ok(Object.values(blocked.comparisons).every(comparison => comparison.routeAccuracy.matches));
});
