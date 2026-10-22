import test from 'node:test';
import assert from 'node:assert/strict';
import {compareDistanceFields, summarize} from '../docs/site/navigation/benchmark.mjs';
import {WIDTH, HEIGHT, preset, distanceField, plan, mapDocument} from '../docs/site/navigation/model.mjs';
import {placePallet} from '../docs/site/navigation/pallet.mjs';
import {compareReplanning} from '../docs/site/navigation/replanning.mjs';
import {SCALE_GRIDS, resampleMap, compareMapSizes} from '../docs/site/navigation/scaling.mjs';

const grid = () => new Uint8Array(WIDTH * HEIGHT);
const field = () => new Float32Array(WIDTH * HEIGHT).fill(1);

test('timing statistics retain samples and interpolate percentiles', () => {
  const samples = [5, 1, 3, 2, 4];
  const result = summarize(samples);
  assert.equal(result.medianMs, 3);
  assert.equal(result.p10Ms, 1.4);
  assert.equal(result.p90Ms, 4.6);
  assert.deepEqual(result.samplesMs, samples);
  assert.throws(() => summarize([]));
  assert.throws(() => summarize([NaN]));
  assert.throws(() => summarize([-1]));
});

test('comparison alternates paired order, excludes warmups and preserves its input snapshot', async () => {
  let clock = 0;
  const order = [];
  const source = grid();
  const evaluate = (name, duration) => snapshot => {
    order.push(name);
    assert.equal(snapshot[0], 0);
    clock += duration;
    return field();
  };
  const result = await compareDistanceFields(source, {
    cpuEvaluate: evaluate('cpu', 2), gpuEvaluate: evaluate('gpu', .5),
    warmups: 2, repetitions: 4, now: () => clock,
    yieldControl: async () => { source[0] = 1; },
  });
  assert.deepEqual(order, ['cpu', 'gpu', 'gpu', 'cpu', 'cpu', 'gpu', 'gpu', 'cpu', 'cpu', 'gpu', 'gpu', 'cpu']);
  assert.deepEqual(result.cpu.samplesMs, [2, 2, 2, 2]);
  assert.deepEqual(result.gpu.samplesMs, [.5, .5, .5, .5]);
  assert.equal(result.speedup, 4);
  assert.equal(result.accuracy.matches, true);
  assert.equal(result.accuracy.checkedCellComparisons, WIDTH * HEIGHT * 4);
});

test('CPU-only devices never receive synthetic GPU timings or a speedup', async () => {
  const result = await compareDistanceFields(grid(), {cpuEvaluate: field, warmups: 1, repetitions: 2,
    yieldControl: async () => {}});
  assert.equal(result.cpu.samplesMs.length, 2);
  assert.equal(result.gpu, null);
  assert.equal(result.accuracy, null);
  assert.equal(result.speedup, null);
});

test('mismatched outputs suppress speedup, even when GPU timings are faster', async () => {
  let clock = 0;
  const result = await compareDistanceFields(grid(), {
    cpuEvaluate: () => { clock += 2; return field(); },
    gpuEvaluate: () => { clock += .5; const output = field(); output[7] += .00001; return output; },
    warmups: 0, repetitions: 2, now: () => clock, yieldControl: async () => {},
  });
  assert.equal(result.accuracy.matches, false);
  assert.equal(result.accuracy.mismatchedCellComparisons, 2);
  assert.ok(result.accuracy.maxAbsoluteError > 1e-6);
  assert.equal(result.speedup, null);
});

test('cancelled or invalid runs cannot produce completed reports', async () => {
  let cancelled = false;
  await assert.rejects(compareDistanceFields(grid(), {cpuEvaluate: field, warmups: 0, repetitions: 2,
    shouldCancel: () => cancelled, yieldControl: async () => { cancelled = true; }}), {name: 'AbortError'});
  cancelled = false;
  await assert.rejects(compareDistanceFields(grid(), {cpuEvaluate: field,
    gpuEvaluate: async () => { cancelled = true; return field(); }, warmups: 0, repetitions: 1,
    shouldCancel: () => cancelled, yieldControl: async () => {}}), {name: 'AbortError'});
  await assert.rejects(compareDistanceFields(grid(), {cpuEvaluate: () => new Float32Array(1),
    warmups: 0, repetitions: 1, yieldControl: async () => {}}), /Invalid CPU distance field/);
  await assert.rejects(compareDistanceFields(grid(), {repetitions: 0}), /Invalid benchmark settings/);
});

test('zero-resolution timers never produce infinite speedup claims', async () => {
  const result = await compareDistanceFields(grid(), {cpuEvaluate: field, gpuEvaluate: field,
    warmups: 0, repetitions: 1, now: () => 0, yieldControl: async () => {}});
  assert.equal(result.accuracy.matches, true);
  assert.equal(result.speedup, null);
});

function palletDocument(name = 'empty') {
  const data = preset(name);
  const before = plan(distanceField(data.occupancy), data.start, data.goal, data.radius, data.margin);
  const pallet = placePallet(data.occupancy, before.path, data.start, data.goal, before.required);
  return mapDocument(pallet.occupancy, data.start, data.goal, data.radius, data.margin);
}

test('full replanning measures paired field, CPU routing and total time separately', async () => {
  const input = palletDocument();
  const expectedStart = [...input.start];
  const expectedMap = [...input.occupancy];
  let clock = 0;
  const order = [];
  const report = await compareReplanning(input, {warmups: 1, repetitions: 3, now: () => clock,
    yieldControl: async () => { input.start[0] = 15; input.occupancy.fill(1); },
    cpuEvaluate: occupancy => { order.push('cpu'); assert.deepEqual(Array.from(occupancy), expectedMap); clock += 4; return distanceField(occupancy); },
    gpuEvaluate: occupancy => { order.push('gpu'); clock += 1; return distanceField(occupancy); },
    planRoute: (...args) => { clock += 3; return plan(...args); }});
  assert.deepEqual(order, ['cpu', 'gpu', 'gpu', 'cpu', 'cpu', 'gpu', 'gpu', 'cpu']);
  assert.deepEqual(report.cpu.samplesMs, [7, 7, 7]);
  assert.deepEqual(report.gpu.samplesMs, [4, 4, 4]);
  assert.equal(report.stages.cpu.field.medianMs, 4);
  assert.equal(report.stages.gpu.field.medianMs, 1);
  assert.equal(report.stages.cpu.route.medianMs, 3);
  assert.equal(report.stages.gpu.route.medianMs, 3);
  assert.equal(report.fieldSpeedup, 4);
  assert.equal(report.speedup, 1.75);
  assert.equal(report.routeAccuracy.matches, true);
  assert.equal(report.routeAccuracy.checkedPairs, 3);
  assert.equal(report.routes.cpu.length, 3);
  assert.deepEqual(report.routes.gpu[0].path[0], expectedStart);
  assert.deepEqual(report.input.start, expectedStart);
});

test('blocked replans are matched classified results, not successful routes', async () => {
  const report = await compareReplanning(palletDocument('doorway'), {gpuEvaluate: distanceField,
    warmups: 0, repetitions: 2, yieldControl: async () => {}});
  assert.equal(report.routeAccuracy.matches, true);
  assert.ok(report.routes.cpu.every(route => route.status === 'no_path' && route.path.length === 0));
  assert.ok(report.routes.gpu.every(route => route.minimumClearance === null));
});

test('full replanning CPU fallback does not fabricate GPU stages, routes or speedup', async () => {
  const report = await compareReplanning(palletDocument(), {warmups: 0, repetitions: 2, yieldControl: async () => {}});
  assert.equal(report.gpu, null);
  assert.equal(report.stages.gpu, null);
  assert.equal(report.routes.gpu, null);
  assert.equal(report.routeAccuracy, null);
  assert.equal(report.speedup, null);
  assert.equal(report.fieldSpeedup, null);
});

test('a route mismatch suppresses speedup even with identical fields', async () => {
  let calls = 0;
  const report = await compareReplanning(palletDocument(), {gpuEvaluate: distanceField, warmups: 0,
    repetitions: 2, yieldControl: async () => {}, planRoute: (...args) => {
      const result = plan(...args);
      if (++calls === 4) result.path[1][0] += .001;
      return result;
    }});
  assert.equal(report.accuracy.matches, true);
  assert.equal(report.routeAccuracy.matches, false);
  assert.equal(report.routeAccuracy.mismatchedPairs, 1);
  assert.equal(report.routeAccuracy.pathMismatches, 1);
  assert.equal(report.speedup, null);
  assert.equal(report.fieldSpeedup, null);
});

test('field mismatches, cancellation, invalid routes and invalid maps cannot claim a full-replan gain', async () => {
  const input = palletDocument();
  const mismatch = await compareReplanning(input, {gpuEvaluate: occupancy => {
    const field = distanceField(occupancy); field[0] += .01; return field;
  }, warmups: 0, repetitions: 1, yieldControl: async () => {}});
  assert.equal(mismatch.accuracy.matches, false);
  assert.equal(mismatch.speedup, null);
  await assert.rejects(compareReplanning(input, {shouldCancel: () => true}), {name: 'AbortError'});
  await assert.rejects(compareReplanning(input, {planRoute: () => ({status: 'ready', length: NaN, path: []}),
    warmups: 0, repetitions: 1, yieldControl: async () => {}}), /Invalid CPU route/);
  await assert.rejects(compareReplanning({...input, start: [NaN, 1]}));
  const zeroClock = await compareReplanning(input, {gpuEvaluate: distanceField, now: () => 0,
    warmups: 0, repetitions: 1, yieldControl: async () => {}});
  assert.equal(zeroClock.speedup, null);
  assert.equal(zeroClock.fieldSpeedup, null);
});

test('route classification, waypoint count and clearance mismatches are checked independently', async () => {
  for (const mutate of [route => ({...route, status: 'no_path', path: [], length: 0, minimumClearance: null}),
    route => ({...route, path: [...route.path, [...route.path.at(-1)]]}),
    route => ({...route, minimumClearance: route.minimumClearance + .001})]) {
    let calls = 0;
    const report = await compareReplanning(palletDocument(), {gpuEvaluate: distanceField, warmups: 0,
      repetitions: 1, yieldControl: async () => {}, planRoute: (...args) => {
        const route = plan(...args);
        return ++calls === 2 ? mutate(route) : route;
      }});
    assert.equal(report.accuracy.matches, true);
    assert.equal(report.routeAccuracy.matches, false);
    assert.equal(report.routeAccuracy.mismatchedPairs, 1);
    assert.equal(report.speedup, null);
    assert.equal(report.fieldSpeedup, null);
  }
});

test('scaled maps retain world coordinates and conservatively resample occupancy', () => {
  const data = preset('empty');
  data.occupancy[33 * WIDTH + 41] = 1;
  const source = mapDocument(data.occupancy, data.start, data.goal, data.radius, data.margin);
  for (const grid of SCALE_GRIDS) {
    const scaled = resampleMap(source, grid);
    assert.equal(scaled.width * scaled.resolution, 16);
    assert.equal(scaled.height * scaled.resolution, 12);
    assert.equal(scaled.occupancy.length, grid.width * grid.height);
    assert.deepEqual(scaled.start, source.start);
    assert.deepEqual(scaled.goal, source.goal);
    assert.equal(scaled.radius, source.radius);
    assert.equal(scaled.margin, source.margin);
    assert.equal(scaled.occupancy.reduce((sum, value) => sum + value, 0), grid.width === 256 ? 4 : 1);
    const field = distanceField(Uint8Array.from(scaled.occupancy), grid);
    assert.equal(field.length, grid.width * grid.height);
    assert.equal(field[0], grid.resolution / 2);
    assert.equal(field[Math.floor(33 * grid.height / HEIGHT) * grid.width + Math.floor(41 * grid.width / WIDTH)], 0);
    if (grid.width === WIDTH) assert.deepEqual(scaled.occupancy, source.occupancy);
  }
  assert.equal(source.occupancy.reduce((sum, value) => sum + value, 0), 1);
  assert.throws(() => resampleMap(source, {width: 1024, height: 768, resolution: .015625}));
});

test('each supported size plans in metres, compares only matching dimensions and disposes GPU resources', async () => {
  const created = [];
  const destroyed = [];
  const source = palletDocument();
  const original = structuredClone(source);
  const report = await compareMapSizes(source, {warmups: 0, repetitions: 1, yieldControl: async () => {},
    createGpu: async grid => {
      created.push(grid.width);
      source.start[0] = 15; source.occupancy.fill(1);
      return {evaluate: occupancy => {
        assert.equal(occupancy.length, grid.width * grid.height);
        return distanceField(occupancy, grid);
      }, destroy: () => destroyed.push(grid.width)};
    }});
  assert.deepEqual(created, [64, 128, 256]);
  assert.deepEqual(destroyed, created);
  assert.deepEqual(report.source, original);
  for (const entry of report.results) {
    assert.equal(entry.report.accuracy.matches, true);
    assert.equal(entry.report.routeAccuracy.matches, true);
    assert.equal(entry.report.accuracy.checkedCellComparisons, entry.grid.width * entry.grid.height);
    assert.deepEqual(entry.report.input.start, original.start);
    assert.equal(entry.report.routes.cpu[0].status, 'ready');
    assert.ok(entry.report.routes.cpu[0].minimumClearance >= original.radius + original.margin);
    assert.deepEqual(entry.report.routes.cpu[0].path[0], original.start);
    assert.deepEqual(entry.report.routes.cpu[0].path.at(-1), original.goal);
  }
});

test('scale runs label absent GPUs and discard cancelled or invalid runs', async () => {
  const source = palletDocument();
  const report = await compareMapSizes(source, {grids: [SCALE_GRIDS[0]], warmups: 0, repetitions: 1,
    yieldControl: async () => {}, createGpu: async () => { throw new Error('WebGPU unavailable'); }});
  assert.equal(report.results[0].report.gpu, null);
  assert.equal(report.results[0].report.speedup, null);
  assert.equal(report.results[0].report.gpuUnavailableReason, 'WebGPU unavailable');
  let cancelled = false;
  let disposed = 0;
  await assert.rejects(compareMapSizes(source, {shouldCancel: () => cancelled,
    createGpu: async grid => { cancelled = true; return {evaluate: input => distanceField(input, grid), destroy: () => disposed++}; }}), {name: 'AbortError'});
  assert.equal(disposed, 1);
  await assert.rejects(compareMapSizes(source, {grids: []}));
  await assert.rejects(compareMapSizes(source, {grids: [SCALE_GRIDS[0], SCALE_GRIDS[0]]}));
  disposed = 0;
  await assert.rejects(compareMapSizes(source, {grids: [SCALE_GRIDS[0]], warmups: 0, repetitions: 1,
    yieldControl: async () => {}, createGpu: async () => ({evaluate() { throw new Error('Device lost'); }, destroy: () => disposed++})}), /Device lost/);
  assert.equal(disposed, 1);
});
