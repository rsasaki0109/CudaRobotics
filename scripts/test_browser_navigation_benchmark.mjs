import test from 'node:test';
import assert from 'node:assert/strict';
import {compareDistanceFields, summarize} from '../docs/site/navigation/benchmark.mjs';
import {WIDTH, HEIGHT} from '../docs/site/navigation/model.mjs';

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
