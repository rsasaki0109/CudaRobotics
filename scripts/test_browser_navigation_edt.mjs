import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_GRID, distanceField, preset, mapDocument} from '../docs/site/navigation/model.mjs';
import {distanceFieldEdt} from '../docs/site/navigation/edt.mjs';
import {cpuBaseline} from '../docs/site/navigation/cpu-baseline.mjs';
import {compareDistanceFields} from '../docs/site/navigation/benchmark.mjs';
import {compareReplanning} from '../docs/site/navigation/replanning.mjs';
import {SCALE_GRIDS, resampleMap, compareMapSizes} from '../docs/site/navigation/scaling.mjs';

function verify(occupancy, grid, name) {
  const original = occupancy.slice();
  const expected = distanceField(occupancy, grid);
  const actual = distanceFieldEdt(occupancy, grid);
  assert.ok(actual instanceof Float32Array);
  assert.equal(actual.length, expected.length);
  assert.equal(actual.findIndex((value, index) => !Object.is(value, expected[index])), -1, `${name}: first unequal cell`);
  assert.deepEqual(occupancy, original);
  assert.notEqual(distanceFieldEdt(occupancy, grid).buffer, actual.buffer);
}

for (const grid of SCALE_GRIDS) {
  test(`${grid.width}×${grid.height}: EDT is bit-identical to brute force on adversarial and seeded maps`, () => {
    const count = grid.width * grid.height;
    verify(new Uint8Array(count), grid, 'empty rows and columns');
    verify(new Uint8Array(count).fill(1), grid, 'fully occupied');
    const corner = new Uint8Array(count); corner[0] = 1;
    verify(corner, grid, 'single corner');
    const symmetric = new Uint8Array(count); symmetric[0] = 1; symmetric[count - 1] = 1;
    verify(symmetric, grid, 'equidistant ties');
    const rowOnly = new Uint8Array(count);
    rowOnly.fill(1, grid.width * Math.floor(grid.height / 2), grid.width * (Math.floor(grid.height / 2) + 1));
    verify(rowOnly, grid, 'one occupied row');
    const columnOnly = new Uint8Array(count);
    for (let row = 0; row < grid.height; row++) columnOnly[row * grid.width + Math.floor(grid.width / 2)] = 1;
    verify(columnOnly, grid, 'one occupied column');
    for (const density of [.005, .1, .5]) {
      let seed = 311;
      const random = Uint8Array.from({length: count}, () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 4294967296 < density ? 1 : 0;
      });
      verify(random, grid, `seeded density ${density}`);
    }
    for (const name of ['warehouse', 'doorway']) {
      const source = preset(name);
      const scaled = resampleMap(mapDocument(source.occupancy, source.start, source.goal, source.radius, source.margin), grid);
      verify(Uint8Array.from(scaled.occupancy), grid, name);
    }
  });
}

test('EDT handles one-cell dimensions, invalid grids and isolated metadata', () => {
  for (const [width, height] of [[1, 1], [1, 9], [8, 1], [3, 7]]) {
    const grid = {width, height, resolution: .125};
    const occupancy = new Uint8Array(width * height);
    verify(occupancy, grid, 'empty tiny grid');
    occupancy[occupancy.length - 1] = 1;
    verify(occupancy, grid, 'tiny grid with one site');
  }
  assert.throws(() => distanceFieldEdt(new Uint8Array(1)));
  assert.throws(() => distanceFieldEdt(new Uint8Array(0), {width: 0, height: 1, resolution: .125}));
  assert.throws(() => distanceFieldEdt(new Uint8Array(1), {width: 1, height: 1, resolution: NaN}));
  assert.throws(() => cpuBaseline('unknown'));
  assert.throws(() => cpuBaseline('__proto__'));
  const description = cpuBaseline('optimized').description;
  description.name = 'changed';
  assert.equal(cpuBaseline('optimized').description.name, 'CPU linear-time EDT');
});

test('field, full-route and scale comparisons retain explicit algorithm identity', async () => {
  const source = preset('warehouse');
  const document = mapDocument(source.occupancy, source.start, source.goal, source.radius, source.margin);
  for (const mode of ['matched', 'optimized']) {
    const options = {cpuBaseline: mode, warmups: 0, repetitions: 1, yieldControl: async () => {}};
    const field = await compareDistanceFields(source.occupancy, {...options, gpuEvaluate: distanceField});
    assert.equal(field.cpuBaseline.id, mode);
    assert.equal(field.accuracy.matches, true);
    const route = await compareReplanning(document, {...options, gpuEvaluate: distanceField});
    assert.equal(route.cpuBaseline.id, mode);
    assert.equal(route.routeAccuracy.matches, true);
    assert.equal(route.routes.cpu[0].status, 'ready');
    const scales = await compareMapSizes(document, {...options, createGpu: async grid => ({
      evaluate: occupancy => distanceField(occupancy, grid), destroy() {},
    })});
    assert.equal(scales.cpuBaseline.id, mode);
    for (const entry of scales.results) {
      assert.equal(entry.report.cpuBaseline.id, mode);
      assert.equal(entry.report.accuracy.matches, true);
      assert.equal(entry.report.routeAccuracy.matches, true);
    }
  }
  assert.equal(cpuBaseline('matched').description.comparisonKind, 'same-algorithm parallelism');
  assert.equal(cpuBaseline('optimized').description.comparisonKind, 'different-algorithm practical baseline');
  assert.equal(cpuBaseline('optimized').description.fieldWorkComplexity, 'O(W×H)');
  assert.equal(cpuBaseline('matched').description.fieldWorkComplexity, 'O(W×H×(W+H))');
  const fallback = await compareReplanning(document, {cpuBaseline: 'optimized', warmups: 0, repetitions: 1, yieldControl: async () => {}});
  assert.equal(fallback.gpu, null); assert.equal(fallback.speedup, null);
  assert.equal(fallback.cpuBaseline.id, 'optimized');
  await assert.rejects(compareDistanceFields(source.occupancy, {cpuBaseline: 'unknown'}));
  await assert.rejects(compareReplanning(document, {cpuBaseline: 'unknown'}));
  await assert.rejects(compareMapSizes(document, {cpuBaseline: 'unknown'}));
  assert.equal(DEFAULT_GRID.width, 128);
});
