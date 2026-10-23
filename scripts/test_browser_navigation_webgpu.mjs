import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {GpuField, createPipelines} from '../docs/site/navigation/gpu.mjs';
import {WIDTH, HEIGHT, preset, distanceField, plan, mapDocument} from '../docs/site/navigation/model.mjs';
import {compareDistanceFields} from '../docs/site/navigation/benchmark.mjs';
import {SCALE_GRIDS, resampleMap} from '../docs/site/navigation/scaling.mjs';
import {compareReplanning} from '../docs/site/navigation/replanning.mjs';
import {distanceFieldEdt} from '../docs/site/navigation/edt.mjs';
import {compareImplementations} from '../docs/site/navigation/implementations.mjs';

const {create, globals} = await import(process.env.WEBGPU_MODULE
  ? pathToFileURL(process.env.WEBGPU_MODULE).href : 'webgpu');
Object.assign(globalThis, globals);
const compileOnly = process.argv.includes('--compile-only');
const gpu = create(compileOnly ? ['backend=null'] : []);
const adapter = await gpu.requestAdapter();
assert.ok(adapter, 'A WebGPU adapter is required');
const device = await adapter.requestDevice();
let evaluator;
try {
  const code = await readFile(new URL('../docs/site/navigation/field.wgsl', import.meta.url), 'utf8');
  const edtCode = await readFile(new URL('../docs/site/navigation/field-edt.wgsl', import.meta.url), 'utf8');
  const pipelines = await createPipelines(device, code);
  console.log('Navigation distance-field compute pipelines: PASS');
  if (!compileOnly) {
    evaluator = new GpuField(device, pipelines);
    const maps = ['warehouse', 'doorway', 'empty'].map(name => ({name, ...preset(name)}));
    maps.push({name: 'occupied', occupancy: new Uint8Array(WIDTH * HEIGHT).fill(1)});
    const sparse = new Uint8Array(WIDTH * HEIGHT);
    for (let index = 97; index < sparse.length; index += 251) sparse[index] = 1;
    maps.push({name: 'sparse', occupancy: sparse});
    for (const data of maps) {
      const expected = distanceField(data.occupancy);
      const actual = await evaluator.evaluate(data.occupancy);
      actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-6, `${data.name}: cell ${index}`));
      if (data.start) {
        const route = plan(actual, data.start, data.goal, data.radius, data.margin);
        assert.equal(route.status, 'ready');
        assert.ok(route.minimumClearance >= route.required - 1e-6);
      }
      console.log(`${data.name}: all 12,288 GPU cells match CPU, route checks PASS`);
      if (data.name === 'warehouse') {
        const comparison = await compareDistanceFields(data.occupancy, {
          gpuEvaluate: snapshot => evaluator.evaluate(snapshot), warmups: 1, repetitions: 3,
          yieldControl: async () => {},
        });
        assert.equal(comparison.accuracy.matches, true);
        assert.equal(comparison.cpu.samplesMs.length, 3);
        assert.equal(comparison.gpu.samplesMs.length, 3);
        console.log('Real WebGPU / CPU comparison, warmups and output agreement: PASS');
      }
    }
  }
  const source = preset('warehouse');
  const document = mapDocument(source.occupancy, source.start, source.goal, source.radius, source.margin);
  for (const grid of SCALE_GRIDS) {
    await createPipelines(device, code, grid);
    await createPipelines(device, edtCode, grid, 'edt');
    console.log(`${grid.width}×${grid.height}: reference and exact EDT compute pipelines PASS`);
    if (compileOnly) continue;
    const scaledAdapter = await gpu.requestAdapter();
    const scaledDevice = await scaledAdapter.requestDevice();
    let scaledEvaluator;
    try {
      scaledEvaluator = new GpuField(scaledDevice, await createPipelines(scaledDevice, code, grid), grid);
      for (const cpuBaseline of ['matched', 'optimized']) {
        const comparison = await compareReplanning(resampleMap(document, grid), {grid, cpuBaseline,
          gpuEvaluate: snapshot => scaledEvaluator.evaluate(snapshot), warmups: 0, repetitions: 1, yieldControl: async () => {}});
        assert.equal(comparison.accuracy.matches, true);
        assert.equal(comparison.routeAccuracy.matches, true);
        assert.equal(comparison.accuracy.checkedCellComparisons, grid.width * grid.height);
        assert.equal(comparison.cpuBaseline.id, cpuBaseline);
        console.log(`${grid.width}×${grid.height} / ${cpuBaseline}: all GPU cells and full route agreement PASS`);
      }
    } finally { scaledEvaluator?.destroy(); scaledDevice.destroy(); }
    const edtAdapter = await gpu.requestAdapter();
    const edtDevice = await edtAdapter.requestDevice();
    let edtEvaluator;
    try {
      edtEvaluator = new GpuField(edtDevice, await createPipelines(edtDevice, edtCode, grid, 'edt'), grid);
      const count = grid.width * grid.height;
      const cases = [new Uint8Array(count), new Uint8Array(count).fill(1)];
      const corner = new Uint8Array(count); corner[0] = 1; cases.push(corner);
      const symmetric = new Uint8Array(count); symmetric[0] = 1; symmetric[count - 1] = 1; cases.push(symmetric);
      const rowOnly = new Uint8Array(count); rowOnly.fill(1, grid.width * 5, grid.width * 6); cases.push(rowOnly);
      const columnOnly = new Uint8Array(count);
      for (let row = 0; row < grid.height; row++) columnOnly[row * grid.width + 7] = 1;
      cases.push(columnOnly);
      for (const density of [.005, .1, .5]) {
        let seed = 311;
        cases.push(Uint8Array.from({length: count}, () => {
          seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
          return seed / 4294967296 < density ? 1 : 0;
        }));
      }
      for (const occupancy of cases) {
        const expected = distanceFieldEdt(occupancy, grid);
        const actual = await edtEvaluator.evaluate(occupancy);
        assert.equal(actual.findIndex((value, index) => Math.abs(value - expected[index]) > 1e-6), -1);
      }
      for (const name of ['warehouse', 'doorway', 'empty']) {
        const data = preset(name);
        const input = resampleMap(mapDocument(data.occupancy, data.start, data.goal, data.radius, data.margin), grid);
        const comparison = await compareReplanning(input, {grid, cpuBaseline: 'optimized',
          gpuEvaluate: snapshot => edtEvaluator.evaluate(snapshot), warmups: 0, repetitions: 1, yieldControl: async () => {}});
        assert.equal(comparison.accuracy.matches, true);
        assert.equal(comparison.routeAccuracy.matches, true);
      }
      console.log(`${grid.width}×${grid.height}: GPU EDT adversarial fields and preset routes PASS`);
      const referenceAdapter = await gpu.requestAdapter();
      const referenceDevice = await referenceAdapter.requestDevice();
      let referenceEvaluator;
      try {
        referenceEvaluator = new GpuField(referenceDevice, await createPipelines(referenceDevice, code, grid), grid);
        const report = await compareImplementations(document, {grid, warmups: 0, repetitions: 1, yieldControl: async () => {},
          gpuReferenceEvaluate: occupancy => referenceEvaluator.evaluate(occupancy),
          gpuEdtEvaluate: occupancy => edtEvaluator.evaluate(occupancy)});
        for (const comparison of Object.values(report.comparisons)) {
          assert.equal(comparison.fieldAccuracy.matches, true);
          assert.equal(comparison.routeAccuracy.matches, true);
        }
        console.log(`${grid.width}×${grid.height}: real three-way pipeline and route agreement PASS`);
      } finally { referenceEvaluator?.destroy(); referenceDevice.destroy(); }
    } finally { edtEvaluator?.destroy(); edtDevice.destroy(); }
  }
} finally { evaluator?.destroy(); device.destroy(); }
