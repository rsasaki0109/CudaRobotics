import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {GpuField, createPipelines} from '../docs/site/navigation/gpu.mjs';
import {WIDTH, HEIGHT, preset, distanceField, plan, mapDocument} from '../docs/site/navigation/model.mjs';
import {compareDistanceFields} from '../docs/site/navigation/benchmark.mjs';
import {SCALE_GRIDS, resampleMap} from '../docs/site/navigation/scaling.mjs';
import {compareReplanning} from '../docs/site/navigation/replanning.mjs';

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
    console.log(`${grid.width}×${grid.height}: specialized compute pipelines PASS`);
    if (compileOnly) continue;
    const scaledAdapter = await gpu.requestAdapter();
    const scaledDevice = await scaledAdapter.requestDevice();
    let scaledEvaluator;
    try {
      scaledEvaluator = new GpuField(scaledDevice, await createPipelines(scaledDevice, code, grid), grid);
      const comparison = await compareReplanning(resampleMap(document, grid), {grid,
        gpuEvaluate: snapshot => scaledEvaluator.evaluate(snapshot), warmups: 0, repetitions: 1, yieldControl: async () => {}});
      assert.equal(comparison.accuracy.matches, true);
      assert.equal(comparison.routeAccuracy.matches, true);
      assert.equal(comparison.accuracy.checkedCellComparisons, grid.width * grid.height);
      console.log(`${grid.width}×${grid.height}: all GPU cells and full route agreement PASS`);
    } finally { scaledEvaluator?.destroy(); scaledDevice.destroy(); }
  }
} finally { evaluator?.destroy(); device.destroy(); }
