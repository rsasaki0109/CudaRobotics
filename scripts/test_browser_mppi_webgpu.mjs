import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {GpuCosts} from '../docs/site/demo/gpu.mjs';
import {Planner, scenes, cpuCosts, safeAdvance, COLLISION} from '../docs/site/demo/planner.mjs';
import {mppiSnapshot, compareMppi} from '../docs/site/demo/benchmark.mjs';

const {create, globals} = await import(process.env.WEBGPU_MODULE
  ? pathToFileURL(process.env.WEBGPU_MODULE).href : 'webgpu');
Object.assign(globalThis, globals);
const compileOnly = process.argv.includes('--compile-only');
const gpu = create(compileOnly ? ['backend=null'] : []);
const adapter = await gpu.requestAdapter();
assert.ok(adapter, 'A WebGPU adapter is required; use --compile-only for shader validation without a GPU');
const device = await adapter.requestDevice();
let evaluator;
try {
  const module = device.createShaderModule({code: await readFile(
    new URL('../docs/site/demo/rollout.wgsl', import.meta.url), 'utf8')});
  const diagnostics = await module.getCompilationInfo();
  assert.deepEqual(diagnostics.messages.filter(message => message.type === 'error')
    .map(message => message.message), []);
  const pipeline = await device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}});
  console.log('WGSL compilation and compute pipeline: PASS');
  if (!compileOnly) {
    evaluator = new GpuCosts(device, pipeline);
    await evaluator.checkReference();
    const input = mppiSnapshot(scenes.gap.start, scenes.gap.goal, scenes.gap.obstacles, 512);
    const comparison = await compareMppi(input, {warmups: 1, repetitions: 3, yieldControl: async () => {},
      gpuEvaluate: () => evaluator.evaluate(input.state, input.goal, input.obstacles, input.controls, input.count)});
    assert.equal(comparison.accuracy.matches, true);
    console.log('Matched MPPI comparison and collision classification: PASS');
    for (const count of [1, 65, 128, 512, 2048, 128]) {
      for (const scene of Object.values(scenes)) {
        const controls = new Planner().sample(scene.start, scene.goal, count);
        const expected = cpuCosts(scene.start, scene.goal, scene.obstacles, controls, count);
        const actual = await evaluator.evaluate(scene.start, scene.goal, scene.obstacles, controls, count);
        assert.equal(actual.length, count);
        expected.forEach((cost, index) => {
          assert.equal(actual[index] >= COLLISION, cost >= COLLISION);
          assert.ok(Math.abs(cost - actual[index]) <= 0.001 * Math.max(1, cost));
        });
      }
    }
    console.log('GPU / CPU cost parity, collision classification and buffer resizing: PASS');
    for (const [name, scene] of Object.entries(scenes)) {
      const planner = new Planner();
      let state = [...scene.start];
      const waypoints = [...(scene.waypoints ?? [])];
      let reached = false;
      for (let step = 0; step < 600; step++) {
        while (waypoints.length && Math.hypot(state[0] - waypoints[0][0], state[1] - waypoints[0][1]) < 0.65) waypoints.shift();
        const target = waypoints[0] ?? scene.goal;
        const controls = planner.sample(state, target, 128);
        const costs = await evaluator.evaluate(state, target, scene.obstacles, controls, 128);
        const result = planner.update(costs, controls, 3);
        assert.ok(result.valid > 0, `${name}: no valid rollout at step ${step}`);
        const next = safeAdvance(state, result.velocity, result.angular, scene.obstacles);
        assert.ok(next, `${name}: collision at step ${step}`);
        state = next;
        if (Math.hypot(state[0] - scene.goal[0], state[1] - scene.goal[1]) < 0.3) {
          reached = true;
          break;
        }
      }
      assert.ok(reached, `${name}: goal not reached`);
      console.log(`WebGPU closed-loop ${name}: PASS`);
    }
  }
} finally {
  evaluator?.destroy();
  device.destroy();
}
