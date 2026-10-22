import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {createPipelines, GpuFilter} from '../docs/site/particles/gpu.mjs';
import {likelihood, move, scan} from '../docs/site/particles/model.mjs';

const {create, globals} = await import(process.env.WEBGPU_MODULE
  ? pathToFileURL(process.env.WEBGPU_MODULE).href : 'webgpu');
Object.assign(globalThis, globals);
const compileOnly = process.argv.includes('--compile-only');
const gpu = create(compileOnly ? ['backend=null'] : []);
const codes = await Promise.all(['filter.wgsl', 'draw.wgsl'].map(name => readFile(
  new URL(`../docs/site/particles/${name}`, import.meta.url), 'utf8')));

for (const count of compileOnly ? [1] : [1, 257, 4096, 65536]) {
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter, 'A WebGPU adapter is required');
  const device = await adapter.requestDevice();
  let filter;
  let texture;
  try {
    const pipelines = await createPipelines(device, ...codes, 'rgba8unorm');
    if (compileOnly) { console.log('Particle compute + render pipeline compilation: PASS'); continue; }
    filter = new GpuFilter(device, pipelines, count);
    texture = device.createTexture({size: [256, 192], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC});
    const truth = [2, 1.5, 0];
    const ranges = scan(truth);
    const result = await filter.step(ranges, [0, 0], .3, true, .03, texture.createView());
    assert.ok(result.every(Number.isFinite));
    assert.ok(result[3] > 0 && result[3] <= count + .01);
    if (count === 65536) {
      const pixels = device.createBuffer({size: 256 * 192 * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      try {
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({texture}, {buffer: pixels, bytesPerRow: 1024}, [256, 192]);
        device.queue.submit([encoder.finish()]);
        await pixels.mapAsync(GPUMapMode.READ);
        const image = new Uint8Array(pixels.getMappedRange());
        assert.ok(image.some((value, index) => index % 4 === 1 && value > 0), 'GPU renderer must produce visible green particles');
      } finally { pixels.destroy(); }
    }
    const scored = await filter.snapshot();
    let total = 0;
    let squared = 0;
    let meanX = 0;
    let meanY = 0;
    const candidates = new Set();
    for (let index = 0; index < count; index++) {
      const pose = scored.slice(index * 4, index * 4 + 4);
      candidates.add(pose.join(','));
      const reference = likelihood(pose, ranges, .3);
      assert.ok(Math.abs(reference - pose[3]) < .001 * Math.max(.1, reference),
        `likelihood mismatch at ${index}: pose=${pose}, CPU=${reference}`);
      total += pose[3]; squared += pose[3] ** 2;
      meanX += pose[0] * pose[3]; meanY += pose[1] * pose[3];
    }
    assert.ok(Math.abs(result[0] - meanX / total) < .0001);
    assert.ok(Math.abs(result[1] - meanY / total) < .0001);
    assert.ok(Math.abs(result[3] - total * total / squared) < .01);
    const resampled = await filter.snapshot(filter.states[filter.frame % 2]);
    for (let index = 0; index < count; index++) {
      assert.ok(candidates.has(resampled.slice(index * 4, index * 4 + 4).join(',')), 'Resampling must select an evaluated pose');
    }
    if (count === 65536) {
      let estimate;
      for (let step = 0; step < 40; step++) estimate = await filter.step(ranges, [0, 0], .3);
      assert.ok(Math.hypot(estimate[0] - truth[0], estimate[1] - truth[1]) < .5, `GPU estimate: ${estimate}`);
      const teleported = [13.8, 10.7, Math.PI];
      for (let step = 0; step < 80; step++) estimate = await filter.step(scan(teleported), [0, 0], .3);
      assert.ok(Math.hypot(estimate[0] - teleported[0], estimate[1] - teleported[1]) < .5, `GPU recovery: ${estimate}`);
      let moving = teleported;
      for (let step = 0; step < 80; step++) {
        moving = move(moving, .06, 0);
        estimate = await filter.step(scan(moving), [.06, 0], .3);
        assert.ok(Math.hypot(estimate[0] - moving[0], estimate[1] - moving[1]) < .5, `GPU tracking: ${estimate}`);
      }
      moving = [2, 1.5, 0];
      for (let step = 0; step < 120; step++) {
        moving = move(moving, .085, 0);
        estimate = await filter.step(scan(moving), [.085, 0], .3, step === 0);
        if (step > 40) assert.ok(Math.hypot(estimate[0] - moving[0], estimate[1] - moving[1]) < .5,
          `Global acquisition while moving: ${estimate}`);
      }
    }
    console.log(`${count} particles: likelihood parity, reductions, resampling and draw PASS`);
  } finally { texture?.destroy(); filter?.destroy(); device.destroy(); }
}
