import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker as NodeWorker} from 'node:worker_threads';
import {runComparison} from '../docs/site/assets/benchmark.mjs';
import {evaluateKernel} from '../docs/site/assets/kernels.mjs';
import {createCpuKernel} from '../docs/site/assets/cpu-kernel.mjs';
import {mppiSnapshot, compareMppi} from '../docs/site/demo/benchmark.mjs';
import {particleSnapshot, compareParticles} from '../docs/site/particles/benchmark.mjs';
import {scan} from '../docs/site/particles/model.mjs';

test('MPPI compares identical seeded controls and collision classifications', async () => {
  const input = mppiSnapshot([1, 5, 0], [9, 5], [[4.8, 0, 5.2, 4]], 128);
  assert.deepEqual(input.controls, mppiSnapshot(input.state, input.goal, input.obstacles, 128).controls);
  const report = await compareMppi(input, {gpuEvaluate: () => evaluateKernel(input), warmups: 0, repetitions: 2, yieldControl: async () => {}});
  assert.equal(report.accuracy.matches, true);
  assert.equal(report.accuracy.checkedCellComparisons, 256);
  assert.equal(report.input.horizon, 32);
  const mismatch = await runComparison({cpuEvaluate: () => [1e6], gpuEvaluate: () => [999999], length: 1,
    absoluteTolerance: .001, relativeTolerance: .001, classify: value => value >= 1e6,
    warmups: 0, repetitions: 1, yieldControl: async () => {}});
  assert.equal(mismatch.accuracy.classificationMismatches, 1);
  assert.equal(mismatch.speedup, null);
});

test('particles use the requested matched count, not the 512-particle live fallback', async () => {
  const input = particleSnapshot(4096, scan([2, 1.5, 0]), .3);
  assert.equal(input.poses.length, 4096 * 4);
  const report = await compareParticles(input, {gpuEvaluate: () => evaluateKernel(input), warmups: 0,
    repetitions: 1, yieldControl: async () => {}});
  assert.equal(report.accuracy.matches, true);
  assert.equal(report.accuracy.checkedCellComparisons, 4096);
  assert.equal(report.gpuReadbackBytes, 4096 * 4);
  assert.match(report.scope, /scoring only/);
});

test('real CPU worker protocol returns the same kernels and main-thread fallback is explicit', async () => {
  class BrowserWorker {
    constructor(url) {
      this.worker = new NodeWorker(`const {parentPort} = require('node:worker_threads');
        globalThis.self = {postMessage: (data, transfer) => parentPort.postMessage(data, transfer)};
        (async () => { await import(${JSON.stringify(url.href)}); parentPort.on('message', data => self.onmessage({data})); })();`, {eval: true});
      this.worker.on('message', data => this.onmessage?.({data}));
      this.worker.on('error', () => this.onerror?.());
    }
    postMessage(data) { this.worker.postMessage(data); }
    terminate() { this.worker.terminate(); }
  }
  const original = globalThis.Worker;
  globalThis.Worker = BrowserWorker;
  try {
    for (const input of [mppiSnapshot([1, 5, 0], [9, 5], [], 65), particleSnapshot(257, scan([2, 1.5, 0]), .3)]) {
      const cpu = await createCpuKernel(input);
      try {
        assert.match(cpu.mode, /single JavaScript worker/);
        assert.deepEqual(await cpu.evaluate(), evaluateKernel(input));
        assert.deepEqual(await cpu.evaluate(), evaluateKernel(input));
      } finally { cpu.destroy(); }
    }
    globalThis.Worker = class { constructor() { throw new Error('CSP denied'); } };
    const input = particleSnapshot(1, scan([2, 1.5, 0]), .3);
    const fallback = await createCpuKernel(input);
    assert.match(fallback.mode, /main-thread/);
    assert.deepEqual(fallback.evaluate(), evaluateKernel(input));
  } finally { if (original === undefined) delete globalThis.Worker; else globalThis.Worker = original; }
});

test('failed and stopped CPU workers reject future requests instead of hanging', async () => {
  const original = globalThis.Worker;
  let worker;
  globalThis.Worker = class {
    constructor() { worker = this; }
    postMessage(message) {
      if (message.type === 'init') queueMicrotask(() => this.onmessage({data: {ready: true}}));
    }
    terminate() {}
  };
  try {
    const cpu = await createCpuKernel({kind: 'unused'});
    const pending = cpu.evaluate();
    await assert.rejects(cpu.evaluate(), /already in progress/);
    worker.onerror();
    await assert.rejects(pending, /CPU worker failed/);
    await assert.rejects(cpu.evaluate(), /CPU worker failed/);
    cpu.destroy();
    await assert.rejects(cpu.evaluate(), {name: 'AbortError'});
  } finally { if (original === undefined) delete globalThis.Worker; else globalThis.Worker = original; }
});
