import test from 'node:test';
import assert from 'node:assert/strict';
import {BEAMS, CpuFilter, free, likelihood, move, randomGenerator, randomPose, scan, surfaceDistance} from '../docs/site/particles/model.mjs';

test('warehouse scan and surface distances agree with known geometry', () => {
  const ranges = scan([2, 1.5, 0]);
  assert.equal(ranges.length, BEAMS);
  assert.ok(ranges.every(value => value > 0 && value < 21));
  assert.equal(ranges[0], 14);
  assert.equal(ranges[8], 10.5);
  assert.equal(ranges[16], 2);
  assert.equal(ranges[24], 1.5);
  assert.equal(surfaceDistance(5, 3), 0);
  assert.equal(surfaceDistance(5, 4), 1);
  assert.equal(surfaceDistance(3, 4), 1);
  assert.equal(free(5, 4), false);
  assert.equal(free(0, 4), false);
});

test('likelihood rewards the measured pose, not an unrelated hypothesis', () => {
  const pose = [2, 1.5, 0];
  const ranges = scan(pose);
  assert.ok(likelihood(pose, ranges, .3) > .999);
  assert.ok(likelihood([12, 8, 1.2], ranges, .3) < .05);
  assert.ok(likelihood([5, 4, 0], ranges, .3) < 1e-7);
});

test('global proposals are reproducible and lie in free space', () => {
  const first = randomGenerator();
  const second = randomGenerator();
  for (let index = 0; index < 100; index++) {
    const pose = randomPose(first);
    assert.deepEqual(pose, randomPose(second));
    assert.ok(free(pose[0], pose[1], .1));
  }
  assert.deepEqual(move([1, 1, 0], 1, 0), [2, 1, 0]);
});

test('CPU posterior remains finite, resamples and converges from a global prior', () => {
  const filter = new CpuFilter(4096);
  const truth = [2, 1.5, 0];
  let result;
  for (let step = 0; step < 50; step++) result = filter.step(scan(truth), [0, 0], .3);
  assert.ok(result.every(Number.isFinite));
  assert.ok(result[3] > 0 && result[3] <= filter.count + 1e-4);
  assert.ok(Math.hypot(result[0] - truth[0], result[1] - truth[1]) < .4, `estimate: ${result}`);
  assert.equal(filter.particles.length, filter.count);
  const teleported = [13.8, 10.7, Math.PI];
  for (let step = 0; step < 100; step++) result = filter.step(scan(teleported), [0, 0], .3);
  assert.ok(Math.hypot(result[0] - teleported[0], result[1] - teleported[1]) < .5, `recovery: ${result}`);
});

test('particle app supports CPU fallback, pause, teleport and scatter', async () => {
  const identifiers = ['map', 'particles', 'fallback', 'overlay', 'population', 'checks', 'latency',
    'error', 'backend', 'kidnap', 'run', 'reset', 'count', 'engine', 'sigma', 'sigma-value',
    'status', 'ess', 'groups', 'transfer', 'time'];
  const elements = Object.fromEntries(identifiers.map(id => [id, {
    id, value: '', textContent: '', listeners: {},
    addEventListener(name, listener) { this.listeners[name] = listener; },
    getContext() { return new Proxy({}, {get: () => () => {}}); },
  }]));
  elements.count.value = '65536'; elements.engine.value = 'auto'; elements.sigma.value = '.3';
  let frame;
  globalThis.document = {hidden: false, querySelectorAll: () => Object.values(elements)};
  globalThis.matchMedia = () => ({matches: false});
  globalThis.requestAnimationFrame = callback => { frame = callback; };
  try {
    await import('../docs/site/particles/app.mjs');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(elements.backend.textContent, 'CPU PREVIEW / 512 PARTICLES');
    assert.equal(elements.population.textContent, '512');
    assert.equal(elements.checks.textContent, '16,384');
    assert.equal(elements.particles.hidden, true);
    assert.equal(elements.run.disabled, false);
    await frame();
    await frame();
    assert.equal(elements.time.textContent, '0.1 s');
    elements.run.listeners.click();
    await frame();
    assert.equal(elements.time.textContent, '0.1 s');
    elements.kidnap.listeners.click();
    assert.match(elements.status.textContent, /no ground-truth pose/);
    elements.reset.listeners.click();
    await frame();
    assert.equal(elements.time.textContent, '0.1 s');
    assert.match(elements.error.textContent, /m$/);
    elements.run.listeners.click();
    await frame();
    assert.equal(elements.time.textContent, '0.2 s');
  } finally {
    delete globalThis.document;
    delete globalThis.matchMedia;
    delete globalThis.requestAnimationFrame;
  }
});
