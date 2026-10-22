import test from 'node:test';
import assert from 'node:assert/strict';
import {Planner, scenes, HORIZON, COLLISION, clearance, safeAdvance, cpuCosts, randomGenerator} from '../docs/site/demo/planner.mjs';

test('seeded sampling is deterministic and respects command limits', () => {
  const first = new Planner().sample(scenes.gap.start, scenes.gap.goal, 128);
  assert.deepEqual(first, new Planner().sample(scenes.gap.start, scenes.gap.goal, 128));
  for (let index = 0; index < first.length; index += 2) {
    assert.ok(first[index] >= 0 && first[index] <= 1.6 + 1e-6);
    assert.ok(Math.abs(first[index + 1]) <= 2.4 + 1e-6);
  }
  const random = randomGenerator();
  for (let index = 0; index < 100; index++) assert.ok(random() > 0 && random() < 1);
});

test('disk clearance, boundaries and collision stops', () => {
  assert.ok(clearance(5, 5, scenes.gap.obstacles) > 0);
  assert.ok(clearance(5, 2, scenes.gap.obstacles) < 0);
  assert.ok(clearance(0.1, 5, []) < 0);
  assert.equal(safeAdvance([4.5, 2, 0], 1.6, 0, scenes.gap.obstacles), null);
  const controls = new Float32Array(HORIZON * 2).fill(0);
  for (let index = 0; index < controls.length; index += 2) controls[index] = 1.6;
  assert.equal(cpuCosts([4.5, 2, 0], [9, 2], scenes.gap.obstacles, controls, 1)[0], COLLISION);
});

test('softmin is stable, rejects invalid samples and shifts warm start', () => {
  const planner = new Planner();
  const controls = new Float32Array(HORIZON * 2 * 3);
  controls.fill(0.5, 0, HORIZON * 2);
  controls.fill(1, HORIZON * 2, HORIZON * 4);
  const result = planner.update(new Float32Array([10000, 10000, COLLISION]), controls, 0.5);
  assert.equal(result.valid, 2);
  assert.equal(result.velocity, 0.75);
  assert.equal(planner.nominal[0], 0.75);
  const stopped = planner.update(new Float32Array([NaN, Infinity, COLLISION]), controls, 3);
  assert.equal(stopped.valid, 0);
  assert.equal(stopped.velocity, 0);
  assert.equal(stopped.angular, 0);
});

for (const [name, scene] of Object.entries(scenes)) {
  test(`closed-loop ${name} reaches goal without collision`, () => {
    const planner = new Planner();
    let state = [...scene.start];
    const waypoints = [...(scene.waypoints ?? [])];
    for (let step = 0; step < 600; step++) {
      while (waypoints.length && Math.hypot(state[0] - waypoints[0][0], state[1] - waypoints[0][1]) < 0.65) waypoints.shift();
      const target = waypoints[0] ?? scene.goal;
      const controls = planner.sample(state, target, 128);
      const result = planner.update(cpuCosts(state, target, scene.obstacles, controls, 128), controls, 3);
      assert.ok(result.valid > 0, `no valid rollout at step ${step}`);
      const next = safeAdvance(state, result.velocity, result.angular, scene.obstacles);
      assert.ok(next, `collision at step ${step}`);
      state = next;
      if (Math.hypot(state[0] - scene.goal[0], state[1] - scene.goal[1]) < 0.3) return;
    }
    assert.fail(`goal not reached: ${state}`);
  });
}

test('app initializes CPU fallback and supports start, pause, goal validation and reset', async () => {
  const identifiers = ['world', 'backend', 'scene', 'engine', 'count', 'temperature',
    'temperature-value', 'goal-x', 'goal-y', 'set-goal', 'run', 'reset', 'status',
    'latency', 'valid', 'distance', 'time'];
  const elements = Object.fromEntries(identifiers.map(id => [id, {
    id, value: '', textContent: '', listeners: {},
    addEventListener(name, listener) { this.listeners[name] = listener; },
  }]));
  Object.assign(elements.scene, {value: 'gap'});
  Object.assign(elements.engine, {value: 'auto'});
  Object.assign(elements.count, {value: '128'});
  Object.assign(elements.temperature, {value: '3'});
  elements.world.getContext = () => new Proxy({}, {get: () => () => {}});
  let frame;
  globalThis.document = {hidden: false, querySelectorAll: () => Object.values(elements)};
  globalThis.requestAnimationFrame = callback => { frame = callback; };
  try {
    await import('../docs/site/demo/app.mjs');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(elements.backend.textContent, 'CPU / JavaScript');
    assert.equal(elements.run.disabled, false);
    assert.match(elements.status.textContent, /CPU fallback/);
    elements.run.listeners.click();
    await frame();
    assert.equal(elements.time.textContent, '0.1 s');
    assert.equal(elements.run.textContent, 'Pause');
    elements.run.listeners.click();
    await frame();
    assert.equal(elements.time.textContent, '0.1 s');
    elements['goal-x'].valueAsNumber = 5;
    elements['goal-y'].valueAsNumber = 2;
    elements['set-goal'].listeners.click();
    assert.match(elements.status.textContent, /free space/);
    elements['goal-y'].valueAsNumber = 5;
    elements['set-goal'].listeners.click();
    assert.match(elements.status.textContent, /New goal set/);
    elements.reset.listeners.click();
    assert.equal(elements.time.textContent, '0.0 s');
    assert.equal(elements.run.textContent, 'Start');
    elements.scene.value = 'slalom';
    elements.scene.listeners.change();
    elements.run.listeners.click();
    for (let step = 0; step < 600 && elements.run.textContent === 'Pause'; step++) await frame();
    assert.match(elements.status.textContent, /Goal reached/);
  } finally {
    delete globalThis.document;
    delete globalThis.requestAnimationFrame;
  }
});
