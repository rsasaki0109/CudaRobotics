import test from 'node:test';
import assert from 'node:assert/strict';
import {WIDTH, HEIGHT, RESOLUTION, preset, distanceField, clearance, segmentClear, plan, follow, mapDocument, parseMap} from '../docs/site/navigation/model.mjs';

test('separable distance field has conservative obstacle and boundary distances', () => {
  const occupancy = new Uint8Array(WIDTH * HEIGHT);
  occupancy[40 * WIDTH + 40] = 1;
  const field = distanceField(occupancy);
  assert.equal(field[40 * WIDTH + 40], 0);
  assert.ok(Math.abs(field[40 * WIDTH + 48] - (1 - Math.SQRT2 * RESOLUTION / 2)) < 1e-6);
  assert.equal(field[0], RESOLUTION / 2);
  assert.equal(clearance(field, [NaN, 1]), 0);
  assert.equal(clearance(field, [-1, 1]), 0);
  assert.equal(clearance(field, [16, 1]), 0);
  assert.ok(!segmentClear(field, [3, 5.0625], [7, 5.0625], .3));
});

for (const name of ['warehouse', 'doorway', 'empty']) {
  test(`${name}: a collision-checked route reaches the destination`, () => {
    const data = preset(name);
    const field = distanceField(data.occupancy);
    const result = plan(field, data.start, data.goal, data.radius, data.margin);
    assert.equal(result.status, 'ready');
    assert.ok(result.minimumClearance >= result.required - 1e-6);
    assert.ok(result.length >= Math.hypot(data.start[0] - data.goal[0], data.start[1] - data.goal[1]));
    let position = data.start;
    let waypoint = 1;
    const obstacles = Array.from(data.occupancy.entries()).filter(([, occupied]) => occupied)
      .map(([index]) => [(index % WIDTH) * RESOLUTION, Math.floor(index / WIDTH) * RESOLUTION]);
    for (let step = 0; step < 1500; step++) {
      const next = follow(field, position, result.path, waypoint, .04, result.required);
      assert.equal(next.blocked, false, `blocked at step ${step}, position=${next.position}`);
      let actualClearance = Math.min(next.position[0], next.position[1], 16 - next.position[0], 12 - next.position[1]);
      for (const [left, bottom] of obstacles) {
        actualClearance = Math.min(actualClearance, Math.hypot(Math.max(left - next.position[0], 0, next.position[0] - left - RESOLUTION),
          Math.max(bottom - next.position[1], 0, next.position[1] - bottom - RESOLUTION)));
      }
      assert.ok(actualClearance >= result.required - 1e-6, `Geometric footprint collision at ${next.position}`);
      position = next.position; waypoint = next.waypoint;
      if (next.reached) { assert.ok(Math.hypot(position[0] - data.goal[0], position[1] - data.goal[1]) < 1e-8); return; }
    }
    assert.fail('Destination was not reached');
  });
}

test('large footprints, occupied goals and blocked aisles produce classified failures', () => {
  const data = preset('doorway');
  let field = distanceField(data.occupancy);
  assert.equal(plan(field, data.start, data.goal, .75, .12).status, 'no_path');
  assert.equal(plan(field, [8, 2], data.goal, .3, .12).status, 'start_unsafe');
  assert.equal(plan(field, data.start, [8, 2], .3, .12).status, 'goal_unsafe');
  for (let row = 0; row < HEIGHT; row++) data.occupancy[row * WIDTH + 64] = 1;
  field = distanceField(data.occupancy);
  assert.equal(plan(field, data.start, data.goal, .3, .12).status, 'no_path');
  const stopped = follow(field, [7, 6], [[7, 6], [9, 6]], 1, 2, .42);
  assert.equal(stopped.blocked, true);
  assert.deepEqual(stopped.position, [7, 6]);
});

test('maps round-trip with units and reject malformed or ambiguous occupancy', () => {
  const data = preset('warehouse');
  const document = mapDocument(data.occupancy, data.start, data.goal, data.radius, data.margin);
  assert.deepEqual(parseMap(JSON.parse(JSON.stringify(document))), data);
  for (const changes of [{resolution: 1}, {width: 10}, {origin: [1, 0]}, {rowOrder: 'top-to-bottom'},
    {occupancy: [1]}, {radius: NaN}, {radius: .333}, {margin: -.1}, {goal: [Infinity, 1]}]) {
    assert.throws(() => parseMap({...document, ...changes}));
  }
});

test('workbench completes a mission, pauses on editing and round-trips exported maps', async () => {
  const identifiers = ['map', 'world', 'controls', 'length', 'clearance', 'field-time', 'plan-time', 'backend',
    'preset', 'tool', 'radius', 'radius-value', 'margin', 'margin-value', 'start-x', 'start-y', 'goal-x', 'goal-y',
    'targets', 'run', 'replan', 'status', 'show-field', 'required', 'expanded', 'time', 'export-map', 'export-plan', 'import-map'];
  const elements = Object.fromEntries(identifiers.map(id => [id, {
    id, value: '', textContent: '', listeners: {},
    get valueAsNumber() { return this.value === '' ? NaN : Number(this.value); },
    addEventListener(name, listener) { this.listeners[name] = listener; },
    getContext() { return new Proxy({}, {get: (_, name) => name === 'createImageData'
      ? (width, height) => ({data: new Uint8ClampedArray(width * height * 4)}) : () => {}}); },
    getBoundingClientRect() { return {left: 0, top: 0, width: 1280, height: 960}; },
    focus() {}, setPointerCapture() {},
  }]));
  elements.preset.value = 'warehouse'; elements.tool.value = 'draw'; elements['show-field'].checked = true;
  let frame;
  let exported;
  const originalUrl = URL.createObjectURL;
  URL.createObjectURL = blob => { exported = blob; return originalUrl(blob); };
  globalThis.document = {hidden: false, querySelectorAll: () => Object.values(elements), createElement: () => ({click() {}})};
  globalThis.requestAnimationFrame = callback => { frame = callback; };
  const ready = async () => {
    for (let attempt = 0; attempt < 200; attempt++) {
      if (!elements.replan.disabled && !elements['export-plan'].disabled) return;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.fail(`Planning did not complete: ${elements.status.textContent}`);
  };
  try {
    await import('../docs/site/navigation/app.mjs');
    await ready();
    assert.equal(elements.backend.textContent, 'CPU DISTANCE FIELD / CPU A*');
    assert.equal(elements.controls.disabled, false);
    assert.equal(elements.run.disabled, false);
    elements.run.listeners.click();
    for (let step = 1; step < 1500 && !elements.run.disabled; step++) frame(step * 50);
    assert.match(elements.status.textContent, /Mission complete/);
    elements['export-plan'].listeners.click();
    const report = JSON.parse(await exported.text());
    assert.equal(report.schema, 'cudarobotics.route.v1');
    assert.equal(report.execution.reached, true);
    assert.ok(report.minimumClearance >= report.required);
    elements.preset.value = 'doorway'; elements.preset.listeners.change();
    await ready();
    elements.run.listeners.click(); frame(100000);
    elements.world.listeners.pointerdown({pointerId: 1, clientX: 160, clientY: 480});
    elements.world.listeners.pointerup();
    assert.equal(elements.run.disabled, true);
    assert.match(elements.status.textContent, /Motion paused/);
    await ready();
    assert.match(elements.status.textContent, /Robot footprint overlaps/);
    elements['export-map'].listeners.click();
    const savedMap = JSON.parse(await exported.text());
    assert.equal(savedMap.schema, 'cudarobotics.costmap.v1');
    assert.ok(parseMap(savedMap));
    const valid = preset('empty');
    const imported = mapDocument(valid.occupancy, valid.start, valid.goal, valid.radius, valid.margin);
    await elements['import-map'].listeners.change({target: {files: [{size: 100000, text: async () => JSON.stringify(imported)}]}});
    await ready();
    assert.equal(elements.run.disabled, false);
    await elements['import-map'].listeners.change({target: {files: [{size: 100, text: async () => '{}'}]}});
    assert.match(elements.status.textContent, /Import rejected/);
  } finally {
    URL.createObjectURL = originalUrl;
    delete globalThis.document; delete globalThis.requestAnimationFrame;
  }
});
