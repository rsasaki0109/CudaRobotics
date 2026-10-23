import test from 'node:test';
import assert from 'node:assert/strict';
import {WIDTH, HEIGHT, RESOLUTION, preset, distanceField, clearance, segmentClear, plan, follow, mapDocument, parseMap} from '../docs/site/navigation/model.mjs';
import {placePallet} from '../docs/site/navigation/pallet.mjs';

for (const name of ['warehouse', 'empty', 'doorway']) {
  test(`${name}: a pallet blocks the former route without mutating the original map`, () => {
    const data = preset(name);
    const original = data.occupancy.slice();
    const before = plan(distanceField(original), data.start, data.goal, data.radius, data.margin);
    const pallet = placePallet(original, before.path, data.start, data.goal, before.required);
    assert.ok(pallet);
    assert.deepEqual(original, data.occupancy);
    assert.equal(pallet.cells.length, 64);
    assert.ok(pallet.cells.every(index => original[index] === 0 && pallet.occupancy[index] === 1));
    const field = distanceField(pallet.occupancy);
    assert.ok(clearance(field, data.start) >= before.required);
    assert.ok(clearance(field, data.goal) >= before.required);
    assert.ok(before.path.slice(1).some((point, index) => !segmentClear(field, before.path[index], point, before.required)));
    const after = plan(field, data.start, data.goal, data.radius, data.margin);
    assert.equal(after.status, name === 'doorway' ? 'no_path' : 'ready');
    if (after.status === 'ready') {
      assert.notDeepEqual(after.path, before.path);
      assert.ok(after.minimumClearance >= before.required);
      let position = [...data.start];
      let waypoint = 1;
      let reached = false;
      for (let step = 0; step < 1500; step++) {
        const next = follow(field, position, after.path, waypoint, .04, after.required);
        assert.equal(next.blocked, false);
        position = next.position; waypoint = next.waypoint;
        if (next.reached) { reached = true; break; }
      }
      assert.equal(reached, true);
      assert.deepEqual(position, data.goal);
    }
  });
}

test('pallet placement rejects occupied or endpoint-only routes without mutation', () => {
  const occupied = new Uint8Array(WIDTH * HEIGHT).fill(1);
  assert.equal(placePallet(occupied, [[2, 6], [14, 6]], [2, 6], [14, 6], .42), null);
  assert.ok(occupied.every(value => value === 1));
  assert.equal(placePallet(new Uint8Array(WIDTH * HEIGHT), [[2, 6], [2.1, 6]], [2, 6], [2.1, 6], .42), null);
  assert.equal(placePallet(new Uint8Array(WIDTH * HEIGHT), [], [2, 6], [14, 6], .42), null);
});

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
    'targets', 'run', 'replan', 'status', 'show-field', 'required', 'expanded', 'time', 'export-map', 'export-plan', 'import-map',
    'engine', 'compare', 'export-comparison', 'comparison-status', 'cpu-median', 'gpu-median', 'cpu-range', 'gpu-range',
    'cpu-bar', 'gpu-bar', 'speedup', 'accuracy', 'add-pallet', 'restore-pallet', 'pallet-status', 'replan-total',
    'route-change', 'replan-clearance', 'before-length', 'after-length', 'compare-replan', 'export-replan-comparison',
    'replan-speedup', 'replan-agreement', 'replan-comparison-status', 'replan-cpu-field', 'replan-cpu-route',
    'replan-cpu-total', 'replan-cpu-path', 'replan-gpu-field', 'replan-gpu-route', 'replan-gpu-total', 'replan-gpu-path',
    'compare-sizes', 'cancel-sizes', 'export-sizes', 'scale-field-chart', 'scale-total-chart', 'scale-detail-rows', 'scale-summary', 'scale-status',
    'cpu-baseline', 'cpu-baseline-label', 'replan-cpu-label', 'field-baseline', 'replan-baseline', 'scale-baseline',
    'compare-implementations', 'cancel-implementations', 'export-implementations', 'implementation-grid', 'implementation-status',
    'impl-cpuEdt-gpuReference', 'impl-cpuEdt-gpuEdt', 'impl-gpuReference-gpuEdt',
    ...['cpuEdt', 'gpuReference', 'gpuEdt'].flatMap(backend => ['total', 'range', 'field', 'route', 'path', 'bar'].map(stage => `impl-${backend}-${stage}`))];
  const elements = Object.fromEntries(identifiers.map(id => [id, {
    id, value: '', textContent: '', listeners: {}, style: {},
    get valueAsNumber() { return this.value === '' ? NaN : Number(this.value); },
    addEventListener(name, listener) { this.listeners[name] = listener; },
    getContext() { return new Proxy({}, {get: (_, name) => name === 'createImageData'
      ? (width, height) => ({data: new Uint8ClampedArray(width * height * 4)}) : () => {}}); },
    getBoundingClientRect() { return {left: 0, top: 0, width: 1280, height: 960}; },
    focus() {}, setPointerCapture() {},
  }]));
  elements.preset.value = 'warehouse'; elements.tool.value = 'draw'; elements['show-field'].checked = true; elements.engine.value = 'auto';
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
    elements['implementation-grid'].value = '64'; elements['implementation-grid'].listeners.change();
    await elements['compare-implementations'].listeners.click();
    elements['export-implementations'].listeners.click();
    const implementationReport = JSON.parse(await exported.text());
    assert.equal(implementationReport.schema, 'cudarobotics.implementation-benchmark.v1');
    assert.equal(implementationReport.source.width, 128);
    assert.equal(implementationReport.input.width, 64);
    assert.equal(implementationReport.backends.cpuEdt.total.samplesMs.length, 15);
    assert.equal(implementationReport.backends.gpuReference, null);
    assert.equal(implementationReport.backends.gpuEdt, null);
    assert.match(elements['impl-cpuEdt-total'].textContent, /ms$/);
    assert.equal(elements['impl-gpuEdt-total'].textContent, 'Unavailable');
    const cancelledImplementation = elements['compare-implementations'].listeners.click();
    elements['cancel-implementations'].listeners.click();
    await cancelledImplementation;
    assert.match(elements['implementation-status'].textContent, /cancelled/);
    assert.equal(elements['export-implementations'].disabled, true);
    assert.equal(elements['compare-implementations'].disabled, false);
    const sizeComparison = elements['compare-sizes'].listeners.click();
    assert.equal(elements['cancel-sizes'].disabled, false);
    assert.equal(elements.compare.disabled, true);
    elements['cancel-sizes'].listeners.click();
    await sizeComparison;
    assert.match(elements['scale-status'].textContent, /cancelled/);
    assert.equal(elements['export-sizes'].disabled, true);
    assert.equal(elements.run.disabled, false);
    elements['export-map'].listeners.click();
    const beforePalletMap = JSON.parse(await exported.text());
    elements.run.listeners.click(); frame(50);
    const capturedStart = [Number(elements['start-x'].value), Number(elements['start-y'].value)];
    elements['add-pallet'].listeners.click();
    assert.equal(elements.run.disabled, true);
    assert.equal(elements['add-pallet'].disabled, true);
    await ready();
    assert.equal(elements['restore-pallet'].disabled, false);
    assert.match(elements['pallet-status'].textContent, /New route ready/);
    assert.match(elements['replan-total'].textContent, /ms$/);
    elements['export-plan'].listeners.click();
    const palletReport = JSON.parse(await exported.text());
    assert.deepEqual(palletReport.palletExperiment.before.path[0], palletReport.palletExperiment.after.path[0]);
    assert.ok(Math.hypot(palletReport.palletExperiment.start[0] - capturedStart[0], palletReport.palletExperiment.start[1] - capturedStart[1]) < .01);
    assert.notDeepEqual(palletReport.palletExperiment.before.path, palletReport.palletExperiment.after.path);
    assert.ok(palletReport.palletExperiment.replanningMs >= 0);
    await elements['compare-replan'].listeners.click();
    assert.match(elements['replan-cpu-total'].textContent, /ms/);
    assert.equal(elements['replan-gpu-total'].textContent, 'Unavailable');
    assert.equal(elements['replan-speedup'].textContent, '—');
    elements['export-replan-comparison'].listeners.click();
    const replanReport = JSON.parse(await exported.text());
    assert.equal(replanReport.schema, 'cudarobotics.replanning-benchmark.v1');
    assert.equal(replanReport.stages.cpu.route.samplesMs.length, 15);
    assert.equal(replanReport.routes.cpu.length, 15);
    assert.equal(replanReport.routes.gpu, null);
    assert.deepEqual(replanReport.input.start, palletReport.palletExperiment.start);
    assert.deepEqual(replanReport.palletBoundsMetres, palletReport.palletExperiment.palletBoundsMetres);
    assert.deepEqual(replanReport.input.occupancy, palletReport.map.occupancy);
    assert.equal(replanReport.cpuBaseline.id, 'matched');
    elements['cpu-baseline'].value = 'optimized'; elements['cpu-baseline'].listeners.change();
    assert.equal(elements['export-replan-comparison'].disabled, true);
    assert.equal(elements['compare-replan'].disabled, false);
    assert.equal(elements['restore-pallet'].disabled, false);
    assert.match(elements['replan-cpu-label'].textContent, /EDT/);
    assert.match(elements['scale-baseline'].textContent, /Different algorithms/);
    await elements['compare-replan'].listeners.click();
    elements['export-replan-comparison'].listeners.click();
    const optimizedReport = JSON.parse(await exported.text());
    assert.equal(optimizedReport.cpuBaseline.id, 'optimized');
    assert.deepEqual(optimizedReport.input, replanReport.input);
    assert.deepEqual(optimizedReport.routes.cpu[0], replanReport.routes.cpu[0]);
    const cancelledBaseline = elements['compare-replan'].listeners.click();
    elements['cpu-baseline'].value = 'matched'; elements['cpu-baseline'].listeners.change();
    await cancelledBaseline;
    assert.equal(elements['export-replan-comparison'].disabled, true);
    assert.match(elements['replan-comparison-status'].textContent, /cancelled/);
    assert.equal(elements['compare-replan'].disabled, false);
    elements['restore-pallet'].listeners.click();
    await ready();
    elements['export-map'].listeners.click();
    assert.deepEqual(JSON.parse(await exported.text()).occupancy, beforePalletMap.occupancy);
    assert.equal(elements['restore-pallet'].disabled, true);
    assert.equal(elements['replan-total'].textContent, '—');
    assert.equal(elements['compare-replan'].disabled, true);
    assert.equal(elements['export-replan-comparison'].disabled, true);
    assert.equal(elements['replan-cpu-total'].textContent, '—');
    await elements.compare.listeners.click();
    assert.match(elements['cpu-median'].textContent, /ms$/);
    assert.equal(elements['gpu-median'].textContent, 'Unavailable');
    assert.equal(elements.speedup.textContent, '—');
    elements['export-comparison'].listeners.click();
    const comparison = JSON.parse(await exported.text());
    assert.equal(comparison.schema, 'cudarobotics.distance-benchmark.v1');
    assert.equal(comparison.cpu.samplesMs.length, 15);
    assert.equal(comparison.gpu, null);
    assert.ok(parseMap(comparison.input));
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
    elements['add-pallet'].listeners.click(); await ready();
    assert.equal(elements.run.disabled, true);
    assert.equal(elements['after-length'].textContent, 'Blocked');
    assert.equal(elements['route-change'].textContent, 'No route');
    assert.match(elements['pallet-status'].textContent, /no safe route/);
    await elements['compare-replan'].listeners.click();
    assert.match(elements['replan-cpu-path'].textContent, /Blocked/);
    const cancelledReplan = elements['compare-replan'].listeners.click();
    elements.margin.listeners.input();
    await cancelledReplan;
    assert.match(elements['replan-comparison-status'].textContent, /cancelled/);
    assert.equal(elements['export-replan-comparison'].disabled, true);
    await ready();
    elements.preset.listeners.change(); await ready();
    elements.run.listeners.click(); frame(100000);
    const cancelledComparison = elements.compare.listeners.click();
    elements.world.listeners.pointerdown({pointerId: 1, clientX: 160, clientY: 480});
    elements.world.listeners.pointerup();
    assert.equal(elements.run.disabled, true);
    assert.equal(elements['export-comparison'].disabled, true);
    assert.equal(elements['cpu-median'].textContent, '—');
    assert.match(elements.status.textContent, /Motion paused/);
    await cancelledComparison;
    assert.match(elements['comparison-status'].textContent, /Comparison cancelled/);
    assert.equal(elements['export-comparison'].disabled, true);
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
