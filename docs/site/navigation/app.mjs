import {WIDTH, HEIGHT, RESOLUTION, WORLD_HEIGHT, preset, distanceField, plan, follow, mapDocument, parseMap} from './model.mjs';
import {GpuField} from './gpu.mjs';
import {compareDistanceFields} from './benchmark.mjs?v=all-comparisons-1';

const elements = Object.fromEntries([...document.querySelectorAll('[id]')].map(element => [element.id, element]));
const base = elements.map.getContext('2d');
const context = elements.world.getContext('2d');
let occupancy, position, goal;
let field = null;
let result = null;
let plannedStart = null;
let gpu = null;
let running = false;
let computing = false;
let benchmarking = false;
let comparison = null;
let pending = false;
let revision = 0;
let timer;
let waypoint = 1;
let elapsed = 0;
let dragging = false;
let lastPaint = null;
let cursor = [12, 12];
let track = [];

const required = () => Number(elements.radius.value) + Number(elements.margin.value);
const useGpu = () => gpu && elements.engine.value !== 'cpu';
const messages = {ready: 'Route ready. Run the mission or edit the map.', no_path: 'No route fits this footprint. Widen the aisle, erase an obstacle or reduce the radius.',
  start_unsafe: 'Robot footprint overlaps an obstacle, safety band or map boundary. Move the robot or edit the map.',
  goal_unsafe: 'Goal has insufficient clearance. Move the goal or edit the map.',
  endpoint_grid_clearance: 'Endpoint is too close to the clearance limit on this grid. Move it slightly into open space.'};

function syncPositions() {
  elements['start-x'].value = position[0].toFixed(2); elements['start-y'].value = position[1].toFixed(2);
  elements['goal-x'].value = goal[0].toFixed(2); elements['goal-y'].value = goal[1].toFixed(2);
}

function paintField() {
  if (!occupancy) return;
  const image = base.createImageData(WIDTH, HEIGHT);
  const threshold = required();
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      const index = row * WIDTH + column;
      const target = ((HEIGHT - row - 1) * WIDTH + column) * 4;
      let color = [13, 29, 36];
      if (occupancy[index]) color = [88, 109, 120];
      else if (field && elements['show-field'].checked) {
        const distance = field[index];
        color = distance < threshold ? [94, 52, 42] : [12, 36 + Math.min(70, distance * 24), 39 + Math.min(40, distance * 14)];
      }
      image.data.set([...color, 255], target);
    }
  }
  base.putImageData(image, 0, 0);
}

function line(points, color, width) {
  if (!points.length) return;
  context.strokeStyle = color; context.lineWidth = width;
  context.beginPath();
  points.forEach(([positionX, positionY], index) => context[index ? 'lineTo' : 'moveTo'](positionX * 80, (WORLD_HEIGHT - positionY) * 80));
  context.stroke();
}

function draw() {
  if (!position) return;
  context.clearRect(0, 0, 1280, 960);
  if (result?.status === 'ready') line(result.path.slice(Math.max(0, waypoint - 1)), '#8ff3b8', 5);
  line(track, '#9bc7ff88', 3);
  context.strokeStyle = '#ffcc7c'; context.lineWidth = 3;
  context.beginPath(); context.arc(goal[0] * 80, (WORLD_HEIGHT - goal[1]) * 80, 18, 0, Math.PI * 2); context.stroke();
  line([[goal[0] - .32, goal[1]], [goal[0] + .32, goal[1]]], '#ffcc7c', 2);
  line([[goal[0], goal[1] - .32], [goal[0], goal[1] + .32]], '#ffcc7c', 2);
  context.setLineDash([6, 6]); context.strokeStyle = '#9bc7ff88';
  context.beginPath(); context.arc(position[0] * 80, (WORLD_HEIGHT - position[1]) * 80, required() * 80, 0, Math.PI * 2); context.stroke();
  context.setLineDash([]); context.fillStyle = '#9bc7ff';
  context.beginPath(); context.arc(position[0] * 80, (WORLD_HEIGHT - position[1]) * 80, Number(elements.radius.value) * 80, 0, Math.PI * 2); context.fill();
  if (document.activeElement === elements.world) {
    context.strokeStyle = '#fff'; context.lineWidth = 2;
    context.strokeRect(cursor[0] * 10, (HEIGHT - cursor[1] - 1) * 10, 10, 10);
  }
}

function schedule() {
  if (!occupancy) return;
  revision++;
  clearComparison();
  running = false;
  result = null;
  elements.run.disabled = true; elements.run.textContent = 'Run mission';
  elements['export-plan'].disabled = true;
  elements.compare.disabled = true;
  elements.status.textContent = 'Motion paused. Rebuilding clearance and route from the current position…';
  elements['radius-value'].value = `${Number(elements.radius.value).toFixed(2)} m`;
  elements['margin-value'].value = `${Number(elements.margin.value).toFixed(2)} m`;
  elements.required.textContent = `${required().toFixed(2)} m`;
  elements.length.textContent = '—'; elements.clearance.textContent = '—';
  pending = true;
  clearTimeout(timer);
  timer = setTimeout(rebuild, 100);
  paintField(); draw();
}

async function rebuild() {
  if (computing || benchmarking) return;
  computing = true;
  pending = false;
  const version = revision;
  const snapshot = occupancy.slice();
  try {
    const start = performance.now();
    let nextField;
    try { nextField = useGpu() ? await gpu.evaluate(snapshot) : distanceField(snapshot); }
    catch (error) {
      gpu?.destroy(); gpu = null;
      elements.backend.textContent = `CPU REFERENCE / GPU unavailable`;
      nextField = distanceField(snapshot);
    }
    if (version !== revision) return;
    elements['field-time'].textContent = `${(performance.now() - start).toFixed(1)} ms`;
    field = nextField;
    plannedStart = [...position];
    const planningStart = performance.now();
    result = plan(field, position, goal, Number(elements.radius.value), Number(elements.margin.value));
    elements['plan-time'].textContent = `${(performance.now() - planningStart).toFixed(1)} ms`;
    elements.length.textContent = result.path.length ? `${result.length.toFixed(2)} m` : 'Blocked';
    elements.clearance.textContent = result.minimumClearance === null ? '—' : `${result.minimumClearance.toFixed(2)} m`;
    elements.expanded.textContent = result.expanded.toLocaleString('en-US');
    elements.status.textContent = messages[result.status];
    elements.run.disabled = result.status !== 'ready';
    elements['export-plan'].disabled = false;
    elements.replan.disabled = false;
    elements.compare.disabled = false;
    elements.backend.textContent = useGpu() ? 'WEBGPU DISTANCE FIELD / CPU A*' : 'CPU DISTANCE FIELD / CPU A*';
    waypoint = 1;
    paintField(); draw();
  } catch (error) { elements.status.textContent = `Planning failed: ${error.message}`; }
  finally { computing = false; if (pending) { clearTimeout(timer); timer = setTimeout(rebuild, 0); } }
}

function clearComparison() {
  comparison = null;
  elements['export-comparison'].disabled = true;
  for (const id of ['cpu-median', 'gpu-median', 'speedup']) elements[id].textContent = '—';
  for (const id of ['cpu-range', 'gpu-range']) elements[id].textContent = 'P10–P90: —';
  for (const id of ['cpu-bar', 'gpu-bar']) elements[id].style.width = '0%';
  elements.accuracy.textContent = 'Result agreement: —';
  elements['comparison-status'].textContent = 'Comparison pauses the mission. Three warmups, then 15 paired measurements of the current map.';
}

async function compare() {
  if (computing || benchmarking || !occupancy || pending) return;
  clearComparison();
  if (running) elements.status.textContent = 'Mission paused for GPU / CPU comparison. Resume it after measurement.';
  benchmarking = true; running = false;
  elements.run.textContent = 'Run mission'; elements.run.disabled = true; elements.compare.disabled = true;
  const version = revision;
  const input = mapDocument(occupancy, position, goal, Number(elements.radius.value), Number(elements.margin.value));
  try {
    const report = await compareDistanceFields(occupancy, {
      gpuEvaluate: gpu && !gpu.failure ? snapshot => gpu.evaluate(snapshot) : null,
      shouldCancel: () => version !== revision || document.hidden,
      onProgress: progress => {
        elements['comparison-status'].textContent = `${progress.phase === 'warmup' ? 'Warming up' : 'Measuring'} ${progress.backend.toUpperCase()}: ${progress.iteration} / ${progress.total}…`;
      },
    });
    if (version !== revision) return;
    comparison = {...report, input};
    const showTiming = (backend, data) => {
      elements[`${backend}-median`].textContent = data ? `${data.medianMs.toFixed(2)} ms` : 'Unavailable';
      elements[`${backend}-range`].textContent = data ? `P10–P90: ${data.p10Ms.toFixed(2)}–${data.p90Ms.toFixed(2)} ms` : 'WebGPU adapter required';
    };
    showTiming('cpu', report.cpu); showTiming('gpu', report.gpu);
    const largest = Math.max(report.cpu.medianMs, report.gpu?.medianMs ?? 0);
    elements['cpu-bar'].style.width = `${largest > 0 ? report.cpu.medianMs / largest * 100 : 0}%`;
    elements['gpu-bar'].style.width = `${largest > 0 && report.gpu ? report.gpu.medianMs / largest * 100 : 0}%`;
    elements.speedup.textContent = report.speedup === null ? '—' : `${report.speedup.toFixed(2)}×`;
    elements.accuracy.textContent = report.accuracy
      ? `${report.accuracy.matches ? 'Match' : 'MISMATCH'} · max error ${report.accuracy.maxAbsoluteError.toExponential(2)} m (tolerance 1e-6 m)`
      : 'Result agreement: GPU unavailable';
    elements['comparison-status'].textContent = !report.gpu ? 'CPU timing complete. GPU comparison is unavailable on this device.'
      : !report.accuracy.matches ? 'Results differ beyond tolerance. No speedup claim is shown.'
        : report.speedup === null ? 'Results match, but timings are below usable timer precision.'
          : report.speedup >= 1 ? `Results match. GPU is ${report.speedup.toFixed(2)}× faster by median end-to-end field time.`
            : `Results match. CPU is ${(1 / report.speedup).toFixed(2)}× faster on this map; GPU overhead is included.`;
    elements['export-comparison'].disabled = false;
  } catch (error) {
    elements['comparison-status'].textContent = error.name === 'AbortError' ? 'Comparison cancelled: the map/settings changed or the page was hidden. Run it again.'
      : `Comparison failed: ${error.message}. No result was saved.`;
  } finally {
    benchmarking = false;
    elements.compare.disabled = computing || pending;
    elements.run.disabled = pending || result?.status !== 'ready' || waypoint >= (result?.path.length ?? 0);
    if (pending) { clearTimeout(timer); timer = setTimeout(rebuild, 0); }
  }
}

function applyPreset() {
  const data = preset(elements.preset.value);
  occupancy = data.occupancy; position = data.start; goal = data.goal;
  elements.radius.value = data.radius; elements.margin.value = data.margin;
  elapsed = 0; track = []; field = null;
  elements.time.textContent = '0.0 s';
  syncPositions(); schedule();
}

function edit(point, previous = null) {
  if (!occupancy) return;
  const column = Math.max(0, Math.min(WIDTH - 1, Math.floor(point[0] / RESOLUTION)));
  const row = Math.max(0, Math.min(HEIGHT - 1, Math.floor(point[1] / RESOLUTION)));
  cursor = [column, row];
  if (elements.tool.value === 'robot') { position = [(column + .5) * RESOLUTION, (row + .5) * RESOLUTION]; track = []; }
  else if (elements.tool.value === 'goal') goal = [(column + .5) * RESOLUTION, (row + .5) * RESOLUTION];
  else {
    const value = elements.tool.value === 'draw' ? 1 : 0;
    const first = previous ?? point;
    const steps = Math.max(1, Math.ceil(Math.hypot(first[0] - point[0], first[1] - point[1]) / (RESOLUTION / 2)));
    for (let step = 0; step <= steps; step++) {
      const brushColumn = Math.floor((first[0] + (point[0] - first[0]) * step / steps) / RESOLUTION);
      const brushRow = Math.floor((first[1] + (point[1] - first[1]) * step / steps) / RESOLUTION);
      for (let deltaRow = -1; deltaRow <= 1; deltaRow++) {
        for (let deltaColumn = -1; deltaColumn <= 1; deltaColumn++) {
          const nextRow = brushRow + deltaRow; const nextColumn = brushColumn + deltaColumn;
          if (nextRow >= 0 && nextRow < HEIGHT && nextColumn >= 0 && nextColumn < WIDTH) occupancy[nextRow * WIDTH + nextColumn] = value;
        }
      }
    }
    field = null;
  }
  syncPositions(); schedule();
}

function pointerPoint(event) {
  const bounds = elements.world.getBoundingClientRect();
  return [(event.clientX - bounds.left) / bounds.width * WIDTH * RESOLUTION,
    (1 - (event.clientY - bounds.top) / bounds.height) * HEIGHT * RESOLUTION];
}

function save(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

elements.world.addEventListener('pointerdown', event => {
  elements.world.focus(); elements.world.setPointerCapture(event.pointerId);
  dragging = true; lastPaint = pointerPoint(event); edit(lastPaint);
});
elements.world.addEventListener('pointermove', event => {
  if (dragging && ['draw', 'erase'].includes(elements.tool.value)) {
    const point = pointerPoint(event); edit(point, lastPaint); lastPaint = point;
  }
});
elements.world.addEventListener('pointerup', () => { dragging = false; lastPaint = null; });
elements.world.addEventListener('pointercancel', () => { dragging = false; lastPaint = null; });
elements.world.addEventListener('keydown', event => {
  const direction = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1]}[event.key];
  if (direction) {
    event.preventDefault();
    cursor = [Math.max(0, Math.min(WIDTH - 1, cursor[0] + direction[0] * (event.shiftKey ? 8 : 1))),
      Math.max(0, Math.min(HEIGHT - 1, cursor[1] + direction[1] * (event.shiftKey ? 8 : 1)))];
    draw();
  } else if (event.key === ' ' || event.key === 'Enter') {
    event.preventDefault(); edit([(cursor[0] + .5) * RESOLUTION, (cursor[1] + .5) * RESOLUTION]);
  }
});
elements.world.addEventListener('focus', draw); elements.world.addEventListener('blur', draw);
elements.preset.addEventListener('change', applyPreset);
elements.engine.addEventListener('change', schedule);
elements.compare.addEventListener('click', compare);
elements['export-comparison'].addEventListener('click', () => { if (comparison) save('cudarobotics-comparison.json', comparison); });
elements.radius.addEventListener('input', schedule); elements.margin.addEventListener('input', schedule);
elements['show-field'].addEventListener('change', paintField);
elements.replan.addEventListener('click', schedule);
elements.targets.addEventListener('click', () => {
  const points = [elements['start-x'].valueAsNumber, elements['start-y'].valueAsNumber,
    elements['goal-x'].valueAsNumber, elements['goal-y'].valueAsNumber];
  if (!points.every(Number.isFinite) || points[0] <= 0 || points[0] >= 16 || points[1] <= 0 || points[1] >= 12
    || points[2] <= 0 || points[2] >= 16 || points[3] <= 0 || points[3] >= 12) {
    elements.status.textContent = 'Enter finite coordinates inside the 16 × 12 m map.'; return;
  }
  position = points.slice(0, 2); goal = points.slice(2); track = []; schedule();
});
for (const id of ['start-x', 'start-y', 'goal-x', 'goal-y']) {
  elements[id].addEventListener('focus', () => {
    running = false; elements.run.textContent = 'Run mission';
    elements.status.textContent = 'Paused for coordinate editing. Apply positions to replan, or resume the existing mission.';
  });
}
elements.run.addEventListener('click', () => {
  if (benchmarking || !result || result.status !== 'ready') return;
  running = !running;
  elements.run.textContent = running ? 'Pause' : 'Run mission';
  elements.status.textContent = running ? 'Following the route. Editing will stop motion and replan.' : 'Paused.';
});
elements['export-map'].addEventListener('click', () => save('cudarobotics-map.json',
  mapDocument(occupancy, position, goal, Number(elements.radius.value), Number(elements.margin.value))));
elements['export-plan'].addEventListener('click', () => {
  if (!result) return;
  save('cudarobotics-plan.json', {schema: 'cudarobotics.route.v1', units: 'metres',
    backend: useGpu() ? 'webgpu-distance-field/cpu-astar' : 'cpu-reference/cpu-astar',
    map: mapDocument(occupancy, plannedStart, goal, Number(elements.radius.value), Number(elements.margin.value)),
    ...result, simulationOnly: true,
    execution: {running, robotPosition: [...position], simulationSeconds: elapsed,
      reached: result.status === 'ready' && waypoint >= result.path.length}});
});
elements['import-map'].addEventListener('change', async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  running = false; elements.run.textContent = 'Run mission';
  try {
    if (file.size > 512000) throw new Error('Map file exceeds 500 KB');
    const data = parseMap(JSON.parse(await file.text()));
    occupancy = data.occupancy; position = data.start; goal = data.goal;
    elements.radius.value = data.radius; elements.margin.value = data.margin;
    field = null; track = []; elapsed = 0; elements.time.textContent = '0.0 s';
    syncPositions(); schedule();
  } catch (error) { elements.status.textContent = `Import rejected: ${error.message}`; }
  finally { elements['import-map'].value = ''; }
});

let previous = 0;
function tick(timestamp) {
  if (running && result?.status === 'ready' && !document.hidden && timestamp - previous >= 50) {
    previous = timestamp;
    const next = follow(field, position, result.path, waypoint, .8 * .05, result.required);
    position = next.position; waypoint = next.waypoint;
    track.push([...position]); if (track.length > 2000) track.shift();
    elapsed += .05; elements.time.textContent = `${elapsed.toFixed(1)} s`;
    syncPositions();
    let remaining = 0;
    for (let index = waypoint; index < result.path.length; index++) {
      const prior = index === waypoint ? position : result.path[index - 1];
      remaining += Math.hypot(result.path[index][0] - prior[0], result.path[index][1] - prior[1]);
    }
    elements.length.textContent = `${remaining.toFixed(2)} m`;
    if (next.blocked || next.reached) {
      running = false; elements.run.textContent = 'Run mission'; elements.run.disabled = true;
      elements.status.textContent = next.reached ? 'Mission complete. Export the route report or choose another destination.' : 'Stopped: clearance check failed. Replan or edit the map.';
    }
    draw();
  }
  requestAnimationFrame(tick);
}

async function initialize() {
  elements.replan.disabled = true;
  try { gpu = await GpuField.create(); elements.backend.textContent = 'WEBGPU DISTANCE FIELD / CPU A*'; }
  catch (error) { elements.backend.textContent = 'CPU DISTANCE FIELD / CPU A*'; }
  applyPreset();
  elements.controls.disabled = false;
}
initialize(); requestAnimationFrame(tick);
