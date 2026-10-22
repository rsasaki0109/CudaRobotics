import {WIDTH, HEIGHT, BEAMS, BOXES, TAU, CpuFilter, free, move, scan} from './model.mjs';
import {GpuFilter} from './gpu.mjs';

const elements = Object.fromEntries([...document.querySelectorAll('[id]')].map(element => [element.id, element]));
const mapContext = elements.map.getContext('2d');
const overlay = elements.overlay.getContext('2d');
const preview = elements.fallback.getContext('2d');
const format = value => value.toLocaleString('en-US');
const route = [[14, 1.5], [14, 10.7], [1.5, 10.7], [1.5, 1.5]];
let truth = [2, 1.5, 0];
let waypoint = 0;
let filter = null;
let usingGpu = false;
let running = false;
let busy = false;
let resetPending = true;
let revision = 0;
let configuration = 0;
let simulationTime = 0;
let estimate = null;
let lastRanges = scan(truth);
let history = [];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
if (matchMedia('(max-width: 760px)').matches) elements.count.value = '16384';

function drawMap() {
  mapContext.fillStyle = '#0b181e';
  mapContext.fillRect(0, 0, 1280, 960);
  mapContext.strokeStyle = '#20343c';
  mapContext.lineWidth = 1;
  for (let grid = 0; grid <= WIDTH; grid++) {
    mapContext.beginPath(); mapContext.moveTo(grid * 80, 0); mapContext.lineTo(grid * 80, 960); mapContext.stroke();
  }
  for (let grid = 0; grid <= HEIGHT; grid++) {
    mapContext.beginPath(); mapContext.moveTo(0, grid * 80); mapContext.lineTo(1280, grid * 80); mapContext.stroke();
  }
  for (const [left, bottom, right, top] of BOXES) {
    mapContext.fillStyle = '#050c10';
    mapContext.fillRect(left * 80 + 10, (HEIGHT - top) * 80 + 12, (right - left) * 80, (top - bottom) * 80);
    mapContext.fillStyle = '#273c46';
    mapContext.fillRect(left * 80, (HEIGHT - top) * 80, (right - left) * 80, (top - bottom) * 80);
    mapContext.strokeStyle = '#71949c'; mapContext.lineWidth = 2;
    mapContext.strokeRect(left * 80, (HEIGHT - top) * 80, (right - left) * 80, (top - bottom) * 80);
    mapContext.fillStyle = '#36505a';
    mapContext.fillRect(left * 80, (HEIGHT - top) * 80, (right - left) * 80, 7);
  }
}

function drawOverlay() {
  overlay.clearRect(0, 0, 1280, 960);
  const robotX = truth[0] * 80;
  const robotY = (HEIGHT - truth[1]) * 80;
  overlay.strokeStyle = '#ffae662b'; overlay.lineWidth = 1;
  overlay.fillStyle = '#ffc078';
  for (let beam = 0; beam < BEAMS; beam++) {
    const angle = truth[2] + beam / BEAMS * TAU;
    const endpointX = robotX + Math.cos(angle) * lastRanges[beam] * 80;
    const endpointY = robotY - Math.sin(angle) * lastRanges[beam] * 80;
    overlay.beginPath(); overlay.moveTo(robotX, robotY); overlay.lineTo(endpointX, endpointY); overlay.stroke();
    overlay.fillRect(endpointX - 3, endpointY - 3, 6, 6);
  }
  overlay.strokeStyle = '#ffb66b80'; overlay.lineWidth = 2;
  overlay.beginPath();
  history.forEach(([poseX, poseY], index) => overlay[index ? 'lineTo' : 'moveTo'](poseX * 80, (HEIGHT - poseY) * 80));
  overlay.stroke();
  if (estimate) {
    const estimateX = estimate[0] * 80;
    const estimateY = (HEIGHT - estimate[1]) * 80;
    overlay.strokeStyle = '#a3caff'; overlay.lineWidth = 3;
    overlay.beginPath(); overlay.arc(estimateX, estimateY, 22, 0, TAU); overlay.stroke();
    overlay.beginPath(); overlay.moveTo(estimateX - 30, estimateY); overlay.lineTo(estimateX + 30, estimateY);
    overlay.moveTo(estimateX, estimateY - 30); overlay.lineTo(estimateX, estimateY + 30); overlay.stroke();
  }
  overlay.save(); overlay.translate(robotX, robotY); overlay.rotate(-truth[2]);
  overlay.shadowColor = '#ffb66b'; overlay.shadowBlur = 18; overlay.fillStyle = '#ffbf7b';
  overlay.beginPath(); overlay.moveTo(22, 0); overlay.lineTo(-14, -13); overlay.lineTo(-9, 0); overlay.lineTo(-14, 13); overlay.closePath(); overlay.fill();
  overlay.restore();
}

function drawCpu() {
  preview.clearRect(0, 0, 1280, 960);
  for (const [positionX, positionY, , weight] of filter.particles) {
    preview.fillStyle = `rgba(100,255,170,${0.12 + Math.sqrt(weight) * 0.7})`;
    preview.fillRect(positionX * 80 - 2, (HEIGHT - positionY) * 80 - 2, 4, 4);
  }
}

async function configure() {
  const ticket = ++configuration;
  revision++;
  running = false;
  for (const id of ['run', 'reset', 'kidnap']) elements[id].disabled = true;
  elements.backend.textContent = 'Initializing…';
  while (busy) await new Promise(resolve => setTimeout(resolve, 20));
  if (ticket !== configuration) return;
  filter?.destroy?.();
  filter = null;
  let reason = '';
  let candidate;
  if (elements.engine.value === 'auto') {
    try { candidate = await GpuFilter.create(elements.particles, Number(elements.count.value)); }
    catch (error) { reason = error.message; }
  }
  if (ticket !== configuration) { candidate?.destroy(); return; }
  usingGpu = Boolean(candidate);
  filter = candidate ?? new CpuFilter(512);
  elements.particles.hidden = !usingGpu;
  elements.fallback.hidden = usingGpu;
  elements.backend.textContent = usingGpu ? 'WEBGPU / COMPUTE + RENDER' : 'CPU PREVIEW / 512 PARTICLES';
  elements.population.textContent = format(filter.count);
  elements.checks.textContent = format(filter.count * BEAMS);
  elements.groups.textContent = usingGpu ? `${Math.ceil(filter.count / 256)} × 256 lanes` : 'CPU reference';
  elements.transfer.textContent = usingGpu ? '32 bytes' : 'Not applicable';
  elements.status.textContent = reason ? `${reason}. Showing a 512-particle CPU preview.` : 'Live likelihood-field localization. Try teleporting the robot.';
  elements.count.disabled = !usingGpu;
  for (const id of ['run', 'reset', 'kidnap']) elements[id].disabled = false;
  resetPending = true;
  estimate = null;
  running = !reducedMotion;
  elements.run.textContent = running ? 'Pause' : 'Start';
}

function robotMotion() {
  const point = route[waypoint];
  if (Math.hypot(truth[0] - point[0], truth[1] - point[1]) < 0.35) waypoint = (waypoint + 1) % route.length;
  const goal = route[waypoint];
  const error = Math.atan2(Math.sin(Math.atan2(goal[1] - truth[1], goal[0] - truth[0]) - truth[2]),
    Math.cos(Math.atan2(goal[1] - truth[1], goal[0] - truth[0]) - truth[2]));
  const rotation = Math.max(-0.12, Math.min(0.12, error * 0.25));
  const translation = Math.max(0, Math.cos(error)) * 0.085;
  const next = move(truth, translation, rotation);
  if (free(next[0], next[1], 0.25)) { truth = next; return [translation, rotation]; }
  truth = move(truth, 0, rotation);
  return [0, rotation];
}

async function tick() {
  if (!filter || busy || document.hidden || (!running && !resetPending)) { requestAnimationFrame(tick); return; }
  busy = true;
  const version = revision;
  try {
    const reset = resetPending;
    resetPending = false;
    const motion = reset || !running ? [0, 0] : robotMotion();
    lastRanges = scan(truth);
    const start = performance.now();
    const result = await filter.step(lastRanges, motion, Number(elements.sigma.value), reset);
    if (version !== revision) return;
    estimate = result;
    elements.latency.textContent = `${(performance.now() - start).toFixed(1)} ms`;
    elements.error.textContent = `${Math.hypot(result[0] - truth[0], result[1] - truth[1]).toFixed(2)} m`;
    elements.ess.textContent = `${Math.round(result[3])} / ${format(filter.count)}`;
    if (!reset && running) simulationTime += 0.1;
    elements.time.textContent = `${simulationTime.toFixed(1)} s`;
    history.push(truth.slice(0, 2));
    if (history.length > 1200) history.shift();
    if (!usingGpu) drawCpu();
    drawOverlay();
  } catch (error) {
    if (version !== revision) return;
    filter?.destroy?.();
    filter = new CpuFilter(512);
    usingGpu = false;
    elements.particles.hidden = true; elements.fallback.hidden = false;
    elements.backend.textContent = 'CPU PREVIEW / 512 PARTICLES';
    elements.population.textContent = '512'; elements.checks.textContent = format(512 * BEAMS);
    elements.groups.textContent = 'CPU reference'; elements.transfer.textContent = 'Not applicable';
    elements.count.disabled = true;
    elements.status.textContent = `${error.message}. Switched to CPU preview.`;
    resetPending = true;
  } finally { busy = false; requestAnimationFrame(tick); }
}

elements.run.addEventListener('click', () => {
  running = !running;
  elements.run.textContent = running ? 'Pause' : 'Start';
});
elements.reset.addEventListener('click', () => { revision++; resetPending = true; estimate = null; });
elements.kidnap.addEventListener('click', () => {
  revision++;
  if (truth[1] < 6) { truth = [13.8, 10.7, Math.PI]; waypoint = 2; }
  else { truth = [2, 1.5, 0]; waypoint = 0; }
  estimate = null; history = [];
  lastRanges = scan(truth);
  elements.status.textContent = 'Robot teleported. The filter receives only odometry and a new scan — no ground-truth pose.';
  drawOverlay();
});
elements.count.addEventListener('change', configure);
elements.engine.addEventListener('change', configure);
elements.sigma.addEventListener('input', () => { elements['sigma-value'].value = `${Number(elements.sigma.value).toFixed(2)} m`; });
drawMap(); drawOverlay();
configure();
requestAnimationFrame(tick);
