import {Planner, scenes, HORIZON, DT, RADIUS, clearance, safeAdvance, rollout, cpuCosts} from './planner.mjs';
import {GpuCosts} from './gpu.mjs';

const elements = Object.fromEntries([...document.querySelectorAll('[id]')].map(element => [element.id, element]));
const canvas = elements.world;
const context = canvas.getContext('2d');
let scene, state, goal, waypoints, planner, history, traces, simulationTime;
let running = false;
let busy = false;
let revision = 0;
let gpu = null;
let backendRevision = 0;

function reset() {
  revision++;
  running = false;
  scene = scenes[elements.scene.value];
  state = [...scene.start];
  goal = [...scene.goal];
  waypoints = (scene.waypoints ?? []).map(point => [...point]);
  planner = new Planner();
  history = [state.slice(0, 2)];
  traces = [];
  simulationTime = 0;
  elements['goal-x'].value = goal[0];
  elements['goal-y'].value = goal[1];
  elements.run.textContent = 'Start';
  elements.status.textContent = 'Ready. Start, or choose a new goal in free space.';
  elements.latency.textContent = '—';
  elements.valid.textContent = '—';
  draw();
}

async function configureBackend() {
  const request = ++backendRevision;
  revision++;
  running = false;
  elements.run.textContent = 'Start';
  elements.run.disabled = true;
  elements.backend.textContent = 'Initializing…';
  while (busy) await new Promise(resolve => setTimeout(resolve, 20));
  if (request !== backendRevision) return;
  gpu?.destroy();
  gpu = null;
  let reason = '';
  if (elements.engine.value === 'auto') {
    try {
      const candidate = await GpuCosts.create();
      if (request !== backendRevision) { candidate.destroy(); return; }
      gpu = candidate;
    } catch (error) { reason = error.message; }
  }
  if (request !== backendRevision) return;
  elements.backend.textContent = gpu ? 'WebGPU / rollout compute' : 'CPU / JavaScript';
  elements.status.textContent = reason ? `${reason}. Using CPU fallback.` : 'Ready. All computation stays on this device.';
  elements.run.disabled = false;
}

function setGoal(x, y) {
  if (!Number.isFinite(x) || !Number.isFinite(y) || clearance(x, y, scene.obstacles) <= 0.08) {
    elements.status.textContent = 'Choose a goal in free space, away from walls.';
    return;
  }
  revision++;
  goal = [x, y];
  waypoints = [];
  planner = new Planner();
  traces = [];
  elements['goal-x'].value = x.toFixed(1);
  elements['goal-y'].value = y.toFixed(1);
  elements.status.textContent = 'New goal set. Custom goals use local planning without guide waypoints.';
  draw();
}

function line(points, color, width = 1) {
  if (!points.length) return;
  context.beginPath();
  points.forEach(([x, y], index) => context[index ? 'lineTo' : 'moveTo'](x * 80, (10 - y) * 80));
  context.strokeStyle = color;
  context.lineWidth = width;
  context.stroke();
}

function draw() {
  context.fillStyle = '#0c1921';
  context.fillRect(0, 0, 800, 800);
  for (let grid = 0; grid <= 10; grid++) {
    line([[grid, 0], [grid, 10]], '#21333e');
    line([[0, grid], [10, grid]], '#21333e');
  }
  for (const [left, bottom, right, top] of scene.obstacles) {
    context.fillStyle = '#354b58';
    context.fillRect(left * 80, (10 - top) * 80, (right - left) * 80, (top - bottom) * 80);
    context.strokeStyle = '#6a8796';
    context.strokeRect(left * 80, (10 - top) * 80, (right - left) * 80, (top - bottom) * 80);
  }
  if (waypoints.length) {
    context.setLineDash([8, 8]);
    line([state, ...waypoints, goal], '#57707e', 2);
    context.setLineDash([]);
  }
  traces.forEach(trace => line(trace, '#527a9550'));
  line(history, '#73e4c0', 3);
  const [goalX, goalY] = [goal[0] * 80, (10 - goal[1]) * 80];
  context.strokeStyle = '#ffcc7c';
  context.lineWidth = 3;
  context.beginPath(); context.arc(goalX, goalY, 16, 0, Math.PI * 2); context.stroke();
  line([[goal[0] - .28, goal[1]], [goal[0] + .28, goal[1]]], '#ffcc7c', 2);
  line([[goal[0], goal[1] - .28], [goal[0], goal[1] + .28]], '#ffcc7c', 2);
  context.save();
  context.translate(state[0] * 80, (10 - state[1]) * 80);
  context.rotate(-state[2]);
  context.fillStyle = '#73e4c0';
  context.beginPath(); context.arc(0, 0, RADIUS * 80, 0, Math.PI * 2); context.fill();
  context.strokeStyle = '#0b2425'; context.lineWidth = 4;
  context.beginPath(); context.moveTo(0, 0); context.lineTo(21, 0); context.stroke();
  context.restore();
  elements.distance.textContent = `${Math.hypot(state[0] - goal[0], state[1] - goal[1]).toFixed(2)} m`;
  elements.time.textContent = `${simulationTime.toFixed(1)} s`;
}

async function tick() {
  if (running && !busy && !document.hidden) {
    busy = true;
    const version = revision;
    try {
      while (waypoints.length && Math.hypot(state[0] - waypoints[0][0], state[1] - waypoints[0][1]) < 0.65) waypoints.shift();
      const target = waypoints[0] ?? goal;
      const count = Number(elements.count.value);
      const start = performance.now();
      const controls = planner.sample(state, target, count);
      let costs;
      try {
        costs = gpu ? await gpu.evaluate(state, target, scene.obstacles, controls, count)
          : cpuCosts(state, target, scene.obstacles, controls, count);
      } catch (error) {
        if (version !== revision) return;
        gpu?.destroy(); gpu = null;
        elements.backend.textContent = 'CPU / JavaScript';
        elements.status.textContent = `WebGPU failed: ${error.message}. Switched to CPU.`;
        costs = cpuCosts(state, target, scene.obstacles, controls, count);
      }
      if (version !== revision) return;
      const result = planner.update(costs, controls, Number(elements.temperature.value));
      elements.latency.textContent = `${(performance.now() - start).toFixed(1)} ms`;
      elements.valid.textContent = `${result.valid} / ${count}`;
      traces = [];
      for (let sample = 0; sample < count; sample += Math.max(1, Math.floor(count / 64))) {
        const trace = [state.slice(0, 2)];
        rollout(state, target, scene.obstacles, controls, sample * HORIZON * 2, trace);
        traces.push(trace);
      }
      const next = safeAdvance(state, result.velocity, result.angular, scene.obstacles);
      if (!result.valid || !next) {
        running = false;
        elements.status.textContent = 'Stopped safely: no valid command. Reset or choose another goal.';
      } else {
        state = next;
        simulationTime += DT;
        history.push(state.slice(0, 2));
        if (history.length > 5000) history.shift();
        if (Math.hypot(state[0] - goal[0], state[1] - goal[1]) < 0.3) {
          running = false;
          elements.status.textContent = 'Goal reached. Choose another goal or reset to replay.';
        }
      }
      elements.run.textContent = running ? 'Pause' : 'Start';
      draw();
    } catch (error) {
      running = false;
      elements.run.textContent = 'Start';
      elements.status.textContent = `Planner stopped: ${error.message}`;
    } finally {
      busy = false;
      requestAnimationFrame(tick);
    }
    return;
  }
  requestAnimationFrame(tick);
}

elements.run.addEventListener('click', () => {
  running = !running;
  if (!running) revision++;
  elements.run.textContent = running ? 'Pause' : 'Start';
  elements.status.textContent = running ? 'Planning live. Click free space to change the goal.' : 'Paused.';
});
elements.reset.addEventListener('click', reset);
elements.scene.addEventListener('change', reset);
elements.engine.addEventListener('change', configureBackend);
elements.temperature.addEventListener('input', () => {
  elements['temperature-value'].value = Number(elements.temperature.value).toFixed(1);
});
elements['set-goal'].addEventListener('click', () => setGoal(
  elements['goal-x'].valueAsNumber, elements['goal-y'].valueAsNumber));
canvas.addEventListener('pointerdown', event => {
  const bounds = canvas.getBoundingClientRect();
  setGoal((event.clientX - bounds.left) / bounds.width * 10, 10 - (event.clientY - bounds.top) / bounds.height * 10);
});
reset();
configureBackend();
requestAnimationFrame(tick);
