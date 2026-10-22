export const HORIZON = 32;
export const DT = 0.1;
export const RADIUS = 0.22;
export const COLLISION = 1e6;
export const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export const scenes = {
  gap: {name: 'Wall gap', start: [1, 5, 0], goal: [9, 5],
    obstacles: [[4.8, 0, 5.2, 4], [4.8, 6, 5.2, 10]]},
  slalom: {name: 'Slalom', start: [1, 5, 0], goal: [9, 5],
    obstacles: [[3, 0, 3.5, 5.6], [6, 4.4, 6.5, 10]],
    waypoints: [[2.3, 6.6], [4.5, 6.6], [5.2, 3.3], [7.3, 3.3]]},
  open: {name: 'Open arena', start: [1, 2, 0], goal: [9, 8], obstacles: []},
};

export function randomGenerator(seed = 311) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

export function clearance(x, y, obstacles) {
  let distance = Math.min(x, y, 10 - x, 10 - y) - RADIUS;
  for (const [left, bottom, right, top] of obstacles) {
    const outsideX = Math.max(left - x, 0, x - right);
    const outsideY = Math.max(bottom - y, 0, y - top);
    distance = Math.min(distance, Math.hypot(outsideX, outsideY) - RADIUS);
  }
  return distance;
}

export function advance(state, velocity, angular, dt = DT) {
  const heading = state[2] + angular * dt / 2;
  return [state[0] + velocity * Math.cos(heading) * dt,
    state[1] + velocity * Math.sin(heading) * dt, state[2] + angular * dt];
}

export function safeAdvance(state, velocity, angular, obstacles) {
  let next = state;
  for (let substep = 0; substep < 2; substep++) {
    next = advance(next, velocity, angular, DT / 2);
    if (clearance(next[0], next[1], obstacles) <= 0) return null;
  }
  return next;
}

export function rollout(state, target, obstacles, controls, offset = 0, trace = null) {
  let cost = 0;
  let previousAngular = 0;
  for (let step = 0; step < HORIZON; step++) {
    const velocity = controls[offset + 2 * step];
    const angular = controls[offset + 2 * step + 1];
    const next = safeAdvance(state, velocity, angular, obstacles);
    if (!next) return COLLISION;
    state = next;
    if (trace) trace.push(state.slice(0, 2));
    const distance = Math.hypot(state[0] - target[0], state[1] - target[1]);
    const margin = Math.max(0, 0.65 - clearance(state[0], state[1], obstacles));
    cost += 0.4 * distance + 8 * margin * margin + 0.04 * angular * angular
      + 0.08 * (angular - previousAngular) ** 2;
    previousAngular = angular;
  }
  return cost + 12 * Math.hypot(state[0] - target[0], state[1] - target[1]) ** 2;
}

export function cpuCosts(state, target, obstacles, controls, count) {
  const costs = new Float32Array(count);
  for (let sample = 0; sample < count; sample++) {
    costs[sample] = rollout(state, target, obstacles, controls, sample * HORIZON * 2);
  }
  return costs;
}

export class Planner {
  constructor(seed = 311) {
    this.random = randomGenerator(seed);
    this.nominal = new Float32Array(HORIZON * 2);
  }

  sample(state, target, count) {
    const controls = new Float32Array(count * HORIZON * 2);
    const headingError = Math.atan2(Math.sin(Math.atan2(target[1] - state[1], target[0] - state[0]) - state[2]),
      Math.cos(Math.atan2(target[1] - state[1], target[0] - state[0]) - state[2]));
    for (let sample = 0; sample < count; sample++) {
      const speedBias = (this.random() - 0.5) * 1.2;
      const turnBias = (this.random() - 0.5) * 3.6;
      for (let step = 0; step < HORIZON; step++) {
        const index = sample * HORIZON * 2 + step * 2;
        const guided = sample < count / 4;
        const baseSpeed = guided ? Math.max(0.25, 1.3 - Math.abs(headingError)) : this.nominal[step * 2];
        const baseTurn = guided ? headingError * Math.exp(-step / 8) : this.nominal[step * 2 + 1];
        controls[index] = clamp(baseSpeed + speedBias + (this.random() - 0.5) * 0.5, 0, 1.6);
        controls[index + 1] = clamp(baseTurn + turnBias + (this.random() - 0.5) * 0.7, -2.4, 2.4);
      }
    }
    return controls;
  }

  update(costs, controls, temperature) {
    let minimum = COLLISION;
    let best = -1;
    let valid = 0;
    costs.forEach((cost, index) => {
      if (Number.isFinite(cost) && cost < COLLISION) {
        valid++;
        if (cost < minimum) { minimum = cost; best = index; }
      }
    });
    const average = new Float32Array(HORIZON * 2);
    let total = 0;
    if (valid) {
      costs.forEach((cost, sample) => {
        if (!Number.isFinite(cost) || cost >= COLLISION) return;
        const weight = Math.exp(-(cost - minimum) / Math.max(0.01, temperature));
        total += weight;
        for (let index = 0; index < average.length; index++) {
          average[index] += weight * controls[sample * average.length + index];
        }
      });
      for (let index = 0; index < average.length; index++) average[index] /= total;
    }
    this.nominal.set(average.subarray(2), 0);
    this.nominal.set(average.subarray(average.length - 2), average.length - 2);
    return {velocity: average[0], angular: average[1], valid, best, minimum, average};
  }
}
