export const WIDTH = 16;
export const HEIGHT = 12;
export const BEAMS = 32;
export const BOXES = [[4, 3, 6, 7], [9, 5, 10.5, 10], [11.5, 3, 13, 4]];
export const TAU = Math.PI * 2;

export function randomGenerator(seed = 311) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

export function free(positionX, positionY, margin = 0) {
  return positionX > margin && positionY > margin && positionX < WIDTH - margin && positionY < HEIGHT - margin
    && BOXES.every(([left, bottom, right, top]) => positionX < left - margin || positionX > right + margin
      || positionY < bottom - margin || positionY > top + margin);
}

export function randomPose(random) {
  for (;;) {
    const pose = [0.3 + random() * (WIDTH - 0.6), 0.3 + random() * (HEIGHT - 0.6), random() * TAU];
    if (free(pose[0], pose[1], 0.1)) return pose;
  }
}

export function surfaceDistance(positionX, positionY) {
  let distance = Math.min(Math.abs(positionX), Math.abs(positionY), Math.abs(WIDTH - positionX), Math.abs(HEIGHT - positionY));
  for (const [left, bottom, right, top] of BOXES) {
    const outsideX = Math.max(left - positionX, 0, positionX - right);
    const outsideY = Math.max(bottom - positionY, 0, positionY - top);
    const inside = Math.max(Math.max(left - positionX, positionX - right), Math.max(bottom - positionY, positionY - top));
    distance = Math.min(distance, Math.hypot(outsideX, outsideY) + Math.min(inside, 0) * -1);
  }
  return distance;
}

export function scan(pose) {
  const ranges = new Float32Array(BEAMS);
  const geometry = [[-0.1, -0.1, 0, HEIGHT + .1], [WIDTH, -.1, WIDTH + .1, HEIGHT + .1],
    [0, -.1, WIDTH, 0], [0, HEIGHT, WIDTH, HEIGHT + .1], ...BOXES];
  for (let beam = 0; beam < BEAMS; beam++) {
    const angle = pose[2] + beam / BEAMS * TAU;
    const direction = [Math.cos(angle), Math.sin(angle)];
    let nearest = 30;
    for (const bounds of geometry) {
      let entry = -Infinity;
      let exit = Infinity;
      for (let axis = 0; axis < 2; axis++) {
        if (Math.abs(direction[axis]) < 1e-10) {
          if (pose[axis] < bounds[axis] || pose[axis] > bounds[axis + 2]) { entry = Infinity; break; }
        } else {
          const first = (bounds[axis] - pose[axis]) / direction[axis];
          const second = (bounds[axis + 2] - pose[axis]) / direction[axis];
          entry = Math.max(entry, Math.min(first, second));
          exit = Math.min(exit, Math.max(first, second));
        }
      }
      if (entry >= 0 && exit >= entry) nearest = Math.min(nearest, entry);
    }
    ranges[beam] = nearest;
  }
  return ranges;
}

export function likelihood(pose, ranges, sigma) {
  if (!free(pose[0], pose[1])) return 1e-8;
  let error = 0;
  for (let beam = 0; beam < BEAMS; beam++) {
    const angle = pose[2] + beam / BEAMS * TAU;
    const distance = surfaceDistance(pose[0] + Math.cos(angle) * ranges[beam], pose[1] + Math.sin(angle) * ranges[beam]);
    error += Math.min(4, distance * distance);
  }
  return Math.exp(-error / BEAMS / (2 * sigma * sigma)) + 1e-8;
}

export function move(pose, translation, rotation) {
  const heading = pose[2] + rotation / 2;
  return [pose[0] + Math.cos(heading) * translation, pose[1] + Math.sin(heading) * translation, pose[2] + rotation];
}

export class CpuFilter {
  constructor(count = 512, seed = 311) {
    this.count = count;
    this.random = randomGenerator(seed);
    this.particles = Array.from({length: count}, () => [...randomPose(this.random), 1 / count]);
  }

  step(ranges, motion, sigma, reset = false, injection = 0.03) {
    const candidates = this.particles.map(pose => {
      const next = reset || this.random() < injection ? randomPose(this.random)
        : move(pose, motion[0] + (this.random() - .5) * .07, motion[1] + (this.random() - .5) * .05);
      return [...next, likelihood(next, ranges, sigma)];
    });
    const summary = new Float64Array(6);
    const cumulative = [];
    let total = 0;
    for (const [positionX, positionY, yaw, weight] of candidates) {
      total += weight;
      cumulative.push(total);
      summary[0] += positionX * weight; summary[1] += positionY * weight;
      summary[2] += Math.sin(yaw) * weight; summary[3] += Math.cos(yaw) * weight;
      summary[4] += weight * weight;
    }
    let source = 0;
    this.particles = candidates.map((_, index) => {
      const threshold = (index + this.random()) / this.count * total;
      while (source < this.count - 1 && cumulative[source] < threshold) source++;
      return [...candidates[source]];
    });
    return [summary[0] / total, summary[1] / total, Math.atan2(summary[2], summary[3]), total * total / summary[4]];
  }
}
