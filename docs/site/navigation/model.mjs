export const WIDTH = 128;
export const HEIGHT = 96;
export const RESOLUTION = 0.125;
export const WORLD_WIDTH = WIDTH * RESOLUTION;
export const WORLD_HEIGHT = HEIGHT * RESOLUTION;
const HALF_DIAGONAL = Math.SQRT2 * RESOLUTION / 2;

export function preset(name) {
  const occupancy = new Uint8Array(WIDTH * HEIGHT);
  const boxes = name === 'warehouse' ? [[3.5, 2, 5, 9.5], [7, 2.5, 8.5, 10], [10.5, 2, 12, 8.5]]
    : name === 'doorway' ? [[7.75, 0, 8.25, 5.25], [7.75, 6.75, 8.25, 12]] : [];
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      const positionX = (column + .5) * RESOLUTION;
      const positionY = (row + .5) * RESOLUTION;
      occupancy[row * WIDTH + column] = boxes.some(([left, bottom, right, top]) =>
        positionX >= left && positionX <= right && positionY >= bottom && positionY <= top) ? 1 : 0;
    }
  }
  return {occupancy, start: name === 'doorway' ? [2, 6] : [1.5, 1.5],
    goal: name === 'doorway' ? [14, 6] : [14, 10.5], radius: .3, margin: .12};
}

export function distanceField(occupancy) {
  const horizontal = new Float32Array(WIDTH * HEIGHT);
  const result = new Float32Array(WIDTH * HEIGHT);
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      let nearest = 1e8;
      for (let obstacle = 0; obstacle < WIDTH; obstacle++) {
        if (occupancy[row * WIDTH + obstacle]) nearest = Math.min(nearest, (column - obstacle) ** 2);
      }
      horizontal[row * WIDTH + column] = nearest;
    }
  }
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      let nearest = 1e8;
      for (let obstacle = 0; obstacle < HEIGHT; obstacle++) {
        nearest = Math.min(nearest, horizontal[obstacle * WIDTH + column] + (row - obstacle) ** 2);
      }
      const boundary = Math.min(column + .5, row + .5, WIDTH - column - .5, HEIGHT - row - .5) * RESOLUTION;
      result[row * WIDTH + column] = Math.max(0, Math.min(boundary, Math.sqrt(nearest) * RESOLUTION - HALF_DIAGONAL));
    }
  }
  return result;
}

export function clearance(field, point) {
  const [positionX, positionY] = point;
  if (!Number.isFinite(positionX) || !Number.isFinite(positionY)
    || positionX <= 0 || positionY <= 0 || positionX >= WORLD_WIDTH || positionY >= WORLD_HEIGHT) return 0;
  const column = Math.floor(positionX / RESOLUTION);
  const row = Math.floor(positionY / RESOLUTION);
  const offset = Math.hypot(positionX - (column + .5) * RESOLUTION, positionY - (row + .5) * RESOLUTION);
  return Math.max(0, field[row * WIDTH + column] - offset);
}

export function segmentClear(field, first, second, required) {
  const length = Math.hypot(first[0] - second[0], first[1] - second[1]);
  const steps = Math.max(1, Math.ceil(length / (RESOLUTION / 4)));
  const samplingAllowance = length / steps / 2;
  for (let step = 0; step <= steps; step++) {
    const fraction = step / steps;
    const point = [first[0] + (second[0] - first[0]) * fraction, first[1] + (second[1] - first[1]) * fraction];
    if (clearance(field, point) < required + samplingAllowance) return false;
  }
  return true;
}

class MinHeap {
  constructor() { this.items = []; }
  push(entry) {
    let index = this.items.length;
    this.items.push(entry);
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.items[parent].score <= entry.score) break;
      this.items[index] = this.items[parent];
      index = parent;
    }
    this.items[index] = entry;
  }
  pop() {
    const result = this.items[0];
    const last = this.items.pop();
    if (this.items.length) {
      let index = 0;
      while (index * 2 + 1 < this.items.length) {
        let child = index * 2 + 1;
        if (child + 1 < this.items.length && this.items[child + 1].score < this.items[child].score) child++;
        if (this.items[child].score >= last.score) break;
        this.items[index] = this.items[child]; index = child;
      }
      this.items[index] = last;
    }
    return result;
  }
}

export function plan(field, start, goal, radius, margin) {
  const required = radius + margin;
  const base = {path: [], length: 0, minimumClearance: null, expanded: 0, required};
  if (clearance(field, start) < required) return {...base, status: 'start_unsafe'};
  if (clearance(field, goal) < required) return {...base, status: 'goal_unsafe'};
  const indexOf = point => Math.floor(point[1] / RESOLUTION) * WIDTH + Math.floor(point[0] / RESOLUTION);
  const pointOf = index => [(index % WIDTH + .5) * RESOLUTION, (Math.floor(index / WIDTH) + .5) * RESOLUTION];
  const first = indexOf(start);
  const last = indexOf(goal);
  if (!segmentClear(field, start, pointOf(first), required) || !segmentClear(field, pointOf(last), goal, required)) {
    return {...base, status: 'endpoint_grid_clearance'};
  }
  const costs = new Float64Array(WIDTH * HEIGHT).fill(Infinity);
  const parents = new Int32Array(WIDTH * HEIGHT).fill(-1);
  const visited = new Uint8Array(WIDTH * HEIGHT);
  const heap = new MinHeap();
  costs[first] = 0;
  heap.push({index: first, score: 0});
  let expanded = 0;
  while (heap.items.length) {
    const current = heap.pop().index;
    if (visited[current]) continue;
    visited[current] = 1;
    expanded++;
    if (current === last) break;
    const column = current % WIDTH;
    const row = Math.floor(current / WIDTH);
    const point = pointOf(current);
    for (let deltaRow = -1; deltaRow <= 1; deltaRow++) {
      for (let deltaColumn = -1; deltaColumn <= 1; deltaColumn++) {
        if ((!deltaColumn && !deltaRow) || column + deltaColumn < 0 || column + deltaColumn >= WIDTH
          || row + deltaRow < 0 || row + deltaRow >= HEIGHT) continue;
        const neighbor = (row + deltaRow) * WIDTH + column + deltaColumn;
        if (visited[neighbor] || field[neighbor] < required) continue;
        if (deltaColumn && deltaRow && (field[row * WIDTH + column + deltaColumn] < required
          || field[(row + deltaRow) * WIDTH + column] < required)) continue;
        const next = pointOf(neighbor);
        if (!segmentClear(field, point, next, required)) continue;
        const edge = Math.hypot(deltaColumn, deltaRow) * RESOLUTION;
        const score = costs[current] + edge * (1 + .08 / Math.max(.1, field[neighbor] - required));
        if (score >= costs[neighbor]) continue;
        costs[neighbor] = score;
        parents[neighbor] = current;
        heap.push({index: neighbor, score: score + Math.hypot(next[0] - goal[0], next[1] - goal[1])});
      }
    }
  }
  if (!visited[last]) return {...base, expanded, status: 'no_path'};
  const raw = [];
  for (let index = last; index !== -1; index = parents[index]) raw.push(pointOf(index));
  raw.reverse(); raw.unshift([...start]); raw.push([...goal]);
  const path = [raw[0]];
  let anchor = 0;
  while (anchor < raw.length - 1) {
    let next = raw.length - 1;
    while (next > anchor + 1 && !segmentClear(field, raw[anchor], raw[next], required)) next--;
    path.push(raw[next]); anchor = next;
  }
  let length = 0;
  let minimumClearance = Infinity;
  for (let index = 1; index < path.length; index++) {
    const firstPoint = path[index - 1];
    const secondPoint = path[index];
    const distance = Math.hypot(firstPoint[0] - secondPoint[0], firstPoint[1] - secondPoint[1]);
    length += distance;
    const steps = Math.max(1, Math.ceil(distance / (RESOLUTION / 4)));
    for (let step = 0; step <= steps; step++) {
      minimumClearance = Math.min(minimumClearance, clearance(field, [firstPoint[0] + (secondPoint[0] - firstPoint[0]) * step / steps,
        firstPoint[1] + (secondPoint[1] - firstPoint[1]) * step / steps]) - distance / steps / 2);
    }
  }
  return {status: 'ready', path, length, minimumClearance, required, expanded};
}

export function follow(field, position, path, waypoint, travel, required) {
  let next = [...position];
  while (travel > 0 && waypoint < path.length) {
    const goal = path[waypoint];
    const distance = Math.hypot(goal[0] - next[0], goal[1] - next[1]);
    const portion = Math.min(distance, travel);
    const candidate = distance < 1e-9 ? [...goal] : [next[0] + (goal[0] - next[0]) * portion / distance,
      next[1] + (goal[1] - next[1]) * portion / distance];
    const anchor = path[Math.max(0, waypoint - 1)];
    const segmentLength = Math.hypot(goal[0] - anchor[0], goal[1] - anchor[1]);
    const fromAnchor = Math.hypot(next[0] - anchor[0], next[1] - anchor[1]);
    if (Math.abs(fromAnchor + distance - segmentLength) > 1e-7
      || !segmentClear(field, anchor, goal, required)) return {position: next, waypoint, blocked: true};
    next = candidate; travel -= portion;
    if (distance <= portion + 1e-9) waypoint++; else break;
  }
  return {position: next, waypoint, blocked: false, reached: waypoint >= path.length};
}

export function mapDocument(occupancy, start, goal, radius, margin) {
  return {schema: 'cudarobotics.costmap.v1', width: WIDTH, height: HEIGHT, resolution: RESOLUTION,
    origin: [0, 0], rowOrder: 'bottom-to-top', occupancy: Array.from(occupancy), start: [...start], goal: [...goal], radius, margin};
}

export function parseMap(document) {
  if (!document || document.schema !== 'cudarobotics.costmap.v1' || document.width !== WIDTH
    || document.height !== HEIGHT || document.resolution !== RESOLUTION
    || document.rowOrder !== 'bottom-to-top' || !Array.isArray(document.origin)
    || document.origin.length !== 2 || document.origin.some(value => value !== 0)) throw new Error('Expected a 128 × 96 map at 0.125 m/cell, origin [0,0], bottom-to-top rows');
  if (!Array.isArray(document.occupancy) || document.occupancy.length !== WIDTH * HEIGHT
    || !document.occupancy.every(value => value === 0 || value === 1)) throw new Error('Occupancy must contain exactly 12,288 binary cells');
  for (const point of [document.start, document.goal]) {
    if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)
      || point[0] <= 0 || point[0] >= WORLD_WIDTH || point[1] <= 0 || point[1] >= WORLD_HEIGHT) throw new Error('Start and goal must be finite world coordinates inside the map');
  }
  if (!Number.isFinite(document.radius) || document.radius < .15 || document.radius > .75
    || Math.abs(document.radius / .05 - Math.round(document.radius / .05)) > 1e-6
    || !Number.isFinite(document.margin) || document.margin < 0 || document.margin > .4
    || Math.abs(document.margin / .02 - Math.round(document.margin / .02)) > 1e-6) {
    throw new Error('Radius must be 0.15–0.75 m in 0.05 m steps; margin 0–0.4 m in 0.02 m steps');
  }
  return {occupancy: new Uint8Array(document.occupancy), start: [...document.start], goal: [...document.goal],
    radius: document.radius, margin: document.margin};
}
