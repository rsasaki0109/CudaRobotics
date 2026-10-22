import {WIDTH, HEIGHT, RESOLUTION} from './model.mjs';

export function placePallet(occupancy, path, start, goal, required) {
  const lengths = path.slice(1).map((point, index) => Math.hypot(point[0] - path[index][0], point[1] - path[index][1]));
  const total = lengths.reduce((sum, length) => sum + length, 0);
  for (const fraction of [.5, .35, .65, .2, .8]) {
    let remaining = total * fraction;
    for (let index = 0; index < lengths.length; index++) {
      if (remaining > lengths[index]) { remaining -= lengths[index]; continue; }
      const portion = lengths[index] > 0 ? remaining / lengths[index] : 0;
      const center = path[index].map((value, axis) => value + (path[index + 1][axis] - value) * portion);
      const column = Math.floor(center[0] / RESOLUTION) - 4;
      const row = Math.floor(center[1] / RESOLUTION) - 4;
      if (column < 0 || row < 0 || column + 8 > WIDTH || row + 8 > HEIGHT) break;
      const bounds = [column * RESOLUTION, row * RESOLUTION, (column + 8) * RESOLUTION, (row + 8) * RESOLUTION];
      const distance = point => Math.hypot(Math.max(bounds[0] - point[0], 0, point[0] - bounds[2]),
        Math.max(bounds[1] - point[1], 0, point[1] - bounds[3]));
      if (distance(start) <= required + RESOLUTION || distance(goal) <= required + RESOLUTION) break;
      const cells = [];
      for (let nextRow = row; nextRow < row + 8; nextRow++) {
        for (let nextColumn = column; nextColumn < column + 8; nextColumn++) cells.push(nextRow * WIDTH + nextColumn);
      }
      if (cells.some(index => occupancy[index])) break;
      const next = occupancy.slice();
      for (const index of cells) next[index] = 1;
      return {occupancy: next, bounds, cells};
    }
  }
  return null;
}
