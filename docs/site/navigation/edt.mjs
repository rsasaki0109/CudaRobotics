import {DEFAULT_GRID} from './model.mjs';

function transformLine(input, output, sites, boundaries, length) {
  let last = 0;
  sites[0] = 0; boundaries[0] = -Infinity; boundaries[1] = Infinity;
  for (let candidate = 1; candidate < length; candidate++) {
    let intersection;
    while (true) {
      const anchor = sites[last];
      intersection = ((input[candidate] + candidate * candidate) - (input[anchor] + anchor * anchor)) / (2 * (candidate - anchor));
      if (intersection > boundaries[last]) break;
      last--;
    }
    last++; sites[last] = candidate;
    boundaries[last] = intersection; boundaries[last + 1] = Infinity;
  }
  last = 0;
  for (let position = 0; position < length; position++) {
    while (boundaries[last + 1] < position) last++;
    const delta = position - sites[last];
    output[position] = delta * delta + input[sites[last]];
  }
}

export function distanceFieldEdt(occupancy, grid = DEFAULT_GRID) {
  const {width, height, resolution} = grid;
  if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1
    || !Number.isFinite(resolution) || resolution <= 0 || occupancy.length !== width * height) throw new Error('Invalid EDT grid');
  const horizontal = new Float64Array(width * height);
  const result = new Float32Array(width * height);
  const longest = Math.max(width, height);
  const input = new Float64Array(longest);
  const output = new Float64Array(longest);
  const sites = new Int32Array(longest);
  const boundaries = new Float64Array(longest + 1);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) input[column] = occupancy[row * width + column] ? 0 : 1e8;
    transformLine(input, output, sites, boundaries, width);
    for (let column = 0; column < width; column++) horizontal[row * width + column] = output[column];
  }
  const halfDiagonal = Math.SQRT2 * resolution / 2;
  for (let column = 0; column < width; column++) {
    for (let row = 0; row < height; row++) input[row] = horizontal[row * width + column];
    transformLine(input, output, sites, boundaries, height);
    for (let row = 0; row < height; row++) {
      const boundary = Math.min(column + .5, row + .5, width - column - .5, height - row - .5) * resolution;
      result[row * width + column] = Math.max(0, Math.min(boundary, Math.sqrt(output[row]) * resolution - halfDiagonal));
    }
  }
  return result;
}
