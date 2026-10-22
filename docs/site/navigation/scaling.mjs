import {WIDTH, HEIGHT, parseMap, mapDocument} from './model.mjs?v=scale-comparison-1';
import {compareReplanning} from './replanning.mjs?v=scale-comparison-1';

export const SCALE_GRIDS = Object.freeze([64, 128, 256].map(width => Object.freeze({width, height: width * 3 / 4, resolution: 16 / width})));

function supportedGrid(grid) {
  return SCALE_GRIDS.some(supported => supported.width === grid?.width && supported.height === grid.height && supported.resolution === grid.resolution);
}

export function resampleMap(document, grid) {
  if (!supportedGrid(grid)) throw new Error('Expected a supported 64×48, 128×96 or 256×192 grid');
  const source = parseMap(document);
  const occupancy = new Uint8Array(grid.width * grid.height);
  for (let row = 0; row < grid.height; row++) {
    for (let column = 0; column < grid.width; column++) {
      for (let sourceRow = Math.floor(row * HEIGHT / grid.height); sourceRow < Math.ceil((row + 1) * HEIGHT / grid.height); sourceRow++) {
        for (let sourceColumn = Math.floor(column * WIDTH / grid.width); sourceColumn < Math.ceil((column + 1) * WIDTH / grid.width); sourceColumn++) {
          if (source.occupancy[sourceRow * WIDTH + sourceColumn]) occupancy[row * grid.width + column] = 1;
        }
      }
    }
  }
  return mapDocument(occupancy, source.start, source.goal, source.radius, source.margin, grid);
}

export async function compareMapSizes(document, {grids = SCALE_GRIDS, createGpu = null,
  shouldCancel = () => false, onProgress = () => {}, ...options} = {}) {
  if (!grids.length || grids.length > 3 || grids.some(grid => !supportedGrid(grid))
    || new Set(grids.map(grid => grid.width)).size !== grids.length) throw new Error('Invalid comparison sizes');
  const source = parseMap(document);
  const snapshot = mapDocument(source.occupancy, source.start, source.goal, source.radius, source.margin);
  const results = [];
  const check = () => { if (shouldCancel()) throw new DOMException('Scale comparison cancelled', 'AbortError'); };
  for (const grid of grids) {
    check();
    onProgress({grid, phase: 'setup', sizeIndex: results.length + 1, sizeCount: grids.length});
    let gpu = null;
    let reason = createGpu ? '' : 'No GPU evaluator requested';
    try {
      if (createGpu) {
        try { gpu = await createGpu(grid); }
        catch (error) { reason = error.message; }
      }
      check();
      const report = await compareReplanning(resampleMap(snapshot, grid), {...options, grid, shouldCancel,
        gpuEvaluate: gpu ? occupancy => gpu.evaluate(occupancy) : null,
        onProgress: progress => onProgress({...progress, grid, sizeIndex: results.length + 1, sizeCount: grids.length})});
      check();
      results.push({grid: {...grid}, report: {...report, gpuUnavailableReason: reason}});
    } finally { gpu?.destroy(); }
  }
  check();
  return {schema: 'cudarobotics.scale-benchmark.v1', source: snapshot, results,
    method: 'fixed 16×12 m world; conservative occupied-cell resampling; same start, goal and footprint; CPU/GPU compared within each grid, not route equality across grids',
    timing: 'small-to-large sequential run; per-size GPU allocation/compilation excluded; per-call field, CPU route and total timing included; not a universal crossover threshold',
    limitations: 'resolution changes discretization, clearance conservatism, route geometry and possibly reachability; editor import stays fixed at 128×96; GPU A* and native CUDA are not measured'};
}
