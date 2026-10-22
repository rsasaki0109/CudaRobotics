import {DEFAULT_GRID, distanceField, plan, parseMap, mapDocument} from './model.mjs?v=scale-comparison-1';
import {runComparison, summarize} from '../assets/benchmark.mjs';

export async function compareReplanning(document, {grid = DEFAULT_GRID, gpuEvaluate = null, cpuEvaluate = distanceField, planRoute = plan,
  now = () => performance.now(), onProgress = () => {}, ...options} = {}) {
  const input = parseMap(document, grid);
  const latest = {};
  const samples = {cpu: {field: [], route: [], routes: []}, gpu: {field: [], route: [], routes: []}};
  const evaluate = backend => async () => {
    const start = now();
    const field = await (backend === 'cpu' ? cpuEvaluate : gpuEvaluate)(input.occupancy, grid);
    const fieldEnd = now();
    const route = planRoute(field, input.start, input.goal, input.radius, input.margin, grid);
    const routeEnd = now();
    latest[backend] = {fieldMs: fieldEnd - start, routeMs: routeEnd - fieldEnd, route};
    return field;
  };
  const report = await runComparison({...options, now, length: grid.width * grid.height, absoluteTolerance: 1e-6,
    outputLabel: 'distance field', cpuEvaluate: evaluate('cpu'), gpuEvaluate: gpuEvaluate ? evaluate('gpu') : null,
    onProgress: progress => {
      const route = latest[progress.backend].route;
      if (!route || !['ready', 'no_path', 'start_unsafe', 'goal_unsafe', 'endpoint_grid_clearance'].includes(route.status)
        || !Number.isFinite(route.length) || route.length < 0 || !Array.isArray(route.path)
        || !route.path.every(point => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite))
        || (route.status === 'ready' ? route.path.length < 2 || !Number.isFinite(route.minimumClearance)
          : route.path.length !== 0 || route.length !== 0 || route.minimumClearance !== null)) {
        throw new Error(`Invalid ${progress.backend.toUpperCase()} route`);
      }
      if (progress.phase === 'measurement') {
        const measured = latest[progress.backend];
        samples[progress.backend].field.push(measured.fieldMs);
        samples[progress.backend].route.push(measured.routeMs);
        samples[progress.backend].routes.push(structuredClone(measured.route));
      }
      onProgress(progress);
    }});
  const routeAccuracy = gpuEvaluate ? {matches: true, checkedPairs: report.repetitions, mismatchedPairs: 0,
    statusMismatches: 0, pathMismatches: 0, clearanceMismatches: 0, maxWaypointErrorMetres: 0, maxLengthErrorMetres: 0,
    maxClearanceErrorMetres: 0, toleranceMetres: 1e-6} : null;
  if (routeAccuracy) {
    for (let index = 0; index < report.repetitions; index++) {
      const cpu = samples.cpu.routes[index];
      const gpu = samples.gpu.routes[index];
      const statusMatches = cpu.status === gpu.status;
      let pathMatches = cpu.path.length === gpu.path.length;
      if (pathMatches) {
        for (let waypoint = 0; waypoint < cpu.path.length; waypoint++) {
          const error = Math.hypot(cpu.path[waypoint][0] - gpu.path[waypoint][0], cpu.path[waypoint][1] - gpu.path[waypoint][1]);
          routeAccuracy.maxWaypointErrorMetres = Math.max(routeAccuracy.maxWaypointErrorMetres, error);
          if (error > routeAccuracy.toleranceMetres) pathMatches = false;
        }
      }
      const lengthError = Math.abs(cpu.length - gpu.length);
      const clearanceError = cpu.minimumClearance === null || gpu.minimumClearance === null
        ? cpu.minimumClearance === gpu.minimumClearance ? 0 : Infinity
        : Math.abs(cpu.minimumClearance - gpu.minimumClearance);
      routeAccuracy.maxLengthErrorMetres = Math.max(routeAccuracy.maxLengthErrorMetres, lengthError);
      if (Number.isFinite(clearanceError)) routeAccuracy.maxClearanceErrorMetres = Math.max(routeAccuracy.maxClearanceErrorMetres, clearanceError);
      if (!statusMatches) routeAccuracy.statusMismatches++;
      if (!pathMatches) routeAccuracy.pathMismatches++;
      if (clearanceError > 1e-6) routeAccuracy.clearanceMismatches++;
      if (!statusMatches || !pathMatches || lengthError > 1e-6 || clearanceError > 1e-6) routeAccuracy.mismatchedPairs++;
    }
    routeAccuracy.matches = routeAccuracy.mismatchedPairs === 0;
  }
  const stages = Object.fromEntries(['cpu', 'gpu'].map(backend => [backend, samples[backend].field.length
    ? {field: summarize(samples[backend].field), route: summarize(samples[backend].route)} : null]));
  const matches = report.accuracy?.matches && routeAccuracy?.matches;
  return {...report, schema: 'cudarobotics.replanning-benchmark.v1',
    input: mapDocument(input.occupancy, input.start, input.goal, input.radius, input.margin, grid),
    stages, routes: {cpu: samples.cpu.routes, gpu: gpuEvaluate ? samples.gpu.routes : null}, routeAccuracy,
    speedup: matches ? report.speedup : null,
    fieldSpeedup: matches && stages.gpu.field.medianMs > 0 && stages.cpu.field.medianMs > 0
      ? stages.cpu.field.medianMs / stages.gpu.field.medianMs : null,
    scope: 'CPU distance field + CPU A* versus WebGPU distance field + the same CPU A*; identical changed map, start, goal and footprint',
    timing: 'performance.now wall time; total includes field computation, transfers/readback, CPU routing and call overhead; stages measured within the same call; setup, warmups, agreement checks, debounce and drawing excluded',
    resources: 'single main-thread JavaScript CPU reference; GPU field buffers reused; CPU temporary and route allocations included; GPU does not run A*; not native CUDA/C++'};
}
