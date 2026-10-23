import {DEFAULT_GRID, plan, parseMap, mapDocument} from './model.mjs?v=scale-comparison-1';
import {cpuBaseline as selectCpuBaseline} from './cpu-baseline.mjs';
import {runComparison, summarize} from '../assets/benchmark.mjs';
import {validateRoute, routeAgreement} from './route-checks.mjs';

export async function compareReplanning(document, {grid = DEFAULT_GRID, gpuEvaluate = null, cpuEvaluate = null, cpuBaseline = 'matched', planRoute = plan,
  now = () => performance.now(), onProgress = () => {}, ...options} = {}) {
  const input = parseMap(document, grid);
  const baseline = selectCpuBaseline(cpuBaseline);
  const evaluateCpu = cpuEvaluate ?? baseline.evaluate;
  const latest = {};
  const samples = {cpu: {field: [], route: [], routes: []}, gpu: {field: [], route: [], routes: []}};
  const evaluate = backend => async () => {
    const start = now();
    const field = await (backend === 'cpu' ? evaluateCpu : gpuEvaluate)(input.occupancy, grid);
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
      validateRoute(route, progress.backend);
      if (progress.phase === 'measurement') {
        const measured = latest[progress.backend];
        samples[progress.backend].field.push(measured.fieldMs);
        samples[progress.backend].route.push(measured.routeMs);
        samples[progress.backend].routes.push(structuredClone(measured.route));
      }
      onProgress(progress);
    }});
  const routeAccuracy = routeAgreement(samples.cpu.routes, gpuEvaluate ? samples.gpu.routes : null);
  const stages = Object.fromEntries(['cpu', 'gpu'].map(backend => [backend, samples[backend].field.length
    ? {field: summarize(samples[backend].field), route: summarize(samples[backend].route)} : null]));
  const matches = report.accuracy?.matches && routeAccuracy?.matches;
  return {...report, schema: 'cudarobotics.replanning-benchmark.v1',
    cpuBaseline: baseline.description,
    input: mapDocument(input.occupancy, input.start, input.goal, input.radius, input.margin, grid),
    stages, routes: {cpu: samples.cpu.routes, gpu: gpuEvaluate ? samples.gpu.routes : null}, routeAccuracy,
    speedup: matches ? report.speedup : null,
    fieldSpeedup: matches && stages.gpu.field.medianMs > 0 && stages.cpu.field.medianMs > 0
      ? stages.cpu.field.medianMs / stages.gpu.field.medianMs : null,
    scope: 'CPU distance field + CPU A* versus WebGPU distance field + the same CPU A*; identical changed map, start, goal and footprint',
    timing: 'performance.now wall time; total includes field computation, transfers/readback, CPU routing and call overhead; stages measured within the same call; setup, warmups, agreement checks, debounce and drawing excluded',
    resources: 'single main-thread JavaScript CPU reference; GPU field buffers reused; CPU temporary and route allocations included; GPU does not run A*; not native CUDA/C++'};
}
