import {DEFAULT_GRID, parseMap, mapDocument, plan} from './model.mjs';
import {distanceFieldEdt} from './edt.mjs';
import {resampleMap} from './scaling.mjs';
import {summarize} from '../assets/benchmark.mjs';
import {validateRoute, routeAgreement} from './route-checks.mjs';

export const IMPLEMENTATIONS = Object.freeze({
  cpuEdt: Object.freeze({name: 'CPU EDT', algorithm: 'lower-envelope-exact-edt', fieldWorkComplexity: 'O(W×H)', execution: 'main-thread JavaScript'}),
  gpuReference: Object.freeze({name: 'GPU brute force', algorithm: 'brute-force-separable', fieldWorkComplexity: 'O(W×H×(W+H))', execution: 'WebGPU'}),
  gpuEdt: Object.freeze({name: 'GPU EDT', algorithm: 'binary-row-sweeps-column-envelope-exact-edt', fieldWorkComplexity: 'O(W×H)', execution: 'WebGPU; one invocation per line'}),
});
const permutations = [
  ['cpuEdt', 'gpuReference', 'gpuEdt'], ['gpuReference', 'gpuEdt', 'cpuEdt'], ['gpuEdt', 'cpuEdt', 'gpuReference'],
  ['cpuEdt', 'gpuEdt', 'gpuReference'], ['gpuEdt', 'gpuReference', 'cpuEdt'], ['gpuReference', 'cpuEdt', 'gpuEdt'],
];
export const IMPLEMENTATION_PAIRS = Object.freeze([
  Object.freeze(['cpuEdt', 'gpuReference']), Object.freeze(['cpuEdt', 'gpuEdt']), Object.freeze(['gpuReference', 'gpuEdt']),
]);

export async function compareImplementations(document, {grid = DEFAULT_GRID, cpuEvaluate = distanceFieldEdt,
  gpuReferenceEvaluate = null, gpuEdtEvaluate = null, planRoute = plan, warmups = 3, repetitions = 15,
  now = () => performance.now(), yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
  shouldCancel = () => false, onProgress = () => {}} = {}) {
  if (!Number.isInteger(warmups) || warmups < 0 || warmups > 20 || !Number.isInteger(repetitions)
    || repetitions < 1 || repetitions > 100 || typeof cpuEvaluate !== 'function'
    || (gpuReferenceEvaluate !== null && typeof gpuReferenceEvaluate !== 'function')
    || (gpuEdtEvaluate !== null && typeof gpuEdtEvaluate !== 'function')) throw new Error('Invalid implementation benchmark settings');
  const source = parseMap(document);
  const snapshot = mapDocument(source.occupancy, source.start, source.goal, source.radius, source.margin);
  const scaled = resampleMap(snapshot, grid);
  const input = parseMap(scaled, grid);
  const evaluators = {cpuEdt: cpuEvaluate, gpuReference: gpuReferenceEvaluate, gpuEdt: gpuEdtEvaluate};
  const ids = Object.keys(evaluators).filter(id => evaluators[id]);
  const samples = Object.fromEntries(ids.map(id => [id, {total: [], field: [], route: [], routes: []}]));
  const fields = Object.fromEntries(IMPLEMENTATION_PAIRS.map(([first, second]) => [`${first}:${second}`,
    evaluators[first] && evaluators[second] ? {matches: true, maxAbsoluteError: 0, mismatchedCellComparisons: 0,
      checkedCellComparisons: 0, toleranceMetres: 1e-6} : null]));
  const check = () => { if (shouldCancel()) throw new DOMException('Implementation comparison cancelled', 'AbortError'); };
  const orders = [];
  for (let iteration = 0; iteration < warmups + repetitions; iteration++) {
    const order = ids.length === 3 ? permutations[iteration % permutations.length]
      : ids.length === 2 && iteration % 2 ? [...ids].reverse() : [...ids];
    orders.push([...order]);
    const outputs = {};
    for (const id of order) {
      check(); await yieldControl(); check();
      const start = now();
      const field = await evaluators[id](input.occupancy, grid);
      const fieldEnd = now();
      const route = planRoute(field, input.start, input.goal, input.radius, input.margin, grid);
      const end = now();
      check();
      const times = {total: end - start, field: fieldEnd - start, route: end - fieldEnd};
      if (!Object.values(times).every(value => Number.isFinite(value) && value >= 0)) throw new Error('Invalid implementation benchmark clock');
      if (field.length !== grid.width * grid.height || !field.every(value => Number.isFinite(value) && value >= 0)) throw new Error(`Invalid ${id} distance field`);
      validateRoute(route, id);
      outputs[id] = field;
      if (iteration >= warmups) {
        for (const stage of ['total', 'field', 'route']) samples[id][stage].push(times[stage]);
        samples[id].routes.push(structuredClone(route));
      }
      onProgress({backend: id, phase: iteration < warmups ? 'warmup' : 'measurement',
        iteration: iteration < warmups ? iteration + 1 : iteration - warmups + 1,
        total: iteration < warmups ? warmups : repetitions});
      check();
    }
    if (iteration >= warmups) {
      for (const [first, second] of IMPLEMENTATION_PAIRS) {
        const accuracy = fields[`${first}:${second}`];
        if (!accuracy) continue;
        for (let index = 0; index < outputs[first].length; index++) {
          const error = Math.abs(outputs[first][index] - outputs[second][index]);
          accuracy.maxAbsoluteError = Math.max(accuracy.maxAbsoluteError, error);
          if (error > accuracy.toleranceMetres) accuracy.mismatchedCellComparisons++;
          accuracy.checkedCellComparisons++;
        }
      }
    }
  }
  check();
  const backends = Object.fromEntries(Object.keys(evaluators).map(id => [id, samples[id] ? {
    ...IMPLEMENTATIONS[id], total: summarize(samples[id].total), field: summarize(samples[id].field),
    route: summarize(samples[id].route), routes: samples[id].routes,
  } : null]));
  const comparisons = Object.fromEntries(IMPLEMENTATION_PAIRS.map(([first, second]) => {
    const accuracy = fields[`${first}:${second}`];
    if (!accuracy) return [`${first}:${second}`, null];
    accuracy.matches = accuracy.mismatchedCellComparisons === 0;
    const routes = routeAgreement(samples[first].routes, samples[second].routes);
    const ratio = stage => accuracy.matches && routes.matches && backends[first][stage].medianMs > 0 && backends[second][stage].medianMs > 0
      ? backends[first][stage].medianMs / backends[second][stage].medianMs : null;
    return [`${first}:${second}`, {first, second, fieldAccuracy: accuracy, routeAccuracy: routes,
      totalRatio: ratio('total'), fieldRatio: ratio('field')}];
  }));
  return {schema: 'cudarobotics.implementation-benchmark.v1', source: snapshot, input: scaled,
    warmups, repetitions, backends, comparisons, orders,
    order: 'six balanced permutations when all three exist; alternating pairs when two exist; yielding before every call',
    timing: 'per-call wall time split into field and CPU A*; upload/readback, CPU allocations and GPU scratch initialization included; GPU setup/compilation, resampling, validation, agreement checks and drawing excluded',
    scope: 'same resampled map, captured start, goal and footprint; all routes use the same CPU A*; independent of the paired-panel CPU baseline selector',
    limitations: 'different exact field procedures, not a pure hardware comparison or fastest-native claim; GPU EDT has sequential line work and may lose to the reference GPU or CPU; blocked outcomes are not successful missions; WebGPU/JavaScript, not CUDA'};
}
