import {WIDTH, HEIGHT} from './model.mjs';
import {cpuBaseline as selectCpuBaseline} from './cpu-baseline.mjs';
import {runComparison} from '../assets/benchmark.mjs';
export {summarize} from '../assets/benchmark.mjs';

export async function compareDistanceFields(occupancy, {gpuEvaluate = null, cpuEvaluate = null, cpuBaseline = 'matched',
  tolerance = 1e-6, ...options} = {}) {
  if (occupancy.length !== WIDTH * HEIGHT || !occupancy.every(value => value === 0 || value === 1)) throw new Error('Expected the fixed binary occupancy grid');
  const snapshot = occupancy.slice();
  const baseline = selectCpuBaseline(cpuBaseline);
  const evaluateCpu = cpuEvaluate ?? baseline.evaluate;
  const report = await runComparison({...options, length: WIDTH * HEIGHT, absoluteTolerance: tolerance,
    outputLabel: 'distance field', cpuEvaluate: () => evaluateCpu(snapshot), gpuEvaluate: gpuEvaluate ? () => gpuEvaluate(snapshot) : null});
  return {...report, schema: 'cudarobotics.distance-benchmark.v1',
    method: 'separable squared distance to occupied cell centres, conservative metre clearance',
    cpuBaseline: baseline.description,
    scope: 'distance field only; A*, rendering, initialization and compilation excluded',
    timing: 'performance.now wall time; GPU includes upload, dispatch, synchronization, readback and output copy',
    resources: 'GPU buffers reused; JavaScript CPU reference allocates temporary and output arrays per call',
    width: WIDTH, height: HEIGHT,
    accuracy: report.accuracy ? {...report.accuracy, toleranceMetres: tolerance} : null};
}
