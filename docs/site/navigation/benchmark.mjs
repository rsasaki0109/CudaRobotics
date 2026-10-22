import {WIDTH, HEIGHT, distanceField} from './model.mjs';

export function summarize(samples) {
  if (!samples.length || !samples.every(value => Number.isFinite(value) && value >= 0)) {
    throw new Error('Timing samples must be finite, non-negative and non-empty');
  }
  const sorted = [...samples].sort((first, second) => first - second);
  const percentile = fraction => {
    const index = (sorted.length - 1) * fraction;
    const low = Math.floor(index);
    return sorted[low] + (sorted[Math.ceil(index)] - sorted[low]) * (index - low);
  };
  return {medianMs: percentile(.5), p10Ms: percentile(.1), p90Ms: percentile(.9), samplesMs: [...samples]};
}

export async function compareDistanceFields(occupancy, {
  gpuEvaluate = null, cpuEvaluate = distanceField, warmups = 3, repetitions = 15,
  now = () => performance.now(), yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
  shouldCancel = () => false, onProgress = () => {}, tolerance = 1e-6,
} = {}) {
  if (occupancy.length !== WIDTH * HEIGHT || !occupancy.every(value => value === 0 || value === 1)) {
    throw new Error('Expected the fixed binary occupancy grid');
  }
  if (!Number.isInteger(warmups) || warmups < 0 || warmups > 20
    || !Number.isInteger(repetitions) || repetitions < 1 || repetitions > 100
    || !Number.isFinite(tolerance) || tolerance < 0) throw new Error('Invalid benchmark settings');
  const snapshot = occupancy.slice();
  const samples = {cpu: [], gpu: []};
  const evaluators = {cpu: cpuEvaluate, gpu: gpuEvaluate};
  let maxAbsoluteError = 0;
  let mismatchedComparisons = 0;
  const checkCancellation = () => {
    if (shouldCancel()) throw new DOMException('Comparison cancelled; input changed or page became hidden', 'AbortError');
  };
  for (let iteration = 0; iteration < warmups + repetitions; iteration++) {
    const outputs = {};
    const order = !gpuEvaluate ? ['cpu'] : iteration % 2 ? ['gpu', 'cpu'] : ['cpu', 'gpu'];
    for (const backend of order) {
      checkCancellation();
      await yieldControl();
      checkCancellation();
      const start = now();
      const output = await evaluators[backend](snapshot);
      const duration = now() - start;
      checkCancellation();
      if (!Number.isFinite(duration) || duration < 0) throw new Error('Invalid benchmark clock');
      if (output.length !== WIDTH * HEIGHT || !output.every(value => Number.isFinite(value) && value >= 0)) {
        throw new Error(`Invalid ${backend.toUpperCase()} distance field`);
      }
      outputs[backend] = output;
      if (iteration >= warmups) samples[backend].push(duration);
      onProgress({phase: iteration < warmups ? 'warmup' : 'measurement',
        iteration: iteration < warmups ? iteration + 1 : iteration - warmups + 1,
        total: iteration < warmups ? warmups : repetitions, backend});
    }
    if (gpuEvaluate && iteration >= warmups) {
      for (let index = 0; index < outputs.cpu.length; index++) {
        const error = Math.abs(outputs.cpu[index] - outputs.gpu[index]);
        maxAbsoluteError = Math.max(maxAbsoluteError, error);
        if (error > tolerance) mismatchedComparisons++;
      }
    }
  }
  checkCancellation();
  const cpu = summarize(samples.cpu);
  const gpu = gpuEvaluate ? summarize(samples.gpu) : null;
  const matches = gpu ? mismatchedComparisons === 0 : null;
  return {schema: 'cudarobotics.distance-benchmark.v1',
    method: 'separable squared distance to occupied cell centres, conservative metre clearance',
    scope: 'distance field only; A*, rendering, initialization and compilation excluded',
    timing: 'performance.now wall time; GPU includes upload, dispatch, synchronization, readback and output copy',
    resources: 'GPU buffers reused; JavaScript CPU reference allocates temporary and output arrays per call',
    order: 'alternating CPU-first / GPU-first pairs, yielding before each backend',
    width: WIDTH, height: HEIGHT, warmups, repetitions, cpu, gpu,
    accuracy: gpu ? {matches, maxAbsoluteError, toleranceMetres: tolerance, mismatchedCellComparisons: mismatchedComparisons,
      checkedCellComparisons: WIDTH * HEIGHT * repetitions} : null,
    speedup: matches && cpu.medianMs > 0 && gpu.medianMs > 0 ? cpu.medianMs / gpu.medianMs : null};
}
