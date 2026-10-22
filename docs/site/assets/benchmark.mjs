export function summarize(samples) {
  if (!samples.length || !samples.every(value => Number.isFinite(value) && value >= 0)) throw new Error('Invalid timing samples');
  const sorted = [...samples].sort((first, second) => first - second);
  const percentile = fraction => {
    const index = (sorted.length - 1) * fraction;
    const low = Math.floor(index);
    return sorted[low] + (sorted[Math.ceil(index)] - sorted[low]) * (index - low);
  };
  return {medianMs: percentile(.5), p10Ms: percentile(.1), p90Ms: percentile(.9), samplesMs: [...samples]};
}

export async function runComparison({cpuEvaluate, gpuEvaluate = null, length, warmups = 3, repetitions = 15,
  absoluteTolerance = 1e-6, relativeTolerance = 0, classify = null, outputLabel = 'kernel output',
  now = () => performance.now(), yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
  shouldCancel = () => false, onProgress = () => {}}) {
  if (!Number.isInteger(length) || length < 1 || !Number.isInteger(warmups) || warmups < 0 || warmups > 20
    || !Number.isInteger(repetitions) || repetitions < 1 || repetitions > 100
    || !Number.isFinite(absoluteTolerance) || absoluteTolerance < 0
    || !Number.isFinite(relativeTolerance) || relativeTolerance < 0) throw new Error('Invalid benchmark settings');
  const samples = {cpu: [], gpu: []};
  const evaluators = {cpu: cpuEvaluate, gpu: gpuEvaluate};
  let maxAbsoluteError = 0;
  let mismatchedCellComparisons = 0;
  let classificationMismatches = 0;
  const check = () => { if (shouldCancel()) throw new DOMException('Comparison cancelled', 'AbortError'); };
  for (let iteration = 0; iteration < warmups + repetitions; iteration++) {
    const outputs = {};
    const order = !gpuEvaluate ? ['cpu'] : iteration % 2 ? ['gpu', 'cpu'] : ['cpu', 'gpu'];
    for (const backend of order) {
      check(); await yieldControl(); check();
      const start = now();
      const output = await evaluators[backend]();
      const duration = now() - start;
      check();
      if (!Number.isFinite(duration) || duration < 0) throw new Error('Invalid benchmark clock');
      if (output.length !== length || !output.every(value => Number.isFinite(value) && value >= 0)) throw new Error(`Invalid ${backend.toUpperCase()} ${outputLabel}`);
      outputs[backend] = output;
      if (iteration >= warmups) samples[backend].push(duration);
      onProgress({phase: iteration < warmups ? 'warmup' : 'measurement',
        iteration: iteration < warmups ? iteration + 1 : iteration - warmups + 1,
        total: iteration < warmups ? warmups : repetitions, backend});
    }
    if (gpuEvaluate && iteration >= warmups) {
      for (let index = 0; index < length; index++) {
        const error = Math.abs(outputs.cpu[index] - outputs.gpu[index]);
        const classificationMismatch = classify && classify(outputs.cpu[index]) !== classify(outputs.gpu[index]);
        if (classificationMismatch) classificationMismatches++;
        maxAbsoluteError = Math.max(maxAbsoluteError, error);
        if (classificationMismatch || error > Math.max(absoluteTolerance, relativeTolerance * Math.abs(outputs.cpu[index]))) mismatchedCellComparisons++;
      }
    }
  }
  check();
  const cpu = summarize(samples.cpu);
  const gpu = gpuEvaluate ? summarize(samples.gpu) : null;
  const matches = gpu ? mismatchedCellComparisons === 0 : null;
  return {warmups, repetitions, cpu, gpu, order: 'alternating CPU-first / GPU-first pairs; yielding before each call',
    accuracy: gpu ? {matches, maxAbsoluteError, absoluteTolerance, relativeTolerance,
      mismatchedCellComparisons, classificationMismatches, checkedCellComparisons: length * repetitions} : null,
    speedup: matches && cpu.medianMs > 0 && gpu.medianMs > 0 ? cpu.medianMs / gpu.medianMs : null};
}
