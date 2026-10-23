function stageShares(measured) {
  const stages = measured.stages?.gpu;
  if (!stages) return null;
  const names = ['sampling', 'rollout', 'reduction'];
  const totals = stages.total?.samplesMs;
  if (!totals?.length || !totals.every(value => Number.isFinite(value) && value >= 0)
    || !names.every(name => stages[name]?.samplesMs?.length === totals.length
      && stages[name].samplesMs.every(value => Number.isFinite(value) && value >= 0))) return null;
  for (let index = 0; index < totals.length; index++) {
    const sum = names.reduce((total, name) => total + stages[name].samplesMs[index], 0);
    if (Math.abs(sum - totals[index]) > Math.max(1e-6, totals[index] * 1e-6)) return null;
  }
  const total = totals.reduce((sum, value) => sum + value, 0);
  if (!(total > 0) || !Number.isFinite(total)) return null;
  return Object.fromEntries(names.map(name => [name, stages[name].samplesMs.reduce((sum, value) => sum + value, 0) / total]));
}

export function workloadInsight(measured, count, measurement) {
  const scope = measurement === 'plan' ? 'Complete cold-start plan' : 'Rollout costs';
  const base = {count, scope, winner: null, ratio: null, rolloutRatio: null, gpuPathShares: null, rangesOverlap: null};
  if (!measured.cpu || !measured.gpu) return {...base, status: 'unavailable',
    verdict: 'CPU measurements only; GPU comparison unavailable.', effect: 'No GPU timing or share is inferred.'};
  if (!measured.accuracy?.matches || (measurement === 'plan' && measured.commandAccuracy?.matches !== true)
    || measured.commandAccuracy?.matches === false) return {...base, status: 'mismatch',
    verdict: 'Outputs disagree. No performance recommendation.', effect: 'Ratios and stage-share interpretation are suppressed.'};
  const cpu = measured.cpu.medianMs;
  const gpu = measured.gpu.medianMs;
  if (!(cpu > 0) || !(gpu > 0) || !Number.isFinite(cpu) || !Number.isFinite(gpu)) return {...base, status: 'timer',
    verdict: 'Timer precision is insufficient for a comparison.', effect: 'No finite positive median ratio is available.'};
  const ratio = cpu / gpu;
  if (!Number.isFinite(ratio) || ratio <= 0 || !Number.isFinite(gpu / cpu)) return {...base, status: 'timer',
    verdict: 'Timer precision is insufficient for a comparison.', effect: 'No finite positive median ratio is available.'};
  const winner = cpu === gpu ? 'tie' : cpu > gpu ? 'gpu' : 'cpu';
  const ranges = [measured.cpu.p10Ms, measured.cpu.p90Ms, measured.gpu.p10Ms, measured.gpu.p90Ms];
  const rangesOverlap = ranges.every(Number.isFinite) ? Math.max(ranges[0], ranges[2]) <= Math.min(ranges[1], ranges[3]) : null;
  const verdict = winner === 'tie' ? `${scope}: equal measured medians.`
    : `${scope}: ${winner.toUpperCase()} ${(winner === 'gpu' ? ratio : 1 / ratio).toFixed(2)}× faster by measured median.`;
  const cpuRollout = measured.stages?.cpu?.rollout?.medianMs;
  const gpuRollout = measured.stages?.gpu?.rollout?.medianMs;
  const candidate = cpuRollout / gpuRollout;
  const rolloutRatio = measurement === 'plan' && cpuRollout > 0 && gpuRollout > 0 && Number.isFinite(candidate) && candidate > 0 ? candidate : null;
  const gpuPathShares = measurement === 'plan' ? stageShares(measured) : null;
  let effect = measurement === 'plan' ? rolloutRatio === null ? 'Rollout-stage ratio is unavailable or below timer precision.'
    : `Rollout-stage CPU/GPU ratio ${rolloutRatio.toFixed(2)}×; whole-plan ratio ${ratio.toFixed(2)}×. These are different measured scopes.`
    : 'Rollout-only timing does not establish whole-plan speed or mission success.';
  if (gpuPathShares) effect += ` CPU sampling + softmin account for ${((gpuPathShares.sampling + gpuPathShares.reduction) * 100).toFixed(1)}% of GPU-path measured wall time.`;
  if (rangesOverlap) effect += ' P10–P90 ranges overlap; this is not a statistically established winner.';
  else effect += ' Descriptive medians, not a statistical significance test.';
  return {...base, status: 'measured', winner, ratio, rolloutRatio, gpuPathShares, rangesOverlap, verdict, effect};
}

export function summarizeWorkloads(report) {
  const workloads = report.results.map(entry => workloadInsight(entry.report, entry.count, report.measurement));
  const gpuWins = workloads.filter(entry => entry.winner === 'gpu').map(entry => entry.count);
  const cpuWins = workloads.filter(entry => entry.winner === 'cpu').map(entry => entry.count);
  const ties = workloads.filter(entry => entry.winner === 'tie').map(entry => entry.count);
  const incomplete = workloads.some(entry => entry.status !== 'measured');
  const conclusion = workloads.every(entry => entry.status === 'unavailable') ? 'CPU measurements only. No GPU/CPU conclusion.'
    : `Measured median wins: GPU ${gpuWins.length}, CPU ${cpuWins.length}, equal ${ties.length} of ${workloads.length} workloads.${incomplete ? ' Some workloads cannot support a conclusion; inspect their warnings.' : ''} This is a fixed-input experiment, not a universal hardware recommendation.`;
  return {conclusion, gpuWins, cpuWins, ties, workloads,
    shareMethod: 'GPU-path stage shares use summed paired raw wall-time samples, not ratios of stage medians; rollout includes uploads, synchronization and readback, not isolated GPU kernel time or utilization',
    caveats: 'P10–P90 overlap is descriptive, not a significance test; cold-start plans exclude motion, safety guard and drawing; mission success is a separate experiment'};
}
