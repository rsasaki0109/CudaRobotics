export function drawBudgetPaths(canvas, scene, trials) {
  const context = canvas.getContext('2d');
  const scale = canvas.width / 10;
  context.fillStyle = '#0c1921'; context.fillRect(0, 0, canvas.width, canvas.height);
  if (!scene) {
    context.fillStyle = '#a6bec8'; context.font = '14px system-ui';
    context.fillText('Run the budget experiment to see paths.', 16, 28); return;
  }
  for (const [left, bottom, right, top] of scene.obstacles) {
    context.fillStyle = '#354b58'; context.fillRect(left * scale, (10 - top) * scale, (right - left) * scale, (top - bottom) * scale);
  }
  context.strokeStyle = '#ffcc7c'; context.lineWidth = 2;
  context.beginPath(); context.arc(scene.goal[0] * scale, (10 - scene.goal[1]) * scale, 8, 0, Math.PI * 2); context.stroke();
  const colors = ['#99c8ff', '#73e4c0', '#ffcc7c'];
  for (const [index, trial] of (trials ?? []).entries()) {
    context.strokeStyle = colors[index % colors.length]; context.lineWidth = 2;
    context.beginPath();
    for (const [step, point] of trial.path.entries()) context[step ? 'lineTo' : 'moveTo'](point[0] * scale, (10 - point[1]) * scale);
    context.stroke();
    const end = trial.path.at(-1);
    context.fillStyle = colors[index % colors.length]; context.beginPath();
    context.arc(end[0] * scale, (10 - end[1]) * scale, 3, 0, Math.PI * 2); context.fill();
  }
}

export function renderBudget(elements, report) {
  elements['budget-calibration'].textContent = report.calibrations.map(({count, report: measured}) => {
    const timing = backend => measured[backend] ? `${measured[backend].p90Ms.toFixed(2)} ms` : 'unavailable';
    const agreement = measured.gpu ? measured.accuracy.matches && measured.commandAccuracy.matches ? 'verified' : 'MISMATCH; excluded' : 'CPU only';
    return `${count.toLocaleString()} candidates: CPU P90 ${timing('cpu')} / GPU P90 ${timing('gpu')} · ${agreement}`;
  }).join('\n');
  for (const backend of ['cpu', 'gpu']) {
    const summary = report.summaries[backend];
    elements[`budget-${backend}-status`].textContent = !summary ? report.selections[backend].reason
      : `${summary.count.toLocaleString()} candidates · ${summary.reached}/${summary.trials} reached · ${summary.safeStops} safe stops · ${summary.deadlineStops} deadline stops · ${summary.timeouts} timeouts. ${summary.deadlineMisses} late plans discarded. Mean recorded path ${summary.meanPathLength.toFixed(2)} m across all trials, including stops.`;
    elements[`budget-${backend}-trials`].textContent = !summary ? 'Not tested. No mission outcome or timing is invented.'
      : report.results[backend].map(trial => `Seed ${trial.seed}: ${trial.status}, ${trial.pathLength.toFixed(2)} m, ${trial.planning.timing?.samplesMs.length ?? 0} plans; max plan ${Math.max(0, ...(trial.planning.timing?.samplesMs ?? [])).toFixed(2)} ms`).join('\n');
    drawBudgetPaths(elements[`budget-${backend}-canvas`], report.scene, report.results[backend]);
  }
}
