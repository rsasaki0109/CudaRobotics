import {WORKLOAD_COUNTS, workloadMatches, compareWorkloads, compareMissions} from './workloads.mjs?v=workload-insights-1';

export class WorkloadPanel {
  constructor(elements, {snapshot, createJob, createGpu, pause, idle, finished}) {
    Object.assign(this, {elements, snapshot, createJob, createGpu, pause, idle, finished});
    this.busy = false; this.ready = false; this.generation = 0; this.reports = {};
    for (const kind of ['workloads', 'missions']) {
      elements[`compare-${kind}`].addEventListener('click', () => this.run(kind));
      elements[`export-${kind}`].addEventListener('click', () => {
        if (!this.reports[kind]) return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(this.reports[kind], null, 2)], {type: 'application/json'}));
        const anchor = document.createElement('a'); anchor.href = url; anchor.download = `mppi-${kind}.json`; anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
    }
    elements['cancel-workloads'].addEventListener('click', () => this.invalidate());
    elements['mission-count'].addEventListener('change', () => this.invalidate());
    elements['workload-scope'].addEventListener('change', () => this.invalidate());
    this.invalidate();
  }
  setReady(ready) {
    this.ready = ready;
    for (const kind of ['workloads', 'missions']) this.elements[`compare-${kind}`].disabled = !ready || this.busy;
  }
  clear(kind) {
    delete this.reports[kind]; this.elements[`export-${kind}`].disabled = true;
    if (kind === 'workloads') {
      for (const count of WORKLOAD_COUNTS) {
        for (const field of ['cpu', 'gpu', 'ratio', 'accuracy', 'stages', 'verdict', 'effect', 'share']) this.elements[`workload-${count}-${field}`].textContent = '—';
        for (const backend of ['cpu', 'gpu']) this.elements[`workload-${count}-${backend}-bar`].style.width = '0%';
        for (const stage of ['sampling', 'rollout', 'reduction']) this.elements[`workload-${count}-${stage}-share`].style.width = '0%';
      }
      this.elements['workload-takeaway'].textContent = 'Run the sweep to see the measured trade-off. No result yet.';
      this.elements['workload-crossover'].textContent = 'No measured crossover yet.';
    } else for (const backend of ['cpu', 'gpu']) this.elements[`mission-${backend}`].textContent = '—';
  }
  invalidate() {
    this.generation++;
    this.clear('workloads'); this.clear('missions');
    this.elements['workload-method'].textContent = this.elements['workload-scope'].value === 'plan'
      ? 'Complete cold-start plan: CPU sampling + CPU/GPU costs + CPU softmin/update. CPU costs run on the main thread, not in a worker. Motion, safety guard and drawing excluded.'
      : 'Rollout costs only: CPU worker round trip versus GPU upload/readback. Sampling, softmin, motion and drawing excluded.';
    this.elements['workload-status'].textContent = this.busy ? 'Cancelling after outstanding work finishes…' : 'Ready. Speed sweep and mission validation are separate experiments.';
  }
  renderWorkloads(report) {
    this.elements['workload-takeaway'].textContent = report.insights.conclusion;
    for (const insight of report.insights.workloads) {
      this.elements[`workload-${insight.count}-verdict`].textContent = insight.verdict;
      this.elements[`workload-${insight.count}-effect`].textContent = insight.effect;
      const shares = insight.gpuPathShares;
      this.elements[`workload-${insight.count}-share`].textContent = shares
        ? `GPU-path wall time: sampling ${(shares.sampling * 100).toFixed(1)}% · rollouts + transfers ${(shares.rollout * 100).toFixed(1)}% · softmin ${(shares.reduction * 100).toFixed(1)}%.`
        : 'No GPU-path stage share available. Complete-plan mode and positive, consistent samples are required.';
      for (const stage of ['sampling', 'rollout', 'reduction']) this.elements[`workload-${insight.count}-${stage}-share`].style.width = `${shares ? shares[stage] * 100 : 0}%`;
    }
    const maximum = Math.max(...report.results.flatMap(entry => [entry.report.cpu.medianMs, entry.report.gpu?.medianMs ?? 0]));
    const timing = value => value ? `${value.medianMs.toFixed(2)} ms · P10–P90 ${value.p10Ms.toFixed(2)}–${value.p90Ms.toFixed(2)}` : 'Unavailable';
    for (const {count, report: measured} of report.results) {
      for (const backend of ['cpu', 'gpu']) {
        this.elements[`workload-${count}-${backend}`].textContent = timing(measured[backend]);
        this.elements[`workload-${count}-${backend}-bar`].style.width = `${maximum > 0 && measured[backend] ? measured[backend].medianMs / maximum * 100 : 0}%`;
      }
      this.elements[`workload-${count}-ratio`].textContent = measured.speedup === null ? '—' : `${measured.speedup.toFixed(2)}×`;
      this.elements[`workload-${count}-accuracy`].textContent = !measured.gpu ? 'CPU only' : workloadMatches(measured)
        ? measured.commandAccuracy ? 'Costs + collision classes + controls match' : 'Costs + collision classes match' : 'MISMATCH · ratio suppressed';
      this.elements[`workload-${count}-stages`].textContent = measured.stages ? ['cpu', 'gpu'].map(backend => {
        const stage = measured.stages[backend];
        return stage ? `${backend.toUpperCase()}: sample ${stage.sampling.medianMs.toFixed(2)} / rollouts ${stage.rollout.medianMs.toFixed(2)} / softmin ${stage.reduction.medianMs.toFixed(2)} ms` : 'GPU unavailable';
      }).join(' · ') : 'Rollout-only timing; no planning-stage measurements.';
    }
    this.elements['workload-crossover'].textContent = report.firstTestedGpuWin ? `First tested GPU median win: ${report.firstTestedGpuWin.toLocaleString()} trajectories. This is not an interpolated crossover or a guarantee.`
      : report.results.every(entry => workloadMatches(entry.report)) ? 'No GPU median win among these tested counts.' : 'No crossover claim: GPU unavailable or output mismatch.';
  }
  renderMissions(report) {
    for (const backend of ['cpu', 'gpu']) {
      const summary = report.summaries[backend];
      this.elements[`mission-${backend}`].textContent = !summary ? `Unavailable · ${report.gpuUnavailableReason}`
        : `${summary.reached}/${summary.trials} reached · ${summary.safeStops} safe stops · ${summary.timeouts} timeouts · min clearance ${summary.minimumClearance.toFixed(3)} m. ${report.results[backend].map(trial => `Seed ${trial.seed}: ${trial.status} (${trial.simulationSeconds.toFixed(1)} s simulated)`).join('; ')}`;
    }
  }
  async run(kind) {
    if (!this.ready || this.busy) return;
    this.generation++; this.clear(kind); this.busy = true; this.setReady(this.ready); this.pause();
    this.elements['cancel-workloads'].disabled = false;
    const generation = this.generation;
    const shouldCancel = () => generation !== this.generation || document.hidden;
    const check = () => {if (shouldCancel()) throw new DOMException('Cancelled', 'AbortError');};
    let gpu;
    try {
      while (!this.idle()) {await new Promise(resolve => setTimeout(resolve, 10)); check();}
      check(); const input = this.snapshot();
      this.elements['workload-status'].textContent = 'Preparing fixed inputs and dedicated resources (not timed)…';
      let report;
      if (kind === 'workloads') {
        report = await compareWorkloads(input, {createJob: this.createJob, shouldCancel, onProgress: progress => {
          this.elements['workload-status'].textContent = `${progress.count.toLocaleString()} trajectories · ${progress.phase} ${progress.backend.toUpperCase()} ${progress.iteration}/${progress.total}`;
        }});
      } else {
        let reason = '';
        try {gpu = await this.createGpu();} catch (error) {reason = error.message;}
        check();
        report = {...await compareMissions(input, {gpuEvaluate: gpu ? (...args) => gpu.evaluate(...args) : null, shouldCancel,
          onProgress: progress => {this.elements['workload-status'].textContent = `${input.count} trajectories · ${progress.backend.toUpperCase()} seed ${progress.seed} · step ${progress.steps}/${progress.maxSteps}`;}}), gpuUnavailableReason: reason};
      }
      check(); this.reports[kind] = report;
      if (kind === 'workloads') this.renderWorkloads(report); else this.renderMissions(report);
      this.elements[`export-${kind}`].disabled = false;
      this.elements['workload-status'].textContent = kind === 'workloads' ? 'Sweep complete. Bars share one time scale; shorter is faster. CPU wins remain visible.' : 'Mission trials complete. Arrival, safe stop and timeout are distinct outcomes; no timing ratio is inferred.';
    } catch (error) {
      this.elements['workload-status'].textContent = error.name === 'AbortError' ? 'Cancelled. No partial report saved.' : `Experiment failed: ${error.message}. No partial report saved.`;
    } finally {
      try {gpu?.destroy();} finally {this.busy = false; this.elements['cancel-workloads'].disabled = true; this.setReady(this.ready); this.finished();}
    }
  }
}
