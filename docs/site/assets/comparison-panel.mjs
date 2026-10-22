export class ComparisonPanel {
  constructor(elements, {prepare, pause, idle, finished = () => {}}) {
    this.elements = elements;
    this.prepare = prepare; this.pause = pause; this.idle = idle; this.finished = finished;
    this.generation = 0; this.busy = false; this.ready = false; this.report = null;
    elements.compare.addEventListener('click', () => this.run());
    elements['export-comparison'].addEventListener('click', () => {
      if (!this.report) return;
      const url = URL.createObjectURL(new Blob([JSON.stringify(this.report, null, 2)], {type: 'application/json'}));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${this.report.demo}-comparison.json`; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    this.invalidate();
  }
  setReady(ready) { this.ready = ready; this.elements.compare.disabled = !ready || this.busy; }
  invalidate() {
    this.generation++; this.report = null;
    const elements = this.elements;
    elements['export-comparison'].disabled = true;
    for (const id of ['cpu-median', 'gpu-median', 'speedup']) elements[id].textContent = '—';
    for (const id of ['cpu-range', 'gpu-range']) elements[id].textContent = 'P10–P90: —';
    for (const id of ['cpu-bar', 'gpu-bar']) elements[id].style.width = '0%';
    elements.accuracy.textContent = 'Output agreement: —';
    elements['comparison-status'].textContent = this.busy ? 'Cancelling after outstanding work finishes…' : 'Ready: 3 warmups, then 15 paired measurements. Live animation pauses.';
  }
  async run() {
    if (this.busy || !this.ready) return;
    this.invalidate(); this.busy = true; this.setReady(this.ready); this.pause();
    const generation = this.generation;
    const cancelled = () => generation !== this.generation || document.hidden;
    let job;
    try {
      while (!this.idle()) { await new Promise(resolve => setTimeout(resolve, 10)); if (cancelled()) throw new DOMException('Cancelled', 'AbortError'); }
      if (cancelled()) throw new DOMException('Cancelled', 'AbortError');
      this.elements['comparison-status'].textContent = 'Preparing fixed inputs, CPU worker and GPU pipeline (not timed)…';
      job = await this.prepare();
      if (cancelled()) throw new DOMException('Cancelled', 'AbortError');
      const report = await job.run({shouldCancel: cancelled, onProgress: progress => {
        this.elements['comparison-status'].textContent = `${progress.phase} ${progress.backend.toUpperCase()}: ${progress.iteration}/${progress.total}`;
      }});
      if (cancelled()) throw new DOMException('Cancelled', 'AbortError');
      this.report = report;
      const elements = this.elements;
      const maximum = Math.max(report.cpu.medianMs, report.gpu?.medianMs ?? 0);
      for (const backend of ['cpu', 'gpu']) {
        const timing = report[backend];
        elements[`${backend}-median`].textContent = timing ? `${timing.medianMs.toFixed(2)} ms` : 'Unavailable';
        elements[`${backend}-range`].textContent = timing ? `P10–P90: ${timing.p10Ms.toFixed(2)}–${timing.p90Ms.toFixed(2)} ms` : 'No WebGPU measurement';
        elements[`${backend}-bar`].style.width = `${timing && maximum > 0 ? timing.medianMs / maximum * 100 : 0}%`;
      }
      elements.speedup.textContent = report.speedup === null ? '—' : `${report.speedup.toFixed(2)}×`;
      elements.accuracy.textContent = report.accuracy ? `${report.accuracy.matches ? 'Match' : 'MISMATCH'} · max absolute error ${report.accuracy.maxAbsoluteError.toExponential(2)}` : 'GPU unavailable; CPU timings only';
      elements['comparison-status'].textContent = !report.gpu ? `CPU timing only. ${report.gpuUnavailableReason}`
        : !report.accuracy.matches ? 'Outputs differ beyond tolerance. No speedup claim.'
          : report.speedup === null ? 'Outputs match; timer precision is insufficient for a ratio.'
            : report.speedup >= 1 ? `GPU ${report.speedup.toFixed(2)}× faster by median call time.` : `CPU ${(1 / report.speedup).toFixed(2)}× faster by median call time.`;
      elements['export-comparison'].disabled = false;
    } catch (error) {
      this.elements['comparison-status'].textContent = error.name === 'AbortError' ? 'Comparison cancelled. Run again with the new settings.' : `Comparison failed: ${error.message}. No result saved.`;
    } finally {
      try { job?.destroy(); }
      finally { this.busy = false; this.setReady(this.ready); this.finished(); }
    }
  }
}
