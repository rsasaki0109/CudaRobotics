import {compareMapSizes} from './scaling.mjs';

const label = entry => `${entry.grid.width}×${entry.grid.height}`;
const matched = entry => entry.report.accuracy?.matches && entry.report.routeAccuracy?.matches;
const milliseconds = timing => timing ? `${timing.medianMs.toFixed(2)} ms` : 'Unavailable';

export class ScalingPanel {
  constructor(elements, {canStart, snapshot, pause, setBusy, createGpu, download}) {
    this.elements = elements;
    Object.assign(this, {canStart, snapshot, pause, setBusy, createGpu, download});
    this.generation = 0; this.busy = false;
    elements['compare-sizes'].addEventListener('click', () => this.run());
    elements['cancel-sizes'].addEventListener('click', () => this.invalidate());
    elements['export-sizes'].addEventListener('click', () => { if (this.report) this.download(this.report); });
    this.invalidate();
  }
  invalidate() {
    this.generation++; this.report = null;
    this.elements['export-sizes'].disabled = true;
    for (const id of ['scale-field-chart', 'scale-total-chart', 'scale-detail-rows']) this.elements[id].innerHTML = '';
    this.elements['scale-summary'].textContent = 'No measurements yet. A faster distance field does not necessarily mean a faster total replan.';
    this.elements['scale-status'].textContent = this.busy ? 'Cancelling after outstanding work finishes…' : 'Ready: same 16 × 12 m world at three grid resolutions. Motion pauses during measurement.';
  }
  chart(report, stage) {
    const timing = (entry, backend) => stage === 'field' ? entry.report.stages[backend]?.field : entry.report[backend];
    const maximum = Math.max(0, ...report.results.flatMap(entry => ['cpu', 'gpu'].map(backend => timing(entry, backend)?.medianMs ?? 0)));
    return `<ol class="scale-groups">${report.results.map(entry => `<li><h4>${label(entry)} <span>· ${(entry.grid.width * entry.grid.height).toLocaleString('en-US')} cells</span></h4>${['cpu', 'gpu'].map(backend => {
      const measured = timing(entry, backend);
      const width = measured && maximum > 0 ? measured.medianMs / maximum * 100 : 0;
      const range = measured ? `P10–P90 ${measured.p10Ms.toFixed(2)}–${measured.p90Ms.toFixed(2)} ms` : 'No GPU measurement';
      return `<div class="scale-bar-row"><span>${backend.toUpperCase()}</span><div class="scale-track" aria-hidden="true"><div class="scale-fill ${backend}" style="width:${width}%"></div></div><span title="${range}">${milliseconds(measured)}</span></div>`;
    }).join('')}</li>`).join('')}</ol>`;
  }
  render(report) {
    this.elements['scale-field-chart'].innerHTML = this.chart(report, 'field');
    this.elements['scale-total-chart'].innerHTML = this.chart(report, 'total');
    this.elements['scale-detail-rows'].innerHTML = report.results.map(entry => {
      const result = entry.report;
      const status = result.gpu ? matched(entry) ? 'Fields + routes match' : 'MISMATCH · no ratio' : 'CPU only';
      const routes = `CPU: ${result.routes.cpu[0].status}; GPU: ${result.routes.gpu?.[0].status ?? 'unavailable'}`;
      return `<tr><th scope="row">${label(entry)}<br>${entry.grid.resolution} m/cell</th><td>${milliseconds(result.stages.cpu.route)}</td><td>${result.stages.gpu ? milliseconds(result.stages.gpu.route) : '—'}</td><td>${status}<br>${routes}<br>Total CPU/GPU: ${result.speedup === null ? '—' : `${result.speedup.toFixed(2)}×`}</td></tr>`;
    }).join('');
    const verified = report.results.filter(matched);
    const sizes = entries => entries.length ? entries.map(label).join(', ') : 'none';
    this.elements['scale-summary'].textContent = !report.results.some(entry => entry.report.gpu)
      ? 'CPU timings only. No GPU crossover can be inferred on this device.'
      : `Observed GPU wins — field: ${sizes(verified.filter(entry => entry.report.fieldSpeedup > 1))}; total: ${sizes(verified.filter(entry => entry.report.speedup > 1))}. CPU routing is the larger GPU-pipeline stage at: ${sizes(verified.filter(entry => entry.report.stages.gpu.route.medianMs > entry.report.stages.gpu.field.medianMs))}. This is a sampled observation, not a universal crossover threshold.`;
  }
  async run() {
    if (this.busy || !this.canStart()) return;
    this.invalidate(); this.busy = true; this.setBusy(true); this.pause();
    this.elements['cancel-sizes'].disabled = false;
    const generation = this.generation;
    try {
      const report = await compareMapSizes(this.snapshot(), {createGpu: this.createGpu,
        shouldCancel: () => generation !== this.generation || document.hidden,
        onProgress: progress => {
          this.elements['scale-status'].textContent = `${progress.grid.width}×${progress.grid.height} (${progress.sizeIndex}/${progress.sizeCount}): ${progress.phase === 'setup' ? 'preparing GPU resources (not timed)' : `${progress.phase} ${progress.backend.toUpperCase()} ${progress.iteration}/${progress.total}`}`;
        }});
      if (generation !== this.generation || document.hidden) throw new DOMException('Cancelled', 'AbortError');
      this.report = report; this.render(report);
      this.elements['scale-status'].textContent = 'Complete: 3 warmups and 15 calls per available backend per grid; CPU/GPU order alternates when both exist. Hover timing labels for P10–P90; all samples are in JSON.';
      this.elements['export-sizes'].disabled = false;
    } catch (error) {
      this.elements['scale-status'].textContent = error.name === 'AbortError' ? 'Scale comparison cancelled. No partial report saved.' : `Scale comparison failed: ${error.message}. No report saved.`;
    } finally {
      this.busy = false; this.elements['cancel-sizes'].disabled = true; this.setBusy(false);
    }
  }
}
