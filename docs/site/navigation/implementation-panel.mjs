import {IMPLEMENTATIONS, IMPLEMENTATION_PAIRS, compareImplementations} from './implementations.mjs';
import {SCALE_GRIDS} from './scaling.mjs';

export class ImplementationPanel {
  constructor(elements, {canStart, snapshot, pause, setBusy, createGpu, download}) {
    this.elements = elements;
    Object.assign(this, {canStart, snapshot, pause, setBusy, createGpu, download});
    this.generation = 0; this.busy = false;
    elements['compare-implementations'].addEventListener('click', () => this.run());
    elements['cancel-implementations'].addEventListener('click', () => this.invalidate());
    elements['implementation-grid'].addEventListener('change', () => this.invalidate());
    elements['export-implementations'].addEventListener('click', () => { if (this.report) this.download(this.report); });
    this.invalidate();
  }
  invalidate() {
    this.generation++; this.report = null;
    this.elements['export-implementations'].disabled = true;
    for (const id of Object.keys(IMPLEMENTATIONS)) {
      for (const stage of ['total', 'range', 'field', 'route', 'path']) this.elements[`impl-${id}-${stage}`].textContent = '—';
      this.elements[`impl-${id}-bar`].style.width = '0%';
    }
    for (const [first, second] of IMPLEMENTATION_PAIRS) this.elements[`impl-${first}-${second}`].textContent = '—';
    this.elements['implementation-status'].textContent = this.busy ? 'Cancelling after outstanding work finishes…'
      : 'Ready: CPU EDT, reference GPU and exact GPU EDT on the same input. Independent of the paired-panel CPU selector.';
  }
  render(report) {
    const maximum = Math.max(0, ...Object.values(report.backends).map(backend => backend?.total.medianMs ?? 0));
    const timing = measured => `${measured.medianMs.toFixed(2)} ms (P10–P90 ${measured.p10Ms.toFixed(2)}–${measured.p90Ms.toFixed(2)})`;
    for (const id of Object.keys(IMPLEMENTATIONS)) {
      const measured = report.backends[id];
      this.elements[`impl-${id}-total`].textContent = measured ? `${measured.total.medianMs.toFixed(2)} ms` : 'Unavailable';
      this.elements[`impl-${id}-range`].textContent = measured ? `P10–P90 ${measured.total.p10Ms.toFixed(2)}–${measured.total.p90Ms.toFixed(2)} ms` : report.unavailableReasons[id] ?? 'No GPU result';
      for (const stage of ['field', 'route']) this.elements[`impl-${id}-${stage}`].textContent = measured ? timing(measured[stage]) : '—';
      const route = measured?.routes[0];
      this.elements[`impl-${id}-path`].textContent = !route ? 'No result' : route.status === 'ready'
        ? `Ready · ${route.length.toFixed(2)} m · clearance ${route.minimumClearance.toFixed(2)} m` : `Blocked · ${route.status}`;
      this.elements[`impl-${id}-bar`].style.width = `${measured && maximum > 0 ? measured.total.medianMs / maximum * 100 : 0}%`;
    }
    for (const [first, second] of IMPLEMENTATION_PAIRS) {
      const comparison = report.comparisons[`${first}:${second}`];
      this.elements[`impl-${first}-${second}`].textContent = !comparison ? 'Unavailable — no invented timing or ratio'
        : !comparison.fieldAccuracy.matches || !comparison.routeAccuracy.matches ? 'MISMATCH — ratio suppressed'
          : `Fields + routes match. Total ratio: ${comparison.totalRatio === null ? '—' : `${comparison.totalRatio.toFixed(2)}×`}; field ratio: ${comparison.fieldRatio === null ? '—' : `${comparison.fieldRatio.toFixed(2)}×`}.`;
    }
  }
  async run() {
    if (this.busy || !this.canStart()) return;
    this.invalidate(); this.busy = true; this.setBusy(true); this.pause();
    this.elements['cancel-implementations'].disabled = false;
    const generation = this.generation;
    const cancelled = () => generation !== this.generation || document.hidden;
    const gpus = {};
    const unavailableReasons = {};
    try {
      const grid = SCALE_GRIDS.find(candidate => candidate.width === Number(this.elements['implementation-grid'].value || 128));
      if (!grid) throw new Error('Unknown implementation comparison grid');
      const input = this.snapshot();
      this.elements['implementation-status'].textContent = 'Preparing dedicated GPU resources and shaders (not timed)…';
      for (const [id, algorithm] of [['gpuReference', 'reference'], ['gpuEdt', 'edt']]) {
        if (cancelled()) throw new DOMException('Cancelled', 'AbortError');
        try { gpus[id] = await this.createGpu(grid, algorithm); }
        catch (error) { unavailableReasons[id] = error.message; }
      }
      if (cancelled()) throw new DOMException('Cancelled', 'AbortError');
      const report = await compareImplementations(input, {grid,
        gpuReferenceEvaluate: gpus.gpuReference ? occupancy => gpus.gpuReference.evaluate(occupancy) : null,
        gpuEdtEvaluate: gpus.gpuEdt ? occupancy => gpus.gpuEdt.evaluate(occupancy) : null,
        shouldCancel: cancelled, onProgress: progress => {
          this.elements['implementation-status'].textContent = `${grid.width}×${grid.height}: ${progress.phase} ${IMPLEMENTATIONS[progress.backend].name} ${progress.iteration}/${progress.total}`;
        }});
      if (cancelled()) throw new DOMException('Cancelled', 'AbortError');
      this.report = {...report, unavailableReasons}; this.render(this.report);
      this.elements['implementation-status'].textContent = 'Complete: 3 warmups and 15 calls per available implementation; balanced rotating order. GPU EDT is not assumed to win. All routes still use CPU A*.';
      this.elements['export-implementations'].disabled = false;
    } catch (error) {
      this.elements['implementation-status'].textContent = error.name === 'AbortError' ? 'Implementation comparison cancelled. No partial report saved.'
        : `Implementation comparison failed: ${error.message}. No report saved.`;
    } finally {
      for (const device of Object.values(gpus)) {
        try { device?.destroy(); }
        catch { this.elements['implementation-status'].textContent += ' GPU resource cleanup failed.'; }
      }
      this.busy = false; this.elements['cancel-implementations'].disabled = true; this.setBusy(false);
    }
  }
}
