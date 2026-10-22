import {evaluateKernel} from './kernels.mjs';

export async function createCpuKernel(input) {
  const direct = () => ({mode: 'main-thread JavaScript reference (worker unavailable)', evaluate: () => evaluateKernel(input), destroy() {}});
  if (typeof Worker === 'undefined') return direct();
  let worker;
  try { worker = new Worker(new URL('./kernel-worker.mjs', import.meta.url), {type: 'module'}); }
  catch (error) { return direct(); }
  let pending;
  let failure;
  let timeout;
  const request = message => new Promise((resolve, reject) => {
    if (failure) { reject(failure); return; }
    if (pending) { reject(new Error('CPU worker request already in progress')); return; }
    pending = {resolve, reject};
    try { worker.postMessage(message); }
    catch (error) { pending = null; reject(error); }
  });
  worker.onmessage = event => {
    if (event.data.error) pending?.reject(new Error(event.data.error));
    else pending?.resolve(event.data.output ?? event.data.ready);
    pending = null;
  };
  worker.onerror = () => { failure = new Error('CPU worker failed'); pending?.reject(failure); pending = null; };
  try {
    await Promise.race([request({type: 'init', input}), new Promise((_, reject) => {
      timeout = setTimeout(() => reject(new Error('CPU worker initialization timed out')), 10000);
    })]);
  } catch (error) { worker.terminate(); return direct(); }
  finally { clearTimeout(timeout); }
  return {mode: 'single JavaScript worker; request and result-message overhead included',
    evaluate: () => request({type: 'run'}), destroy: () => {
      failure = new DOMException('Worker stopped', 'AbortError');
      worker.terminate(); pending?.reject(failure); pending = null;
    }};
}
