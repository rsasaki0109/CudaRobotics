import {evaluateKernel} from './kernels.mjs';
let input;
self.onmessage = event => {
  try {
    if (event.data.type === 'init') { input = event.data.input; self.postMessage({ready: true}); }
    else { const output = evaluateKernel(input); self.postMessage({output}, [output.buffer]); }
  } catch (error) { self.postMessage({error: error.message}); }
};
