import {createCpuKernel} from './cpu-kernel.mjs';

export async function prepareComparison(input, createGpu, compare) {
  const cpu = await createCpuKernel(input);
  let gpu;
  let reason = '';
  try { gpu = await createGpu(); } catch (error) { reason = error.message; }
  return {run: async options => ({...await compare(input, {...options,
    cpuEvaluate: () => cpu.evaluate(), gpuEvaluate: gpu ? () => gpu.evaluate(input) : null}),
    cpuExecution: cpu.mode, gpuUnavailableReason: reason,
    resources: 'input generation, worker creation, shader compilation and GPU allocation excluded; GPU buffers reused; CPU request/result-message overhead included when worker is used'}),
  destroy() { cpu.destroy(); gpu?.destroy(); }};
}
