import {WIDTH, HEIGHT} from './model.mjs';

export async function createPipelines(device, code) {
  const module = device.createShaderModule({code});
  const diagnostics = await module.getCompilationInfo();
  const errors = diagnostics.messages.filter(message => message.type === 'error');
  if (errors.length) throw new Error(errors.map(message => message.message).join('; '));
  const bindLayout = device.createBindGroupLayout({entries: [0, 1, 2].map(binding => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: {type: binding ? 'storage' : 'read-only-storage'},
  }))});
  const layout = device.createPipelineLayout({bindGroupLayouts: [bindLayout]});
  const pipelines = await Promise.all(['row_distance', 'column_distance'].map(entryPoint =>
    device.createComputePipelineAsync({layout, compute: {module, entryPoint}})));
  return {bindLayout, pipelines};
}

export class GpuField {
  static async create() {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice();
    try {
      const response = await fetch(new URL('./field.wgsl', import.meta.url));
      if (!response.ok) throw new Error(`Shader HTTP ${response.status}`);
      return new GpuField(device, await createPipelines(device, await response.text()));
    } catch (error) { device.destroy(); throw error; }
  }

  constructor(device, pipelines) {
    this.device = device;
    this.pipelines = pipelines.pipelines;
    this.failure = null;
    const size = WIDTH * HEIGHT * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.buffers = [0, 1, 2].map(() => device.createBuffer({size, usage}));
    this.readback = device.createBuffer({size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    this.group = device.createBindGroup({layout: pipelines.bindLayout,
      entries: this.buffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
    device.lost.then(info => { this.failure = new Error(`GPU lost: ${info.message}`); });
    device.addEventListener('uncapturederror', event => { this.failure = event.error; });
  }

  async evaluate(occupancy) {
    if (this.failure) throw this.failure;
    if (occupancy.length !== WIDTH * HEIGHT) throw new Error('Invalid occupancy dimensions');
    this.device.queue.writeBuffer(this.buffers[0], 0, new Uint32Array(occupancy));
    const encoder = this.device.createCommandEncoder();
    for (const pipeline of this.pipelines) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, this.group);
      pass.dispatchWorkgroups(Math.ceil(WIDTH * HEIGHT / 64)); pass.end();
    }
    encoder.copyBufferToBuffer(this.buffers[2], 0, this.readback, 0, WIDTH * HEIGHT * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    let field;
    try { field = new Float32Array(this.readback.getMappedRange()).slice(); }
    finally { this.readback.unmap(); }
    if (this.failure) throw this.failure;
    if (!field.every(value => Number.isFinite(value) && value >= 0)) throw new Error('Invalid GPU distance field');
    return field;
  }

  destroy() { this.buffers.forEach(buffer => buffer.destroy()); this.readback.destroy(); this.device.destroy(); }
}
