import {BEAMS, BOXES} from './model.mjs';

export async function createPipelines(device, computeCode, drawCode, format) {
  const compute = device.createShaderModule({code: computeCode});
  const draw = device.createShaderModule({code: drawCode});
  for (const module of [compute, draw]) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(message => message.message).join('; '));
  }
  const computeLayout = device.createBindGroupLayout({entries: Array.from({length: 9}, (_, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE,
    buffer: {type: binding === 0 ? 'uniform' : [1, 6, 8].includes(binding) ? 'read-only-storage' : 'storage'},
  }))});
  const layout = device.createPipelineLayout({bindGroupLayouts: [computeLayout]});
  const passes = await Promise.all(['predict_score', 'scan_particles', 'scan_groups', 'resample']
    .map(entryPoint => device.createComputePipelineAsync({layout, compute: {module: compute, entryPoint}})));
  const render = await device.createRenderPipelineAsync({layout: 'auto',
    vertex: {module: draw, entryPoint: 'vertex_main'},
    fragment: {module: draw, entryPoint: 'fragment_main', targets: [{format,
      blend: {color: {srcFactor: 'one', dstFactor: 'one', operation: 'add'},
        alpha: {srcFactor: 'one', dstFactor: 'one', operation: 'add'}}}]},
    primitive: {topology: 'triangle-list'},
  });
  const score = await device.createComputePipelineAsync({layout, compute: {module: compute, entryPoint: 'score_snapshot'}});
  return {passes, computeLayout, render, score};
}

export class GpuFilter {
  static async create(canvas, count) {
    if (!navigator.gpu) throw new Error('WebGPU is not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice();
    try {
      const codes = await Promise.all(['filter.wgsl', 'draw.wgsl'].map(async name => {
        const response = await fetch(new URL(`${name}?v=all-comparisons-1`, import.meta.url));
        if (!response.ok) throw new Error(`Shader HTTP ${response.status}`);
        return response.text();
      }));
      const format = navigator.gpu.getPreferredCanvasFormat();
      const pipelines = await createPipelines(device, ...codes, format);
      const context = canvas.getContext('webgpu');
      if (!context) throw new Error('WebGPU canvas is unavailable');
      context.configure({device, format, alphaMode: 'premultiplied'});
      const filter = new GpuFilter(device, pipelines, count);
      filter.context = context;
      filter.width = canvas.width;
      filter.height = canvas.height;
      return filter;
    } catch (error) { device.destroy(); throw error; }
  }

  constructor(device, pipelines, count) {
    if (!Number.isInteger(count) || count < 1 || count > 65536) throw new Error('Particle count must be 1–65536');
    this.device = device;
    this.pipelines = pipelines;
    this.count = count;
    this.frame = 0;
    this.width = 1280;
    this.height = 960;
    this.failure = null;
    this.buffers = [];
    device.lost.then(info => { this.failure = new Error(`GPU device lost: ${info.message}`); });
    device.addEventListener('uncapturederror', event => { this.failure = event.error; });
    const make = (size, usage) => {
      const buffer = device.createBuffer({size, usage});
      this.buffers.push(buffer);
      return buffer;
    };
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.params = make(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.states = [make(count * 16, storage), make(count * 16, storage)];
    this.scored = make(count * 16, storage);
    this.cdf = make(count * 4, storage);
    this.groups = make(Math.ceil(count / 256) * 32, storage);
    this.ranges = make(BEAMS * 4, storage);
    this.summary = make(32, storage);
    this.geometry = make(BOXES.length * 16, storage);
    this.readback = make(32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.geometry, 0, new Float32Array(BOXES.flat()));
    this.computeGroups = [0, 1].map(index => device.createBindGroup({layout: pipelines.computeLayout,
      entries: [this.params, this.states[index], this.scored, this.cdf, this.groups,
        this.states[1 - index], this.ranges, this.summary, this.geometry]
        .map((buffer, binding) => ({binding, resource: {buffer}})),
    }));
    this.renderGroups = this.states.map(buffer => device.createBindGroup({layout: pipelines.render.getBindGroupLayout(0),
      entries: [{binding: 0, resource: {buffer: this.params}}, {binding: 1, resource: {buffer}}],
    }));
  }

  async step(ranges, motion, sigma, reset = false, injection = 0.03, view = null) {
    if (this.failure) throw this.failure;
    const data = new ArrayBuffer(48);
    new Float32Array(data).set([motion[0], motion[1], sigma, injection]);
    new Uint32Array(data).set([this.count, BEAMS, this.frame + 1, reset || this.frame === 0 ? 1 : 0], 4);
    new Float32Array(data).set([16, 12, this.width, this.height], 8);
    this.device.queue.writeBuffer(this.params, 0, data);
    this.device.queue.writeBuffer(this.ranges, 0, ranges);
    const index = this.frame % 2;
    const encoder = this.device.createCommandEncoder();
    for (const [stage, pipeline] of this.pipelines.passes.entries()) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.computeGroups[index]);
      pass.dispatchWorkgroups(stage === 2 ? 1 : Math.ceil(this.count / 256));
      pass.end();
    }
    const textureView = view ?? this.context?.getCurrentTexture().createView();
    if (textureView) {
      const pass = encoder.beginRenderPass({colorAttachments: [{view: textureView,
        clearValue: {r: 0, g: 0, b: 0, a: 0}, loadOp: 'clear', storeOp: 'store'}]});
      pass.setPipeline(this.pipelines.render);
      pass.setBindGroup(0, this.renderGroups[1 - index]);
      pass.draw(6, this.count);
      pass.end();
    }
    encoder.copyBufferToBuffer(this.summary, 0, this.readback, 0, 32);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    let result;
    try { result = new Float32Array(this.readback.getMappedRange()).slice(); }
    finally { this.readback.unmap(); }
    if (this.failure) throw this.failure;
    if (!result.every(Number.isFinite) || result[4] <= 0) throw new Error('Invalid GPU posterior');
    this.frame++;
    return result;
  }

  async snapshot(buffer = this.scored) {
    const readback = this.device.createBuffer({size: this.count * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, readback, 0, this.count * 16);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return new Float32Array(readback.getMappedRange()).slice();
    } finally { readback.destroy(); }
  }

  async scoreSnapshot(poses, ranges, sigma) {
    if (this.failure) throw this.failure;
    if (poses.length !== this.count * 4 || ranges.length !== BEAMS) throw new Error('Invalid scoring snapshot');
    if (!this.scoreReadback) {
      this.scoreReadback = this.device.createBuffer({size: this.count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
      this.buffers.push(this.scoreReadback);
    }
    const data = new ArrayBuffer(48);
    new Float32Array(data).set([0, 0, sigma, 0]);
    new Uint32Array(data).set([this.count, BEAMS, 0, 0], 4);
    this.device.queue.writeBuffer(this.params, 0, data);
    this.device.queue.writeBuffer(this.states[0], 0, poses);
    this.device.queue.writeBuffer(this.ranges, 0, ranges);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.score); pass.setBindGroup(0, this.computeGroups[0]);
    pass.dispatchWorkgroups(Math.ceil(this.count / 256)); pass.end();
    encoder.copyBufferToBuffer(this.cdf, 0, this.scoreReadback, 0, this.count * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.scoreReadback.mapAsync(GPUMapMode.READ);
    let output;
    try { output = new Float32Array(this.scoreReadback.getMappedRange()).slice(); }
    finally { this.scoreReadback.unmap(); }
    if (this.failure) throw this.failure;
    return output;
  }

  destroy() { this.buffers.forEach(buffer => buffer.destroy()); this.device.destroy(); }
}

export async function createGpuScorer(count) {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();
  try {
    const codes = await Promise.all(['filter.wgsl', 'draw.wgsl'].map(async name => {
      const response = await fetch(new URL(`${name}?v=all-comparisons-1`, import.meta.url));
      if (!response.ok) throw new Error(`Shader HTTP ${response.status}`);
      return response.text();
    }));
    const filter = new GpuFilter(device, await createPipelines(device, ...codes, 'rgba8unorm'), count);
    return {evaluate: input => filter.scoreSnapshot(input.poses, input.ranges, input.sigma), destroy: () => filter.destroy()};
  } catch (error) { device.destroy(); throw error; }
}
