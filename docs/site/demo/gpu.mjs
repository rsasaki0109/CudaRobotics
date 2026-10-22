import {Planner, scenes, cpuCosts, COLLISION} from './planner.mjs';

export class GpuCosts {
  static async create() {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter available');
    const device = await adapter.requestDevice();
    try {
      const response = await fetch(new URL('./rollout.wgsl', import.meta.url));
      if (!response.ok) throw new Error(`Shader download: HTTP ${response.status}`);
      const module = device.createShaderModule({code: await response.text()});
      const diagnostics = await module.getCompilationInfo();
      const errors = diagnostics.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(errors.map(message => message.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}});
      const evaluator = new GpuCosts(device, pipeline);
      await evaluator.checkReference();
      return evaluator;
    } catch (error) {
      device.destroy();
      throw error;
    }
  }

  async checkReference() {
    for (const scene of Object.values(scenes)) {
      const count = 65;
      const controls = new Planner().sample(scene.start, scene.goal, count);
      const reference = cpuCosts(scene.start, scene.goal, scene.obstacles, controls, count);
      const actual = await this.evaluate(scene.start, scene.goal, scene.obstacles, controls, count);
      reference.forEach((expected, index) => {
        const matchingCollision = (expected >= COLLISION) === (actual[index] >= COLLISION);
        if (!matchingCollision || Math.abs(actual[index] - expected) > 0.001 * Math.max(1, expected)) {
          throw new Error(`WebGPU reference check failed (${scene.name}, sample ${index})`);
        }
      });
    }
  }

  constructor(device, pipeline) {
    this.device = device;
    this.pipeline = pipeline;
    this.buffers = [];
    this.capacity = 0;
    this.failure = null;
    device.lost.then(info => { this.failure = new Error(`GPU device lost: ${info.message}`); });
    device.addEventListener('uncapturederror', event => { this.failure = event.error; });
  }

  allocate(count) {
    this.buffers.forEach(buffer => buffer.destroy());
    const device = this.device;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.buffers = [
      device.createBuffer({size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST}),
      device.createBuffer({size: count * 32 * 2 * 4, usage: storage}),
      device.createBuffer({size: 64 * 16, usage: storage}),
      device.createBuffer({size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}),
      device.createBuffer({size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST}),
    ];
    this.group = device.createBindGroup({layout: this.pipeline.getBindGroupLayout(0),
      entries: this.buffers.slice(0, 4).map((buffer, binding) => ({binding, resource: {buffer}}))});
    this.capacity = count;
  }

  async evaluate(state, target, obstacles, controls, count) {
    if (this.failure) throw this.failure;
    if (obstacles.length > 64) throw new Error('At most 64 obstacles are supported');
    if (count !== this.capacity) this.allocate(count);
    const [params, samples, geometry, costs, readback] = this.buffers;
    const data = new ArrayBuffer(32);
    new Float32Array(data).set([...state, 0, ...target]);
    new Uint32Array(data).set([count, obstacles.length], 6);
    this.device.queue.writeBuffer(params, 0, data);
    this.device.queue.writeBuffer(samples, 0, controls);
    if (obstacles.length) this.device.queue.writeBuffer(geometry, 0, new Float32Array(obstacles.flat()));
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(costs, 0, readback, 0, count * 4);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    try {
      if (this.failure) throw this.failure;
      const result = new Float32Array(readback.getMappedRange()).slice();
      if (!result.every(Number.isFinite)) throw new Error('Non-finite GPU rollout costs');
      return result;
    } finally {
      readback.unmap();
    }
  }

  destroy() {
    this.buffers.forEach(buffer => buffer.destroy());
    this.device.destroy();
  }
}
