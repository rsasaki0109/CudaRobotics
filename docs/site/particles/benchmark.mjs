import {randomGenerator, randomPose, BEAMS, BOXES, WIDTH, HEIGHT} from './model.mjs';
import {runComparison} from '../assets/benchmark.mjs';
import {evaluateKernel} from '../assets/kernels.mjs';

export function particleSnapshot(count, ranges, sigma) {
  const random = randomGenerator(311);
  const poses = new Float32Array(count * 4);
  for (let index = 0; index < count; index++) poses.set(randomPose(random), index * 4);
  return {kind: 'particles', count, poses, ranges: new Float32Array(ranges), sigma};
}
export async function compareParticles(input, {gpuEvaluate = null, cpuEvaluate = () => evaluateKernel(input), ...options} = {}) {
  const report = await runComparison({...options, cpuEvaluate, gpuEvaluate, length: input.count,
    absoluteTolerance: .0001, relativeTolerance: .001});
  return {...report, schema: 'cudarobotics.kernel-benchmark.v1', demo: 'particles',
    scope: 'likelihood scoring only; identical Float32 poses and 32-beam scan; prediction, reduction, resampling and drawing excluded',
    input: {count: input.count, poseGenerator: 'randomPose Float32 v1', seed: 311, ranges: Array.from(input.ranges), sigma: input.sigma, beams: BEAMS, boxes: BOXES, worldMetres: [WIDTH, HEIGHT]},
    timing: 'CPU call/worker round trip versus WebGPU scoring call including pose/scan upload, synchronization and all-weight readback; setup excluded',
    gpuReadbackBytes: input.count * 4};
}
