import {cpuCosts} from '../demo/planner.mjs';
import {likelihood} from '../particles/model.mjs';

export function evaluateKernel(input) {
  if (input.kind === 'mppi') return cpuCosts(input.state, input.goal, input.obstacles, input.controls, input.count);
  if (input.kind === 'particles') {
    const output = new Float32Array(input.count);
    for (let index = 0; index < input.count; index++) output[index] = likelihood(input.poses.subarray(index * 4, index * 4 + 3), input.ranges, input.sigma);
    return output;
  }
  throw new Error('Unknown comparison kernel');
}
