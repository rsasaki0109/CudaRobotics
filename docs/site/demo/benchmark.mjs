import {Planner, HORIZON, DT, COLLISION} from './planner.mjs';
import {runComparison} from '../assets/benchmark.mjs';
import {evaluateKernel} from '../assets/kernels.mjs';

export function mppiSnapshot(state, goal, obstacles, count) {
  return {kind: 'mppi', state: [...state], goal: [...goal], obstacles: obstacles.map(box => [...box]), count,
    controls: new Planner(311).sample(state, goal, count)};
}
export async function compareMppi(input, {gpuEvaluate = null, cpuEvaluate = () => evaluateKernel(input), ...options} = {}) {
  const report = await runComparison({...options, cpuEvaluate, gpuEvaluate, length: input.count,
    absoluteTolerance: .001, relativeTolerance: .001, classify: cost => cost >= COLLISION});
  return {...report, schema: 'cudarobotics.kernel-benchmark.v1', demo: 'mppi',
    scope: 'rollout cost evaluation only; identical seeded controls; sampling, weighting, motion and drawing excluded',
    input: {state: input.state, goal: input.goal, obstacles: input.obstacles, count: input.count, seed: 311, horizon: HORIZON, dt: DT},
    timing: 'CPU call/worker round trip versus WebGPU call including upload, synchronization, all-cost readback and copy; setup excluded'};
}
