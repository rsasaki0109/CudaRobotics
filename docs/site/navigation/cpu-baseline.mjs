import {distanceField} from './model.mjs';
import {distanceFieldEdt} from './edt.mjs';

const baselines = {
  matched: {evaluate: distanceField, description: {id: 'matched', name: 'CPU brute force', cpuAlgorithm: 'brute-force-separable',
    gpuAlgorithm: 'brute-force-separable', comparisonKind: 'same-algorithm parallelism', fieldWorkComplexity: 'O(W×H×(W+H))',
    note: 'Same brute-force field algorithm on CPU and GPU; useful for isolating parallelism, not a best-CPU claim.'}},
  optimized: {evaluate: distanceFieldEdt, description: {id: 'optimized', name: 'CPU linear-time EDT', cpuAlgorithm: 'linear-time-squared-euclidean-edt',
    gpuAlgorithm: 'brute-force-separable', comparisonKind: 'different-algorithm practical baseline', fieldWorkComplexity: 'O(W×H)',
    note: 'Different algorithms: optimized CPU EDT versus brute-force GPU field. Same distance/clearance definition; not a pure hardware speedup or native C++ comparison.'}},
};

export function cpuBaseline(id = 'matched') {
  const baseline = Object.hasOwn(baselines, id) ? baselines[id] : null;
  if (!baseline) throw new Error('Unknown CPU comparison baseline');
  return {evaluate: baseline.evaluate, description: {...baseline.description}};
}
