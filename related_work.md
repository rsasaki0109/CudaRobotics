# Related Work and Novelty Check

Date: 2026-04-01

This note is a first-pass literature review for the new research-style additions in this repository.
It is intentionally pragmatic: the goal is not to survey everything, but to answer one question clearly.

Question:
- Which parts of this repo still have room for a paper-level novelty claim, and which parts are already well-covered by prior work?

Scope:
- Differentiable / sampling-based MPC
- Neural SDFs for navigation and planning
- GPU-native simulation and RL
- GPU neuroevolution

## Papers Checked

### Differentiable MPC / MPPI

1. Path Integral Networks: End-to-End Differentiable Optimal Control
   Link: https://arxiv.org/abs/1706.09597
   Date: 2017-06-29
   Why it matters:
   - PI-Net already makes the path-integral control computation differentiable end-to-end.
   - This means "differentiable path-integral control" as a broad idea is not new.

2. Differentiable MPC for End-to-end Planning and Control
   Link: https://arxiv.org/abs/1810.13400
   Date: 2018-10-31
   Why it matters:
   - This is a canonical differentiable MPC paper.
   - It learns dynamics and cost through the controller, so "differentiate through a planner/controller" is already established.

3. Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation
   Link: https://arxiv.org/abs/2506.14855
   Date: 2025-06-17, revised 2025-12-12
   Why it matters:
   - This is the closest recent overlap to our current `diff_mppi`.
   - It explicitly augments MPPI with sensitivity-analysis-based feedback gains derived from rollout differentiation.

4. GPU Based Path Integral Control with Learned Dynamics
   Link: https://arxiv.org/abs/1503.00330
   Date: 2015-03-01
   Why it matters:
   - GPU-accelerated MPPI itself is old.
   - So "MPPI on GPU" alone is not a novelty claim.

### Neural SDFs for Navigation

5. iSDF: Real-Time Neural Signed Distance Fields for Robot Perception
   Link: https://arxiv.org/abs/2204.02296
   Date: 2022-04-05
   Why it matters:
   - Real-time neural SDF reconstruction for robotics is already established.
   - The paper explicitly highlights downstream collision costs and gradients for planners.

6. Stochastic Implicit Neural Signed Distance Functions for Safe Motion Planning under Sensing Uncertainty
   Link: https://arxiv.org/abs/2309.16862
   Date: 2023-09-28
   Why it matters:
   - Neural SDFs have already been used directly inside motion planning under uncertainty.
   - This pushes beyond representation learning into planning-time safety reasoning.

7. Differentiable Composite Neural Signed Distance Fields for Robot Navigation in Dynamic Indoor Environments
   Link: https://arxiv.org/abs/2502.02664
   Date: 2025-02-04, revised 2025-03-06
   Why it matters:
   - This is the closest recent overlap to our `neural_sdf` / `sdf_potential_field` / `sdf_mppi` line.
   - It directly targets robot navigation with differentiable neural SDFs and trajectory optimization.

### GPU Simulation / RL

8. Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning
   Link: https://arxiv.org/abs/2108.10470
   Date: 2021-08-24
   Why it matters:
   - GPU-resident simulation plus GPU-resident policy training is already a known research direction.
   - This is the central prior work for anything framed as "MiniIsaacGym".

9. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
   Link: https://arxiv.org/abs/2109.11978
   Date: 2021-09-24
   Why it matters:
   - Massive parallel RL with thousands of environments on one GPU is already well demonstrated.
   - So "many environments + RL on one GPU" is not novel by itself.

### GPU Neuroevolution

10. EvoRL: A GPU-accelerated Framework for Evolutionary Reinforcement Learning
    Link: https://arxiv.org/abs/2501.15129
    Date: 2025-01-25
    Why it matters:
    - End-to-end GPU EvoRL frameworks now exist.
    - This weakens any broad novelty claim around "GPU evolutionary RL".

11. TensorNEAT: A GPU-accelerated Library for NeuroEvolution of Augmenting Topologies
    Link: https://arxiv.org/abs/2504.08339
    Date: 2025-04-11
    Why it matters:
    - GPU neuroevolution libraries with strong acceleration claims are already current.
    - This especially weakens "GPU neuroevolution framework" as a standalone paper claim.

## What Prior Work Already Covers

### 1. `diff_mppi`

Prior work already covers:
- differentiable control/planning in general
- path-integral control made differentiable
- rollout-differentiation ideas near MPPI
- GPU MPPI

Implication:
- The phrase "Differentiable MPPI" by itself is too broad to claim as novel.

What may still be claimable:
- a very lightweight hybrid algorithm that keeps vanilla MPPI intact and adds a cheap local autodiff refinement stage in plain CUDA
- a practical compute-quality tradeoff result:
  for the same control quality, the hybrid method may need fewer samples than vanilla MPPI
- a minimal educational systems contribution:
  showing how to bolt first-order sensitivity onto a sampling controller without a full differentiable simulator stack

Current novelty score:
- Medium at repo/demo level
- Low to medium at paper level unless we add stronger empirical evidence

### 2. `neural_sdf`, `sdf_potential_field`, `sdf_mppi`, `comparison_sdf_nav`

Prior work already covers:
- neural SDF reconstruction
- neural SDF gradients for planning
- safety-aware motion planning with neural SDFs
- recent differentiable composite neural SDF navigation in indoor environments

Implication:
- A 2D learned SDF heatmap plus planner integration is not enough for a paper claim.
- Right now this line is best viewed as a compact demonstration, not a research contribution.

What may still be claimable:
- almost nothing in the current 2D toy setup
- maybe an education/demo paper or tutorial artifact, but not a strong research novelty claim

Current novelty score:
- High as a didactic repo addition
- Low as a research-paper contribution

### 3. `mini_isaac`, `mini_isaac_rl`

Prior work already covers:
- all-GPU simulation loops
- all-GPU policy learning
- thousands of parallel environments
- minute-scale training in massively parallel settings

Implication:
- "MiniIsaacGym" is not a research novelty claim.
- It is useful as a minimal implementation, teaching artifact, or systems reproduction.

What may still be claimable:
- a minimal-from-scratch CUDA teaching implementation
- a compact reproduction-style benchmark

Current novelty score:
- High as an educational engineering artifact
- Very low as a paper contribution

### 4. `neuroevo`, `comparison_neuroevo`

Prior work already covers:
- GPU-accelerated neuroevolution libraries
- GPU-resident evolutionary RL frameworks
- population-scale benchmarking on standard control tasks

Implication:
- "4096 policies evolved on GPU" is not enough for a new paper.

What may still be claimable:
- only if we introduce a genuinely new evolutionary operator, representation, or hardware-aware scaling result

Current novelty score:
- Medium as a repo demo
- Low as a paper contribution

## Bottom Line

As of 2026-04-01, the strongest paper candidate in this repo is still `diff_mppi`.

Why:
- It is the only line where our implementation direction still has some room between prior work and a clean, compact claim.
- It sits at the intersection of:
  sampling-based control
  first-order sensitivity
  lightweight GPU implementation
- The nearest overlap, `Feedback-MPPI` from 2025, is close enough that we must be precise, but not so close that the direction is dead.

The other new projects are not strong novelty candidates right now:
- `neural_sdf_*`: good demo, weak paper novelty
- `mini_isaac*`: good educational systems artifact, weak paper novelty
- `neuroevo`: useful demo, weak paper novelty after 2025 GPU EvoRL / TensorNEAT papers

## If We Want a Paper, the Claim Must Get Narrower

Bad claim:
- "We propose differentiable MPPI."

Why bad:
- too broad
- conflicts with PI-Net, differentiable MPC literature, and especially Feedback-MPPI

Better claim:
- "We study a minimal hybrid controller that combines vanilla MPPI with a local autodiff refinement stage, and show that this refinement improves the quality-vs-samples tradeoff under a fixed GPU budget."

Even better claim:
- "For fixed wall-clock or fixed sample budget, a plain-CUDA MPPI + autodiff refinement controller reaches lower trajectory cost / better goal success than vanilla MPPI across several navigation tasks."

That is a narrower and more defensible systems/control paper.

## What Evidence Is Missing for `diff_mppi`

To move from demo to paper direction, we still need:

1. Controlled baselines
- vanilla MPPI
- our current hybrid `diff_mppi`
- if possible, a simple feedback baseline or gradient-only ablation

2. Fixed-budget comparisons
- same wall-clock budget
- same number of rollouts
- same horizon
- same GPU

3. Multiple tasks
- easy cluttered navigation
- narrow passage
- higher-speed scenario
- dynamic obstacle variant if feasible

4. Metrics
- success rate
- final goal distance
- average cumulative cost
- collision count
- wall-clock per control step
- rollouts required to reach a given success level

5. Ablations
- no gradient refinement
- one gradient step vs multiple
- autodiff step size / weight
- sampling count sweep

Without these, `diff_mppi` is still mostly a nice demo.

## Recommended Direction

If the goal is publication rather than just a strong repo, I would focus the repo around one paper target:

Title sketch:
- Lightweight Rollout-Differentiated MPPI in Plain CUDA
- Hybrid Sampling-and-Gradient MPPI for Budget-Constrained GPU Control

Core thesis:
- Keep MPPI as the main optimizer.
- Do not replace it with a full differentiable planner.
- Add only a cheap local sensitivity-based refinement step.
- Show better quality per unit compute.

That framing is much stronger than trying to claim novelty for:
- neural SDF navigation
- mini Isaac Gym
- generic GPU neuroevolution

## Immediate Next Work

1. Turn `diff_mppi` into an experiment suite instead of a single demo
- fixed seeds
- CSV logging
- benchmark configs

2. Add a clean ablation binary
- `benchmark_diff_mppi`
- output success/cost/time tables

3. Add 3 to 5 environments designed for paper figures
- open field
- cluttered maze
- narrow passage
- moving obstacle

4. Write a sharper novelty statement
- explicitly position against PI-Net, Differentiable MPC, and Feedback-MPPI

## Honest Assessment

The repo now contains several high-quality research-style demos.
But only one line currently looks plausibly paper-worthy with additional work: `diff_mppi`.

Everything else is better treated as:
- strong engineering
- useful educational artifacts
- supporting material for the repo

Not as the main novelty claim.
