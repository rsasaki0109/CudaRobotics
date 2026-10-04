# GPU Batched Inverse Kinematics with Random Restarts

`gpu_batched_ik` solves numerical inverse kinematics for a 7-DOF arm on the
GPU, one thread per (target pose, initial configuration) pair.

Damped-least-squares IK converges from a good initial guess, but from a bad one
it stalls in a local minimum or against a joint limit. The standard remedy is
random restarts: solve from many initial configurations and keep the best. On a
CPU the restarts multiply the cost. On a GPU every (target, restart) pair is an
independent problem, so restarts are close to free. This follows the repo's
canonical idiom: **one thread = one IK solve**.

## Setup

- **Arm:** Franka-Panda-like, using the published modified-DH table, joint limits and flange offset. Forward kinematics gives the full flange pose.
- **Targets:** `1024` target poses, each the forward kinematics of a random in-limit configuration, so every target is reachable.
- **Restarts:** `32` random initial configurations per target.
- **Solver:** damped least squares on the 6-D pose error (position, and the rotation vector of `R_target * R^T`), using the geometric Jacobian.
  - damping `lambda^2 = 0.01`;
  - joint-step cap of `0.3 rad`;
  - joints clamped to their limits after every step;
  - at most `80` iterations.
- **Success:** position error < `1 mm` and orientation error < `1 deg`.
- **Shared code:** the solver is a single `__host__ __device__` routine. A serial CPU loop and the batch CUDA kernel both call it, built with `--fmad=false`.

## Results

| Restarts per target (best of) | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| targets solved | 42.2% | 85.6% | 96.7% | 99.1% | **99.8%** |

| | CPU serial | GPU batch |
|---|---:|---:|
| all 32768 solves | 1.9-2.8 s | 8-19 ms |
| per solve | ~70 us | ~0.3-0.6 us |

- **Restarts carry the success rate.** One damped-least-squares solve from a random start reaches the target 42% of the time. The best of 32 reaches it 99.8% of the time.
- **On the GPU the 32 restarts for all 1024 targets cost about 10 ms**, roughly 120-300x faster than the serial CPU batch. The range reflects a GPU shared with other work while this was measured.
- **CPU and GPU agree:** they give the same outcome (solved or not) on 100% of the 32768 (target, restart) pairs.

## Collision-aware IK

The second part of the demo adds six sphere obstacles in front of the arm. The arm's links are approximated by 0.06 m spheres, three per link segment.

**Targets.** The 1024 targets are taken from configurations that are collision-free with at least 2 cm of clearance and whose flange is within 12 cm of an obstacle surface. So every target has a collision-free solution, and it sits next to an obstacle. 55266 random configurations were drawn to find them.

**Success** here also needs a collision-free final configuration.

**Two solvers, both run with the same restarts:**
- **pose-only:** the damped-least-squares IK above, ignoring the obstacles.
- **null-space avoidance:** the same IK plus a step down the collision cost's gradient, projected into the null space of the pose task (`g - J^T (J J^T + lambda^2 I)^-1 J g`, gain 20). It uses the arm's redundant seventh degree of freedom to move the links away from the obstacles without disturbing the pose.

| Restarts (best of) | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| pose-only, solved collision-free | 28.1% | 73.0% | 90.1% | 97.5% | 99.4% |
| null-space avoidance | 29.2% | 74.6% | 91.7% | 97.9% | 99.4% |

**Reading:**
- **Collisions are common.** Next to an obstacle, 27.8% of the pose-only IK solutions collide.
- **Restarts and selection solve it anyway.** Keeping the best collision-free restart reaches 99.4% with 32 restarts: the restarts land in different IK branches, and some of those clear the obstacles.
- **Null-space avoidance adds 1-2 points at 1-16 restarts and nothing at 32.** Gains of 50-200 gave the same picture. With one redundant degree of freedom, self-motion rarely moves a colliding branch into a free one, while a different restart often lands in one.
- **On a GPU the cheap answer is more restarts.**
- **CPU / GPU:** the collision-aware batch takes 3.7-3.9 s on the CPU and 16 ms on the GPU, with the same outcome on 100% of solves.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_batched_ik -j$(nproc)
./bin/gpu_batched_ik                       # also writes the GIF
./bin/gpu_batched_ik --check --no-video    # the CTest gate (gpu_batched_ik_gate)
```

`--check` exits non-zero unless the best-of-32 success rate is at least 99%
(pose-only) and at least 95% (collision-aware), and CPU and GPU agree on at least
99% of the solves. CTest runs it as
`gpu_batched_ik_gate` (labels `gpu;manipulation;ik`).

Generated files:

- `tmp/gpu_batched_ik.avi`
- `gif/gpu_batched_ik.gif`

## Output

The GIF shows one target in side and top views as its restarts iterate: four
that reach the target (green) and four that stall (grey). The info panel shows
the batch headline: success for the best of 1 and of 32 restarts, the CPU and GPU
times, and the speedup.
