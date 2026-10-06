# GPU relocalization for Monte Carlo localization

A seeded benchmark for recovering 2D-LiDAR Monte Carlo localization (MCL) from a kidnapping and for global localization, plus a GPU relocalization method.

The repository's localization demos (AMCL, expansion-reset MCL, global localization) each run one scripted scenario. This benchmark runs many random worlds and pairs every method on the same episode.

Source: [`src/benchmark_mcl_relocalization.cu`](../src/benchmark_mcl_relocalization.cu). Test report: [results/mcl_relocalization_test_2026-10-06.md](results/mcl_relocalization_test_2026-10-06.md).

## Benchmark

**World.** Each seed draws one world:
- A 20 m × 20 m map at 5 cm, with 3 × 3 rooms. Each wall between rooms has a 1 m door, or is left open.
- Up to three pieces of furniture per room.
- A drive at 0.5 m/s along breadth-first paths between random goals.
- Odometry with 5 % noise.
- A scan every 0.1 s: 180 beams with 2 cm noise, and 5 % short readings from unmodeled obstacles.

The filter uses every third beam on a likelihood field (σ = 0.1 m). It has 2000 particles and carries log weights between resamplings, which happen when the effective sample size falls below half.

**Two environments.**

| | open | hard |
|---|---|---|
| field of view | 360° | 270° |
| range | 8 m | 5 m |
| rooms | random sizes and furniture | equal sizes, the same furniture layout in every room |
| unmapped boxes | none | 25 (0.2-0.4 m, against walls and furniture) |

**Two cells.**
- **kidnap:** the filter starts at the true pose. At step 100 of 400, the robot is carried to a random pose at least 5 m away, and odometry sees nothing.
- **global:** the filter starts with particles spread over the free space, for 300 steps.

**Metrics.**
- **Localized:** position error < 0.3 m and heading error < 0.2 rad.
- **Recovered:** localized for 10 steps in a row.
- Also reported: steps to recover, the fraction of localized steps after the kidnap and at the final step, and, before the kidnap, the fraction of localized steps (does a reset method break good tracking?).

## Methods

| Method | What happens when the filter is lost |
|---|---|
| `mcl` | nothing |
| `aug` | augmented MCL (AMCL): random particles are injected when the short-term average likelihood (rate 0.1) falls below the long-term one (0.001) |
| `er` | expansion resetting (Ueda et al. 2004): when the mean per-beam likelihood falls below 0.45, every particle is scattered (σ 0.2 m, 0.2 rad) and reweighted |
| `reloc` | GPU relocalization, on the same trigger as `er`; see below |
| `reloc_lr` | the same, triggered by a likelihood ratio instead of a threshold |

**How `reloc` works.**
1. **Grid search.** The scan is scored at every pose of a grid over the free space, every 0.2 m and 5°: about 6000 positions × 72 headings. Scoring uses a smoother field (σ = 0.3 m). One CUDA block covers one position, with one thread per heading.
2. **New particles.** Half of the particles, the least likely ones, are replaced by samples around the 32 best poses.
3. **Kidnap prior.** The new particles enter with the belief's mean weight times a prior that the robot was moved (1e-4).
   - A real kidnapping overrules the prior at once: the old belief explains the scan tens of nats worse.
   - A look-alike pose found on a false trigger does not.

**How `reloc_lr` triggers.** It runs the grid search every step and resets when the best grid pose beats the belief's best particles by more than 0.05 per beam, on the same coarse field.

## Development (seeds 0-29)

Choices made on these seeds only:
- **The threshold** (0.45, chosen in the open environment).
  - At 0.5 to 0.6, false triggers before the kidnap appear in 11 to 30 of the 30 runs.
  - At 0.45, one.
- **Carrying weights.** A first version weighted each step by its scan alone when it did not resample, which drops the history that tells look-alike rooms apart. Carrying log weights fixed it.
- **The kidnap prior.** In the hard environment, the share of localized steps before the kidnap went 0.92 → 0.93 → 0.94 → 0.98 for priors 1 → 1e-2 → 1e-3 → 1e-4, with recovery unchanged.
- **The trigger.** The likelihood-ratio trigger was no better than the threshold on these seeds: 30/30 recoveries for both, and 0.97 against 0.98 localized before the kidnap. It is reported, not chosen.

## Test on fresh seeds (100-199)

No earlier step ran these seeds. The criterion was fixed in [`scripts/mcl_relocalization_eval.py`](../scripts/mcl_relocalization_eval.py) before they ran. In each environment, `reloc` against `aug`, the strongest baseline:
1. recovers from the kidnapping in more episodes (paired sign test, p < 0.05);
2. recovers in global localization in more episodes (paired sign test, p < 0.05);
3. its fraction of localized steps before the kidnap is not lower than `aug`'s by more than 0.03.

| Environment | Method | kidnap: recovered | median steps | final | localized before the kidnap | global: recovered | ms per step |
|---|---|---:|---:|---:|---:|---:|---:|
| open | `mcl` | 1 | 198 | 1 | 1.000 | 13 | 0.36 |
| open | `aug` | 80 | 64 | 83 | 1.000 | 39 | 0.39 |
| open | `er` | 2 | 267 | 3 | 1.000 | 21 | 0.45 |
| open | `reloc` | **100** | **0** | **100** | 1.000 | **100** | 0.37 |
| open | `reloc_lr` | 100 | 0 | 100 | 1.000 | 100 | 1.85 |
| hard | `mcl` | 1 | 247 | 1 | 0.996 | 5 | 0.34 |
| hard | `aug` | 54 | 89 | 54 | 0.988 | 19 | 0.37 |
| hard | `er` | 3 | 157 | 3 | 0.981 | 16 | 0.41 |
| hard | `reloc` | **99** | **4** | **84** | 0.946 | **100** | 0.38 |
| hard | `reloc_lr` | 97 | 3 | 84 | 0.945 | 99 | 1.32 |

Counts are out of 100 episodes. One grid search takes 1.3-1.8 ms on the GPU, and the per-step time includes the host-side resampling.

`reloc` against `aug`, paired (only `reloc` / only `aug` recovered):

| Environment | kidnap | global |
|---|---|---|
| open | 20 / 0, p = 2e-6 | 61 / 0, p = 9e-19 |
| hard | 45 / 0, p = 6e-14 | 81 / 0, p = 8e-25 |

**The criterion is not met.** Conditions 1 and 2 hold everywhere. Condition 3 fails in the hard environment: before the kidnap, `reloc` is localized 0.946 of the time against 0.988 for `aug`. Half of the hard runs (50/100) see a false trigger, and some of those move the estimate into a look-alike room.

## Reading

- **Exhaustive relocalization is what makes MCL recover.**
  - In the open environment it recovers every kidnapping, usually on the step it happens, and every global localization. The best baseline (`aug`) manages 80 and 39 of 100.
  - Expansion resetting is built for small displacements. It recovers almost none of these 5 m+ kidnappings.
  - The full grid search costs about 1.5 ms on the GPU, little enough to run on every trigger.
- **The price in a repetitive, cluttered building is false resets.**
  - Unmapped clutter and a short-range sensor push the mean likelihood under the threshold while the filter is still right.
  - In equal rooms with the same furniture, the grid search then offers poses in another room that explain the scan as well. The kidnap prior suppresses most of them (0.92 → 0.98 on development), but not enough on the test seeds.
- **Recovery in the hard environment is not the same as staying right.** `reloc` recovers 99 kidnappings, but only 84 end localized. In identical rooms the scan cannot tell which room the robot is in, and the estimate can flip between copies until a door comes into view.
- **The likelihood-ratio trigger did not fix the false resets.** In identical rooms another room's pose really does score as well as the true one.

## Follow-up: a separate candidate hypothesis (negative)

`reloc_dual` keeps the belief whole and runs the relocalization particles as a second filter beside it (1000 particles). The candidate set replaces the belief only when it wins a sequential test:
- **Log odds.** The log odds that the robot was moved start at the kidnap prior, log 1e-4.
- **Per scan.** Each scan adds the log of the two sets' marginal likelihoods (candidate over belief).
- **Switch and drop.** The set switches when the odds turn positive. It is dropped when it falls 20 nats below the prior or runs 30 steps without winning.

After a real kidnapping, the belief explains the scan tens of nats worse per step, so the switch comes one step later. The idea was that look-alike poses would never win.

Development seeds 0-29, hard environment, kidnap cell. Every row recovers all 30 kidnappings. Columns: runs with a reset or switch before the kidnap, and the share of steps localized before it.

| Variant | runs with a reset / switch before the kidnap | localized before the kidnap |
|---|---:|---:|
| `reloc` (reference) | 18 | 0.98 |
| candidate set, switch on positive odds | 3 | 0.97 |
| + candidates kept 1 m away from the belief's estimate | 4 | 0.93 |
| + a CUSUM drift: the candidates must win by 2 nats per step | 3 | 0.97 |
| + switch only while the belief's own alpha is below 0.45 | 3 | 0.97 |
| + keep the old belief as the alternative after a switch | 3 | 0.97 |

A drift of 1 / 2 / 3 / 5 nats per step gives 0.94 / 0.97 / 0.97 / 0.97. The open environment stays perfect (30/30, no switch before the kidnap) in every variant. The last row is the code's `reloc_dual`.

- **Fewer false switches, but not less harm.** The separate set cuts the runs with a false reset from 18 to 3. Each remaining false switch is a full jump to another room, though, and the share of localized steps does not improve. No fresh-seed test was run: development already shows no gain on the failed condition.
- **Why the test cannot help.** It is the likelihood model, not the reset rule.
  - On seed 9, at step 10, the true pose scores -23.8 and a pose in an identical room -15. Many beams hit unmapped clutter, and the look-alike pose explains the short readings better.
  - The trace of seed 29 shows the same thing in smaller steps: a copy room beats the true pose by 2-3.5 nats per scan for several scans.
  - With the scan model preferring the wrong room, every rule built on it can be talked into the switch.
  - Switching back does not happen either: in an identical room the true pose never wins clearly.

## Follow-up: an occlusion-aware beam model

`--sensor beam` weighs every method with the beam model of Probabilistic Robotics (6.3) instead of the likelihood field.
- **Expected range.** Each beam's expected range is ray-cast in the map: sphere tracing on the distance field, one GPU thread per particle.
- **Short readings.** A reading short of the expected range has its own term (z_short = 0.2, λ = 0.5/m), so a box in front of a wall is not evidence against the true pose.
- **Other parameters.** z_hit = 0.75, σ = 0.1 m, z_rand = 0.05.
- **Threshold.** The reset threshold for this model's per-beam likelihood scale (0.55) was chosen on development seeds 0-29. While tracking, the scale runs 0.75-2.1; when lost, about 0.2.

### Is the GPU worth it here?

Timing (`--time-grid`) on the same scans. The best scores and log-likelihood sums agree between CPU and GPU.

| Work per scan | GPU | CPU, 1 thread | CPU, 12 threads |
|---|---:|---:|---:|
| relocalization grid search (open: 546k poses × 55 beams) | 1.6 ms | 348 ms | 123 ms |
| relocalization grid search (hard: 521k poses × 34 beams) | 1.0 ms | 210 ms | 75 ms |
| beam-model weighting, 2000 particles (open / hard) | 0.23 / 0.18 ms | 28 / 18 ms | 8.7 / 5.7 ms |
| beam-model weighting, 10000 particles (open / hard) | 0.27 / 0.24 ms | 153 / 107 ms | 41 / 36 ms |

- **The grid search is where the GPU matters.** On the GPU it takes 1-1.6 ms. Even with 12 threads the CPU needs 75-123 ms, the whole budget of a 10 Hz scanner. The GPU makes the exhaustive search affordable on every trigger.
  - The CPU code here is the plain loop. A branch-and-bound matcher (as in Cartographer) prunes most poses, so the honest claim is "exhaustive search at 1 ms without pruning", not "impossible on a CPU".
- **The beam model is affordable on either at 2000 particles.** A CPU thread pool handles it, so at this size it is a modeling choice, not a GPU one.
  - The GPU makes it nearly free (0.2 ms), and at 10000 particles a single CPU thread no longer fits a 10 Hz loop (107-153 ms).
- **The filter itself (2D MCL with a few thousand particles) does not need a GPU.**

### Development (seeds 0-29)

With the beam model, no method is harmed before the kidnap in either environment: the localized share is 1.00 for every method.
- **Hard environment, kidnap cell.** `reloc` recovers 30/30 and ends localized in 29.
- **Open environment.** At a threshold of 0.4, some recoveries in the open environment end in a look-alike pose (localized after 0.83). At 0.55 it is 1.00.

### Test on fresh seeds (200-299)

No earlier step ran these seeds. Same three-part criterion as before, fixed before they ran. Report: [results/mcl_relocalization_beam_test_2026-10-07.md](results/mcl_relocalization_beam_test_2026-10-07.md).

| Environment | Method | kidnap: recovered | median steps | final | localized before the kidnap | global: recovered | final |
|---|---|---:|---:|---:|---:|---:|---:|
| open | `aug` | 96 | 58 | 96 | 1.000 | 40 | 41 |
| open | `reloc` | **100** | **0** | **100** | 1.000 | **100** | **100** |
| hard | `aug` | 53 | 72 | 55 | 0.980 | 26 | 28 |
| hard | `reloc` | **100** | **0** | 94 | 0.979 | **100** | 97 |
| hard | `reloc_lr` | 99 | 0 | **96** | **0.990** | 100 | **100** |
| hard | `reloc_dual` | 99 | 1 | 91 | 0.987 | 100 | 98 |

Paired, `reloc` against `aug`:

| Environment | kidnap | global |
|---|---|---|
| open | 4 / 0, p = 0.12 | 60 / 0, p = 2e-18 |
| hard | 47 / 0, p = 1e-14 | 74 / 0, p = 1e-22 |

**The criterion is not met.**
- **The failure:** condition 1 fails in the open environment. With the beam model, augmented MCL also recovers 96 of the 100 kidnappings. Only 4 seeds differ, too few for the sign test.
- **What still holds:**
  - The safety condition that failed with the likelihood field now holds: 0.979 against 0.980.
  - Global localization and the hard environment stay far apart.
  - `reloc` recovers on the kidnap step itself, where `aug` takes a median of 58 steps.

**The same seeds with the likelihood field.** Run with `--sensor field --seed-offset 200`, outside the criterion:

| Environment | Method | kidnap: recovered | final | localized before the kidnap | global: final |
|---|---|---:|---:|---:|---:|
| open | `aug`, field → beam | 82 → 96 | 83 → 96 | 1.000 → 1.000 | 35 → 41 |
| hard | `aug`, field → beam | 41 → 53 | 40 → 55 | 0.979 → 0.980 | 21 → 28 |
| hard | `reloc`, field → beam | 100 → 100 | 91 → 94 | **0.932 → 0.979** | 90 → 97 |

### Reading

- **The occlusion-aware model removes the cost of relocalization.** Relocalization's damage to good tracking in the hard environment (0.932 localized before the kidnap) goes away (0.979, the same as `aug`). The cause found in the follow-up above was the scan model, and fixing the model fixed it.
- **It helps every method.** Augmented MCL gains most in the open environment (82 → 96 kidnap recoveries), which is why the pre-registered count comparison there lost its power.
- **The likelihood-ratio trigger looks best in the hard environment:** 0.990 localized before the kidnap, 96 final, 100 global. It was not the pre-registered method, so this is a lead, not a result.

## Limitations and next steps

- Simulated worlds, one map size, one sensor model, 2000 particles, CPU resampling.
- The hard environment is extreme: every room is an exact copy.
- **Done: a scan model that knows occlusion** (the beam model above). It fixed the safety condition.
- **Next:**
  - Real data: a 2D LiDAR bag with a map, where the clutter is real rather than drawn.
  - A branch-and-bound CPU baseline for the grid search, to state the GPU's advantage against the best CPU method rather than a plain loop.

## Reproduce

```bash
./bin/benchmark_mcl_relocalization --env open --seed-count 30 [--methods mcl,aug,er,reloc,reloc_lr] [--trace]
./bin/benchmark_mcl_relocalization --env hard --seed-count 30 --inject-prior 1e-2   # development sweeps
./bin/benchmark_mcl_relocalization --env hard --seed-count 30 --methods reloc,reloc_dual   # first ablation row: --dual-exclude 0 --dual-drift 0 --dual-gate 0 --dual-swap 0
python scripts/mcl_relocalization_eval.py     # the test on fresh seeds 100-199 (writes the CSVs and the report)
python scripts/mcl_relocalization_eval.py --sensor beam --seed-offset 200 --date 2026-10-07   # beam model, seeds 200-299
./bin/benchmark_mcl_relocalization --env hard --sensor beam --time-grid 10 [--particles 10000]   # CPU vs GPU timing
```
