# 3D ESDF-MPPI with Moving Obstacles

Raw `--trials` tables for [`gpu_esdf_mppi_3d.md`](../gpu_esdf_mppi_3d.md#moving-obstacles). Every table runs the same 30 mover scenarios (episode seeds 1000-1029) for all three modes, so rows are paired. Per-episode lines come from the same runs.

## 4 movers, 1x speed

```bash
./bin/gpu_esdf_mppi_3d --movers 4 --trials 30 --seed 1000
```

| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 26/30 | 4 | 0 | 84.0 | 0.31 | 0.46 |
| rebuild | 29/30 | 0 | 1 | 97.5 | 0.41 | 2.33 |
| predict | 30/30 | 0 | 0 | 87.0 | 0.41 | 0.46 |

## 6 movers, 1x speed

```bash
./bin/gpu_esdf_mppi_3d --movers 6 --trials 30 --seed 1000
```

| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 23/30 | 7 | 0 | 84.0 | 0.21 | 0.45 |
| rebuild | 30/30 | 0 | 0 | 92.2 | 0.40 | 2.42 |
| predict | 29/30 | 0 | 1 | 100.6 | 0.41 | 0.46 |

## 8 movers, 1x speed

```bash
./bin/gpu_esdf_mppi_3d --movers 8 --trials 30 --seed 1000
```

| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 22/30 | 8 | 0 | 84.0 | 0.18 | 0.45 |
| rebuild | 30/30 | 0 | 0 | 90.1 | 0.40 | 2.42 |
| predict | 29/30 | 0 | 1 | 104.0 | 0.41 | 0.46 |

## 6 movers, 2x speed

```bash
./bin/gpu_esdf_mppi_3d --movers 6 --mover-speed 2 --trials 30 --seed 1000
```

| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 21/30 | 9 | 0 | 84.0 | 0.16 | 0.46 |
| rebuild | 25/30 | 5 | 0 | 89.6 | 0.24 | 2.35 |
| predict | 30/30 | 0 | 0 | 90.6 | 0.39 | 0.45 |

## 6 movers, 3x speed

```bash
./bin/gpu_esdf_mppi_3d --movers 6 --mover-speed 3 --trials 30 --seed 1000
```

| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 23/30 | 7 | 0 | 84.0 | 0.20 | 0.46 |
| rebuild | 27/30 | 3 | 0 | 89.4 | 0.28 | 2.37 |
| predict | 29/30 | 1 | 0 | 88.1 | 0.39 | 0.46 |

## Bounce-aware prediction (episode seeds from 2000)

Fresh scenarios, all four modes. The 6-mover 3x and 8-mover 3x runs use 30 scenarios (the first 60-scenario attempt was stopped for low memory).

```
6 movers at 2.0x speed, 60 episodes per mode (episode seeds 2000..2059)
| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 44/60 | 16 | 0 | 81.7 | 0.19 | 0.45 |
| rebuild | 53/60 | 7 | 0 | 90.1 | 0.29 | 2.40 |
| predict | 60/60 | 0 | 0 | 90.3 | 0.40 | 0.47 |
| predict_bounce | 60/60 | 0 | 0 | 91.0 | 0.42 | 0.48 |
```

```
6 movers at 3.0x speed, 30 episodes per mode (episode seeds 2000..2029)
| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 22/30 | 8 | 0 | 78.8 | 0.17 | 0.46 |
| rebuild | 25/30 | 5 | 0 | 87.8 | 0.24 | 2.38 |
| predict | 30/30 | 0 | 0 | 91.9 | 0.40 | 0.46 |
| predict_bounce | 29/30 | 0 | 1 | 95.8 | 0.41 | 0.47 |
```

```
8 movers at 3.0x speed, 30 episodes per mode (episode seeds 2000..2029)
| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 22/30 | 8 | 0 | 78.8 | 0.17 | 0.47 |
| rebuild | 23/30 | 7 | 0 | 89.1 | 0.22 | 2.40 |
| predict | 30/30 | 0 | 0 | 89.4 | 0.38 | 0.47 |
| predict_bounce | 29/30 | 0 | 1 | 105.5 | 0.41 | 0.51 |
```

## Local ESDF update (episode seeds 1000-1029)

All five modes on the original scenarios, including `--mode 4` (rebuild_local).

```
Local ESDF update (6 windows): 0.151 ms; max |local - full rebuild| 0.125 m over 575419 voxels in the clearance band
6 movers at 1.0x speed, 30 episodes per mode (episode seeds 1000..1029)
| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 23/30 | 7 | 0 | 84.0 | 0.21 | 0.45 |
| rebuild | 30/30 | 0 | 0 | 92.2 | 0.40 | 2.38 |
| predict | 29/30 | 0 | 1 | 100.6 | 0.41 | 0.46 |
| predict_bounce | 29/30 | 0 | 1 | 91.1 | 0.41 | 0.47 |
| rebuild_local | 30/30 | 0 | 0 | 88.8 | 0.41 | 0.57 |
```

```
Local ESDF update (6 windows): 0.187 ms; max |local - full rebuild| 0.125 m over 575419 voxels in the clearance band
6 movers at 2.0x speed, 30 episodes per mode (episode seeds 1000..1029)
| mode | success | collisions | timeouts | mean steps | mean min clearance (m) | ms per step |
|---|---:|---:|---:|---:|---:|---:|
| static | 21/30 | 9 | 0 | 84.0 | 0.16 | 0.47 |
| rebuild | 25/30 | 5 | 0 | 89.6 | 0.24 | 2.40 |
| predict | 30/30 | 0 | 0 | 90.6 | 0.39 | 0.46 |
| predict_bounce | 30/30 | 0 | 0 | 84.7 | 0.41 | 0.48 |
| rebuild_local | 25/30 | 5 | 0 | 92.1 | 0.26 | 0.61 |
```
