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
