# GPU LiDAR Ground Segmentation and Object Pipeline

`gpu_ground_segmentation` runs a LiDAR object pipeline on the GPU, on synthetic scans with exact ground truth. The pipeline:
1. separates ground from non-ground points;
2. clusters the rest into objects;
3. fits an oriented box to each cluster, classes it (car, van or other) and completes it with a size prior;
4. tracks the vehicles across scans, parked or moving.

The production path is also a library, `CudaRobotics::lidar_objects_gpu` (see [Library](#library)). It complements the repo's KISS-ICP, voxel mapping and DBSCAN demos.

## Current performance

Measured with the current build and class on held-out scenes and drives (seeds 1-20), by `scripts/lidar_objects_summary.py` ([results/lidar_objects_summary_2026-10-06.md](results/lidar_objects_summary_2026-10-06.md)). The sensor is 1.8 m above the ground unless a height is given. Boxes are the vehicles', completed with the size prior.

| Stage | measure | value |
|---|---|---:|
| ground segmentation | F1 (a plain height threshold: 0.835) | **0.983** |
| clustering | objects found as one cluster | **97%** (2242 / 2309) |
| boxes, single scan | BEV IoU: axis-aligned / L-shape / + size prior / + free space | 0.402 / 0.500 / 0.728 / **0.777** |
| boxes, tracked along a drive | BEV IoU | **0.763** |
| boxes, moving traffic | BEV IoU: single scan / motion tracker / hybrid | 0.754 / 0.761 / **0.781** |
| motion tracker | velocity error, moving / parked vehicles | 0.63 / 0.63 m/s |
| motion tracker | identity switches over 20 drives | 30 |
| time per scan (`--moving`) | mean / max over a drive's scans, without / with the free-space refinement | 10.1 / 19.7 ms; about +10 ms (this run, shared GPU) |

| Sensor height | cars classed car | vans classed van | others classed car or van | box IoU with the prior |
|---:|---:|---:|---:|---:|
| 0.8 m | 367 / 407 | 34 / 80 | 140 / 2330 | 0.581 |
| 1.2 m | 428 / 450 | 71 / 107 | 89 / 2894 | 0.637 |
| 1.8 m | 454 / 472 | 122 / 148 | 46 / 4469 | 0.728 |
| 2.5 m | 470 / 476 | 154 / 160 | 24 / 6171 | 0.789 |

**Reading the rest of this page.** The sections below follow the development in order, each with the numbers measured at the time:
- every result on held-out seeds, with the negative results kept;
- the boxes and trackers before "Sensor height and the class" were measured with the first class, which the class in use has since replaced.

The table above is the current state.

## Ground segmentation

A single height threshold fails as soon as the ground is not flat: a ramp rises
above it and a curb splits it. The demo implements a concentric-zone ground model
in the spirit of Patchwork.

### Method

1. **Bin.** Each point gets a polar bin (24 rings with geometrically growing width, from 2 m to 60 m, × 72 sectors). Points are stably sorted by bin: thrust on the GPU, `std::stable_sort` on the CPU, giving the same order.
2. **Fit, one warp per bin.**
   - Seeds are the points within 0.25 m of the bin's lowest point.
   - A plane is fitted by PCA (the smallest eigenvector of the covariance, by Jacobi iterations) and refitted twice on the points within 0.12 m of it.
   - The plane counts if it is within 25° of horizontal.
   - The 32 lanes share the bin's points. Near bins hold hundreds to thousands of points, so one thread per bin left a few threads doing most of the work (GPU 1.8 → 0.8-1.2 ms per scan).
3. **Check, one thread per sector.** The sector's rings are walked outward. A plane is kept only if its height at the bin centre is within 0.25 m + 0.18 × (distance from the last kept plane) of that plane, or of the flat ground under the sensor for the first one. This stops a flat car roof from passing for ground.
4. **Label.** A point is ground if its bin's plane is kept and the point lies within 0.12 m of it.

The check and the labelling are `__host__ __device__` routines shared with the
CPU reference. The fit is the same algorithm, with the GPU's sums taken
lane-strided.

### Scene

A synthetic 64 × 1024 LiDAR (-24.8° to +2° elevation, 1.8 m mounting height, 2 cm range noise), ray-cast per beam with exact ground-truth labels.

- **Terrain:** a 6° ramp beyond x = 10 m, a 0.15 m curb and sidewalk beyond y = 6 m, and gentle undulation.
- **Objects:** cars (one on the ramp), a van, a wall, a crate, a bench, poles and pedestrians. The boxes stand at headings from -15° to 35°, so that the box fitting below has headings to find.
- **Scans:** eight scans from sensor poses on the flat part, on the ramp and on the sidewalk side.

### Results (8 scans, 482643 returns)

| Method | precision | recall | F1 |
|---|---:|---:|---:|
| concentric-zone model | 0.983 | 0.982 | **0.982** |
| height threshold (0.25 m above flat ground under the sensor) | 0.970 | 0.735 | 0.836 |

- **The height threshold collapses once the sensor faces the ramp.** Per-scan F1 falls from 0.94 on flat ground to 0.62-0.66 at x = 16-20 m, because the rising ground is labelled non-ground.
- **The model stays at F1 0.97-0.99 on every scan.**
- **CPU and GPU agree** on 100% of the labels.
- **Time per scan:** CPU 7-13 ms, GPU 0.8-1.2 ms (about 9-16x). The GPU time is dominated by the sort and the kernel launches, which is ample for a 10 Hz LiDAR. The ranges come from a GPU shared with other work.

## Object clustering after ground removal

The second stage clusters the remaining points into objects: Euclidean clustering with 0.5 m connectivity, dropping clusters under 10 points (the same idea as PCL's `EuclideanClusterExtraction`).

**GPU implementation.**
- The points are sorted by 0.5 m grid cell, and neighbouring cells are found by binary search.
- Each point unites itself with every lower-indexed neighbour in a lock-free union-find. The union-find always hooks the larger root under the smaller one (`atomicCAS`), so each component's root is its smallest point index.
- The CPU reference runs a BFS over the same grid and assigns the same labels.
- The two partitions are compared exactly, and they are identical on every scan.

**Scoring.** Clusters are scored against the ground-truth object of every return (an object counts if the scan sees it with at least 20 points). An object is *found* if one cluster holds at least half of the object's points and at least half of that cluster's points belong to the object.

| Ground removal before clustering | objects found | split objects | clusters mostly of ground | clusters |
|---|---:|---:|---:|---:|
| concentric-zone model | **109 / 116** | 14 | 107 | 269 |
| height threshold | 87 / 116 | 11 | 361 | 481 |
| none | 32 / 116 | 12 | 869 | 938 |

- **Ground removal decides the clustering.** Without it, the ground connects or swamps most objects.
- **With the height threshold,** the ramp's ground survives as hundreds of spurious clusters, and objects standing on it merge with that ground.
- **With the model,** 94% of the objects come out as their own cluster.

**Voxel clustering, same partition.** Points in one voxel of side `0.5 m / sqrt(3)` are always within 0.5 m of each other, so a voxel can be a node. Two voxels are connected if any pair of their points is within 0.5 m.

| Step | What it does |
|---|---|
| graph | about 1550 occupied voxels per scan instead of about 50k points |
| unite | one warp per (voxel, neighbour offset): 62 offsets within 2 voxels per axis, each pair once |
| skip | pairs already in one set, and pairs whose voxels' point bounds are more than 0.5 m apart |
| test | point pairs otherwise, shared by the warp's 32 lanes, stopping as soon as one lane finds a pair within 0.5 m |
| label | each component takes its smallest point index |

The result is the same partition and the same labels as the CPU's BFS on every scan.

**Time per scan** (GPU: minimum of 5 runs; three runs of the 8 scans):

| Clusterer | time | same partition as the CPU |
|---|---:|---|
| CPU BFS over the points | 44-48 ms | — |
| GPU point-level union-find | 7.7-8.5 ms | yes |
| GPU voxel-level union-find | **0.91-1.00 ms** | yes |

- **Point-level:** near the sensor a point has thousands of neighbours within 0.5 m, and every one is tested.
- **Voxel-level:** collapses those neighbourhoods and stops at the first close pair, about 8x faster than point-level and about 45x faster than the CPU.
- **One warp per voxel pair.** With one thread per (voxel, offset), a voxel pair near the sensor holds hundreds of points on each side. When the pair turns out not to be connected, every point pair is tested by that one thread. This took 3.8-4.2 ms per scan here, and up to 37 ms on the drive below when the sensor passed 2 m from a parked car. Sharing the tests over a warp gives 0.9-1.0 ms here and at most 1.9 ms on the drive.
- **What did not help:**
  - Sorting the points instead of scanning a dense grid, and union-find instead of iterated label propagation, left the point-level clusterer unchanged.
  - One thread per voxel instead of per (voxel, offset) was slower than point-level (too few threads).

## Oriented boxes: L-shape fitting

The third stage fits an oriented box to every cluster of the model's ground
removal, by the search-based L-shape fitting of Zhang et al. (2017) with the
closeness criterion.

- **Search.** The cluster's points are projected on the axes of each of 90 headings in [0°, 90°) (a rectangle repeats every 90°).
- **Score.** Each point's distance to the nearer edge of the bounding rectangle in that frame is clamped below at 1 cm, and the heading scores the sum of the inverse distances. Points hugging two perpendicular edges, the L a LiDAR sees of a car, score high.
- **Box.** The best heading's bounding rectangle is the box.
- **GPU: one warp per (cluster, heading).** Lane *l* takes points *l*, *l* + 32, …, and the partial sums are combined by an xor butterfly. The clusters are formed on the GPU (a stable sort of the voxel clusterer's labels and `reduce_by_key`), then one thread per cluster picks the best heading.
- **CPU: the same arithmetic.** The CPU runs the same 32 lanes and butterfly with the same heading table, and its boxes are bit-identical to the GPU's on every scan.

**Scoring.** Each ground-truth box whose cluster is found (as above) is one
observation, 50 over the 8 scans. The fit is compared with the axis-aligned
box of the same points:

| Box | heading error | BEV IoU | IoU ≥ 0.5 | centre error | long / short side error |
|---|---:|---:|---:|---:|---:|
| L-shape | **1.3°** | **0.47** | **26 / 50** | 1.11 m | 1.42 / 0.70 m |
| axis-aligned | 15.2° | 0.40 | 19 / 50 | 1.19 m | 1.48 / 0.89 m |

| Box (L-shape vs axis-aligned) | observations | heading error | BEV IoU |
|---|---:|---:|---:|
| car, 20° | 7 | 3.4° vs 20.0° | 0.55 vs 0.39 |
| car on the ramp, -15° | 8 | 0.5° vs 15.0° | 0.61 vs 0.46 |
| car, 35° | 8 | 0.9° vs 35.0° | 0.52 vs 0.31 |
| van on the ramp, 10° | 7 | 0.3° vs 10.0° | 0.44 vs 0.38 |
| crate, 30° | 5 | 0.2° vs 30.0° | 0.14 vs 0.17 |
| wall, 0° | 7 | 2.0° vs 0.0° | 0.19 vs 0.29 |
| bench, 0° | 8 | 1.4° vs 0.0° | 0.70 vs 0.70 |

- **The heading is what L-shape fitting buys.** The heading error drops from 15° to about 1°, and on the rotated cars the IoU rises by 0.15-0.2.
- **The extent is limited by what the scan sees.** The box covers only the visible surface: the far sides of a car, the far end of the 12 m wall, and most of the distant low crate are hidden, so the centre and side errors stay near 1 m for both boxes. The size prior below fills in the hidden part.
- **Faces visible.** With one face visible (11 observations) the heading error is 0.9°, and with two (39) it is 1.4°: the closeness criterion also aligns a single wall-like face.
- **Where axis-aligned wins.** On the 0° wall the axis-aligned box is exact by construction, while the L-shape fits are off by 1-2° on it and on the bench.

**Time per scan** (about 33 fits; GPU minimum of 5 runs, measured while the GPU was 40-55% busy with other work):

| Fitter | time |
|---|---:|
| CPU | 54-88 ms |
| GPU, one thread per (cluster, heading) | 7.1 ms |
| GPU, one warp per (cluster, heading) | **1.6-2.1 ms** |

One thread per heading left a few threads looping over the thousands of points
of the largest clusters. A warp per heading spreads them over 32 lanes.

## Completing the boxes with a size prior

The L-shape box covers only what the scan sees. A class size prior fills in
the rest (host side, one step per cluster):

- **Class.** The cluster's height and footprint stand in for a classifier:
  - a *car* (4.5 × 1.8 m) is 1.0-2.0 m tall;
  - a *van* (6.0 × 2.0 m) is 2.0-3.2 m tall and at least 0.8 m wide, which keeps thin walls out;
  - in both cases the footprint must be at least 1.2 m long and within 1.2× the prior.
- **Axes.** The longer observed side is the length, unless neither side exceeds the class width (× 1.2). Then the sensor sees one end, and the length runs along the axis closer to the line of sight.
- **Growth.** A side shorter than the prior grows away from the sensor, keeping the edge the sensor sees. A side the sensor stands across grows about its centre.

| Box | heading error | BEV IoU | IoU ≥ 0.5 | centre error | long / short side error |
|---|---:|---:|---:|---:|---:|
| L-shape | 1.3° | 0.47 | 26 / 50 | 1.11 m | 1.42 / 0.70 m |
| L-shape + size prior | 1.3° | **0.60** | **34 / 50** | **0.82 m** | 0.85 / 0.49 m |
| prior 10% too small | 1.3° | 0.58 | 33 / 50 | 0.87 m | 0.98 / 0.52 m |
| prior 10% too large | 1.3° | 0.57 | 34 / 50 | 0.83 m | 0.97 / 0.54 m |

| Box | L-shape IoU | + size prior |
|---|---:|---:|
| car, 20° | 0.55 | 0.70 |
| car on the ramp, -15° | 0.61 | 0.82 |
| car, 35° | 0.52 | **0.92** |
| van on the ramp, 10° | 0.44 | 0.51 |
| one face visible (11 observations) | 0.41 | 0.68 (centre error 0.85 → 0.29 m) |
| two faces visible (39 observations) | 0.49 | 0.58 |

- **Best case.** The scene's cars match the car prior exactly. With a prior 10% off in both sides, the mean IoU still rises from 0.47 to 0.57-0.58.
- **The end-view rule matters.** Taking the longer observed side as the length every time gives a mean IoU of 0.53. A car seen end-on then grows sideways.
- **The height-based class is the weak link.**
  - 17 of the 23 car observations are classed as cars.
  - Five misses have an observed footprint wider than 1.2× the prior, and one car is mostly hidden (0.5 m of it visible).
  - The van is never classed as a van. Its roof is above the scan's +2° upper beam, so only 1.1-2.1 m of it is seen. It gets the car prior 4 times out of 7, which still helps it, but it stays short of its true length.
  - The wall, crate and bench get no class and keep their L-shape boxes.

## Held-out check of the box fitting

The L-shape fitting and the size prior were designed on the scene above.
`--seed N` (N > 0) draws a held-out scene:
- every box except the wall moves within 1 m and takes a new heading in [0°, 180°);
- the cars take sizes of 4.0-5.0 × 1.7-1.9 m and the van 5.5-6.5 × 1.9-2.1 m, while the prior stays 4.5 × 1.8 and 6.0 × 2.0 m;
- the sensor takes 8 new poses on the road.

`scripts/box_fitting_heldout.py` runs seeds 1-20 and compares the methods per observation with exact sign tests. Observations of one seed share objects, so it also compares the per-seed means across the 20 seeds. Full report: [results/box_fitting_heldout_2026-10-05.md](results/box_fitting_heldout_2026-10-05.md).

| Held-out, 996 observations | heading error | BEV IoU | IoU ≥ 0.5 | centre error |
|---|---:|---:|---:|---:|
| axis-aligned | 18.6° | 0.376 | 237 | 1.09 m |
| L-shape | 3.4° (median 0.6°) | 0.448 | 450 | 0.99 m |
| L-shape + size prior | 3.4° | **0.565** | **612** | **0.77 m** |
| prior 10% too small / too large | 3.4° | 0.552 / 0.552 | 596 / 617 | 0.80 / 0.78 m |
| prior without the end-view rule | 3.4° | 0.522 | 540 | 0.90 m |

| Comparison (held-out) | seeds better / worse |
|---|---:|
| L-shape vs axis-aligned, heading error | 20 / 0 |
| L-shape vs axis-aligned, IoU | 19 / 1 |
| size prior vs L-shape, IoU and centre error | 20 / 0 each |
| size prior 10% off vs L-shape, IoU | 20 / 0 (both directions) |
| end-view rule vs none, IoU | 19 / 1 |

- **Every gain replicates.** The sizes of the gains are close to the dev scene's: +0.12 IoU from the prior (dev +0.13), +0.04 from the end-view rule (dev +0.08).
- **The heading error has a tail.** The median is 0.6°, but 10% of the observations are off by more than 10°. These are mostly the 2 × 1 m bench and the car nearest the sensor poses.
- **The L-shape's IoU gain over the axis-aligned box is modest per observation** (498 better, 422 worse). The heading gain is not (755 vs 165).
- **The height-based class is still the weak link.**
  - 346 of 472 car observations are classed as cars.
  - Of 148 van observations, 8 are classed as vans and 65 as cars.
  - The learned class below replaces this rule.

## Learned class

> This section describes the first learned class, trained at 1.8 m on random scenes. The class in use replaced it: it is trained on drives too and across sensor heights (see "Sensor height and the class"). The numbers in this section and in the tracking sections below were measured with the first class.

A small classifier replaces the height rule. It takes six features of a cluster's L-shape box:
- the long side, the short side and the visible height;
- log(point count) and the range;
- the **top margin**: how far the highest point's elevation lies below the scan's +2° upper beam. A margin near 0 means the top may be cut off, as for the van.

**Training.** `scripts/train_box_classifier.py` runs training scenes, seeds 101-200, which are disjoint from the evaluated seed 0 and seeds 1-20. That gives 25017 clusters: 2358 cars, 753 vans and 21906 others. A cluster is labelled car or van if it is the found cluster of a car or the van, and none otherwise. The script trains a 6-16-3 tanh MLP (numpy, full-batch Adam, fixed seed) and writes the weights to `include/lidar_box_classifier.h`. The host classifies each cluster.

**Classification, held-out seeds 1-20** (5089 clusters):

| | accuracy | cars classed car (of 472) | vans classed van (of 148) | others classed car or van (of 4469) |
|---|---:|---:|---:|---:|
| height rule | 0.925 | 346 | 8 | 114 |
| learned | **0.987** | **446** | **134** | **25** |

**Boxes** (L-shape + size prior):

| Class | dev IoU | held-out IoU | held-out IoU ≥ 0.5 | held-out centre error |
|---|---:|---:|---:|---:|
| height rule | 0.600 | 0.565 | 612 | 0.77 m |
| learned | **0.622** | **0.585** | **631** | **0.75 m** |

The learned class is better in 16 of 20 held-out seeds (sign test p = 0.012). Per observation it is better in 130 and worse in 72.

- **The box gain is smaller than the classification gain.** Most of it comes from the van.
  - Van observations the rule left unclassified gain 0.42 IoU on average (50 observations).
  - Those it called cars gain 0.16 (45 observations).
- **Losses.**
  - The rule had called the 4 × 3 m crate a car, and the car prior happened to fit it. Classing it correctly as none loses 0.32 IoU on 8 observations.
  - The 15 car and van observations that the rule classed but the learned class leaves unclassified lose 0.4-0.7 on average.
- **Display.** The GIF's completed boxes now use the learned class.

## Tracking along a drive

A single scan sees only the near faces of an object. A sensor that drives past
sees the others in turn. `--sequence` drives the sensor along the road: y = 0.5 m,
x = -12 to 24 m, 37 scans 1 m apart (10 m/s at 10 Hz). The tracker works in the
world frame, because the objects are static and the sensor pose is known.

- **Association.** A cluster that the learned class calls a car or a van joins the track whose box lies within 1 m of the cluster's box (rectangle-to-rectangle distance). Closest pairs go first, with one cluster per track. Otherwise the cluster starts a new track.
- **Accumulation.** A track keeps its clusters' world points on a 0.1 m voxel grid and counts the scans each voxel was seen in. It refits its L-shape box to the voxels seen in at least K = 3 scans (fewer while the track is young). The track's class is the majority of its clusters' learned classes, and the size prior completes its box as before.

**What did not work.**
- **Plain union of the points** (K = 1). Every scan leaves a few ground points beside an object, and their union keeps growing. The tracked box swelled to 16 m for a car, and the mean IoU fell below the single-scan box.
- **Gating on the box of all voxels.** That box includes the stray points, and a track grew large enough to absorb a second car. The rectangle-to-rectangle gate on the filtered box gives no identity switches on the dev drive.

**Results.** `scripts/box_tracking_eval.py` runs the dev drive (seed 0) and held-out drives (seeds 1-20; the boxes keep 1 m clear of the road). Only the observations of the cars and the van count. Full report: [results/box_tracking_2026-10-05.md](results/box_tracking_2026-10-05.md).

| Held-out, 2906 vehicle observations | BEV IoU | IoU ≥ 0.5 | centre error | heading error |
|---|---:|---:|---:|---:|
| single-scan L-shape | 0.471 | 1475 | 0.95 m | 3.7° |
| single-scan L-shape + size prior | 0.716 | 2418 | 0.47 m | 3.7° |
| tracked L-shape | 0.562 | 1977 | 0.66 m | 3.2° |
| tracked L-shape + size prior | **0.757** | **2689** | **0.29 m** | 3.2° |

- **Tracking replicates.** With the size prior, the tracked box beats the single-scan box in IoU in 18 of 20 held-out seeds and in centre error in all 20. Without the prior, the tracked box beats the single-scan L-shape in 19 of 20.
- **The prior is still needed.** On a drive past, some faces stay hidden, and the tracked L-shape alone (0.56) stays below the single-scan box with the prior (0.72).
- **Few identity switches.** There are 4 over the 20 held-out drives and none on the dev drive.

| K | dev IoU (tracked + prior) | held-out IoU | held-out centre error | held-out identity switches |
|---:|---:|---:|---:|---:|
| 1 (plain union) | 0.433 | 0.450 | 1.04 m | 6 |
| 3 (default) | 0.839 | 0.757 | 0.29 m | 4 |
| 5 | 0.851 | 0.807 | 0.29 m | 4 |

K = 3 was set before the held-out drives ran. K = 5 does better on both the dev and the held-out drives and is a candidate for the default.

## Moving traffic

The tracker above assumes that nothing moves. `--moving` adds traffic to the drive: a lead car and a following car in the sensor's lane.
- They start 12 m ahead and 12 m behind.
- The lead car drives at 8-13 m/s and the following car at 7-12 m/s (11.5 and 9 m/s on the dev drive).
- The other vehicles stay parked.

A second tracker, the motion tracker, runs next to the static one:

- **Kalman filter.** A constant-velocity filter on (x, y, vx, vy) per track, with acceleration noise 2 m/s² and measurement noise 0.3 m. The measurement is the centre of the cluster's box completed with the size prior of the track's majority class. That centre depends less on the viewpoint than the box of the visible points.
- **Association.** A cluster may join a track whose predicted centre is within 2 m of the measurement, or whose predicted box is within 1 m of the cluster's box. Pairs go closest first by the box distance.
- **Accumulation in the object's frame.** A track keeps every scan's points with their time. A refit tries two hypotheses:
  - the object stands still;
  - the object moves with the filter's velocity (tried only above 1 m/s).

  Under each hypothesis, the points are moved to the current time and counted on the voxel grid. The hypothesis with more voxels seen in at least 3 scans wins. The filter alone cannot tell a parked car from a moving one: as the sensor passes a parked car, the visible part changes, and the measured centre drifts as if the car moved.

**Development notes.**
- Gating on the centre alone let the parked car beside the road split into new tracks: its measured centre jumps when the view turns from its rear to its side.
- Gating on the box alone gave the lead car's cluster to a parked car's track, as it passed 5 cm beside that car.
- Without the stand-still test, a parked car's drifting velocity smeared its accumulated points.

**Results.** `scripts/box_motion_eval.py` runs the dev drive and held-out drives (seeds 1-20). All boxes are completed with the size prior. Full report: [results/box_motion_2026-10-05.md](results/box_motion_2026-10-05.md).

| Held-out drives | observations | single scan | static tracker | motion tracker |
|---|---:|---:|---:|---:|
| moving vehicles, BEV IoU | 1426 | **0.729** | 0.382 | 0.686 |
| moving vehicles, centre error | 1426 | **0.52 m** | 3.22 m | 0.71 m |
| parked vehicles, BEV IoU | 2897 | 0.701 | 0.742 | **0.745** |
| identity switches (all vehicles) | | — | 253 | **26** |
| velocity error, moving vehicles | 1271 | — | 9.82 m/s | **0.69 m/s** |

- **The motion tracker fixes what the static tracker breaks on moving traffic.**
  - IoU rises by 0.30 and centre error drops by 2.5 m, in 19 of 20 seeds.
  - Identity switches drop from 253 to 26.
  - It estimates the traffic's velocity to 0.7 m/s.
- **On the parked vehicles it matches the static tracker** (12 vs 8 seeds, p = 0.5). The stand-still test keeps their drifting velocity (error 0.65 m/s) out of the boxes. Both trackers beat the single scan there.
- **On the moving vehicles it does not beat the single scan.** IoU is 0.04 lower and centre error 0.19 m higher; the single scan is better in 18 of 20 seeds. Traffic in the sensor's lane shows only its rear or its front, so the scans add no new faces. Each scan already completes the box from the visible face, while the track lags and smears by its velocity error.
  - I expected the motion tracker to beat the single scan here and set the gate that way before the held-out drives ran. The gate now checks only what held.
  - A tracker that outputs the single-scan box for tracks judged moving and its own box for the others would combine the two. That rule comes from these held-out drives, so the next section tests it on fresh drives.

## Hybrid boxes, tested on fresh drives

The **hybrid box** combines the two:
- for a track the stand-still test calls moving, it is the single-scan box (with the size prior);
- for every other track, it is the motion tracker's box.

The rule was read off seeds 1-20, so `scripts/box_hybrid_eval.py` tests it on **fresh drives, seeds 21-40**, which no earlier step ran. The criteria were fixed in the script before those drives ran:
1. moving vehicles: hybrid ≥ single scan − 0.01 IoU;
2. parked vehicles: hybrid ≥ motion tracker − 0.01;
3. all vehicles: hybrid above both.

Full report: [results/box_hybrid_2026-10-06.md](results/box_hybrid_2026-10-06.md).

| Fresh drives (seeds 21-40), BEV IoU | observations | single scan | motion tracker | hybrid |
|---|---:|---:|---:|---:|
| moving vehicles | 1453 | **0.751** | 0.729 | 0.743 |
| parked vehicles | 2902 | 0.688 | 0.726 | **0.728** |
| all vehicles | 4355 | 0.709 | 0.727 | **0.733** |
| all vehicles, centre error | 4355 | 0.52 m | 0.46 m | **0.44 m** |

- **All three criteria hold on the fresh drives.**
  - Over all vehicles, the hybrid beats the single scan by 0.024 IoU (18 of 20 seeds).
  - It beats the motion tracker by 0.006 IoU (14 of 20 seeds, p = 0.12, not significant per seed).
- **On the moving vehicles it still trails the single scan a little:** −0.008 IoU, lower in 19 of 20 seeds. A young track is not yet called moving, so it gets the track's box for its first scans. On seeds 1-20, where the rule came from, the gap was −0.011, which misses criterion 1.

## Pipeline time per scan

`--sequence` and `--moving` print the time of each stage per scan. The pipeline is ground segmentation, voxel clustering, L-shape boxes, the learned class, and the motion tracker with its box refits. On the `--moving` drive (37 scans; GPU stages are the minimum of 5 runs, the host stages one run):

| Stage | before | after | change |
|---|---:|---:|---|
| segmentation (GPU) | 0.85 / 1.1 ms | 0.79 / 1.0 ms | |
| voxel clustering (GPU) | 7.7 / 36.7 ms | 1.03 / 1.9 ms | one warp per voxel pair |
| L-shape boxes (GPU) | 2.4 / 4.1 ms | 2.2 / 3.9 ms | |
| classification (host) | 0.23 / 0.38 ms | 0.17 / 0.23 ms | |
| motion tracker (host) | 24.9 / 48.4 ms | 1.9 / 5.3 ms | box refits moved to the GPU; stand-still grid kept up to date |
| motion tracker's box refits (GPU) | (in the line above) | 0.58 / 0.89 ms | |
| **pipeline** | **36.1 / 78.4 ms** | **6.6-6.7 / 10.9-11.3 ms** | |

Each cell is the mean / max over the scans, and "after" is from two runs.

- **Tracker refits on the GPU.** Each tracker now collects the point sets of its refits and fits them in one batch, with the same kernels as the clusters (one warp per set and heading). Before, 85% of the tracker's time went to these L-shape fits on the CPU. With `--check`, the CPU fits them too: the boxes are bit-identical, and the gates require it.
- **Stand-still grid.** The motion tracker's stand-still hypothesis keeps its voxel grid up to date as points arrive instead of recounting all of them each scan. A zero velocity leaves the points where they are, so the grid and the boxes are the same, and the output files are byte-identical.
- **Budget.** The worst scan now takes 11 ms of a 10 Hz LiDAR's 100 ms. The remaining host work is the moving hypothesis's recount and the trackers' voxel grids.

## Library

The production path is also a library: `CudaRobotics::lidar_objects_gpu`, with the interface in `include/cudarobotics/lidar_objects_gpu.hpp` (no CUDA headers needed). It takes one call per scan:

```cpp
#include "cudarobotics/lidar_objects_gpu.hpp"

cudarobotics::LidarObjectPipeline pipe;   // up to 131072 points per scan
cudarobotics::LidarObjectsResult r = pipe.process(xyz, n, sensor_x, sensor_y, sensor_z, t);
// r.ground[i], r.cluster[i]   per point: ground flag, cluster label (-1 for ground)
// r.clusters                   per cluster of >= 10 points: L-shape box, learned class, size-prior box,
//                              and that box refined with the free space
// r.tracks                     per track this scan updated: hybrid box, class, velocity, moving flag
```

- **Steps.**
  - GPU: ground segmentation → voxel clustering → L-shape boxes → (after the class) free-space refinement.
  - Host: learned class and size prior → motion tracker, whose box refits are batched on the GPU → hybrid box.
- **Input.** Points in the sensor's frame, z up. Each scan comes with the sensor's world position, its heading (`sensor_yaw`) and the scan time.
  - The pipeline rotates the points by the heading into a world-aligned frame and rotates the boxes back.
  - The tracks' velocities are in the world frame.
- **Sensor settings** (`LidarObjectsConfig`):
  - `sensor_height` (1.8 m): the ground model starts from flat ground this far below the sensor.
  - `upper_beam_deg` (+2°): the class's top-margin feature.
  - `free_space_refinement` (on): refine the completed boxes (see "Free-space refinement").

  With the defaults, the demo's outputs are byte-identical to before.
- **Tested setups.** `tests/lidar_objects_gpu_smoke.cu` drives a sensor past a car at 1.8 m / 0°, 2.2 m / 30°, 1.5 m / −60° and 0.8 m / 45°. In every case the ground, the car's cluster, its heading in the sensor's frame and its track come out right. How well the class does at each height is measured on held-out scenes, in the next section.
- **The height setting matters.** The same 0.8 m scans with the default 1.8 m label 98.2% of the ground instead of 99.96%, and the test checks this.
- **Code layout.** The algorithms live in `include/cudarobotics/lidar_objects_core.cuh`. It is header-only with internal linkage, so the library and this demo share it.
- **Checks.**
  - On every scan, the demo runs the library on the scan's returns. It checks that the ground labels, the clusters, the boxes and, on the drives, the tracks are exactly its own; every gate requires this.
  - `tests/lidar_objects_gpu_smoke.cu` uses only the public interface: flat ground and a car passed by the sensor (CTest `lidar_objects_gpu_smoke`).
- **Limitations.**
  - A level sensor is assumed: roll and pitch are not modelled.
  - Calls run on the default CUDA stream and are not thread-safe.

## Sensor height and the class

The first class was trained on scans from 1.8 m. From lower sensors it fails: the roof of a car reaches the upper beam, as a van's does from 1.8 m. `--sensor-height H` scans from other heights, and `scripts/box_class_heights_eval.py` measures the class on held-out scenes (seeds 1-20) at 0.8, 1.2, 1.8 and 2.5 m.

A second class, the **height-aware** one, adds the sensor's height as a seventh feature. It was trained on the same training seeds 101-200, with each scene's height drawn from 0.8-2.5 m (`train_box_classifier.py --drive-seeds ""` rebuilds it). Full report: [results/box_class_heights_2026-10-06.md](results/box_class_heights_2026-10-06.md).

| Height | cars classed car | vans classed van | others classed car or van | box IoU with the prior |
|---:|---:|---:|---:|---:|
| 0.8 m | 61 → **353** / 407 | 36 → 48 / 80 | 294 → 101 / 2330 | 0.461 → **0.566** (20 / 20 seeds) |
| 1.2 m | 211 → **415** / 450 | 54 → 64 / 107 | 140 → 44 / 2894 | 0.553 → **0.608** (19 / 20) |
| 1.8 m | 446 → 450 / 472 | 134 → 126 / 148 | 25 → 28 / 4469 | 0.720 → 0.715 (8 / 8) |
| 2.5 m | 470 → 458 / 476 | 96 → **150** / 160 | 29 → 17 / 6171 | 0.774 → 0.776 (10 / 10) |

Each cell is first class → height-aware.

**At 1.8 m it loses on the drives,** although per cluster it looks the same. Rerunning the evaluations above with the height-aware class (`results/*_v2_2026-10-06.md`) gives, on the held-out seeds:

| At 1.8 m, boxes with the prior | first class | height-aware | seeds better / worse |
|---|---:|---:|---:|
| single scan, held-out scenes | 0.585 | 0.581 | 7 / 9 |
| tracked along a drive | **0.757** | 0.739 | 2 / 18 (p = 0.0004) |
| motion tracker, moving traffic (all vehicles) | 0.725 | 0.714 | 9 / 11 |
| hybrid, fresh drives (all vehicles) | 0.733 | 0.725 | |

- **Why the drives differ.** The scenes the class is measured and trained on have no lane traffic seen only from its rear. On the drives, the height-aware class calls more of those scans vans or nothing.
- **More data did not fix it.** Training on 200 scenes (seeds 101-300) made it worse on the drives: the motion tracker fell from 0.725 to 0.684, and the single-scan boxes on the drives from 0.711 to 0.676, both lower in 20 of 20 seeds.
- **So the height-aware class did not become the default.** For a while it was opt-in (`height_aware_class`).
- **Lesson.** A class that matches per cluster can still change what the trackers downstream do. The per-cluster report alone would have passed it.

### Training on drives too: the class in use

The diagnosis points at the training data, so the class in use is trained on drives too.
- **Training data.** The same random scenes (seeds 101-200) plus the `--moving` drives of seeds 101-200. Every scene's sensor height is drawn from 0.8-2.5 m. That gives 146482 clusters.
- **Features.** The same seven as the height-aware class.
- **Test.** It was tested on **fresh seeds 41-60**, which no earlier step ran. The criteria were fixed in `scripts/box_class_drives_eval.py` before the run:
  - on the 1.8 m drives, every box metric drops by at most 0.005 IoU against the first class, with no significant per-seed loss;
  - at 0.8 and 1.2 m, it stays within 0.01 IoU of the height-aware class.

| 1.8 m drives (seeds 41-60), boxes with the prior | first class | class in use | seeds better / worse |
|---|---:|---:|---:|
| tracked along a drive | 0.754 | **0.764** | 13 / 7 |
| motion tracker, moving traffic | 0.729 | **0.757** | 15 / 5 (p = 0.04) |
| single scan, moving traffic | 0.720 | **0.759** | 19 / 1 |
| hybrid, moving traffic | 0.741 | **0.779** | 18 / 2 |

| Random scenes (seeds 41-60) | cars classed car (first → in use) | box IoU with the prior (first → height-aware → in use) |
|---:|---:|---:|
| 0.8 m | 57 → **362** / 409 | 0.477 → 0.568 → 0.566 |
| 1.2 m | 204 → **418** / 449 | 0.533 → 0.594 → 0.602 |
| 1.8 m | 452 → 453 / 472 | 0.708 → 0.704 → 0.719 |
| 2.5 m | 469 → 472 / 477 (vans 89 → 153 / 160) | 0.762 → 0.782 → 0.787 |

- **Both criteria hold.** On the drives it does not just hold the line: every box metric improves, with the single-scan and hybrid boxes better in 19 and 18 of 20 seeds.
- **Cost.** At 1.8 m on the random scenes, 54 instead of 29 of 4424 other clusters are classed as vehicles.
- **One class for every height.** It replaces both earlier classes, and the `height_aware_class` option is gone.
- **Reports:** [results/box_class_drives_2026-10-06.md](results/box_class_drives_2026-10-06.md), [results/box_class_heights_drivedata_2026-10-06.md](results/box_class_heights_drivedata_2026-10-06.md).

## Free-space refinement

The size prior fills in a box's hidden extent, but it does not look at the scan. Yet a ray that passed somewhere proves that place empty. A box completed into space that rays crossed is too big there, or in the wrong place.

**Free-space grid.**
- The scan's rays are marched over a 0.1 m bird's-eye grid around the sensor, one GPU thread per ray.
- Each cell keeps the lowest height at which a ray crossed it, up to 0.3 m before the ray's end, the surface it hit (`atomicMin`).
- A box reaches the ground and rises to the top of its cluster. It contradicts every cell under it that a ray crossed below that top.

**Search.** Around each completed box of a car or van, about 31,000 candidates vary:
- the position by ±1 m along it and ±0.5 m across it, in 0.1 m steps;
- the heading by ±4°;
- the length by ±10% and the width by ±10%.

One GPU thread scores each (box, candidate). The cost is:
- w_free × the free area inside the box (m²);
- \+ w_out × the mean distance of the cluster's points outside it (m);
- \+ w_size × the squared relative change of length and width.

The cheapest candidate wins. The CPU twin (`fs_grid_cpu`, `fs_refine_cpu`) runs the same arithmetic. With `--check`, the grid and the first box of two scans must be bit-identical to the GPU's.

**Weights.** They were chosen on the dev scene and the training seeds 101-110 only (`scripts/box_freespace_eval.py tune`).
- Every setting tried improved the boxes there.
- w_free = 1, w_out = 10, w_size = 3 did best: IoU 0.724 → 0.776.
- Without the size term (w_size = 0) it was worse: 0.764.

**Test on fresh seeds 61-80.** No earlier step ran these. The criterion was fixed in the script: the IoU rises, it is better in significantly more seeds, and the centre error does not rise. Report: [results/box_freespace_2026-10-06.md](results/box_freespace_2026-10-06.md).

| Vehicles' boxes, seeds 61-80 (618) | BEV IoU | IoU ≥ 0.5 | centre error |
|---|---:|---:|---:|
| L-shape + size prior | 0.703 | 509 | 0.49 m |
| **+ free-space refinement** | **0.758** | **544** | **0.38 m** |

- **The criterion holds.** The refinement is better in 20 of 20 seeds (per observation, 381 better and 121 worse).
- **Cost.** About 1 ms per scan for the grid and about 2 ms per refined box: 10 ms per scan on the dev scene, more than the rest of the pipeline together. The search runs one thread per candidate with serial loops over the footprint and the points. A warp per candidate, with the CPU replaying its reduction order as for the L-shape fits, would cut that.
- **Library.** It refines the completed boxes into `LidarCluster::refined` (`LidarObjectsConfig::free_space_refinement`, on by default). The tracks still use the completed boxes.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target gpu_ground_segmentation -j$(nproc)
./bin/gpu_ground_segmentation                       # also writes the GIF
./bin/gpu_ground_segmentation --check --no-video    # the CTest gate
python scripts/lidar_objects_summary.py             # the current-performance table (seeds 1-20)
python scripts/box_freespace_eval.py tune           # free-space weights on the dev scene and seeds 101-110
python scripts/box_freespace_eval.py test --weights 1,10,3   # the test on fresh seeds 61-80
python scripts/box_fitting_heldout.py               # held-out scenes, seeds 1-20
python scripts/train_box_classifier.py              # retrain the class (random scenes and drives, 0.8-2.5 m), then rebuild
python scripts/box_class_heights_eval.py run --tag A --seeds 41-60   # a class at 0.8 / 1.2 / 1.8 / 2.5 m
python scripts/box_class_drives_eval.py run --tag A                  # a class on the 1.8 m drives (seeds 41-60)
# ... rebuild with another class, run with --tag B, then: report --base A --new B (both scripts)
python scripts/train_box_classifier.py --eval-seeds 1-20   # its confusion on the held-out scenes
./bin/gpu_ground_segmentation --sequence            # tracking along a drive; writes the tracking GIF
python scripts/box_tracking_eval.py                 # tracking, dev and held-out drives, K = 1, 3, 5
./bin/gpu_ground_segmentation --moving              # moving traffic; writes the motion GIF
python scripts/box_motion_eval.py                   # moving traffic, dev and held-out drives
python scripts/box_hybrid_eval.py                   # hybrid boxes, fresh drives (seeds 21-40)
```

`--check` exits non-zero unless the model's F1 is at least 0.95, above the
height threshold's, CPU and GPU agree on at least 99.9% of the labels, at least
90% of the objects come out as one cluster, the CPU and GPU clusterings are
the same partition, the L-shape boxes beat the axis-aligned ones in mean heading
error and mean IoU, the CPU and GPU boxes are identical, and the size prior
raises the mean IoU and lowers the mean centre error of the L-shape boxes, with
either the height rule's class or the learned class, the free-space refinement
improves the mean IoU and centre error further, and the CPU and GPU refinements
are identical. CTest runs it as
`gpu_ground_segmentation_gate` (labels `gpu;pointcloud;ground`).

With `--sequence`, `--check` instead requires:
- F1 ≥ 0.95 and the CPU/GPU agreements;
- tracked L-shape boxes that beat the single-scan ones in IoU;
- tracked boxes with the size prior that beat the single-scan ones with the prior in IoU and centre error.

CTest runs it as `gpu_ground_segmentation_track_gate`.

With `--moving`, `--check` requires:
- F1 ≥ 0.95 and the CPU/GPU agreements;
- motion-tracker boxes (with the size prior) that beat the static tracker's on the moving vehicles in IoU and centre error, and match them on the parked ones (within 0.01 IoU);
- no more identity switches than the static tracker;
- a velocity error under 1.5 m/s on the moving vehicles;
- hybrid boxes that beat the single-scan and the motion tracker's boxes over all vehicles.

CTest runs it as `gpu_ground_segmentation_motion_gate`.

Generated files: `tmp/gpu_ground_segmentation.avi` and `gif/gpu_ground_segmentation.gif`; with `--sequence`, `tmp/gpu_ground_segmentation_track.avi` and `gif/gpu_ground_segmentation_track.gif`; with `--moving`, `tmp/gpu_ground_segmentation_motion.avi` and `gif/gpu_ground_segmentation_motion.gif`.

## Output

The GIF shows a bird's-eye view (40 m × 40 m) of each scan: the height threshold on the left, the model on the right. The right panel also shows the true boxes (blue), the L-shape boxes of the clusters (magenta) and, for clusters the learned class calls cars or vans, the boxes completed with the size prior (orange).

| Colour | Meaning |
|---|---|
| green | ground, labelled ground |
| white | object, labelled object |
| red | object labelled ground |
| yellow | ground missed |

The tracking GIF (`--sequence`) shows the model's view along the drive with the true boxes (blue), the L-shape boxes (magenta), the single-scan boxes with the size prior (orange) and the tracked boxes with the size prior (green). The motion GIF (`--moving`) shows the motion tracker's boxes in green.
