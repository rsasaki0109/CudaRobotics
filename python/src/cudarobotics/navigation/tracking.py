"""Scan-based moving-object tracker for CudaNav.

Finds moving objects (people, carts) in a LiDAR scan by comparing it with the
map built so far, and tracks them with constant-velocity Kalman filters. With
``CudaNavPipeline(tracker=ScanObjectTracker())`` their points stay out of the
voxel map and CUDA MPPI avoids their predicted motion.

Per frame:

1. Candidates (:meth:`OccupancyAge.classify`): points whose cell has no
   long-standing structure (a cell occupied for ``static_frames`` frames) in
   its 3 x 3 block. Wall surfaces sit on cell borders, so the block rule also
   rejects the half of their returns that fall into the free cell in front.
   Free and never-observed cells both qualify: anything new is an object
   first. :meth:`ScanObjectTracker.update` adds points near a live track's
   prediction unless they touch structure, and points inside a confirmed
   track's own disc even next to a wall, so people brushing past walls stay
   tracked.
2. Clusters: candidates linked within ``cluster_distance`` (on distinct xy
   returns). Clusters wider than ``maximum_extent`` or hit by fewer than
   ``minimum_rays`` rays are dropped. A scan sees only an object's near side,
   so the centre is pushed away from the sensor by ``2 r / pi`` (the centroid
   offset of a visible half circle), ``r`` being half the cluster's width.
3. Association: greedy nearest neighbour inside ``gate``; unmatched clusters
   start tentative tracks.
4. Confirmation: after ``confirm_hits`` detections, once the track has moved
   ``confirm_displacement`` from where it was first seen. Wall fragments
   jitter in place as different rays hit them; a person walks away. A track
   is deleted after ``maximum_misses`` frames without a detection, or after
   ``commit_seconds`` supported only by points touching structure.
5. Map gating: every detection's points are in ``dynamic_mask`` and should
   be kept out of the map, because a moving object integrated even for a few
   frames leaves lethal traces. An object that does not move within
   ``commit_seconds`` is static: its points are released to the map, where
   they age into structure, and until then it is reported in
   ``static_objects`` so a planner never loses it. Confirmed tracks keep a
   ``history_seconds`` trail of positions for clearing earlier traces.

It is a 2-D, model-free tracker. A person standing still for
``commit_seconds`` before ever moving becomes map structure, and one who
stays in contact with a wall can merge into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    cluster_distance: float = 0.30      # [m] linking distance between candidate points
    minimum_rays: int = 2               # distinct xy returns a cluster needs
    maximum_extent: float = 1.0         # [m] wider clusters are treated as static
    gate: float = 0.8                   # [m] association and candidate distance to a prediction
    confirm_hits: int = 3               # detections before a track is reported
    commit_seconds: float = 2.0         # [s] a still object is released to the map after this
    confirm_displacement: float = 0.3   # [m] net motion since birth required for confirmation
    maximum_misses: int = 5             # frames without a detection before deletion
    acceleration_noise: float = 1.0     # [m/s^2] white-acceleration process noise
    measurement_noise: float = 0.05     # [m] centroid measurement noise
    minimum_radius: float = 0.2         # [m] reported radius floor
    radius_margin: float = 0.0          # [m] added to the cluster's spread
    history_seconds: float = 10.0       # [s] recent positions kept for map clearing
    static_frames: int = 20             # consecutive occupied frames that make a cell structure
    # Overlap the planner tolerates with a tracked disc (MppiParams
    # dynamic_obstacle_overlap_allowance): tracked positions are off by about
    # 0.1 m, which without it can place the robot inside a disc and leave no
    # valid rollout.
    planner_overlap_allowance: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            self.cluster_distance,
            self.maximum_extent,
            self.gate,
            self.acceleration_noise,
            self.measurement_noise,
            self.minimum_radius,
        )
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in positive)
            or self.minimum_rays < 1
            or self.commit_seconds < 0.0
            or self.confirm_displacement < 0.0
            or self.static_frames < 1
            or self.planner_overlap_allowance < 0.0
            or self.confirm_hits < 1
            or self.maximum_misses < 1
            or self.radius_margin < 0.0
            or self.history_seconds < 0.0
        ):
            raise ValueError("invalid tracker configuration")


@dataclass
class _Track:
    identifier: int
    state: np.ndarray            # x, y, vx, vy
    covariance: np.ndarray       # 4 x 4
    radius: float
    birth: tuple[float, float] = (0.0, 0.0)
    born: float = 0.0
    open_support: float = 0.0    # last time a detection had points away from structure
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)


@dataclass(frozen=True, eq=False)
class TrackerResult:
    tracks: np.ndarray           # (N, 5) confirmed moving tracks: x, y, vx, vy, radius
    track_ids: np.ndarray        # (N,) identifiers of the confirmed tracks
    static_objects: np.ndarray   # (M, 5) still objects not yet released to the map (v = 0)
    dynamic_mask: np.ndarray     # (P,) points to keep out of the map
    history: np.ndarray          # (H, 3) recent confirmed positions: x, y, radius
    clusters: int                # candidate clusters this frame
    tentative: int               # live tracks not yet confirmed


def _close_pairs(xy: np.ndarray, distance: float) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs ``i < j`` of points no farther apart than ``distance``.

    Points are hashed into square cells slightly larger than ``distance``, so
    only points in the same or a neighbouring cell are compared, instead of
    all pairs.
    """
    count = xy.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cell = np.floor(xy / np.float32(distance * 1.001)).astype(np.int64)
    cell -= cell.min(axis=0)
    span = int(cell[:, 1].max()) + 3
    key = (cell[:, 0] + 1) * span + (cell[:, 1] + 1)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    limit = np.float32(distance) * np.float32(distance)
    firsts, seconds = [], []
    for dx, dy in ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):  # each neighbour pair once
        target = key + dx * span + dy
        low = np.searchsorted(sorted_key, target, side="left")
        high = np.searchsorted(sorted_key, target, side="right")
        counts = high - low
        total = int(counts.sum())
        if total == 0:
            continue
        i = np.repeat(np.arange(count), counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        j = order[np.repeat(low, counts) + offsets]
        delta = xy[i] - xy[j]
        close = (delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]) <= limit
        keep = close & (i != j)
        firsts.append(np.minimum(i[keep], j[keep]))
        seconds.append(np.maximum(i[keep], j[keep]))
    if not firsts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(firsts), np.concatenate(seconds)


def _clusters(xy: np.ndarray, distance: float) -> list[np.ndarray]:
    """Connected components of points closer than ``distance`` (index arrays)."""
    count = xy.shape[0]
    if count == 0:
        return []
    first, second = _close_pairs(xy, distance)
    # Each point's label converges to the smallest index in its component:
    # propagate minima along the edges, then jump pointers, until stable.
    labels = np.arange(count, dtype=np.int64)
    while True:
        previous = labels.copy()
        np.minimum.at(labels, first, labels[second])
        np.minimum.at(labels, second, labels[first])
        labels = labels[labels]
        if np.array_equal(labels, previous):
            break
    # Components in order of their smallest index, members sorted, as a
    # depth-first search seeded in index order would produce.
    order = np.argsort(labels, kind="stable")
    roots, starts = np.unique(labels[order], return_index=True)
    return [np.sort(part) for part in np.split(order, starts[1:])]


class ScanObjectTracker:
    """Detect and track moving objects in world-frame scans (see module doc)."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = TrackerConfig() if config is None else config
        self._tracks: list[_Track] = []
        self._next_id = 0
        self._time: float | None = None

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0
        self._time = None

    @property
    def live_tracks(self) -> int:
        return len(self._tracks)

    def _predict(self, dt: float) -> None:
        q = self.config.acceleration_noise ** 2
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        noise = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ]
        )
        for track in self._tracks:
            track.state = transition @ track.state
            track.covariance = transition @ track.covariance @ transition.T + noise

    def _correct(self, track: _Track, measurement: np.ndarray) -> None:
        observation = np.zeros((2, 4))
        observation[0, 0] = observation[1, 1] = 1.0
        r = self.config.measurement_noise ** 2 * np.eye(2)
        innovation = measurement - observation @ track.state
        s = observation @ track.covariance @ observation.T + r
        gain = track.covariance @ observation.T @ np.linalg.inv(s)
        track.state = track.state + gain @ innovation
        track.covariance = (np.eye(4) - gain @ observation) @ track.covariance

    def _new_track(self, centre: np.ndarray, radius: float) -> _Track:
        born = 0.0 if self._time is None else self._time
        covariance = np.diag([0.05, 0.05, 1.0, 1.0]) ** 2
        covariance[2, 2] = covariance[3, 3] = 1.0  # unknown initial velocity
        track = _Track(
            identifier=self._next_id,
            state=np.array([centre[0], centre[1], 0.0, 0.0]),
            covariance=covariance,
            radius=radius,
            birth=(float(centre[0]), float(centre[1])),
            born=born,
            open_support=born,
        )
        self._next_id += 1
        return track

    def update(
        self,
        points_xy: Any,
        candidates: Any,
        time_s: float,
        structure: Any = None,
        sensor_xy: tuple[float, float] | None = None,
    ) -> TrackerResult:
        """Process one scan.

        ``points_xy`` are world-frame ``(P, 2)`` point positions,
        ``candidates`` marks the points that may belong to a moving object and
        ``structure`` the points touching long-standing structure (both from
        :meth:`OccupancyAge.classify`; ``structure`` defaults to none).
        ``sensor_xy`` is the scan origin, used to correct the near-side bias
        of cluster centres (no correction when omitted).
        """
        config = self.config
        xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        candidate = np.asarray(candidates, dtype=bool).reshape(-1).copy()
        on_structure = (
            np.zeros(xy.shape[0], dtype=bool)
            if structure is None
            else np.asarray(structure, dtype=bool).reshape(-1)
        )
        if candidate.shape[0] != xy.shape[0] or on_structure.shape[0] != xy.shape[0]:
            raise ValueError("candidates and structure must have one entry per point")
        dt = 0.0 if self._time is None else float(time_s) - self._time
        if dt < 0.0:
            raise ValueError("time must not go backwards")
        self._time = float(time_s)
        if dt > 0.0:
            self._predict(dt)

        predictions = np.array([track.state[:2] for track in self._tracks]).reshape(-1, 2)
        if predictions.shape[0] and xy.shape[0]:
            near = np.min(
                np.hypot(xy[:, None, 0] - predictions[None, :, 0], xy[:, None, 1] - predictions[None, :, 1]),
                axis=1,
            )
            candidate |= (near <= config.gate) & ~on_structure
            for track in self._tracks:
                if track.confirmed:
                    inside = np.hypot(xy[:, 0] - track.state[0], xy[:, 1] - track.state[1])
                    candidate |= inside <= track.radius + config.cluster_distance
        candidate_index = np.flatnonzero(candidate)

        # Points repeat once per height layer; cluster distinct xy returns.
        rounded = np.round(xy[candidate_index], 4)
        unique_xy, inverse = np.unique(rounded, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        detections: list[tuple[np.ndarray, float, np.ndarray, int]] = []
        for members in _clusters(unique_xy, config.cluster_distance):
            index = candidate_index[np.isin(inverse, members)]
            unique = unique_xy[members]
            extent = float(np.max(np.ptp(unique, axis=0))) if unique.shape[0] > 1 else 0.0
            if unique.shape[0] < config.minimum_rays or extent > config.maximum_extent:
                continue
            centre = unique.mean(axis=0)
            if sensor_xy is not None:
                away = centre - np.asarray(sensor_xy, dtype=np.float64)
                length = float(np.hypot(*away))
                if length > 1.0e-6:
                    half_width = max(extent / 2.0, config.minimum_radius)
                    centre = centre + away / length * (2.0 / math.pi) * half_width
            spread = float(np.max(np.hypot(*(unique - centre).T))) if unique.shape[0] > 1 else 0.0
            # A scan sees only the near side of an object, so the centroid is
            # biased towards the sensor; the reported radius covers that.
            radius = max(config.minimum_radius, spread + config.radius_margin)
            open_points = int(np.count_nonzero(~on_structure[index]))
            detections.append((centre, radius, index, open_points))

        # Greedy nearest-neighbour association inside the gate.
        pairs = sorted(
            (float(np.hypot(*(centre - track.state[:2]))), track_index, detection_index)
            for detection_index, (centre, _r, _i, _o) in enumerate(detections)
            for track_index, track in enumerate(self._tracks)
        )
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        owner: dict[int, _Track] = {}
        for distance, track_index, detection_index in pairs:
            if distance > config.gate:
                break
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)
            track = self._tracks[track_index]
            centre, radius, _index, open_points = detections[detection_index]
            self._correct(track, centre)
            track.radius = max(radius, 0.7 * track.radius + 0.3 * radius)
            track.hits += 1
            track.misses = 0
            if open_points > 0:
                track.open_support = self._time
            owner[detection_index] = track

        for track_index, track in enumerate(self._tracks):
            if track_index not in assigned_tracks:
                track.misses += 1
        self._tracks = [
            track
            for track in self._tracks
            if track.misses <= config.maximum_misses
            and self._time - track.open_support <= config.commit_seconds
        ]
        for detection_index, (centre, radius, _index, _open) in enumerate(detections):
            if detection_index not in assigned_detections:
                track = self._new_track(centre, radius)
                self._tracks.append(track)
                owner[detection_index] = track

        rows, identifiers, history, still = [], [], [], []
        for track in self._tracks:
            if not track.confirmed:
                moved = math.hypot(track.state[0] - track.birth[0], track.state[1] - track.birth[1])
                track.confirmed = track.hits >= config.confirm_hits and moved >= config.confirm_displacement
            if not track.confirmed:
                if track.hits >= config.confirm_hits and self._time - track.born < config.commit_seconds:
                    still.append((track.state[0], track.state[1], 0.0, 0.0, track.radius))
                continue
            if track.misses == 0:
                track.history.append((self._time, float(track.state[0]), float(track.state[1])))
            track.history = [
                entry for entry in track.history if self._time - entry[0] <= config.history_seconds
            ]
            rows.append((*track.state, track.radius))
            identifiers.append(track.identifier)
            history.extend((x, y, track.radius) for _t, x, y in track.history)
        dynamic = np.zeros(xy.shape[0], dtype=bool)
        for detection_index, (_centre, _radius, index, _open) in enumerate(detections):
            track = owner.get(detection_index)
            if track is None:
                continue
            if track.confirmed or self._time - track.born < config.commit_seconds:
                dynamic[index] = True
        return TrackerResult(
            tracks=np.asarray(rows, dtype=np.float32).reshape(-1, 5),
            track_ids=np.asarray(identifiers, dtype=np.int64),
            static_objects=np.asarray(still, dtype=np.float32).reshape(-1, 5),
            dynamic_mask=dynamic,
            history=np.asarray(history, dtype=np.float32).reshape(-1, 3),
            clusters=len(detections),
            tentative=sum(not track.confirmed for track in self._tracks),
        )


class OccupancyAge:
    """Consecutive frames each cell of a rolling occupancy grid has been occupied.

    Long-standing occupied cells are structure; a moving object occupies
    cells only briefly. The age grid follows the occupancy grid when it
    rolls with the robot.
    """

    def __init__(self, cap: int = 255) -> None:
        self.cap = int(cap)
        self.age: np.ndarray | None = None
        self.data: np.ndarray | None = None
        self.origin: tuple[float, float] | None = None
        self.resolution: float | None = None

    def update(self, grid: Any) -> None:
        data = np.asarray(grid.data)
        occupied = data == 100
        resolution = float(grid.resolution)
        origin = (float(grid.origin[0]), float(grid.origin[1]))
        if self.age is None or self.age.shape != data.shape or self.resolution != resolution:
            age = np.zeros(data.shape, dtype=np.int32)
        else:
            # Shift the previous ages by the grid's whole-cell origin change.
            dx = int(round((origin[0] - self.origin[0]) / resolution))
            dy = int(round((origin[1] - self.origin[1]) / resolution))
            age = np.zeros_like(self.age)
            height, width = data.shape
            if abs(dx) < width and abs(dy) < height:
                source = self.age[max(dy, 0) : height + min(dy, 0), max(dx, 0) : width + min(dx, 0)]
                age[max(-dy, 0) : height + min(-dy, 0), max(-dx, 0) : width + min(-dx, 0)] = source
        self.age = np.where(occupied, np.minimum(age + 1, self.cap), 0).astype(np.int32)
        self.data = data.astype(np.int8)
        self.origin = origin
        self.resolution = resolution

    def classify(self, points_xy: Any, minimum_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """``(candidates, structure)`` masks for world points.

        ``candidates``: the point is outside the grid or its cell has no
        long-standing structure (occupied for ``minimum_frames``) in its
        3 x 3 block. ``structure``: such structure is in the block.
        """
        xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if self.age is None:
            return np.ones(xy.shape[0], dtype=bool), np.zeros(xy.shape[0], dtype=bool)
        resolution = np.float32(self.resolution)
        column = np.floor((xy[:, 0] - np.float32(self.origin[0])) / resolution)
        row = np.floor((xy[:, 1] - np.float32(self.origin[1])) / resolution)
        height, width = self.age.shape
        inside = (column >= 0) & (row >= 0) & (column < width) & (row < height)
        r = row[inside].astype(np.int64)
        c = column[inside].astype(np.int64)
        result = np.zeros(xy.shape[0], dtype=bool)
        structure = self.age >= minimum_frames
        padded = np.pad(structure, 1)
        near_structure = np.zeros_like(structure)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                near_structure |= padded[dr : dr + height, dc : dc + width]
        # Free or unknown space away from structure: anything new is an object
        # first and only joins the map once it has stayed still.
        result[inside] = ~near_structure[r, c]
        result[~inside] = True
        touching = np.zeros(xy.shape[0], dtype=bool)
        touching[inside] = near_structure[r, c]
        return result, touching

