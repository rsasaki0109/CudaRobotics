#!/usr/bin/env python3
"""Export a ROS 2 LiDAR bag with ground-truth poses for map-based localization.

Reads PointCloud2 scans and their paired geometry_msgs/PoseStamped reference
(the sensor's pose in the map frame) from a ROS 2 SQLite bag, voxel-downsamples
each scan in the sensor frame, and writes a binary sequence for
bin/benchmark_ndt_localization:

  magic "CRLOC1\\0\\0", uint32 version (1), uint32 frame count
  per frame: uint64 stamp_ns, 7 x float64 pose (x y z qx qy qz qw, map <- sensor),
             uint32 point count, count x 3 x float32 (sensor frame)

Scans are not deskewed.
"""

import argparse
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pointcloud2_clearance import parse_pointcloud2  # noqa: E402
from export_rosbag_motion import messages, parse_pose_stamped  # noqa: E402

MAGIC = b"CRLOC1\x00\x00"


def xyz(cloud):
    prefix = ">" if cloud["is_bigendian"] else "<"

    def values(name):
        field = cloud["fields"][name]
        return np.ndarray(shape=(cloud["height"], cloud["width"]), dtype=np.dtype(prefix + "f4"),
                          buffer=cloud["data"], offset=field["offset"],
                          strides=(cloud["row_step"], cloud["point_step"])).reshape(-1)

    return np.stack([values("x"), values("y"), values("z")], axis=1).astype(np.float32)


def voxel_downsample(points, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database", type=Path, required=True)
    ap.add_argument("--pointcloud-topic", default="/os_cloud_node/points")
    ap.add_argument("--pose-topic", default="/mcd/ground_truth/os_sensor_pose")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--max-pose-age-ms", type=float, default=50.0)
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th scan")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.database.as_posix()}?mode=ro", uri=True)
    poses = [parse_pose_stamped(bytes(d)) for _, d in messages(con, args.pose_topic)]
    stamps = np.array([p["stamp_ns"] for p in poses], dtype=np.int64)
    frames, counts, ages = 0, [], []
    with open(args.output, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", 1, 0))
        for k, (_, data) in enumerate(messages(con, args.pointcloud_topic)):
            if k % args.stride:
                continue
            cloud = parse_pointcloud2(bytes(data))
            stamp = cloud["stamp_ns"]
            j = int(np.argmin(np.abs(stamps - stamp)))
            age = abs(int(stamps[j]) - stamp) / 1e6
            if age > args.max_pose_age_ms:
                continue
            pts = xyz(cloud)
            r = np.linalg.norm(pts, axis=1)
            pts = pts[np.isfinite(r) & (r >= args.min_range) & (r <= args.max_range)]
            pts = voxel_downsample(pts, args.voxel)
            p = poses[j]
            out.write(struct.pack("<Q7dI", stamp, p["x"], p["y"], p["z"], p["qx"], p["qy"], p["qz"], p["qw"],
                                  len(pts)))
            out.write(pts.astype("<f4").tobytes())
            frames += 1
            counts.append(len(pts))
            ages.append(age)
        out.seek(len(MAGIC) + 4)
        out.write(struct.pack("<I", frames))
    report = {"database": str(args.database), "frames": frames, "voxel_m": args.voxel, "stride": args.stride,
              "mean_points": float(np.mean(counts)), "max_pose_age_ms": float(np.max(ages))}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
